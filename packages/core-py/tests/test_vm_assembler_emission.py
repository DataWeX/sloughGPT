"""
Byte-exact emission tests for the X86Assembler in domains/shell/vm.py.

Covers every previously-uncovered branch of the opcode/encoding layer:
string ops, push/pop variants, in/out, condition-code jumps, jmp/call/far,
mov (regs, seg/CR/DR, memory, accumulator forms), ModR/M+SIB addressing,
the ALU table, inc/dec, unary ops, shifts, lea/xchg/lgdt/lidt/ltr, data
directives (db/dw/dd/times/equ/org), immediate parsing quirks, X86Assembler.run,
and the keyboard/scancode helpers.

All assertions pin the exact bytes the assembler currently emits, including
documented quirks:
  * ``adc``/``sbb`` are not in the opcode dispatch list -> NOP placeholder (0x90)
  * 16-bit CC-jump near fallback is off-by-one (target-3 vs target-2)
  * ``[bx]``/``inc [mem]`` encode as ``[disp32]`` with disp=0
  * ``xchg``/``ltr`` accept only specific register widths (silently emit nothing)
  * db items do not process backslash escapes (``db '\\n'`` -> 0x5C 0x6E)
  * expressions in immediates resolve to 0 (pass-2 eval path is unreachable)
"""

import pytest

import numpy as np  # imported first to avoid a coverage+numpy extension reload quirk

from domains.shell.vm import (
    X86Assembler,
    X86CPU,
    _parity,
    _char_to_scancode,
    _scancode_to_char,
    _default_kbd_handler,
)


def _hex(source, bits=None):
    if bits:
        source = f"[BITS {bits}]\n{source}"
    return X86Assembler().assemble(source).hex()


# ══════════════════════════════════════════════════════════════════════════════
# Prefix handling
# ══════════════════════════════════════════════════════════════════════════════

def test_operand_size_prefix_16bit_regs_in_32bit_mode():
    assert _hex("lodsw", 32) == "66ad"
    assert _hex("stosw", 32) == "66ab"
    assert _hex("movsw", 32) == "66a5"
    assert _hex("push ax", 32) == "6650"
    assert _hex("pop ax", 32) == "6658"


def test_no_prefix_for_16bit_regs_in_16bit_mode():
    assert _hex("lodsw") == "ad"
    assert _hex("stosw") == "ab"
    assert _hex("movsw") == "a5"
    assert _hex("push ax") == "50"
    assert _hex("pop ax") == "58"
    assert _hex("push cx") == "51"
    assert _hex("pop cx") == "59"


# ══════════════════════════════════════════════════════════════════════════════
# String operations
# ══════════════════════════════════════════════════════════════════════════════

def test_misc_one_byte_ops():
    assert _hex("cld") == "fc"
    assert _hex("std") == "fd"
    assert _hex("retf") == "cb"
    assert _hex("iret") == "cf"
    assert _hex("pusha") == "60"
    assert _hex("popa") == "61"
    assert _hex("pushad") == "60"
    assert _hex("popad") == "61"


def test_string_ops_byte():
    assert _hex("lodsb") == "ac"
    assert _hex("stosb") == "aa"
    assert _hex("movsb") == "a4"
    assert _hex("cmpsb") == "a6"
    assert _hex("scasb") == "ae"


def test_string_ops_word_16bit():
    assert _hex("lodsw") == "ad"
    assert _hex("stosw") == "ab"
    assert _hex("movsw") == "a5"


def test_rep_prefixes():
    assert _hex("rep movsb") == "f3a4"
    assert _hex("rep movsw") == "f3a5"
    assert _hex("rep stosb") == "f3aa"
    assert _hex("rep stosw") == "f3ab"
    assert _hex("rep lodsb") == "f3ac"
    assert _hex("rep lodsw") == "f3ad"
    assert _hex("rep cmpsb") == "f3a6"
    assert _hex("rep scasb") == "f3ae"
    assert _hex("rep movsw", 32) == "f366a5"
    assert _hex("rep lodsw", 32) == "f366ad"


def test_rep_unknown_target_emits_only_prefix():
    assert _hex("rep foo") == "f3"


# ══════════════════════════════════════════════════════════════════════════════
# Push / Pop
# ══════════════════════════════════════════════════════════════════════════════

def test_push_registers():
    assert _hex("push eax") == "6650"
    assert _hex("push ebx") == "6653"
    assert _hex("push ax") == "50"
    assert _hex("push cx") == "51"
    assert _hex("push eax", 32) == "50"
    assert _hex("push ebx", 32) == "53"


def test_push_segment_registers():
    assert _hex("push ds") == "1e"
    assert _hex("push es") == "06"
    assert _hex("push cs") == "0e"
    assert _hex("push ss") == "16"
    assert _hex("push fs") == "26"
    assert _hex("push gs") == "2e"


def test_push_immediates():
    assert _hex("push 5") == "6a05"
    assert _hex("push 200") == "68c800"
    assert _hex("push 0x7FFF") == "68ff7f"
    assert _hex("push 0x12345678") == "687856"
    assert _hex("push 0x7FFF", 32) == "68ff7f0000"
    assert _hex("push 0x12345678", 32) == "6878563412"


def test_push_no_operands_emits_nothing():
    assert _hex("push") == ""


def test_pop_registers_and_segments():
    assert _hex("pop eax") == "6658"
    assert _hex("pop ax") == "58"
    assert _hex("pop eax", 32) == "58"
    assert _hex("pop ds") == "1f"
    assert _hex("pop es") == "07"
    assert _hex("pop fs") == "27"
    assert _hex("pop gs") == "2f"


def test_pop_no_operands_emits_nothing():
    assert _hex("pop") == ""


# ══════════════════════════════════════════════════════════════════════════════
# int / in / out
# ══════════════════════════════════════════════════════════════════════════════

def test_int_immediate():
    assert _hex("int 0x10") == "cd10"
    assert _hex("int 33") == "cd21"


def test_in_forms():
    assert _hex("in al, dx") == "ec"
    assert _hex("in ax, dx") == "ed"
    assert _hex("in eax, dx") == "ed"
    assert _hex("in al, 0x60") == "e460"
    assert _hex("in ax, 0x60") == "e560"
    assert _hex("in eax, 0x60") == "e560"


def test_out_forms():
    assert _hex("out dx, al") == "ee"
    assert _hex("out dx, ax") == "ef"
    assert _hex("out dx, eax") == "ef"
    assert _hex("out 0x20, al") == "e620"
    assert _hex("out 0x20, ax") == "e720"
    assert _hex("out 0x20, eax") == "e720"


# ══════════════════════════════════════════════════════════════════════════════
# Jumps / calls
# ══════════════════════════════════════════════════════════════════════════════

def test_cc_jumps_short():
    assert _hex("jz 0") == "74fe"
    assert _hex("jnz 0") == "75fe"
    assert _hex("js 0") == "78fe"
    assert _hex("jns 0") == "79fe"
    assert _hex("jo 0") == "70fe"
    assert _hex("jno 0") == "71fe"
    assert _hex("jp 0") == "7afe"
    assert _hex("jnp 0") == "7bfe"
    assert _hex("jg 0") == "7ffe"
    assert _hex("jge 0") == "7dfe"
    assert _hex("jl 0") == "7cfe"
    assert _hex("jle 0") == "7efe"
    assert _hex("ja 0") == "77fe"
    assert _hex("jb 0") == "72fe"
    assert _hex("jae 0") == "73fe"
    assert _hex("jbe 0") == "76fe"


def test_loop_and_jcxz():
    assert _hex("loop 0") == "e2fe"
    assert _hex("loope 0") == "e1fe"
    assert _hex("loopne 0") == "e0fe"
    assert _hex("jcxz 0") == "e3fe"
    assert _hex("loopz 0") == "e1fe"
    assert _hex("loopnz 0") == "e0fe"


def test_cc_jump_16bit_far_fallback_is_off_by_one():
    # Documented quirk: near_offset computed after EB is appended -> target-3.
    assert _hex("jz 0x100") == "ebfd"


def test_cc_jump_32bit_near():
    assert _hex("jz 0x100", 32) == "0f84fa000000"
    assert _hex("jnz 0x100", 32) == "0f85fa000000"


def test_loop_overflow_falls_back_to_jmp():
    assert _hex("loop 0x100") == "ebfd"


def test_jmp_near_16bit():
    assert _hex("jmp 0x100") == "e9fd00"
    assert _hex("jmp 0") == "e9fdff"


def test_jmp_32bit_short_and_near():
    assert _hex("jmp 0", 32) == "ebfe"
    assert _hex("jmp 0x100", 32) == "e9fb000000"


def test_far_jump():
    assert _hex("jmp 0x08:0x12345678") == "66ea785634120800"
    assert _hex("jmp 0x08:0x12345678", 32) == "ea785634120800"


def test_call():
    assert _hex("call 0x100") == "e8fd00"
    assert _hex("call 0x100", 32) == "e8fb000000"


def test_forward_label_resolution():
    assert _hex("jmp done\nnop\nnop\nnop\nnop\ndone:\n hlt") == "e9040090909090f4"
    assert _hex("jz done\nnop\nnop\nnop\nnop\ndone:\n hlt") == "740490909090f4"
    assert _hex("lgdt [gdt]\ngdt: dw 0") == "670f0115080000000000"
    assert _hex("lidt [idt]\nidt: dw 0") == "670f011d080000000000"


# ══════════════════════════════════════════════════════════════════════════════
# MOV
# ══════════════════════════════════════════════════════════════════════════════

def test_mov_reg_imm():
    assert _hex("mov al, 0x7F") == "b07f"
    assert _hex("mov ch, 5") == "b505"
    assert _hex("mov ax, 0x1234") == "b83412"
    assert _hex("mov cx, 0x1234") == "b93412"
    assert _hex("mov edx, 0x12345678") == "66ba78563412"
    assert _hex("mov eax, 0x12345678") == "66b878563412"
    assert _hex("mov eax, 0x12345678", 32) == "b878563412"


def test_mov_reg_reg():
    assert _hex("mov al, bl") == "88d8"
    assert _hex("mov dl, cl") == "88ca"
    assert _hex("mov ax, bx") == "89d8"
    assert _hex("mov eax, ebx") == "6689d8"
    assert _hex("mov eax, ebx", 32) == "89d8"


def test_mov_segment_registers():
    assert _hex("mov ds, ax") == "8ed8"
    assert _hex("mov es, bx") == "8ec3"
    assert _hex("mov ss, cx") == "8ed1"
    assert _hex("mov fs, dx") == "8ee2"
    assert _hex("mov gs, di") == "8eef"
    assert _hex("mov ax, ds") == "8cd8"


def test_mov_control_and_debug_registers():
    assert _hex("mov eax, cr0") == "0f20c0"
    assert _hex("mov eax, cr4") == "0f20e0"
    assert _hex("mov cr0, eax") == "0f22c0"
    assert _hex("mov cr3, eax") == "0f22d8"
    assert _hex("mov eax, dr2") == "0f21d0"
    assert _hex("mov eax, dr0") == "0f21c0"
    assert _hex("mov dr1, eax") == "0f23c8"
    assert _hex("mov dr7, eax") == "0f23f8"


def test_mov_byte_memory():
    assert _hex("mov byte [eax], bl") == "8818"
    assert _hex("mov [eax], bl") == "8818"
    assert _hex("mov [ebx], al") == "8803"
    assert _hex("mov al, [eax]") == "8a00"
    assert _hex("mov al, [ebx]") == "8a03"
    assert _hex("mov al, [ebx+2]") == "8a4302"


def test_mov_word_dword_memory():
    assert _hex("mov word [eax], bx") == "8900"
    assert _hex("mov dword [ebx+4], eax") == "66894304"
    assert _hex("mov eax, [ebx]") == "668b03"
    assert _hex("mov eax, [ebx+4]") == "668b4304"
    assert _hex("mov eax, [ebx+0x12345678]") == "668b8378563412"
    assert _hex("mov [eax], ebx") == "668918"


def test_mov_byte_word_dword_imm_memory():
    assert _hex("mov byte [ebx+4], 0x7F") == "c643047f"
    assert _hex("mov byte [0x1234], 5") == "c6053412000005"
    assert _hex("mov word [ebx+4], 0x1234") == "66c743043412"
    assert _hex("mov word [0x1234], 5") == "66c705341200000500"
    assert _hex("mov dword [ebx+4], 0x12345678") == "c7430478563412"
    assert _hex("mov dword [0x1234], 5") == "c7053412000005000000"


def test_mov_accumulator_absolute_16bit():
    assert _hex("mov ax, [0x1234]") == "a13412"
    assert _hex("mov [0x1234], ax") == "a33412"


def test_mov_direct_address_32bit():
    assert _hex("mov eax, [0x1234]") == "668b0534120000"
    assert _hex("mov eax, [0x1234]", 32) == "8b0534120000"
    assert _hex("mov ax, [0x1234]", 32) == "a13412"


def test_mov_label_memory_forms():
    assert _hex("mov eax, [BUF]\nBUF equ 0x100") == "668b0500010000"
    assert _hex("mov eax, [BUF+ebx]\nBUF equ 0x100") == "668b8300010000"
    assert _hex("mov eax, [BUF+ebx]\nBUF equ 4") == "668b4304"
    assert _hex("mov eax, [label+esi*4]") == "668b04b500000000"
    assert _hex("mov eax, [label1+label2]") == "668b0500000000"


def test_mov_esp_special_sib():
    assert _hex("mov eax, [esp]") == "668b04"
    assert _hex("mov eax, [esp+4]") == "668b4404"
    assert _hex("mov eax, [esp+0x100]") == "668b8400010000"


def test_mov_indexed_addressing_drops_scale():
    # Documented quirk: [base+idx*scale] where base is a reg ignores the index.
    assert _hex("mov eax, [ebx+ecx*2]") == "668b03"
    assert _hex("mov eax, [ebx+esi*4+8]") == "668b03"
    assert _hex("mov eax, [esi+eax]") == "668b06"
    assert _hex("mov eax, [ecx*4]") == "668b0500000000"
    assert _hex("mov eax, [eax*8]") == "668b0500000000"


# ══════════════════════════════════════════════════════════════════════════════
# ALU
# ══════════════════════════════════════════════════════════════════════════════

def test_alu_reg_reg():
    assert _hex("add al, bl") == "00d8"
    assert _hex("or ax, bx") == "09d8"
    assert _hex("sub cx, dx") == "29d1"
    assert _hex("xor cl, dl") == "30d1"
    assert _hex("cmp ah, bh") == "38fc"
    assert _hex("and eax, ebx", 32) == "21d8"
    assert _hex("xor eax, ebx", 32) == "31d8"
    assert _hex("cmp eax, ebx", 32) == "39d8"


def test_alu_reg_imm():
    assert _hex("add al, 5") == "80c005"
    assert _hex("add al, 200") == "80c0c8"
    assert _hex("add ax, 0x1234") == "81c03412"
    assert _hex("add ax, 0x1234", 32) == "6681c03412"
    assert _hex("add eax, 5") == "6683c005"
    assert _hex("add eax, 5", 32) == "83c005"
    assert _hex("add eax, 0x1234") == "6681c034120000"
    assert _hex("add eax, 0x12345678") == "6681c078563412"
    assert _hex("or eax, 5") == "6683c805"
    assert _hex("and eax, 5") == "6683e005"
    assert _hex("sub eax, 5") == "6683e805"
    assert _hex("xor eax, 5") == "6683f005"
    assert _hex("cmp eax, 5") == "6683f805"
    assert _hex("sub eax, 0x12345678") == "6681e878563412"
    assert _hex("sub ax, 5") == "83e805"
    assert _hex("sub al, 5") == "80e805"
    assert _hex("add bl, 5") == "80c305"
    assert _hex("add bx, 5") == "83c305"


def test_alu_reg8_large_imm_raises_typeerror():
    # Documented bug: bytearray.append(bytes) in the REG8 imm>255 branch.
    with pytest.raises(TypeError):
        _hex("add al, 0x1234")


def test_test_forms():
    assert _hex("test eax, ebx") == "6685d8"
    assert _hex("test ax, bx") == "85d8"
    assert _hex("test al, bl") == "84d8"
    assert _hex("test eax, 0x12345678") == "66f7c078563412"
    assert _hex("test ax, 0x1234") == "f7c03412"
    assert _hex("test ax, 0x1234", 32) == "66f7c03412"
    assert _hex("test al, 5") == "f6c005"
    assert _hex("test bl, 0x7F") == "f6c37f"
    assert _hex("test bl, 0x1FF") == "f6c3ff"


def test_alu_memory_forms():
    assert _hex("add [ebx], eax") == "0103"
    assert _hex("add [ebx+4], eax") == "010304"
    assert _hex("add [0x1234], ax") == "66010534120000"
    assert _hex("or [ebx], bl") == "081b"
    assert _hex("and [ebx], bl") == "201b"
    assert _hex("sub [ebx], bl") == "281b"
    assert _hex("xor [ebx], bl") == "301b"
    assert _hex("cmp [ebx], bl") == "381b"
    assert _hex("add [ebx], 5") == "830305"
    assert _hex("add byte [ebx], 5") == "830305"
    assert _hex("add word [ebx], 5") == "830305"
    assert _hex("add dword [ebx], 5") == "830305"
    assert _hex("add [0x1234], 5") == "83053412000005"
    assert _hex("test [ebx], eax") == "0103"
    assert _hex("test dword [ebx], 5") == "f70305000000"


def test_adc_sbb_emit_nop_placeholder():
    # Documented quirk: adc/sbb are missing from the opcode dispatch tuple.
    assert _hex("adc eax, 5") == "90"
    assert _hex("sbb eax, 5") == "90"


# ══════════════════════════════════════════════════════════════════════════════
# inc / dec / unary / shift
# ══════════════════════════════════════════════════════════════════════════════

def test_inc_dec_registers():
    assert _hex("inc eax") == "6640"
    assert _hex("dec eax") == "6648"
    assert _hex("inc ax") == "40"
    assert _hex("dec ax") == "48"
    assert _hex("inc eax", 32) == "40"
    assert _hex("dec ebx", 32) == "4b"


def test_inc_dec_memory_uses_disp32_zero():
    # Documented quirk: [mem] form always encodes modrm=05 with disp=0.
    assert _hex("inc word [bx]") == "ff0500000000"
    assert _hex("dec byte [eax]") == "ff0d00000000"
    assert _hex("inc dword [ebx+4]") == "ff0500000000"


def test_unary_ops():
    assert _hex("not eax") == "66f7d0"
    assert _hex("neg ax") == "f7d8"
    assert _hex("mul bx") == "f7e3"
    assert _hex("imul ax") == "f7e8"
    assert _hex("div cx") == "f7f1"
    assert _hex("idiv dx") == "f7fa"
    assert _hex("not bl") == "f6d3"
    assert _hex("neg dl") == "f6da"


def test_unary_memory_emits_nothing():
    assert _hex("not byte [eax]") == ""


def test_shift_imm_and_cl():
    assert _hex("shl eax, 1") == "66c1e001"
    assert _hex("shl eax, 5", 32) == "c1e005"
    assert _hex("shr eax, cl") == "66d3e8"
    assert _hex("sal ax, 1") == "c1e001"
    assert _hex("sar al, 1") == "c0f801"
    assert _hex("rol eax, 1") == "66c1c001"
    assert _hex("ror eax, cl") == "66d3c8"
    assert _hex("rcl eax, 1") == "66c1d001"
    assert _hex("rcr ax, cl") == "d3d8"
    assert _hex("shl eax, 0x1234") == "66c1e034"
    assert _hex("shl bx, cl") == "d3e3"
    assert _hex("shl al, 1") == "c0e001"
    assert _hex("shr cl, cl") == "d2e9"


def test_shift_memory_forms():
    assert _hex("shl word [ebx], 1") == "c12301"
    assert _hex("shr byte [ebx], cl") == "d22b"
    assert _hex("rol dword [ebx+4], 3") == "c1030403"
    assert _hex("shl dword [ebx], cl") == "d323"
    assert _hex("rcr word [0x100], 2") == "c11d0001000002"
    assert _hex("sal byte [ebx], 1") == "c02301"
    assert _hex("shl byte [ebx], cl") == "d223"


# ══════════════════════════════════════════════════════════════════════════════
# lea / xchg / lgdt / lidt / ltr
# ══════════════════════════════════════════════════════════════════════════════

def test_lea():
    assert _hex("lea eax, [ebx+4]") == "8d4304"
    assert _hex("lea eax, [ebx+0x100]") == "8d8300010000"
    assert _hex("lea eax, [ebx+0x100]", 32) == "8d8300010000"


def test_xchg_reg32_only():
    assert _hex("xchg eax, ebx") == "87c3"
    assert _hex("xchg ecx, edx", 32) == "87ca"


def test_xchg_word_emits_nothing():
    assert _hex("xchg ax, bx") == ""


def test_lgdt_lidt_16bit_prefix_and_32bit():
    assert _hex("lgdt [gdt]\ngdt: dw 0") == "670f0115080000000000"
    assert _hex("lidt [idt]\nidt: dw 0") == "670f011d080000000000"
    assert _hex("lgdt [gdt]\ngdt: dw 0", 32) == "0f0115070000000000"
    assert _hex("lidt [idt]\nidt: dw 0", 32) == "0f011d070000000000"


def test_ltr():
    assert _hex("ltr ax") == "0f00d8"
    assert _hex("ltr bx") == "0f00db"


def test_ltr_memory_emits_nothing():
    assert _hex("ltr [gdt]\ngdt: dw 0") == "0000"


# ══════════════════════════════════════════════════════════════════════════════
# Data directives
# ══════════════════════════════════════════════════════════════════════════════

def test_db_forms():
    assert _hex("db 1, 2, 0x3F") == "01023f"
    assert _hex('db "Hello", 0') == "48656c6c6f00"
    assert _hex("db 'A','B',0") == "414200"
    assert _hex("db 'unterminated") == "756e7465726d696e61746564"
    assert _hex("db 0x90,,0x90") == "9090"
    assert _hex("db 'AB'") == "4142"


def test_db_escape_quirk():
    # Documented quirk: backslash escapes are not processed in db items.
    assert _hex(r"db '\n'") == "5c6e"
    assert _hex(r"db '\\'") == "5c5c"


def test_dw_dd():
    assert _hex("dw 0x1234") == "3412"
    assert _hex("dw 0xAA55, 0x01") == "55aa0100"
    assert _hex("dd 0x11223344") == "44332211"
    assert _hex("dd 0x12345678, 0x01") == "7856341201000000"


def test_times():
    assert _hex("times 3 db 0x90") == "909090"
    assert _hex("times 2 dw 0x1234") == "34123412"
    assert _hex("times 4 nop") == "90909090"
    assert _hex("times 2 db 0x41") == "4141"
    assert _hex("times 2+3 db 0x90") == "9090909090"
    assert _hex("times 0 nop") == ""


# ══════════════════════════════════════════════════════════════════════════════
# Directives and immediate parsing
# ══════════════════════════════════════════════════════════════════════════════

def test_bits_org_equ_directives():
    asm = X86Assembler()
    assert asm.assemble("[BITS 32]\nnop") == b"\x90"
    assert asm._bits == 32
    assert asm.assemble("[ORG 0x1000]\nnop") == b"\x90"
    assert asm._org == 0x1000
    assert asm.assemble("[ORG 0x7C00]\nnop") == b"\x90"
    assert asm._org == 0x7C00
    assert asm.assemble("X equ 0x42\nmov ax, X") == b"\xb8\x42\x00"


def test_section_directive_is_ignored():
    assert _hex("section .text\nnop") == "9090"


def test_dollar_and_expressions_resolve_to_zero():
    assert _hex("jmp $\nnop") == "e9fdff90"
    assert _hex("mov ax, 2+3*4") == "b80000"
    assert _hex("mov ax, a+2\na equ 4") == "b80000"


def test_imm_parsing_forms():
    assert _hex("mov ax, 0x1234") == "b83412"
    assert _hex("mov ax, 1234h") == "b83412"
    assert _hex("mov al, 0b1010") == "b00a"
    assert _hex("mov al, 0o17") == "b00f"
    assert _hex("mov al, 0b1100") == "b00c"


def test_imm_char_literals():
    assert _hex("mov al, 'A'") == "b041"
    assert _hex(r"mov al, '\n'") == "b00a"
    assert _hex(r"mov al, '\r'") == "b00d"
    assert _hex(r"mov al, '\t'") == "b009"
    assert _hex(r"mov al, '\0'") == "b000"
    assert _hex(r"mov al, '\\'") == "b05c"


def test_hex_suffix_with_0x_prefix_raises():
    with pytest.raises(ValueError):
        _hex("mov ax, 0x1234h")


def test_asm_run():
    cpu = X86Assembler().run("[BITS 32]\nmov eax, 5\nmov ebx, 3\nadd eax, ebx\nhlt", org=0x1000)
    assert cpu.eax == 8
    assert cpu.ebx == 3


# ══════════════════════════════════════════════════════════════════════════════
# Execution smoke tests (real assembly -> real execution)
# ══════════════════════════════════════════════════════════════════════════════

def test_execute_bits32_program():
    asm = X86Assembler()
    cpu = X86CPU()
    cpu.load(asm.assemble("[BITS 32]\n"
                          "mov eax, 0x1234\n"
                          "push eax\n"
                          "pop ebx\n"
                          "inc eax\n"
                          "dec eax\n"
                          "hlt"), org=0)
    cpu.run(max_steps=1000)
    assert cpu.eax == 0x1234
    assert cpu.ebx == 0x1234
    assert cpu.esp_val == len(cpu._mem) - 4


def test_execute_jump_and_condition():
    asm = X86Assembler()
    cpu = X86CPU()
    cpu.load(asm.assemble("[BITS 32]\n"
                          "xor eax, eax\n"
                          "jz done\n"
                          "mov eax, 0xDEAD\n"
                          "done:\n"
                          "mov ebx, 1\n"
                          "hlt"), org=0)
    cpu.run(max_steps=1000)
    assert cpu.eax == 0
    assert cpu.ebx == 1


# ══════════════════════════════════════════════════════════════════════════════
# Keyboard / scancode helpers and parity
# ══════════════════════════════════════════════════════════════════════════════

def test_parity():
    assert _parity(0x03) is True
    assert _parity(0x02) is False
    assert _parity(0x00) is True


def test_char_to_scancode():
    assert _char_to_scancode("a") == 0x1E
    assert _char_to_scancode("A") == 0x1E
    assert _char_to_scancode("\n") == 0x1C
    assert _char_to_scancode(" ") == 0x39
    assert _char_to_scancode("?") == -1


def test_scancode_to_char():
    assert _scancode_to_char(0x1E) == "a"
    assert _scancode_to_char(0x02) == "1"
    assert _scancode_to_char(0x39) == " "
    assert _scancode_to_char(0x0E) == "\x08"  # backspace as control int
    assert _scancode_to_char(0x2A) == "\x00"  # unknown
    assert _scancode_to_char(999) == "\x00"   # out of range


def test_default_kbd_handler():
    cpu = X86CPU(memory_size=1024 * 1024)
    cpu._kbd_buffer = [0x1E]
    _default_kbd_handler(cpu)
    assert cpu._mem[0x400] == ord("a")
    assert cpu._mem[0x401] == 0x1E
    cpu._kbd_buffer = []
    _default_kbd_handler(cpu)
    cpu._kbd_buffer = [0x2A]  # unknown -> '\0' -> no write
    _default_kbd_handler(cpu)
    assert cpu._mem[0x400] == ord("a")


# ══════════════════════════════════════════════════════════════════════════════
# Dead-code: _estimate_* and _pfx private-method direct calls
# These methods are never called by assemble() at runtime (grep-proven),
# so they can only be covered via direct private-method invocation.
# ══════════════════════════════════════════════════════════════════════════════

def test_pfx_8bit_reg_returns_false():
    asm = X86Assembler()
    assert asm._pfx("al") is False
    assert asm._pfx("cl") is False
    assert asm._pfx("ah") is False


def test_estimate_data_size_db_string():
    asm = X86Assembler()
    assert asm._estimate_data_size("db 'hi'") == 2
    assert asm._estimate_data_size('db "hello"') == 5
    assert asm._estimate_data_size("db 'x'") == 1
    # leading quote only (no closing quote) -> len - 1
    assert asm._estimate_data_size("db 'abc") == 3


def test_estimate_data_size_db_numeric():
    asm = X86Assembler()
    assert asm._estimate_data_size("db 5") == 1
    assert asm._estimate_data_size("db 5, 6, 7") == 3


def test_estimate_data_size_dw():
    asm = X86Assembler()
    assert asm._estimate_data_size("dw 1, 2") == 4
    assert asm._estimate_data_size("dw 0x1234") == 2


def test_estimate_data_size_dd():
    asm = X86Assembler()
    assert asm._estimate_data_size("dd 1, 2, 3") == 12
    assert asm._estimate_data_size("dd 0xDEADBEEF") == 4


def test_estimate_data_size_unknown_returns_1():
    asm = X86Assembler()
    assert asm._estimate_data_size("dq 1") == 1  # unknown directive -> 1


def test_estimate_times_size():
    asm = X86Assembler()
    assert asm._estimate_times_size("times 5 nop") == 5
    assert asm._estimate_times_size("times 3 db 0") == 3
    assert asm._estimate_times_size("times 2 dw 1") == 4


def test_estimate_insn_size_simple_ops():
    asm = X86Assembler()
    assert asm._estimate_insn_size("nop") == 1
    assert asm._estimate_insn_size("hlt") == 1
    assert asm._estimate_insn_size("cli") == 1
    assert asm._estimate_insn_size("ret") == 1
    assert asm._estimate_insn_size("iret") == 1
    assert asm._estimate_insn_size("pusha") == 1
    assert asm._estimate_insn_size("popa") == 1
    assert asm._estimate_insn_size("cld") == 1
    assert asm._estimate_insn_size("std") == 1
    assert asm._estimate_insn_size("lodsb") == 1
    assert asm._estimate_insn_size("stosw") == 1
    assert asm._estimate_insn_size("movsb") == 1


def test_estimate_insn_size_rep():
    asm = X86Assembler()
    assert asm._estimate_insn_size("rep nop") == 2


def test_estimate_insn_size_retf():
    asm = X86Assembler()
    assert asm._estimate_insn_size("retf") == 1


def test_estimate_insn_size_cc_jump():
    asm = X86Assembler()
    assert asm._estimate_insn_size("je") == 2
    assert asm._estimate_insn_size("jnz") == 2


def test_estimate_insn_size_int():
    asm = X86Assembler()
    assert asm._estimate_insn_size("int 0x80") == 2


def test_estimate_insn_size_push_pop():
    asm = X86Assembler()
    assert asm._estimate_insn_size("push eax") == 1
    assert asm._estimate_insn_size("push ax") == 1
    assert asm._estimate_insn_size("push 42") == 3
    assert asm._estimate_insn_size("pop ebx") == 1


def test_estimate_insn_size_jmp():
    asm = X86Assembler()
    # Default _bits=16 -> jmp returns 3 (rel16 worst case)
    assert asm._estimate_insn_size("jmp eax") == 3
    assert asm._estimate_insn_size("jmp label") == 3  # 16-bit mode
    # far jump: seg:off without brackets
    assert asm._estimate_insn_size("jmp 0x1000:0x2000") == 5


def test_estimate_insn_size_call():
    asm = X86Assembler()
    # Default _bits=16 -> call returns 3 (rel16)
    assert asm._estimate_insn_size("call eax") == 3  # 16-bit mode
    asm._bits = 32
    assert asm._estimate_insn_size("call label") == 5  # 32-bit mode
    asm._bits = 16


def test_estimate_insn_size_in_out():
    asm = X86Assembler()
    assert asm._estimate_insn_size("in al, 0x60") == 2
    assert asm._estimate_insn_size("out 0x60, al") == 2


def test_estimate_insn_size_mov_delegates():
    asm = X86Assembler()
    # Default _bits=16; _estimate_mov_size returns based on operand regs
    assert asm._estimate_insn_size("mov eax, ebx") == 5  # r32 → 5
    assert asm._estimate_insn_size("mov ax, bx") == 3  # r16 → 3
    assert asm._estimate_insn_size("mov es, ax") == 2  # Sreg → 2


def test_estimate_insn_size_alu_delegates():
    asm = X86Assembler()
    # Default _bits=16; _estimate_alu_size returns 2 for reg,reg
    assert asm._estimate_insn_size("add eax, ebx") == 2  # r32,r32 → 2
    assert asm._estimate_insn_size("add al, 5") == 3  # reg,imm fallback → 3


def test_estimate_insn_size_unary():
    asm = X86Assembler()
    assert asm._estimate_insn_size("inc eax") == 2
    assert asm._estimate_insn_size("neg ebx") == 2
    assert asm._estimate_insn_size("shl eax, 1") == 2


def test_estimate_insn_size_default():
    asm = X86Assembler()
    assert asm._estimate_insn_size("xchg") == 3  # unknown → default 3


def test_estimate_mov_size_seg_reg():
    asm = X86Assembler()
    assert asm._estimate_mov_size("es, ax") == 2  # Sreg → 2
    assert asm._estimate_mov_size("ax, ds") == 2  # Sreg → 2


def test_estimate_mov_size_no_operands():
    asm = X86Assembler()
    assert asm._estimate_mov_size("") == 2  # len < 2 → 2


def test_estimate_alu_size_no_operands():
    asm = X86Assembler()
    assert asm._estimate_alu_size("") == 2  # len < 2 → 2


def test_estimate_alu_size_reg_reg_8bit():
    asm = X86Assembler()
    assert asm._estimate_alu_size("al, bl") == 2  # r8,r8 → 2


def test_estimate_alu_size_reg_imm():
    asm = X86Assembler()
    assert asm._estimate_alu_size("eax, 5") == 3  # reg,imm → 3
    assert asm._estimate_alu_size("ax, 0x1234") == 3
