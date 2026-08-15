"""
Execution tests for the X86CPU in domains/shell/vm.py.

Drives the 32-bit CPU with raw machine-code bytes (loaded at 0x1000) to cover
every instruction-decoder branch that assembled programs do not reach:
group-1/3/4/5 encodings (FE/FF/80/81/83/8F/C0/C1/D0-D3/F6/F7), the 0F prefix
family (near Jcc, RDTSC, LGDT/LIDT, MOV CRn/DRn, MOVZX/MOVSX, IMUL, BSF/BSR),
the 66-prefix word MOV forms, string ops with both direction-flag paths, REP
loops, IN/OUT on immediate and DX ports, ALU imm/reg/mem forms, SIB + disp
addressing, seg-register moves, high-byte register moves, interrupts/IRET/IRQ
dispatch, and the ALU/shift/cc helpers.

Uses ``_exec_one`` directly (private) only where the ``step`` wrapper would
swallow a deliberate InsFault (DIV by zero / overflow).
"""

import pytest

from domains.shell.vm import (
    X86CPU,
    InsFault,
    FLAG_CF,
    FLAG_PF,
    FLAG_ZF,
    FLAG_SF,
    FLAG_OF,
    FLAG_DF,
    FLAG_IF,
)


def _cpu(hexcode, regs=None, mem=None, org=0x1000, mem_size=0x400000):
    cpu = X86CPU(memory_size=mem_size)
    cpu.load(bytes.fromhex(hexcode.replace(" ", "")), org)
    for name, val in (regs or {}).items():
        if name == "eflags":
            cpu._eflags = val
        else:
            setattr(cpu, name, val)
    for addr, val in (mem or {}).items():
        if isinstance(val, int):
            cpu._write32(addr, val)
        else:
            for i, b in enumerate(val):
                cpu._mem[addr + i] = b
    return cpu


def _steps(cpu, n):
    for _ in range(n):
        assert cpu.step()


def _store16(cpu, addr, val):
    cpu._mem[addr & 0xFFFFFFFF] = val & 0xFF
    cpu._mem[(addr + 1) & 0xFFFFFFFF] = (val >> 8) & 0xFF


# ══════════════════════════════════════════════════════════════════════════════
# Register / segment helpers
# ══════════════════════════════════════════════════════════════════════════════

def test_mov_seg_reg_forms():
    cpu = _cpu("8edb8e1b")
    cpu.ebx = 0x5678
    _steps(cpu, 1)  # MOV DS, BX
    assert cpu._segregs[3] == 0x5678
    cpu.ebx = 0x3000
    _store16(cpu, 0x3000, 0x7777)
    _steps(cpu, 1)  # MOV DS, word [EBX]
    assert cpu._segregs[3] == 0x7777


def test_mov_seg_to_reg_and_mem():
    cpu = _cpu("8cdb8c1b")
    cpu._segregs[3] = 0x1234
    _steps(cpu, 1)  # MOV BX, DS
    assert cpu.ebx == 0x1234
    cpu.ebx = 0x3000
    _steps(cpu, 1)  # MOV word [EBX], DS
    assert cpu._mem[0x3000] == 0x34 and cpu._mem[0x3001] == 0x12


def test_reg_name_helpers():
    cpu = _cpu("")
    assert cpu._reg_index("eax") == 0
    assert cpu._reg_index("ax") == 0
    assert cpu._reg_index("al") == 0
    assert cpu._reg_index("bh") == 7
    with pytest.raises(ValueError):
        cpu._reg_index("xyz")
    cpu._read_reg("eax")
    cpu._read_reg("ax")
    cpu._read_reg("al")
    cpu._read_reg("ah")
    with pytest.raises(ValueError):
        cpu._read_reg("zz")
    cpu._write_reg("eax", 5)
    cpu._write_reg("ax", 7)
    cpu._write_reg("al", 3)
    cpu._write_reg("ah", 9)
    with pytest.raises(ValueError):
        cpu._write_reg("zz", 1)


def test_high_byte_register_moves():
    cpu = _cpu("8ae3")  # MOV AH, BL
    cpu.ebx = 0x0000005A
    _steps(cpu, 1)
    assert (cpu.eax >> 8) & 0xFF == 0x5A
    cpu = _cpu("88c7")  # MOV BH, AL
    cpu.eax = 0x0000000B
    _steps(cpu, 1)
    assert (cpu.ebx >> 8) & 0xFF == 0x0B


# ══════════════════════════════════════════════════════════════════════════════
# ModR/M addressing: SIB, disp8, disp32
# ══════════════════════════════════════════════════════════════════════════════

def test_modrm_sib_addressing():
    cpu = _cpu("8b048b")  # MOV EAX, [EBX+ECX*4]
    cpu.ebx = 0x1000
    cpu.ecx = 0x100
    cpu._write32(0x1400, 0xDEADBEEF)
    _steps(cpu, 1)
    assert cpu.eax == 0xDEADBEEF
    cpu = _cpu("8b0423")  # SIB index=4 (none)
    cpu.ebx = 0x2000
    cpu._write32(0x2000, 7)
    _steps(cpu, 1)
    assert cpu.eax == 7
    cpu = _cpu("8b042534120000")  # SIB base=5 mod=0 -> disp32
    cpu._write32(0x1234, 9)
    _steps(cpu, 1)
    assert cpu.eax == 9


def test_modrm_disp8_disp32():
    cpu = _cpu("8b4305", mem_size=0x20000000)  # MOV EAX, [EBX+5]
    cpu.ebx = 0x3000
    cpu._write32(0x3005, 11)
    _steps(cpu, 1)
    assert cpu.eax == 11
    cpu = _cpu("8b8378563412", mem_size=0x20000000)  # MOV EAX, [EBX+0x12345678]
    cpu.ebx = 0x2000
    cpu._write32(0x12347678, 13)
    _steps(cpu, 1)
    assert cpu.eax == 13


def test_resolve_rm_helper():
    cpu = _cpu("")
    cpu.eax = 0x11
    assert cpu._resolve_rm(0, True, 0, 32) == 0x11
    cpu._mem[0x3000] = 0x7F
    assert cpu._resolve_rm(0, False, 0x3000, 8) == 0x7F


# ══════════════════════════════════════════════════════════════════════════════
# MOV forms
# ══════════════════════════════════════════════════════════════════════════════

def test_mov_imm_forms():
    cpu = _cpu("b005b80000000066b83412c6c005c7c0ffffffff")
    _steps(cpu, 1)  # MOV AL, 5
    assert cpu._get8l(0) == 5
    _steps(cpu, 1)  # MOV EAX, 0
    assert cpu.eax == 0
    _steps(cpu, 1)  # MOV AX, 0x1234
    assert cpu.eax == 0x1234
    _steps(cpu, 1)  # MOV AL, 5
    assert cpu.eax == 0x1205
    _steps(cpu, 1)  # MOV EAX, 0xFFFFFFFF
    assert cpu.eax == 0xFFFFFFFF


def test_mov_c6_c7_mem():
    cpu = _cpu("c60305c70305000000")
    cpu.ebx = 0x3000
    _steps(cpu, 1)  # MOV byte [EBX], 5
    assert cpu._mem[0x3000] == 5
    _steps(cpu, 1)  # MOV dword [EBX], 5
    assert cpu._read32(0x3000) == 5


def test_mov_accumulator_forms():
    cpu = _cpu("a134120000a334120000")
    cpu._write32(0x1234, 0xCAFEBABE)
    _steps(cpu, 1)  # MOV EAX, [0x1234]
    assert cpu.eax == 0xCAFEBABE
    cpu.eax = 0x11223344
    _steps(cpu, 1)  # MOV [0x1234], EAX
    assert cpu._read32(0x1234) == 0x11223344


def test_lea():
    cpu = _cpu("8d43048dc3")
    cpu.ebx = 0x1000
    _steps(cpu, 1)  # LEA EAX, [EBX+4]
    assert cpu.eax == 0x1004
    _steps(cpu, 1)  # LEA EAX, [EBX] (register rm -> no-op)
    assert cpu.eax == 0x1004


def test_66_word_mov_forms():
    cpu = _cpu("66c70334126689c3668bc3668903668b03")
    cpu.ebx = 0x3000
    cpu.eax = 0x0000ABCD
    _steps(cpu, 1)  # MOV word [EBX], 0x1234
    assert cpu._mem[0x3000] == 0x34 and cpu._mem[0x3001] == 0x12
    _steps(cpu, 1)  # MOV BX, AX
    assert cpu.ebx & 0xFFFF == 0xABCD
    _steps(cpu, 1)  # MOV AX, BX
    assert cpu.eax & 0xFFFF == 0xABCD
    cpu.ebx = 0x3000  # MOV BX, AX above clobbered EBX's base
    _steps(cpu, 1)  # MOV word [EBX], AX
    assert cpu._mem[0x3000] == 0xCD and cpu._mem[0x3001] == 0xAB
    _steps(cpu, 1)  # MOV AX, word [EBX]
    assert cpu.eax & 0xFFFF == 0xABCD


def test_66_unknown_opcode2_noop():
    cpu = _cpu("6690")
    _steps(cpu, 1)
    assert cpu.eip == 0x1002


# ══════════════════════════════════════════════════════════════════════════════
# XCHG, MOVSXD
# ══════════════════════════════════════════════════════════════════════════════

def test_xchg_short_forms():
    cpu = _cpu("9197")
    cpu.ecx = 0x11111111
    _steps(cpu, 1)  # XCHG EAX, ECX
    assert cpu.eax == 0x11111111 and cpu.ecx == 0
    cpu.edi = 0x22222222
    _steps(cpu, 1)  # XCHG EAX, EDI
    assert cpu.eax == 0x22222222 and cpu.edi == 0x11111111


def test_movsxd():
    cpu = _cpu("63c36303")
    cpu.ebx = 0x3000
    _steps(cpu, 1)  # MOVSXD EAX, EBX
    assert cpu.eax == 0x3000
    cpu._write32(0x3000, 0x01020304)
    _steps(cpu, 1)  # MOVSXD EAX, [EBX]
    assert cpu.eax == 0x01020304


# ══════════════════════════════════════════════════════════════════════════════
# ALU forms
# ══════════════════════════════════════════════════════════════════════════════

def test_alu_imm_81_83():
    cpu = _cpu("81c00500000083c080810305000000830305")
    _steps(cpu, 1)  # ADD EAX, 5
    assert cpu.eax == 5
    _steps(cpu, 1)  # ADD EAX, -128
    assert cpu.eax == (5 - 128) & 0xFFFFFFFF
    cpu.ebx = 0x3000
    cpu._write32(0x3000, 10)
    _steps(cpu, 1)  # ADD [EBX], 5
    assert cpu._read32(0x3000) == 15
    _steps(cpu, 1)  # ADD [EBX], 5
    assert cpu._read32(0x3000) == 20


def test_alu_imm_80_8bit():
    cpu = _cpu("80c005800305")
    cpu.eax = 0x10
    _steps(cpu, 1)  # ADD AL, 5
    assert cpu._get8l(0) == 0x15
    cpu.ebx = 0x3000
    cpu._mem[0x3000] = 10
    _steps(cpu, 1)  # ADD byte [EBX], 5
    assert cpu._mem[0x3000] == 15


def test_imul_imm_69_6b():
    cpu = _cpu("69c305000000")
    cpu.ebx = 7
    _steps(cpu, 1)  # IMUL EAX, EBX, 5
    assert cpu.eax == 35
    cpu = _cpu("6bc305")
    cpu.ebx = 7
    _steps(cpu, 1)  # IMUL EAX, EBX, 5 (imm8)
    assert cpu.eax == 35


def test_imul_three_operand_mem():
    # IMUL EAX, [EBX], 5 — 6B 03 05
    cpu = _cpu("6b0305", regs={"ebx": 0x3000, "eip": 0x1000})
    cpu._write32(0x3000, 10)
    _steps(cpu, 1)
    assert cpu.eax == 50

    # IMUL EAX, [EBX+4], -1 — 6B 43 04 FF
    cpu = _cpu("6b4304ff", regs={"ebx": 0x3000, "eip": 0x1000})
    cpu._write32(0x3004, 7)
    _steps(cpu, 1)
    assert cpu.eax == 0xFFFFFFF9  # -7 sign-extended

    # IMUL EAX, [EBX+4], 0x100 — 69 43 04 00010000
    cpu = _cpu("69430400010000", regs={"ebx": 0x3000, "eip": 0x1000})
    cpu._write32(0x3004, 3)
    _steps(cpu, 1)
    assert cpu.eax == 0x300


def test_alu_reg_forms():
    cpu = _cpu("0bc323c32bc33bc303c0")
    cpu.eax = 0xF0F0
    cpu.ebx = 0x0F0F
    _steps(cpu, 1)  # OR EAX, EBX -> 0xFFFF
    assert cpu.eax == 0xFFFF
    _steps(cpu, 1)  # AND EAX, EBX -> 0x0F0F
    assert cpu.eax == 0x0F0F
    _steps(cpu, 1)  # SUB EAX, EBX -> 0
    assert cpu.eax == 0
    _steps(cpu, 1)  # CMP EAX, EBX -> no write
    assert cpu.eax == 0
    _steps(cpu, 1)  # ADD EAX, EAX -> 0
    assert cpu.eax == 0


def test_alu_mem_forms():
    cpu = _cpu("0103090321032903310339030003")
    cpu.ebx = 0x3000
    cpu.eax = 5
    cpu._write32(0x3000, 10)
    _steps(cpu, 1)  # ADD [EBX], EAX -> 15
    assert cpu._read32(0x3000) == 15
    _steps(cpu, 1)  # OR [EBX], EAX -> 15
    assert cpu._read32(0x3000) == 15
    _steps(cpu, 1)  # AND [EBX], EAX -> 5
    assert cpu._read32(0x3000) == 5
    _steps(cpu, 1)  # SUB [EBX], EAX -> 0
    assert cpu._read32(0x3000) == 0
    _steps(cpu, 1)  # XOR [EBX], EAX -> 5
    assert cpu._read32(0x3000) == 5
    _steps(cpu, 1)  # CMP [EBX], EAX -> no write
    assert cpu._read32(0x3000) == 5
    _steps(cpu, 1)  # ADD [EBX], AL -> 10
    assert cpu._read8(0x3000) == 10


def test_alu_8bit_reg():
    cpu = _cpu("02c3")
    cpu.eax = 0x10
    cpu.ebx = 0x20
    _steps(cpu, 1)  # ADD AL, BL
    assert cpu._get8l(0) == 0x30


def test_test_forms():
    cpu = _cpu("85c3850384c38403")
    cpu.ebx = 0x3000
    cpu.eax = 1
    cpu._write32(0x3000, 3)
    _steps(cpu, 1)  # TEST EBX, EAX (0x3000 & 1 = 0) -> ZF
    assert cpu.zf
    _steps(cpu, 1)  # TEST [EBX], EAX (3 & 1 = 1)
    assert not cpu.zf
    _steps(cpu, 1)  # TEST BL, AL (0 & 1 = 0)
    assert cpu.zf
    _steps(cpu, 1)  # TEST [EBX], AL (3 & 1 = 1)
    assert not cpu.zf


def test_test_mem_imm_byte_word():
    # F6 /0 ib — TEST byte [ebx], imm8 (mem=0 -> ZF)
    cpu = _cpu("f60305", regs={"ebx": 0x3000})
    cpu._mem[0x3000] = 0
    cpu._set_flag(FLAG_ZF, False)
    _steps(cpu, 1)
    assert cpu.zf
    # F6 /0 ib — TEST byte [ebx], imm8 (mem=4 & 5 = 4 -> no ZF)
    cpu = _cpu("f60305", regs={"ebx": 0x3000})
    cpu._mem[0x3000] = 4
    cpu._set_flag(FLAG_ZF, True)
    _steps(cpu, 1)
    assert not cpu.zf
    # 66 F7 /0 iw — TEST word [ebx], imm16 (mem=0 -> ZF)
    cpu = _cpu("66f7033412", regs={"ebx": 0x3000})
    cpu._write16(0x3000, 0)
    cpu._set_flag(FLAG_ZF, False)
    _steps(cpu, 1)
    assert cpu.zf
    # 66 F7 /0 iw — TEST word [ebx], imm16 (mem=0x1234 -> no ZF)
    cpu = _cpu("66f7033412", regs={"ebx": 0x3000})
    cpu._write16(0x3000, 0x1234)
    cpu._set_flag(FLAG_ZF, True)
    _steps(cpu, 1)
    assert not cpu.zf


def test_f6_not_neg_mem_byte():
    # F6 /2 — NOT byte [ebx] (0x55 -> 0xAA)
    cpu = _cpu("f613", regs={"ebx": 0x3000})
    cpu._mem[0x3000] = 0x55
    _steps(cpu, 1)
    assert cpu._mem[0x3000] == 0xAA
    # F6 /3 — NEG byte [ebx] (1 -> 0xFF, CF set)
    cpu = _cpu("f61b", regs={"ebx": 0x3000})
    cpu._mem[0x3000] = 1
    _steps(cpu, 1)
    assert cpu._mem[0x3000] == 0xFF
    assert cpu.cf


# ══════════════════════════════════════════════════════════════════════════════
# INC/DEC / PUSH/POP
# ══════════════════════════════════════════════════════════════════════════════

def test_inc_dec_short():
    cpu = _cpu("404f")
    _steps(cpu, 1)  # INC EAX
    assert cpu.eax == 1
    cpu.edi = 0x10
    _steps(cpu, 1)  # DEC EDI
    assert cpu.edi == 0x0F


def test_push_pop_short():
    cpu = _cpu("5058")
    cpu.eax = 0x12345678
    _steps(cpu, 1)  # PUSH EAX
    assert cpu._read32(cpu.esp) == 0x12345678
    cpu.eax = 0
    _steps(cpu, 1)  # POP EAX
    assert cpu.eax == 0x12345678


def test_push_imm():
    cpu = _cpu("6a056a806878563412")
    _steps(cpu, 1)  # PUSH 5
    assert cpu._read32(cpu.esp) == 5
    _steps(cpu, 1)  # PUSH -128
    assert cpu._read32(cpu.esp) == 0xFFFFFF80
    _steps(cpu, 1)  # PUSH 0x12345678
    assert cpu._read32(cpu.esp) == 0x12345678


def test_pop_rm():
    cpu = _cpu("8fc38f03")
    cpu.ebx = 0x3000
    cpu._push32(0x0A0B0C0D)
    _steps(cpu, 1)  # POP EBX
    assert cpu.ebx == 0x0A0B0C0D
    cpu.ebx = 0x3000  # restore base clobbered by POP EBX
    cpu._push32(0x11223344)
    _steps(cpu, 1)  # POP [EBX]
    assert cpu._read32(0x3000) == 0x11223344


# ══════════════════════════════════════════════════════════════════════════════
# Group 3/4/5 encodings (FE/FF)
# ══════════════════════════════════════════════════════════════════════════════

def test_fe_inc_dec_rm8():
    cpu = _cpu("fec0fec8fe03fe0b")
    cpu.eax = 0x01
    cpu.ebx = 0x3000
    cpu._mem[0x3000] = 0x10
    _steps(cpu, 1)  # INC AL
    assert cpu._get8l(0) == 0x02
    _steps(cpu, 1)  # DEC AL
    assert cpu._get8l(0) == 0x01
    _steps(cpu, 1)  # INC byte [EBX]
    assert cpu._mem[0x3000] == 0x11
    _steps(cpu, 1)  # DEC byte [EBX]
    assert cpu._mem[0x3000] == 0x10


def test_ff_push_rm():
    cpu = _cpu("fff6ff33")
    cpu.esi = 0x11223344
    _steps(cpu, 1)  # PUSH ESI
    assert cpu._read32(cpu.esp) == 0x11223344
    cpu.ebx = 0x3000
    cpu._write32(0x3000, 0x55667788)
    _steps(cpu, 1)  # PUSH [EBX]
    assert cpu._read32(cpu.esp) == 0x55667788


def test_ff_jmp_rm():
    cpu = _cpu("ffe3ff23")
    cpu.ebx = 0x5000
    _steps(cpu, 1)  # JMP EBX
    assert cpu.eip == 0x5000
    cpu._eip = 0x1002
    cpu.ebx = 0x3000
    cpu._write32(0x3000, 0x6000)
    _steps(cpu, 1)  # JMP [EBX]
    assert cpu.eip == 0x6000


def test_ff_call_rm():
    cpu = _cpu("ffd3")
    cpu.ebx = 0x5000
    _steps(cpu, 1)  # CALL EBX
    assert cpu.eip == 0x5000
    assert cpu._read32(cpu.esp) == 0x1002


def test_ff_inc_dec_rm32():
    cpu = _cpu("ff03ffc3ff0bffcb")
    cpu.ebx = 0x3000
    cpu._write32(0x3000, 10)
    _steps(cpu, 1)  # INC [EBX]
    assert cpu._read32(0x3000) == 11
    _steps(cpu, 1)  # INC EBX
    assert cpu.ebx == 0x3001
    cpu.ebx = 0x3000  # INC EBX above moved the base
    _steps(cpu, 1)  # DEC [EBX]
    assert cpu._read32(0x3000) == 10
    _steps(cpu, 1)  # DEC EBX
    assert cpu.ebx == 0x2FFF


# ══════════════════════════════════════════════════════════════════════════════
# Control flow
# ══════════════════════════════════════════════════════════════════════════════

def test_call_rel32_forward_backward():
    cpu = _cpu("e805000000e8fbffffff")
    _steps(cpu, 1)  # CALL +5
    assert cpu._read32(cpu.esp) == 0x1005
    assert cpu.eip == 0x100A
    cpu._eip = 0x1005
    _steps(cpu, 1)  # CALL -5 -> next_eip 0x100A - 5 = 0x1005
    assert cpu.eip == 0x1005


def test_jmp_rel32_backward():
    cpu = _cpu("e9fbffffff")
    _steps(cpu, 1)
    assert cpu.eip == 0x1000


def test_jmp_rel8():
    cpu = _cpu("eb02", mem={0x1004: [0xEB, 0xFE]})
    _steps(cpu, 1)  # JMP +2 at 0x1000 -> 0x1004
    assert cpu.eip == 0x1004
    _steps(cpu, 1)  # JMP -2 at 0x1004 -> next_eip 0x1006 - 2 = 0x1004
    assert cpu.eip == 0x1004


def test_jcc_rel8_taken_and_not():
    cpu = _cpu("7602")  # JBE +2
    cpu._eflags = FLAG_CF | 0x02
    _steps(cpu, 1)
    assert cpu.eip == 0x1004
    cpu = _cpu("7602")
    cpu._eflags = 0x02
    _steps(cpu, 1)
    assert cpu.eip == 0x1002


def test_0f_near_jcc():
    cpu = _cpu("0f84ffffffff")  # JZ rel32 = -1
    cpu._eflags = FLAG_ZF | 0x02
    _steps(cpu, 1)
    assert cpu.eip == 0x1005
    cpu = _cpu("0f84ffffffff")
    cpu._eflags = 0x02
    _steps(cpu, 1)
    assert cpu.eip == 0x1006


# ══════════════════════════════════════════════════════════════════════════════
# 0F prefix family
# ══════════════════════════════════════════════════════════════════════════════

def test_rdtsc():
    cpu = _cpu("0f31")
    cpu.eax = 0x11111111
    cpu.edx = 0x22222222
    _steps(cpu, 1)
    assert cpu.eax == 0 and cpu.edx == 0


def test_lgdt_lidt():
    cpu = _cpu("0f0115002000000f011d00200000")
    cpu._mem[0x2000] = 0xFF
    cpu._mem[0x2001] = 0x00
    cpu._mem[0x2002:0x2006] = bytes([0x00, 0x10, 0x00, 0x00])
    _steps(cpu, 1)  # LGDT [0x2000]
    assert cpu._gdt_limit == 0xFF and cpu._gdt_base == 0x1000
    _steps(cpu, 1)  # LIDT [0x2000]
    assert cpu._idt_limit == 0xFF and cpu._idt_base == 0x1000


def test_mov_cr_dr():
    cpu = _cpu("0f20c00f20c80f22c00f21c00f23c0")
    cpu._cr[0] = 0xAABBCCDD
    cpu._cr[1] = 0x11223344
    _steps(cpu, 1)  # MOV EAX, CR0
    assert cpu.eax == 0xAABBCCDD
    _steps(cpu, 1)  # MOV EAX, CR1
    assert cpu.eax == 0x11223344
    cpu.eax = 0x55667788
    _steps(cpu, 1)  # MOV CR0, EAX
    assert cpu._cr[0] == 0x55667788
    cpu._dr[0] = 0x01020304
    _steps(cpu, 1)  # MOV EAX, DR0
    assert cpu.eax == 0x01020304
    cpu.eax = 0x090A0B0C
    _steps(cpu, 1)  # MOV DR0, EAX
    assert cpu._dr[0] == 0x090A0B0C


def test_movzx():
    cpu = _cpu("0fb6c30fb6030fb7c30fb703")
    cpu.ebx = 0x0000FFA1
    _steps(cpu, 1)  # MOVZX EAX, BL
    assert cpu.eax == 0xA1
    cpu.ebx = 0x3000
    cpu._mem[0x3000] = 0x80
    _steps(cpu, 1)  # MOVZX EAX, byte [EBX]
    assert cpu.eax == 0x80
    cpu.ebx = 0xFFFF
    _steps(cpu, 1)  # MOVZX EAX, BX
    assert cpu.eax == 0xFFFF
    cpu.ebx = 0x3000
    _store16(cpu, 0x3000, 0x1234)
    _steps(cpu, 1)  # MOVZX EAX, word [EBX]
    assert cpu.eax == 0x1234


def test_movsx():
    cpu = _cpu("0fbec30fbe030fbfc30fbf03")
    cpu.ebx = 0x00000080
    _steps(cpu, 1)  # MOVSX EAX, BL
    assert cpu.eax == 0xFFFFFF80
    cpu.ebx = 0x3000
    cpu._mem[0x3000] = 0xFE
    _steps(cpu, 1)  # MOVSX EAX, byte [EBX]
    assert cpu.eax == 0xFFFFFFFE
    cpu.ebx = 0x8000
    _steps(cpu, 1)  # MOVSX EAX, BX
    assert cpu.eax == 0xFFFF8000
    cpu.ebx = 0x3000
    _store16(cpu, 0x3000, 0x8000)
    _steps(cpu, 1)  # MOVSX EAX, word [EBX]
    assert cpu.eax == 0xFFFF8000


def test_0f_af_imul():
    cpu = _cpu("0fafc30faf03")
    cpu.ebx = 0x0000000A
    cpu.eax = 0x00000005
    _steps(cpu, 1)  # IMUL EAX, EBX
    assert cpu.eax == 50
    cpu.ebx = 0x3000
    cpu._write32(0x3000, 7)
    _steps(cpu, 1)  # IMUL EAX, [EBX]
    assert cpu.eax == 350


def test_bsf_bsr():
    cpu = _cpu("0fbcc30fbcc00fbdc30fbdc0")
    cpu.ebx = 0x00001020
    _steps(cpu, 1)  # BSF EAX, EBX -> 5
    assert cpu.eax == 5 and not cpu.zf
    cpu.eax = 0
    _steps(cpu, 1)  # BSF EAX, EAX(=0) -> ZF
    assert cpu.zf
    cpu.ebx = 0x00001020
    _steps(cpu, 1)  # BSR EAX, EBX -> 12
    assert cpu.eax == 12
    cpu.eax = 0
    _steps(cpu, 1)  # BSR EAX, EAX(=0) -> ZF
    assert cpu.zf


# ══════════════════════════════════════════════════════════════════════════════
# SAHF/LAHF/CDQ/BCD
# ══════════════════════════════════════════════════════════════════════════════

def test_sahf_lahf():
    cpu = _cpu("9e9f")
    cpu.eax = 0x0000D500  # AH = 0xD5
    _steps(cpu, 1)  # SAHF
    assert cpu._eflags & 0xFF == 0xD7
    _steps(cpu, 1)  # LAHF
    assert (cpu.eax >> 8) & 0xFF == 0xD7


def test_cdq():
    cpu = _cpu("9999")
    cpu.eax = 0x80000000
    _steps(cpu, 1)
    assert cpu.edx == 0xFFFFFFFF
    cpu.eax = 0x7FFFFFFF
    _steps(cpu, 1)
    assert cpu.edx == 0


def test_bcd_stubs():
    cpu = _cpu("272f373f")
    for _ in range(4):
        assert cpu.step()
    assert cpu.eip == 0x1004


# ══════════════════════════════════════════════════════════════════════════════
# String ops
# ══════════════════════════════════════════════════════════════════════════════

def test_lodsb_forward_backward():
    cpu = _cpu("ac")
    cpu.esi = 0x3000
    cpu._mem[0x3000] = 0x2A
    _steps(cpu, 1)
    assert cpu.eax == 0x2A and cpu.esi == 0x3001
    cpu = _cpu("fdac")
    cpu.esi = 0x3001
    cpu._mem[0x3001] = 0x2B
    _steps(cpu, 1)  # STD
    assert cpu.df
    _steps(cpu, 1)  # LODSB loads [ESI] then decrements
    assert cpu.eax == 0x2B and cpu.esi == 0x3000


def test_stosb_stosw():
    cpu = _cpu("aaab")
    cpu.edi = 0x3000
    cpu.eax = 0x0000AABB
    _steps(cpu, 1)  # STOSB
    assert cpu._mem[0x3000] == 0xBB and cpu.edi == 0x3001
    _steps(cpu, 1)  # STOSW
    assert cpu._mem[0x3001] == 0xBB and cpu._mem[0x3002] == 0xAA
    assert cpu.edi == 0x3003


def test_cmpsb_cmpsw():
    cpu = _cpu("a6a7")
    cpu.esi = 0x3000
    cpu.edi = 0x4000
    cpu._mem[0x3000] = 0x41
    cpu._mem[0x4000] = 0x41
    _steps(cpu, 1)  # CMPSB (equal)
    assert cpu.zf and cpu.esi == 0x3001 and cpu.edi == 0x4001
    cpu._mem[0x3001:0x3003] = bytes([0x00, 0x42])
    cpu._mem[0x4001:0x4003] = bytes([0x00, 0x42])
    _steps(cpu, 1)  # CMPSW (equal)
    assert cpu.zf and cpu.esi == 0x3003 and cpu.edi == 0x4003


def test_scasb_scasw():
    cpu = _cpu("aeaf")
    cpu.edi = 0x3000
    cpu.eax = 0x00000055
    cpu._mem[0x3000] = 0x55
    _steps(cpu, 1)  # SCASB (equal)
    assert cpu.zf and cpu.edi == 0x3001
    cpu._mem[0x3001] = 0x55
    cpu._mem[0x3002] = 0x00
    _steps(cpu, 1)  # SCASW (AX=0x0055 == word 0x0055)
    assert cpu.zf and cpu.edi == 0x3003


# ══════════════════════════════════════════════════════════════════════════════
# REP prefix
# ══════════════════════════════════════════════════════════════════════════════

def test_rep_movsb():
    cpu = _cpu("f3a4")
    cpu.ecx = 3
    cpu.esi = 0x3000
    cpu.edi = 0x4000
    for i, b in enumerate([0x01, 0x02, 0x03]):
        cpu._mem[0x3000 + i] = b
    _steps(cpu, 1)
    assert cpu.ecx == 0
    assert bytes(cpu._mem[0x4000:0x4003]) == bytes([0x01, 0x02, 0x03])


def test_rep_lodsb():
    cpu = _cpu("f3ac")
    cpu.ecx = 2
    cpu.esi = 0x3000
    cpu._mem[0x3000] = 0x77
    cpu._mem[0x3001] = 0x88
    _steps(cpu, 1)
    assert cpu.ecx == 0
    assert cpu.eax == 0x88 and cpu.esi == 0x3002


def test_rep_stosb():
    cpu = _cpu("f3aa")
    cpu.ecx = 3
    cpu.edi = 0x3000
    cpu.eax = 0x41
    _steps(cpu, 1)
    assert cpu.ecx == 0
    assert bytes(cpu._mem[0x3000:0x3003]) == bytes([0x41] * 3)


def test_rep_ret():
    cpu = _cpu("f3c3")
    cpu.esp = 0x3000
    cpu._write32(0x3000, 0x5000)
    _steps(cpu, 1)
    assert cpu.eip == 0x5000


def test_rep_unknown_target():
    cpu = _cpu("f3a6")
    _steps(cpu, 1)
    assert cpu.eip == 0x1002


# ══════════════════════════════════════════════════════════════════════════════
# IN / OUT
# ══════════════════════════════════════════════════════════════════════════════

def test_in_out_ports():
    out = []
    cpu = _cpu("e460e560ecede660e760eeef")
    cpu.register_io_in(0x60, lambda: 0x5A)
    cpu.register_io_out(0x60, lambda v: out.append(v))
    _steps(cpu, 1)  # IN AL, 0x60
    assert cpu._get8l(0) == 0x5A
    _steps(cpu, 1)  # IN EAX, 0x60
    assert cpu.eax == 0x5A
    cpu.edx = 0x61
    cpu.register_io_in(0x61, lambda: 0x7B)
    _steps(cpu, 1)  # IN AL, DX
    assert cpu._get8l(0) == 0x7B
    _steps(cpu, 1)  # IN EAX, DX
    assert cpu.eax == 0x7B
    cpu.eax = 0x11
    _steps(cpu, 1)  # OUT 0x60, AL
    assert out == [0x11]
    _steps(cpu, 1)  # OUT 0x60, EAX
    assert out == [0x11, 0x11]
    cpu.eax = 0x22
    cpu.register_io_out(0x61, lambda v: out.append(v))
    _steps(cpu, 1)  # OUT DX, AL
    assert out == [0x11, 0x11, 0x22]
    _steps(cpu, 1)  # OUT DX, EAX
    assert out == [0x11, 0x11, 0x22, 0x22]


def test_port_in_default_and_unhandled_out():
    cpu = _cpu("e440e640")
    _steps(cpu, 1)  # IN AL, 0x40 (no handler -> 0xFF)
    assert cpu.eax == 0xFF
    _steps(cpu, 1)  # OUT 0x40, AL (no handler -> no-op)
    assert cpu.eip == 0x1004


# ══════════════════════════════════════════════════════════════════════════════
# Shift / rotate groups
# ══════════════════════════════════════════════════════════════════════════════

def test_shift_groups_c1():
    cpu = _cpu("c1c003c1c803c1e003c1e803c1f803c1d003c1e000")
    cpu.eax = 0x00000001
    _steps(cpu, 1)  # ROL EAX, 3
    assert cpu.eax == 0x8
    _steps(cpu, 1)  # ROR EAX, 3
    assert cpu.eax == 0x1
    _steps(cpu, 1)  # SHL EAX, 3
    assert cpu.eax == 0x8
    _steps(cpu, 1)  # SHR EAX, 3
    assert cpu.eax == 0x1
    _steps(cpu, 1)  # SAR EAX, 3 (positive)
    assert cpu.eax == 0
    _steps(cpu, 1)  # RCL (unsupported op) -> no-op
    assert cpu.eax == 0
    _steps(cpu, 1)  # SHL EAX, 0 (count 0)
    assert cpu.eax == 0


def test_shift_d1_and_mem():
    cpu = _cpu("d1e0c12303d123")
    cpu.eax = 2
    _steps(cpu, 1)  # SHL EAX, 1
    assert cpu.eax == 4
    cpu.ebx = 0x3000
    cpu._write32(0x3000, 8)
    _steps(cpu, 1)  # SHL dword [EBX], 3
    assert cpu._read32(0x3000) == 64
    _steps(cpu, 1)  # SHL dword [EBX], 1
    assert cpu._read32(0x3000) == 128


def test_shift_8bit_groups():
    cpu = _cpu("c0e003d0e0d2e0d3e0d023")
    cpu.eax = 0x10
    _steps(cpu, 1)  # SHL AL, 3
    assert cpu._get8l(0) == 0x80
    _steps(cpu, 1)  # SHL AL, 1
    assert cpu._get8l(0) == 0x00
    cpu.eax = 0x01
    cpu.ecx = 2
    _steps(cpu, 1)  # SHL AL, CL
    assert cpu._get8l(0) == 0x04
    _steps(cpu, 1)  # SHL EAX, CL
    assert cpu.eax == 0x10
    cpu.ebx = 0x3000
    cpu._mem[0x3000] = 1
    _steps(cpu, 1)  # SHL byte [EBX], 1
    assert cpu._mem[0x3000] == 2


def test_sar_sign_extension():
    cpu = _cpu("b800000080c1f802")
    _steps(cpu, 2)  # MOV EAX, 0x80000000; SAR EAX, 2
    assert cpu.eax == 0xE0000000


# ══════════════════════════════════════════════════════════════════════════════
# Group 3 (F6 / F7) multiply/divide
# ══════════════════════════════════════════════════════════════════════════════

def test_f6_mul8():
    cpu = _cpu("f6e3")
    cpu.eax = 0x00000005
    cpu.ebx = 0x00000007
    _steps(cpu, 1)  # MUL BL -> AX = 35
    assert cpu._regs[0] == 35


def test_f6_imul8_negative():
    cpu = _cpu("f6eb")
    cpu.eax = 0x00000002
    cpu.ebx = 0x000000FE  # BL = -2
    _steps(cpu, 1)  # IMUL: 2 * -2 = -4 -> AX = 0xFFFC (fits in 8 bits)
    assert cpu._regs[0] & 0xFFFF == 0xFFFC
    assert not cpu.cf


def test_f6_div8():
    cpu = _cpu("f6f3")
    cpu.eax = 0x00000F0F  # AX = 0x0F0F -> q=0xF0, r=0x0F
    cpu.ebx = 0x00000010
    _steps(cpu, 1)  # DIV BL -> AL=0xF0, AH=0x0F
    assert cpu._get8l(0) == 0xF0
    assert cpu._get8h(0) == 0x0F


def test_f6_idiv8():
    cpu = _cpu("f6fb")
    cpu.eax = 0x0000FF9C  # AX = -100
    cpu.ebx = 0x0000000A
    _steps(cpu, 1)  # IDIV: -100 / 10 -> AL=0xF6
    assert cpu._get8l(0) == 0xF6
    assert cpu._get8h(0) == 0x00


def test_f6_div8_by_zero():
    cpu = _cpu("f6f3")
    cpu.eax = 0x00000001
    cpu.ebx = 0
    with pytest.raises(InsFault):
        cpu._exec_one()


def test_f6_div8_overflow():
    cpu = _cpu("f6f3")
    cpu.eax = 0x0000FFFF
    cpu.ebx = 0x00000001
    with pytest.raises(InsFault):
        cpu._exec_one()


def test_f6_mul8_mem():
    cpu = _cpu("f623")
    cpu.eax = 0x00000003
    cpu.ebx = 0x3000
    cpu._mem[0x3000] = 4
    _steps(cpu, 1)
    assert cpu._regs[0] == 12


def test_f7_test_not_neg():
    cpu = _cpu("f7c001000000f7d0f7d8")
    cpu.eax = 0xFFFFFFFF
    _steps(cpu, 1)  # TEST EAX, 1
    assert not cpu.zf
    _steps(cpu, 1)  # NOT EAX
    assert cpu.eax == 0
    _steps(cpu, 1)  # NEG EAX
    assert cpu.eax == 0


def test_f7_mem_forms():
    cpu = _cpu("f70301000000f710f718")
    cpu.ebx = 0x3000
    cpu.eax = 0
    cpu._write32(0x3000, 3)
    _steps(cpu, 1)  # TEST [EBX], 1
    assert not cpu.zf
    cpu.eax = 0x3000
    _steps(cpu, 1)  # NOT [EAX]
    assert cpu._read32(0x3000) == 0xFFFFFFFC
    _steps(cpu, 1)  # NEG [EAX]
    assert cpu._read32(0x3000) == 4


def test_f7_mul_imul():
    cpu = _cpu("f7e3f7eb")
    cpu.eax = 0x00000005
    cpu.ebx = 0x00000007
    _steps(cpu, 1)  # MUL EBX
    assert cpu.eax == 35 and cpu.edx == 0
    cpu.eax = 0x00010000
    cpu.ebx = 0x00010000
    _steps(cpu, 1)  # IMUL EBX -> EDX:EAX = 0x100000000
    assert cpu.eax == 0 and cpu.edx == 1


def test_f7_div():
    cpu = _cpu("f7f3")
    cpu.edx = 0
    cpu.eax = 100
    cpu.ebx = 7
    _steps(cpu, 1)
    assert cpu.eax == 14 and cpu.edx == 2


def test_f7_idiv():
    cpu = _cpu("f7fb")
    cpu.edx = 0xFFFFFFFF
    cpu.eax = 0xFFFFFF9C  # -100
    cpu.ebx = 10
    _steps(cpu, 1)  # IDIV: -100 / 10 -> -10
    assert cpu.eax == 0xFFFFFFF6 and cpu.edx == 0


def test_f7_div_by_zero():
    cpu = _cpu("f7f3")
    cpu.eax = 5
    cpu.ebx = 0
    with pytest.raises(InsFault):
        cpu._exec_one()


def test_66_f7_test():
    # 66 F7 C0 0100 — TEST AX, 1
    cpu = _cpu("66f7c00100", regs={"eip": 0x1000})
    cpu._set16(0, 0x0003)  # AX = 3
    _steps(cpu, 1)
    assert not cpu.zf  # 3 & 1 = 1, not zero

def test_66_f7_not():
    # 66 F7 D0 — NOT AX
    cpu = _cpu("66f7d0", regs={"eip": 0x1000})
    cpu._set16(0, 0x1234)  # AX = 0x1234
    _steps(cpu, 1)
    assert cpu._get16(0) == 0xEDCB

def test_66_f7_neg():
    # 66 F7 D8 — NEG AX
    cpu = _cpu("66f7d8", regs={"eip": 0x1000})
    cpu._set16(0, 5)  # AX = 5
    _steps(cpu, 1)
    assert cpu._get16(0) == 0xFFFB  # -5 in 16-bit
    assert cpu.cf

def test_66_f7_mul():
    # 66 F7 E3 — MUL BX (DX:AX = AX * BX)
    cpu = _cpu("66f7e3", regs={"eip": 0x1000})
    cpu._set16(0, 5)   # AX = 5
    cpu._set16(3, 7)   # BX = 7
    _steps(cpu, 1)
    assert cpu._get16(0) == 35   # AX = low 16 bits
    assert cpu._get16(2) == 0    # DX = high 16 bits
    assert not cpu.cf  # result fits in 16 bits

def test_66_f7_div():
    # 66 F7 F3 — DIV BX (AX / BX)
    cpu = _cpu("66f7f3", regs={"eip": 0x1000})
    cpu._set16(0, 100)  # AX = 100
    cpu._set16(3, 7)    # BX = 7
    _steps(cpu, 1)
    assert cpu._get16(0) == 14   # AX = quotient
    assert cpu._get16(2) == 2    # DX = remainder


def test_66_group1_add_sub_cmp():
    # 66 83 C0 05 — ADD AX, 5 (sign-extended imm8)
    cpu = _cpu("6683c005", regs={"eip": 0x1000})
    cpu._set16(0, 10)
    _steps(cpu, 1)
    assert cpu._get16(0) == 15

    # 66 83 E8 03 — SUB AX, 3
    cpu = _cpu("6683e803", regs={"eip": 0x1000})
    cpu._set16(0, 10)
    _steps(cpu, 1)
    assert cpu._get16(0) == 7

    # 66 81 C0 1027 — ADD AX, 0x2710 (imm16)
    cpu = _cpu("6681c01027", regs={"eip": 0x1000})
    cpu._set16(0, 100)
    _steps(cpu, 1)
    assert cpu._get16(0) == 10100

    # 66 83 F8 05 — CMP AX, 5 (sets flags, no write)
    cpu = _cpu("6683f805", regs={"eip": 0x1000})
    cpu._set16(0, 5)
    _steps(cpu, 1)
    assert cpu.zf  # 5 == 5

    # 66 83 C0 FB — ADD AX, -5 (sign-extended imm8, tests fix: 0xFFFFFF00 not 0xFFFFFFF0)
    cpu = _cpu("6683c0fb", regs={"eip": 0x1000})
    cpu._set16(0, 10)
    _steps(cpu, 1)
    assert cpu._get16(0) == 5  # 10 + (-5) = 5


def test_66_div_uses_dx_ax():
    # 66 F7 F3 — DIV BX: DX:AX / BX
    cpu = _cpu("66f7f3", regs={"eip": 0x1000})
    cpu._set16(0, 0)  # AX low
    cpu._set16(2, 1)  # DX = 1 → dividend = 0x00010000 = 65536
    cpu._set16(3, 100)  # BX = 100
    _steps(cpu, 1)
    assert cpu._get16(0) == 655  # AX = 65536 / 100
    assert cpu._get16(2) == 36   # DX = 65536 % 100


def test_66_alu_reg_reg():
    # 66 01 D8 — ADD AX, BX (r/m16, r16)
    cpu = _cpu("6601d8", regs={"eip": 0x1000})
    cpu._set16(0, 10)  # AX
    cpu._set16(3, 20)  # BX
    _steps(cpu, 1)
    assert cpu._get16(0) == 30

    # 66 03 C3 — ADD AX, BX (r16, r/m16)
    cpu = _cpu("6603c3", regs={"eip": 0x1000})
    cpu._set16(0, 10)
    cpu._set16(3, 20)
    _steps(cpu, 1)
    assert cpu._get16(0) == 30

    # 66 29 D8 — SUB AX, BX (r/m16, r16)
    cpu = _cpu("6629d8", regs={"eip": 0x1000})
    cpu._set16(0, 30)
    cpu._set16(3, 20)
    _steps(cpu, 1)
    assert cpu._get16(0) == 10

    # 66 39 C3 — CMP AX, BX (r/m16, r16, no write)
    cpu = _cpu("6639c3", regs={"eip": 0x1000})
    cpu._set16(0, 20)
    cpu._set16(3, 20)
    _steps(cpu, 1)
    assert cpu.zf
    assert cpu._get16(0) == 20  # CMP doesn't write


def test_66_inc_dec():
    # 66 40 — INC AX
    cpu = _cpu("6640", regs={"eip": 0x1000})
    cpu._set16(0, 0xFFFE)
    _steps(cpu, 1)
    assert cpu._get16(0) == 0xFFFF

    # 66 48 — DEC AX
    cpu = _cpu("6648", regs={"eip": 0x1000})
    cpu._set16(0, 1)
    _steps(cpu, 1)
    assert cpu._get16(0) == 0


def test_66_push_pop():
    # 66 50 — PUSH AX; 66 5B — POP BX
    cpu = _cpu("6650665b", regs={"eip": 0x1000, "esp": 0x2000})
    cpu._set16(0, 0x1234)  # AX = 0x1234
    _steps(cpu, 1)  # PUSH AX
    assert cpu.esp == 0x1FFE
    _steps(cpu, 1)  # POP BX
    assert cpu._get16(3) == 0x1234  # BX = 0x1234


def test_66_xchg():
    # 66 91 — XCHG AX, CX
    cpu = _cpu("6691", regs={"eip": 0x1000})
    cpu._set16(0, 0xAAAA)  # AX
    cpu._set16(1, 0xBBBB)  # CX
    _steps(cpu, 1)
    assert cpu._get16(0) == 0xBBBB  # AX now has CX's value
    assert cpu._get16(1) == 0xAAAA  # CX now has AX's value


# ══════════════════════════════════════════════════════════════════════════════
# Interrupts / IRQ / flag ops / RET
# ══════════════════════════════════════════════════════════════════════════════

def test_flag_ops():
    cpu = _cpu("fafbfcfd")
    _steps(cpu, 1)  # CLI
    assert not cpu.if_
    _steps(cpu, 1)  # STI
    assert cpu.if_
    _steps(cpu, 1)  # CLD
    assert not cpu.df
    _steps(cpu, 1)  # STD
    assert cpu.df


def test_ret_retf():
    cpu = _cpu("c3cb")
    cpu.esp = 0x3000
    cpu._write32(0x3000, 0x4000)
    cpu._write32(0x3004, 0x5000)
    cpu._write32(0x3008, 0x1234)
    _steps(cpu, 1)  # RET
    assert cpu.eip == 0x4000 and cpu.esp == 0x3004
    cpu._eip = 0x1001
    _steps(cpu, 1)  # RETF
    assert cpu.eip == 0x5000 and cpu.esp == 0x300C


def test_int_soft():
    cpu = _cpu("fbcd10")  # STI; INT 0x10 (>= 16 -> IF untouched)
    _steps(cpu, 1)
    _steps(cpu, 1)
    assert cpu.eip == 0x1003
    assert cpu.if_


def test_int_with_handler():
    cpu = _cpu("cd10")
    calls = []
    cpu.register_handler(0x10, lambda c: calls.append(c._get32(0)))
    _steps(cpu, 1)
    assert calls == [0]
    assert cpu.eip == 0x1002


def test_iret():
    cpu = _cpu("cf")
    cpu.esp = 0x3000
    cpu._write32(0x3000, 0x5000)
    cpu._write32(0x3004, 0)
    cpu._write32(0x3008, 0x246)
    _steps(cpu, 1)
    assert cpu.eip == 0x5000
    assert cpu._eflags == 0x246


def test_irq_dispatch_clears_and_restores_if():
    cpu = _cpu("fb90")  # STI; NOP
    observed = []
    cpu.register_handler(1, lambda c: observed.append(c.if_))
    cpu.fire_irq(1)
    cpu.fire_irq(1)
    _steps(cpu, 1)  # STI
    assert cpu.if_
    _steps(cpu, 1)  # both pending IRQs dispatched in pre-step check, then NOP runs
    assert observed == [False, False]
    assert cpu.if_
    assert cpu.eip == 0x1002


def test_irq_held_when_if_cleared():
    cpu = _cpu("9090")
    cpu.fire_irq(1)
    _steps(cpu, 1)  # IF cleared -> IRQ stays pending
    assert cpu._irq_pending == [1]
    _steps(cpu, 1)
    assert cpu._irq_pending == [1]


def test_fire_irq_and_push_key_helpers():
    cpu = _cpu("")
    cpu.fire_irq(3)
    assert cpu._irq_pending == [3]
    cpu.push_key("A")
    assert cpu._kbd_buffer
    cpu.push_scancode(0x1E)
    assert len(cpu._kbd_buffer) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# Misc / helpers
# ══════════════════════════════════════════════════════════════════════════════

def test_unknown_opcode_raises():
    cpu = _cpu("62")
    with pytest.raises(InsFault, match="unknown opcode"):
        cpu.step()


def test_load_overflow_raises():
    cpu = X86CPU(memory_size=0x1000)
    with pytest.raises(ValueError):
        cpu.load(b"\x90" * 0x1001, 0)


def test_dump_helpers():
    cpu = _cpu("")
    cpu.eax = 0x12345678
    cpu._eflags = FLAG_CF | FLAG_ZF | FLAG_SF | FLAG_OF | FLAG_PF | FLAG_IF
    dump = cpu.reg_dump()
    assert "EAX=12345678" in dump
    assert "EIP=" in dump
    flags = cpu.eflags_str()
    assert "C" in flags and "Z" in flags and "S" in flags
    assert "P" in flags and "O" in flags and "I" in flags
    cpu._mem[0x4000] = 0x41
    d = cpu.mem_dump(0x4000, 16)
    assert "4000" in d and "A" in d


def test_cc_condition_all_codes():
    cpu = _cpu("")
    for mask in (0x02, 0x02 | FLAG_CF | FLAG_ZF | FLAG_SF | FLAG_OF | FLAG_PF):
        cpu._eflags = mask
        for cc in range(16):
            cpu._cc_condition(cc)
    cpu._eflags = FLAG_CF | 0x02
    assert cpu._cc_condition(0x2) and cpu._cc_condition(0x6)
    assert not cpu._cc_condition(0x7)
    cpu._eflags = FLAG_ZF | 0x02
    assert cpu._cc_condition(0x6) and not cpu._cc_condition(0x7)
    cpu._eflags = 0x02
    assert cpu._cc_condition(0x7)
    cpu._eflags = FLAG_SF | 0x02
    assert cpu._cc_condition(0x8) and not cpu._cc_condition(0x9)
    cpu._eflags = FLAG_PF | 0x02
    assert cpu._cc_condition(0xA) and not cpu._cc_condition(0xB)
    cpu._eflags = FLAG_OF | 0x02
    assert cpu._cc_condition(0xC) and not cpu._cc_condition(0xD)
    assert cpu._cc_condition(0xE) and not cpu._cc_condition(0xF)


# ══════════════════════════════════════════════════════════════════════════════
# Targeted raw-byte tests for remaining missed lines
# ══════════════════════════════════════════════════════════════════════════════

def test_ff_call_r_m32_mem():
    cpu = _cpu("ff10", regs={"eax": 0x5000, "eip": 0x1000})
    cpu._write32(0x5000, 0x7000)
    _steps(cpu, 1)
    assert cpu.eip == 0x7000

def test_ff_jmp_r_m32_mem():
    cpu = _cpu("ff20", regs={"eax": 0x5000, "eip": 0x1000})
    cpu._write32(0x5000, 0x7000)
    _steps(cpu, 1)
    assert cpu.eip == 0x7000

def test_ff_push_r_m32_mem():
    cpu = _cpu("ff30", regs={"eax": 0x5000, "eip": 0x1000})
    cpu._write32(0x5000, 0xDEADBEEF)
    old_esp = cpu.esp
    _steps(cpu, 1)
    assert cpu.esp == old_esp - 4
    assert cpu._read32(cpu.esp) == 0xDEADBEEF

def test_ff_inc_r_m32_mem():
    cpu = _cpu("ff00", regs={"eax": 0x5000, "eip": 0x1000})
    cpu._write32(0x5000, 41)
    _steps(cpu, 1)
    assert cpu._read32(0x5000) == 42

def test_ff_dec_r_m32_mem():
    cpu = _cpu("ff08", regs={"eax": 0x5000, "eip": 0x1000})
    cpu._write32(0x5000, 42)
    _steps(cpu, 1)
    assert cpu._read32(0x5000) == 41

def test_jcc_backward_offset():
    cpu = _cpu("74fe", regs={"eip": 0x1000})
    cpu._set_flag(FLAG_ZF, True)
    _steps(cpu, 1)
    assert cpu.eip == 0x1000

def test_mov_r8_from_r_m8_mem():
    cpu = _cpu("8a00", regs={"eax": 0x5000, "eip": 0x1000})
    cpu._mem[0x5000] = 0xAB
    _steps(cpu, 1)
    assert cpu._get8l(0) == 0xAB

def test_mov_r32_from_r_m32_mem():
    cpu = _cpu("8b00", regs={"eax": 0x5000, "eip": 0x1000})
    cpu._write32(0x5000, 0x12345678)
    _steps(cpu, 1)
    assert cpu.eax == 0x12345678

def test_mov_r16_imm16_reg_form():
    cpu = _cpu("66c7c33412", regs={"eip": 0x1000})
    _steps(cpu, 1)
    assert cpu._get16(3) == 0x1234

def test_mov_r16_from_r_m16_mem():
    # 66 8B 19 — MOV BX, [ECX]
    cpu = _cpu("668b19", regs={"ecx": 0x3000, "eip": 0x1000})
    cpu._write16(0x3000, 0xBEEF)
    _steps(cpu, 1)
    assert cpu._get16(3) == 0xBEEF

def test_mov_r16_to_r_m16_mem():
    # 66 89 19 — MOV [ECX], BX
    cpu = _cpu("668919", regs={"ecx": 0x3000, "ebx": 0xCAFE, "eip": 0x1000})
    _steps(cpu, 1)
    assert cpu._read16(0x3000) == 0xCAFE

def test_mov_r16_from_r_m16_mem_disp8():
    # 66 8B 72 04 — MOV SI, [EDX+4]
    cpu = _cpu("668b7204", regs={"edx": 0x3000, "eip": 0x1000})
    cpu._write16(0x3004, 0x1234)
    _steps(cpu, 1)
    assert cpu._get16(6) == 0x1234

def test_mov_r16_from_r_m16_high_reg():
    # 66 8B 19 — MOV BX, [ECX] — verifies BX (reg=3, not 0)
    cpu = _cpu("668b19", regs={"ecx": 0x3000, "eip": 0x1000})
    cpu._write16(0x3000, 0x5678)
    _steps(cpu, 1)
    assert cpu._get16(3) == 0x5678  # BX = register 3

def test_group1_sign_extend_imm32():
    cpu = _cpu("81e800000080", regs={"eip": 0x1000})
    cpu.eax = 0
    _steps(cpu, 1)
    assert cpu.eax == 0x80000000

def test_alu_adc_byte():
    cpu = _cpu("80d005", regs={"eip": 0x1000})  # ADC AL, 5
    cpu._set8l(0, 10)
    cpu._set_flag(FLAG_CF, False)
    _steps(cpu, 1)
    assert cpu._get8l(0) == 15
    cpu = _cpu("80d005", regs={"eip": 0x1000})  # ADC AL, 5 with carry
    cpu._set8l(0, 10)
    cpu._set_flag(FLAG_CF, True)
    _steps(cpu, 1)
    assert cpu._get8l(0) == 16

def test_alu_sbb_via_0x81():
    cpu = _cpu("81d834120000", regs={"eip": 0x1000})  # SBB EAX, 0x1234
    cpu.eax = 0x1234 + 100
    cpu._set_flag(FLAG_CF, False)
    _steps(cpu, 1)
    assert cpu.eax == 100
    cpu = _cpu("81d834120000", regs={"eip": 0x1000})  # SBB with borrow
    cpu.eax = 0x1234 + 100
    cpu._set_flag(FLAG_CF, True)
    _steps(cpu, 1)
    assert cpu.eax == 99

def test_shift_rcl_fallback():
    cpu = _cpu("d0d2", regs={"eip": 0x1000})
    cpu._set8l(2, 0x80)
    cpu._set_flag(FLAG_CF, False)
    _steps(cpu, 1)
    assert cpu._get8l(2) == 0x80

def test_shift_rcr_fallback():
    cpu = _cpu("d0da", regs={"eip": 0x1000})
    cpu._set8l(2, 0x80)
    cpu._set_flag(FLAG_CF, False)
    _steps(cpu, 1)
    assert cpu._get8l(2) == 0x80

def test_shift_rcl_mem():
    cpu = _cpu("c01302", regs={"ebx": 0x5000, "eip": 0x1000})
    cpu._mem[0x5000] = 0x42
    cpu._set_flag(FLAG_CF, False)
    _steps(cpu, 1)
    assert cpu._mem[0x5000] == 0x42

def test_shift_rcr_mem():
    cpu = _cpu("c11b03", regs={"ebx": 0x5000, "eip": 0x1000})
    cpu._write32(0x5000, 0xAABBCCDD)
    cpu._set_flag(FLAG_CF, False)
    _steps(cpu, 1)
    assert cpu._read32(0x5000) == 0xAABBCCDD

def test_shift_d2_mem():
    cpu = _cpu("d223", regs={"ebx": 0x5000, "ecx": 3, "eip": 0x1000})
    cpu._mem[0x5000] = 0x05
    _steps(cpu, 1)
    assert cpu._mem[0x5000] == 0x28

def test_shift_d3_mem():
    cpu = _cpu("d323", regs={"ebx": 0x5000, "ecx": 4, "eip": 0x1000})
    cpu._write32(0x5000, 0x0000000F)
    _steps(cpu, 1)
    assert cpu._read32(0x5000) == 0x000000F0

def test_mul_r_m8_mem():
    cpu = _cpu("f621", regs={"ecx": 0x5000, "eip": 0x1000})
    cpu._set8l(0, 7)  # AL = 7
    cpu._mem[0x5000] = 0x10  # [ECX] = 0x10
    _steps(cpu, 1)
    assert cpu._get16(0) == 0x70

def test_div_r_m8_mem_zero():
    cpu = _cpu("f631", regs={"ecx": 0x5000, "eip": 0x1000})
    cpu._mem[0x5000] = 0
    with pytest.raises(InsFault):
        cpu._exec_one()

def test_idiv_r_m8_mem():
    cpu = _cpu("f639", regs={"ecx": 0x5000, "eip": 0x1000})
    cpu._mem[0x5000] = 3
    cpu._set16(0, 10)  # AX = 10
    _steps(cpu, 1)
    assert cpu._get8l(0) == 3  # AL = quotient
    assert cpu._get8h(0) == 1  # AH = remainder

def test_idiv_r_m8_mem_zero():
    cpu = _cpu("f639", regs={"ecx": 0x5000, "eip": 0x1000})
    cpu._mem[0x5000] = 0
    with pytest.raises(InsFault):
        cpu._exec_one()

def test_f6_unary_test_fallback():
    cpu = _cpu("f6c005", regs={"eip": 0x1000})
    cpu._set8l(0, 0xFF)
    _steps(cpu, 1)
    assert cpu._get8l(0) == 0xFF

def test_mul_r_m32_mem():
    cpu = _cpu("f721", regs={"ecx": 0x5000, "eip": 0x1000})
    cpu._write32(0x5000, 7)
    cpu.eax = 6
    _steps(cpu, 1)
    assert cpu.eax == 42
    assert cpu.edx == 0

def test_imul_r_m32_mem():
    cpu = _cpu("f729", regs={"ecx": 0x5000, "eip": 0x1000})
    cpu._write32(0x5000, 5)
    cpu.eax = -3
    _steps(cpu, 1)
    assert cpu.eax == 0xFFFFFFF1

def test_div_r_m32_mem():
    cpu = _cpu("f731", regs={"ecx": 0x5000, "eip": 0x1000})
    cpu._write32(0x5000, 7)
    cpu.eax = 20
    cpu.edx = 0
    _steps(cpu, 1)
    assert cpu.eax == 2
    assert cpu.edx == 6

def test_div_r_m32_overflow():
    cpu = _cpu("f731", regs={"ecx": 0x5000, "eip": 0x1000})
    cpu._write32(0x5000, 1)
    cpu.eax = 0xFFFFFFFF
    cpu.edx = 0xFFFFFFFF
    with pytest.raises(InsFault):
        cpu._exec_one()

def test_div_r_m32_zero():
    cpu = _cpu("f731", regs={"ecx": 0x5000, "eip": 0x1000})
    cpu._write32(0x5000, 0)
    with pytest.raises(InsFault):
        cpu._exec_one()

def test_idiv_r_m32_mem():
    cpu = _cpu("f739", regs={"ecx": 0x5000, "eip": 0x1000})
    cpu._write32(0x5000, 7)
    cpu.eax = 20
    cpu.edx = 0
    _steps(cpu, 1)
    assert cpu.eax == 2
    assert cpu.edx == 6

def test_idiv_r_m32_zero():
    cpu = _cpu("f739", regs={"ecx": 0x5000, "eip": 0x1000})
    cpu._write32(0x5000, 0)
    with pytest.raises(InsFault):
        cpu._exec_one()

def test_f7_unary_neg_mem():
    cpu = _cpu("f719", regs={"ecx": 0x5000, "eip": 0x1000})
    cpu._write32(0x5000, 5)
    _steps(cpu, 1)
    assert cpu._read32(0x5000) == 0xFFFFFFFB

def test_f7_unary_not_mem():
    cpu = _cpu("f711", regs={"ecx": 0x5000, "eip": 0x1000})
    cpu._write32(0x5000, 0x12345678)
    _steps(cpu, 1)
    assert cpu._read32(0x5000) == 0xEDCBA987

def test_unknown_opcode():
    cpu = _cpu("dd", regs={"eip": 0x1000})
    with pytest.raises(InsFault, match="unknown opcode"):
        cpu.step()

def test_mov_r_m8_to_r8_high():
    cpu = _cpu("8a20", regs={"eax": 0x5000, "eip": 0x1000})
    cpu._mem[0x5000] = 0xCD
    _steps(cpu, 1)
    assert cpu._get8h(0) == 0xCD

def test_mov_r8_high_to_r_m8():
    cpu = _cpu("8821", regs={"ecx": 0x5000, "eip": 0x1000})
    cpu._set8h(0, 0xBE)  # AH = 0xBE
    _steps(cpu, 1)
    assert cpu._mem[0x5000] == 0xBE


def test_xchg_reg32_reg32():
    cpu = _cpu("87c3", regs={"eax": 111, "ebx": 222, "eip": 0x1000})
    _steps(cpu, 1)
    assert cpu.eax == 222
    assert cpu.ebx == 111


def test_xchg_eax_ecx():
    cpu = _cpu("87c1", regs={"eax": 0xAAAAAAAA, "ecx": 0xBBBBBBBB, "eip": 0x1000})
    _steps(cpu, 1)
    assert cpu.eax == 0xBBBBBBBB
    assert cpu.ecx == 0xAAAAAAAA


def test_xchg_reg_mem():
    cpu = _cpu("874500", regs={"eax": 0xCAFEBABE, "ebp": 0x50000, "eip": 0x1000})
    cpu._write32(0x50000, 0x12345678)
    _steps(cpu, 1)
    assert cpu.eax == 0x12345678
    assert cpu._read32(0x50000) == 0xCAFEBABE


def test_xchg_mem_reg():
    cpu = _cpu("874d00", regs={"ecx": 0xDEADBEEF, "ebp": 0x50000, "eip": 0x1000})
    cpu._write32(0x50000, 0x11111111)
    _steps(cpu, 1)
    assert cpu.ecx == 0x11111111
    assert cpu._read32(0x50000) == 0xDEADBEEF
