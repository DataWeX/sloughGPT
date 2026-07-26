"""Assembly programs and self-test for the AI Networking Processor VM."""

from __future__ import annotations

from .vm import VMRunner

# ── Example Programs ─────────────────────────────────────────────────────────

HELLO_ASM = """\
    LOAD_CONST R0, "Hello, AI Compteur!"
    PRINT R0
    HALT
"""

CLASSICAL_ASM = """\
    LOAD_CONST R0, 40
    LOAD_CONST R1, 22
    IADD R2, R0, R1
    PRINT R2
    IMUL R3, R0, R1
    PRINT R3
    LOAD_CONST R4, 100
    ISUB R5, R4, R2
    PRINT R5
    HALT
"""

TENSOR_MATH_ASM = """\
    LOAD_CONST R0, [1, 2, 3, 4]
    LOAD_CONST R1, [10, 20, 30, 40]
    ADD R2, R0, R1
    PRINT R2
    MUL R3, R0, R1
    PRINT R3
    SUM R4, R2
    PRINT R4
    MEAN R5, R2
    PRINT R5
    HALT
"""

MATRIX_MUL_ASM = """\
    LOAD_CONST R0, [[1, 2], [3, 4]]
    LOAD_CONST R1, [[5, 6], [7, 8]]
    MATMUL R2, R0, R1
    PRINT R2
    TRANSPOSE R3, R0
    PRINT R3
    DOT R4, R0, R1
    PRINT R4
    HALT
"""

NEURAL_NET_ASM = """\
    LOAD_CONST R0, [0.5, 0.3, 0.9, 0.1]
    RELU R1, R0
    PRINT R1
    SIGMOID R2, R0
    PRINT R2
    TANH R3, R0
    PRINT R3
    LOAD_CONST R4, [1.0, 2.0, 3.0]
    SOFTMAX R5, R4
    PRINT R5
    HALT
"""

LOOP_ASM = """\
    LOAD_CONST R0, 0
    LOAD_CONST R1, 5
loop:
    IADD R0, R0, R1
    DEC R1
    TEST R1
    JNZ loop
    PRINT R0
    HALT
"""

FUNCTION_ASM = """\
    CALL add_42
    HALT
add_42:
    LOAD_CONST R0, 20
    LOAD_CONST R1, 22
    IADD R0, R0, R1
    PRINT R0
    RET
"""

MIXED_ASM = """\
    LOAD_CONST R0, 10
    LOAD_CONST R1, [1, 2, 3, 4, 5]
    IADD R2, R0, R0
    SUM R3, R1
    MEAN R4, R1
    ADD R5, R1, R1
    PRINT R2
    PRINT R3
    PRINT R4
    PRINT R5
    HALT
"""

NPU_PROGRAM_ASM = """\
    ; Open the Neural Processing Unit device
    DEV_OPEN   R0, npu

    ; Load a model into the NPU
    DEV_CALL   R1, R0, load_model, qwen, qwen2.5-0.5B-Instruct

    ; Tokenize input text
    DEV_CALL   R2, R0, tokenize, qwen, Hello, how are you?

    ; Generate response (autoregressive inference)
    DEV_CALL   R3, R0, generate, qwen, Hello, how are you?, 50

    ; Detokenize generated tokens back to text
    DEV_CALL   R4, R0, detokenize, qwen, R3

    ; Print the generated text
    PRINT R4

    ; Get model info
    DEV_CALL   R5, R0, info

    ; Get embeddings for the input
    DEV_CALL   R6, R0, embed, qwen, Hello, how are you?

    ; Forward pass (raw logits)
    DEV_CALL   R7, R0, forward, qwen, R2

    ; Unload the model
    DEV_CALL   R8, R0, unload_model, qwen

    ; Release the device handle
    DEV_CLOSE  R0

    HALT
"""

COUNTER_ASM = """; Counter: counts 0..9 and prints each
.data
    newline: db 10, 0

.text
start:
    MOV R1, 0          ; counter = 0
loop:
    MOV R0, 2          ; syscall: write char
    MOV R1, R1         ; value to print
    ADD R1, R1, 48     ; convert to ASCII '0'
    SYSCALL
    SUB R1, R1, 48     ; convert back
    ADD R1, R1, 1      ; counter++
    CMP R1, 10         ; compare with 10
    JL loop            ; if < 10, continue

    MOV R0, 0          ; exit
    SYSCALL
"""

FIB_ASM = """; Fibonacci: prints first 12 numbers
.data
    space: db 32, 0
    newline: db 10, 0

.text
start:
    MOV R1, 0          ; a = 0
    MOV R2, 1          ; b = 1
    MOV R3, 12         ; count = 12

loop:
    ; print a
    MOV R0, 1          ; syscall: write string
    MOV R8, R1
    ADD R8, R8, 48
    MOV R7, R8
    ; print manually via char
    MOV R0, 2
    MOV R1, R7
    SYSCALL

    ; print space
    MOV R0, 2
    MOV R1, 32
    SYSCALL

    ; a, b = b, a+b
    MOV R4, R1
    MOV R5, R2
    ADD R1, R5, R4     ; new a = old b
    ADD R2, R4, R5     ; new b = old a + old b
    MOV R6, R1

    SUB R3, R3, 1      ; count--
    CMP R3, 0
    JNZ loop

    MOV R0, 0
    SYSCALL
"""

COLLATZ_ASM = """; Collatz sequence starting from 27
.data
    space: db 32, 0
    newline: db 10, 0

.text
start:
    MOV R1, 27         ; n = 27
    MOV R3, 0          ; steps = 0

loop:
    ; print n as char (only works for n < 10 for simplicity)
    MOV R0, 2
    MOV R2, R1
    ADD R2, R2, 48
    MOV R1, R2
    SYSCALL

    ; print space
    MOV R0, 2
    MOV R1, 32
    SYSCALL

    ; restore n
    SUB R1, R2, 48

    CMP R1, 1
    JZ done

    ; check if even: n & 1
    MOV R4, R1
    AND R4, R1, 1
    CMP R4, 0
    JZ even

odd:
    MUL R1, R1, 3
    ADD R1, R1, 1
    ADD R3, R3, 1
    JMP loop

even:
    SHR R1, R1, 1
    ADD R3, R3, 1
    JMP loop

done:
    MOV R0, 0
    SYSCALL
"""


# ── Self Test ────────────────────────────────────────────────────────────────

def self_test() -> list[str]:
    """Run built-in programs and report results."""
    results = []
    runner = VMRunner()

    out = runner.assemble_and_run(HELLO_ASM)
    hello_ok = "Hello" in " ".join(out)
    results.append(f"  hello: {'PASS' if hello_ok else 'FAIL'} — output: {out}")

    runner2 = VMRunner()
    out2 = runner2.assemble_and_run(COUNTER_ASM)
    results.append(f"  counter: {'PASS' if out2 else 'FAIL'} — steps: {runner2.cpu._step_count}")

    runner3 = VMRunner()
    out3 = runner3.assemble_and_run(FIB_ASM)
    results.append(f"  fib: {'PASS' if out3 else 'FAIL'} — steps: {runner3.cpu._step_count}")

    runner4 = VMRunner()
    out4 = runner4.assemble_and_run(
        "MOV R0, 10\nPUSH R0\nMOV R0, 20\nPOP R1\nIADD R2, R0, R1\nPRINT R2\nHALT"
    )
    stack_ok = out4 == ["30"]
    results.append(f"  stack_push_pop: {'PASS' if stack_ok else 'FAIL'} — output: {out4}")

    runner5 = VMRunner()
    out5 = runner5.assemble_and_run(
        "LOAD_CONST R0, 3.0\nLOAD_CONST R1, 4.0\nFMUL R2, R0, R1\nPRINT R2\nHALT"
    )
    float_ok = out5 and float(out5[0]) == 12.0
    results.append(f"  float_mul: {'PASS' if float_ok else 'FAIL'} — output: {out5}")

    runner6 = VMRunner()
    out6 = runner6.assemble_and_run(
        "ALLOC R0, 256\nMEMINFO R1\nPRINT R1\nHALT"
    )
    mem_ok = out6 and int(out6[0]) >= 1
    results.append(f"  alloc_meminfo: {'PASS' if mem_ok else 'FAIL'} — output: {out6}")

    return results


# ── Boot & Shell Programs ───────────────────────────────────────────────────

BOOT_ASM = """\
    ; AI Compteur Boot Sequence
    ; Initializes system, prints banner, loads shell
    LOAD_CONST R0, "AI Compteur v0.1"
    OUT 1, R0
    LOAD_CONST R0, "Booting kernel..."
    OUT 1, R0
    LOAD_CONST R0, 0
    MEMINFO R1
    LOAD_CONST R2, "Memory: "
    OUT 1, R2
    OUT 1, R1
    LOAD_CONST R2, " blocks available"
    OUT 1, R2
    LOAD_CONST R0, "Initializing devices..."
    OUT 1, R0
    LOAD_CONST R0, "Console: port 0 (in), port 1 (out)"
    OUT 1, R0
    LOAD_CONST R0, "Ready."
    OUT 1, R0
    HALT
"""

SHELL_ASM = """\
    ; AI Compteur Interactive Shell
    ; Reads commands from port 0, executes, prints to port 1
    ; Simple REPL: read line, echo it back
    LOAD_CONST R0, "ai-compteur> "
    OUT 1, R0
    IN R1, 0
    LOAD_CONST R2, "exec: "
    OUT 1, R2
    OUT 1, R1
    LOAD_CONST R3, "ok"
    OUT 1, R3
    JMP 0
"""


# ── x86 Boot Programs ───────────────────────────────────────────────────────

X86_BOOTLOADER_ASM = """\
; AI Compteur x86 Bootloader
; 512-byte MBR — loads kernel from floppy, switches to protected mode
; Loaded by BIOS at 0x0000:0x7C00

[BITS 16]
[ORG 0x7C00]

; ══════════════════════════════════════════════════════════════════════════
; Stage 1: Real Mode Setup
; ══════════════════════════════════════════════════════════════════════════

boot_start:
    cli
    ; ── EARLIEST serial debug: send 'X' to COM1 ──
    mov dx, 0x3F8
    mov al, 0x58
    out dx, al

    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00
    sti
    cld

    ; Save boot drive number (passed in DL by BIOS)
    mov [boot_drive], dl

    ; ── DEBUG: Send 'Y' to COM1 (reached bootloader setup) ──
    push dx
    push ax
    mov dx, 0x3F8
    mov al, 0x59
    out dx, al
    pop ax
    pop dx

    ; ── Kernel already loaded at 0x1000 by BIOS ──
    ; BIOS copied kernel_code from ROM to 0x0000:0x1000
    ; (skipping INT 10h banner — testing PM switch only)

    ; ── DEBUG: Send 'Z' to COM1 (before PM switch) ──
    push dx
    push ax
    mov dx, 0x3F8
    mov al, 0x5A
    out dx, al
    pop ax
    pop dx

    ; ── Switch to Protected Mode ──
    cli

    ; Load GDT
    lgdt [gdt_desc]

    ; Enable A20 line (fast A20, port 0x92)
    in al, 0x92
    or al, 2
    out 0x92, al

    ; Set PE (Protection Enable) bit in CR0
    ; IMPORTANT: The far jump to 32-bit code MUST immediately follow
    ; mov cr0, eax — no instructions in between, per Intel specification.
    mov eax, cr0
    or eax, 1
    mov cr0, eax
    jmp 0x08:protected_mode

; ══════════════════════════════════════════════════════════════════════════
; INT 10h Teletype Print (used before protected mode switch)
; ══════════════════════════════════════════════════════════════════════════

bios_print:
    pusha
.loop:
    lodsb
    or al, al
    jz .done
    mov ah, 0x0E
    mov bx, 0x0007
    int 0x10
    jmp .loop
.done:
    popa
    ret

; ══════════════════════════════════════════════════════════════════════════
; 32-bit Protected Mode Entry
; ══════════════════════════════════════════════════════════════════════════

[BITS 32]
protected_mode:
    ; ── DEBUG: Send '7' to COM1 at 32-bit entry ──
    ; Note: we're in protected mode, so use 32-bit out
    mov dx, 0x3F8
    mov al, 0x37
    out dx, al

    ; Set up segment registers for flat 32-bit mode
    mov ax, 0x10            ; Data segment selector
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    mov esp, 0x90000        ; Stack at 576KB

    ; Jump to kernel at 0x1000
    jmp 0x1000

; ══════════════════════════════════════════════════════════════════════════
; GDT (Global Descriptor Table)
; ══════════════════════════════════════════════════════════════════════════

gdt_start:
    ; Null descriptor (required)
    dd 0x0
    dd 0x0

gdt_code:                   ; Code segment: base=0, limit=4GB, execute/read
    dw 0xFFFF               ; Limit low
    dw 0x0                  ; Base low
    db 0x0                  ; Base middle
    db 10011010b            ; Access: present, ring 0, code, readable
    db 11001111b            ; Flags: 4KB粒度, 32-bit, limit high=0xF
    db 0x0                  ; Base high

gdt_data:                   ; Data segment: base=0, limit=4GB, read/write
    dw 0xFFFF               ; Limit low
    dw 0x0                  ; Base low
    db 0x0                  ; Base middle
    db 10010010b            ; Access: present, ring 0, data, writable
    db 11001111b            ; Flags: 4KB粒度, 32-bit, limit high=0xF
    db 0x0                  ; Base high

gdt_end:

gdt_desc:
    dw gdt_end - gdt_start - 1   ; GDT limit
    dd gdt_start                   ; GDT base address

; ══════════════════════════════════════════════════════════════════════════
; Data
; ══════════════════════════════════════════════════════════════════════════

boot_drive: db 0
msg_boot:   db "AI Compteur Bootloader v0.1", 0x0D, 0x0A, 0
msg_loaded: db "Kernel loaded. Switching to protected mode...", 0x0D, 0x0A, 0
msg_err:    db "Failed to load kernel from floppy.", 0x0D, 0x0A, 0

; ══════════════════════════════════════════════════════════════════════════
; Padding + MBR Signature
; ══════════════════════════════════════════════════════════════════════════

    times 510-($-$$) db 0
    dw 0xAA55               ; MBR boot signature
"""

X86_KERNEL_ASM = """\
; AI Compteur x86 Kernel
; Loaded by bootloader at 0x1000 in 32-bit protected mode
; Features: VGA output, keyboard input, timer

[BITS 32]
[ORG 0x1000]

; ── Constants ────────────────────────────────────────────────────────────

VGA_BUFFER    equ 0xB8000
VGA_COLS      equ 80
VGA_ROWS      equ 25

; ── Kernel Entry Point ──────────────────────────────────────────────────

kernel_start:
    ; ── Minimal serial test: write 'A' to COM1 immediately ──
    mov dx, 0x3F8          ; COM1 data register
    mov al, 0x41           ; 'A'
    out dx, al

    ; ── Initialize COM1 serial port (115200 baud, 8N1) ──
    mov dx, 0x3F9
    mov al, 0x00
    out dx, al

    mov dx, 0x3FB
    mov al, 0x80
    out dx, al
    mov dx, 0x3F8
    mov al, 0x01
    out dx, al
    mov dx, 0x3F9
    mov al, 0x00
    out dx, al
    mov dx, 0x3FB
    mov al, 0x03
    out dx, al
    mov dx, 0x3FA
    mov al, 0xC7
    out dx, al
    mov dx, 0x3FC
    mov al, 0x0B
    out dx, al

    ; ── Print banner to serial ──
    mov esi, msg_banner_serial
    call serial_print

    ; VGA calls disabled until VGA_BUFFER addressing fixed in PM
    ; call vga_clear

    ; ── Print shell prompt to serial ──
    mov esi, msg_shell_serial
    call serial_print

; ── Interactive shell loop ──
; Reads characters from COM1, echoes them back,
; handles basic commands
shell_loop:
    ; Try to read a character from serial (non-blocking)
    call serial_read
    test al, al
    jz shell_loop           ; No character, keep polling

    ; Got a character in AL
    ; Echo it back
    push eax
    call serial_write
    pop eax

    ; Store in input buffer
    cmp al, 13              ; Enter?
    je shell_enter
    cmp al, 10              ; Linefeed?
    je shell_enter

    ; Buffer the character
    mov ebx, [input_pos]
    cmp ebx, 127
    jge shell_loop          ; Buffer full, ignore
    mov [input_buf + ebx], al
    inc dword [input_pos]
    jmp shell_loop

shell_enter:
    ; Null-terminate the input
    mov ebx, [input_pos]
    mov byte [input_buf + ebx], 0

    ; Echo newline to serial
    mov al, 13
    call serial_write
    mov al, 10
    call serial_write

    ; Check for commands
    ; "help" command
    mov esi, input_buf
    mov edi, cmd_help
    call str_cmp
    je cmd_do_help

    ; "reboot" command
    mov esi, input_buf
    mov edi, cmd_reboot
    call str_cmp
    je cmd_do_reboot

    ; "ticks" command
    mov esi, input_buf
    mov edi, cmd_ticks
    call str_cmp
    je cmd_do_ticks

    ; "clear" command
    mov esi, input_buf
    mov edi, cmd_clear
    call str_cmp
    je cmd_do_clear

    ; Unknown command - print error
    mov esi, msg_unknown
    call serial_print

    ; Reset input buffer
    mov dword [input_pos], 0

    ; Print prompt again
    mov esi, msg_prompt
    call serial_print
    jmp shell_loop

cmd_do_help:
    mov esi, msg_help
    call serial_print
    mov dword [input_pos], 0
    mov esi, msg_prompt
    call serial_print
    jmp shell_loop

cmd_do_reboot:
    mov esi, msg_rebooting
    call serial_print
    ; Triple fault to reboot
    lidt [idt_zero]
    int 3
    jmp shell_loop

cmd_do_ticks:
    ; Print tick count as decimal
    mov eax, [tick_count]
    call print_uint32_serial
    mov al, 13
    call serial_write
    mov al, 10
    call serial_write
    mov dword [input_pos], 0
    mov esi, msg_prompt
    call serial_print
    jmp shell_loop

cmd_do_clear:
    ; Send ANSI clear screen
    mov esi, msg_ansi_clear
    call serial_print
    call vga_clear
    mov dword [input_pos], 0
    mov esi, msg_prompt
    call serial_print
    jmp shell_loop

; ── Serial Functions ──────────────────────────────────────────────────────

serial_init:
    ; Initialize COM1 (115200 baud, 8N1)
    mov dx, 0x3F9
    mov al, 0x00
    out dx, al
    mov dx, 0x3FB
    mov al, 0x80
    out dx, al
    mov dx, 0x3F8
    mov al, 0x01
    out dx, al
    mov dx, 0x3F9
    mov al, 0x00
    out dx, al
    mov dx, 0x3FB
    mov al, 0x03
    out dx, al
    mov dx, 0x3FA
    mov al, 0xC7
    out dx, al
    mov dx, 0x3FC
    mov al, 0x0B
    out dx, al
    ret

serial_write:
    ; Write character in AL to COM1
    push edx
    push ax
    mov dx, 0x3FD           ; Line status register (0x3F8+5)
.wait: in al, dx            ; Read line status
    test al, 0x20           ; Bit 5 = transmitter holding register empty
    jz .wait                ; Loop until ready
    pop ax
    mov dx, 0x3F8           ; Data register
    out dx, al
    pop edx
    ret

serial_read:
    ; Read character from COM1 into AL (0 if nothing available)
    push edx
    mov dx, 0x3FD           ; Line status register (0x3F8+5)
    in al, dx
    test al, 1              ; Bit 0 = data ready
    jz .no_data
    mov dx, 0x3F8           ; Data register
    in al, dx
    pop edx
    ret
.no_data:
    xor al, al
    pop edx
    ret

serial_print:
    ; Print null-terminated string at ESI to COM1
    pusha
.loop:
    lodsb
    test al, al
    jz .done
    call serial_write
    jmp .loop
.done:
    popa
    ret

; ── Utility Functions ─────────────────────────────────────────────────────

str_cmp:
    ; Compare strings at ESI and EDI, ZF=1 if equal
    pusha
.loop:
    mov al, [esi]
    mov bl, [edi]
    cmp al, bl
    jne .ne
    test al, al
    jz .eq                  ; Both null = equal
    inc esi
    inc edi
    jmp .loop
.ne:
    or al, 1            ; Set ZF=0 (al is always nonzero here)
    popa
    ret
.eq:
    xor ax, ax              ; Set ZF=1
    test ax, 1
    popa
    ret

print_uint32_serial:
    ; Print EAX as unsigned decimal to serial
    pusha
    mov ecx, 10
    xor ebx, ebx            ; Digit count
.digit_loop:
    xor edx, edx
    div ecx                 ; EAX = quotient, EDX = remainder
    push edx                ; Save digit
    inc ebx
    test eax, eax
    jnz .digit_loop
.print_loop:
    pop eax
    add al, '0'
    call serial_write
    dec ebx
    jnz .print_loop
    popa
    ret

; ── VGA Functions ────────────────────────────────────────────────────────

vga_print:
    ; ESI = string address, EDI = VGA buffer position, AH = color attribute
    push eax
    push ebx
.vga_loop:
    lodsb                   ; Load byte from [ESI] into AL
    test al, al
    jz .vga_done
    cmp al, 10              ; Newline?
    je .vga_newline
    stosw                   ; Store AX at [EDI], advance EDI by 2
    jmp .vga_loop
.vga_newline:
    push eax
    mov eax, edi
    sub eax, VGA_BUFFER
    mov ebx, VGA_COLS * 2
    xor edx, edx
    div ebx                 ; EAX = current row
    inc eax                 ; Next row
    mul ebx                 ; EAX = offset of next row
    add eax, VGA_BUFFER
    mov edi, eax
    pop eax
    jmp .vga_loop
.vga_done:
    pop ebx
    pop eax
    ret

vga_clear:
    push edi
    push ecx
    mov edi, VGA_BUFFER
    mov ecx, VGA_COLS * VGA_ROWS
    mov ax, 0x0720          ; Space with light gray attribute
    cld
    rep stosw
    pop ecx
    pop edi
    ret

; ── Interrupt Handlers ──────────────────────────────────────────────────

; Timer interrupt handler (IRQ0)
timer_handler:
    push eax
    inc dword [tick_count]
    mov al, 0x20
    out 0x20, al
    pop eax
    iret

; Keyboard interrupt handler (IRQ1)
keyboard_handler:
    push eax
    in al, 0x60             ; Read scancode
    mov [last_scancode], al
    mov al, 0x20
    out 0x20, al
    pop eax
    iret

; ── Data ────────────────────────────────────────────────────────────────

msg_banner_serial: db "AI Compteur v0.3 - Serial Console Active", 13, 10, 0
msg_shell_serial:  db "Type 'help' for commands.", 13, 10, 0
msg_prompt:        db "> ", 0
msg_kernel:   db "AI Compteur Kernel v0.3", 0
msg_serial:   db "Serial: 115200 baud, 8N1", 0
msg_ready:    db "Protected mode active.", 0
msg_pmode:    db "VGA: 80x25, 16 colors.", 0
msg_irq:      db "Interrupts enabled.", 0
msg_help:     db "Commands:", 13, 10
              db "  help    - Show this help", 13, 10
              db "  ticks   - Show timer tick count", 13, 10
              db "  clear   - Clear serial screen", 13, 10
              db "  reboot  - Reboot the system", 13, 10, 0
msg_rebooting: db "Rebooting...", 13, 10, 0
msg_unknown:  db "Unknown command. Type 'help'.", 13, 10, 0
msg_ansi_clear: db 27, "[2J", 27, "[H", 0  ; ESC[2J ESC[H

cmd_help:   db "help", 0
cmd_reboot: db "reboot", 0
cmd_ticks:  db "ticks", 0
cmd_clear:  db "clear", 0

idt_zero:   dw 0, 0, 0     ; Null IDT for triple fault reboot

input_buf:  times 128 db 0
input_pos:  dd 0

tick_count:   dd 0
last_scancode: db 0

; ── Padding ─────────────────────────────────────────────────────────────

times 4096-($-$$) db 0          ; Pad to 4KB total
"""


# ── Binary Export ────────────────────────────────────────────────────────────

def export_x86_binary(source: str) -> bytes:
    """Assemble x86 source to raw binary bytes.

    Strips [BITS], [ORG], labels, and directives.
    Returns raw machine code.
    """
    import re

    lines = source.split('\n')
    code_lines = []

    for line in lines:
        line = line.split(';')[0].strip()
        if not line:
            continue
        if line.startswith('['):
            continue
        if line.startswith('times') and 'db' in line:
            continue
        if line.startswith('dw ') or line.startswith('dd '):
            continue
        if ':' in line and not line.startswith('db'):
            line = line.split(':', 1)[1].strip()
            if not line:
                continue
        code_lines.append(line)

    return '\n'.join(code_lines).encode('utf-8')


def build_disk_image(bootloader: bytes, kernel: bytes, size_mb: int = 1) -> bytes:
    """Build a bootable disk image from bootloader + kernel.

    Layout:
      Sector 0 (512 bytes): Bootloader (MBR)
      Sector 1+ (up to 2KB): Kernel
      Rest: zeroed to size_mb
    """
    image = bytearray(size_mb * 1024 * 1024)

    # Write bootloader to sector 0
    image[:len(bootloader)] = bootloader

    # Write kernel starting at sector 1 (offset 512)
    image[512:512 + len(kernel)] = kernel

    return bytes(image)


def export_disk_image(source: str, output_path: str, size_mb: int = 1) -> str:
    """Assemble x86 source and build a bootable disk image.

    Uses X86Assembler to compile assembly to real machine code.
    Returns path to the created .img file.
    """
    from .vm import X86Assembler

    asm = X86Assembler()
    kernel_bytes = asm.assemble(source or X86_KERNEL_ASM)

    # Pad kernel to at least 4KB
    if len(kernel_bytes) < 4096:
        kernel_bytes = kernel_bytes + b'\x00' * (4096 - len(kernel_bytes))

    image = bytearray(size_mb * 1024 * 1024)

    # Bootloader is raw binary (already valid x86 machine code)
    bootloader = export_x86_binary(X86_BOOTLOADER_ASM)
    image[:len(bootloader)] = bootloader

    # Kernel is assembled machine code
    image[512:512 + len(kernel_bytes)] = kernel_bytes

    with open(output_path, 'wb') as f:
        f.write(image)

    return output_path


def build_boot_image(output_path: str = "boot.img") -> str:
    """Build a bootable disk image from bootloader + kernel.

    Assembles both bootloader and kernel using X86Assembler,
    builds a 1MB disk image.
    """
    from .vm import X86Assembler

    asm = X86Assembler()
    bootloader = asm.assemble(X86_BOOTLOADER_ASM)
    kernel = asm.assemble(X86_KERNEL_ASM)

    # Pad kernel to at least 4KB
    if len(kernel) < 4096:
        kernel = kernel + b'\x00' * (4096 - len(kernel))

    image = bytearray(1024 * 1024)  # 1MB

    # Write bootloader to sector 0
    image[:len(bootloader)] = bootloader

    # Write kernel starting at sector 1 (offset 512)
    image[512:512 + len(kernel)] = kernel

    with open(output_path, 'wb') as f:
        f.write(image)

    return output_path


X86_BIOS_ASM = """\
; AI Compteur Custom BIOS
; 64KB BIOS with INT 10h (video) + INT 13h (disk) handlers
; Copies bootloader from ROM to 0x7C00, then jumps to it
; The bootloader in turn copies the kernel from ROM to 0x1000

[BITS 16]
[ORG 0x0000]

; ══════════════════════════════════════════════════════════════════════════
; BIOS Entry Point (called from reset vector at 0xF000:0x0000)
; ══════════════════════════════════════════════════════════════════════════

bios_entry:
    cli
    xor ax, ax
    mov ss, ax
    mov sp, 0x0400
    cld

    ; Set up segments for IVT writes (DS=0)
    mov ds, ax

    ; ── Set up IVT for INT 10h (video) at IVT[0x40] ──
    mov ax, int10h
    mov [0x40], ax
    mov ax, 0xF000
    mov [0x42], ax

    ; ── Set up IVT for INT 13h (disk) at IVT[0x4C] ──
    mov ax, int13h
    mov [0x4C], ax
    mov ax, 0xF000
    mov [0x4E], ax

    ; ── Set DS = BIOS data segment (0xF000) ──
    mov ax, 0xF000
    mov ds, ax

    ; ── Initialize VGA to 80x25 text mode ──
    ; Program Miscellaneous Output Register (color mode)
    mov dx, 0x3C2
    mov al, 0x63
    out dx, al

    ; Reset Sequencer
    mov dx, 0x3C4
    mov al, 0x00
    out dx, al
    mov dx, 0x3C5
    mov al, 0x03
    out dx, al

    ; Unlock CRTC registers
    mov dx, 0x3D4
    mov al, 0x03
    out dx, al
    mov dx, 0x3D5
    mov al, 0x80
    out dx, al

    ; Set CRTC start address to 0x0000
    mov dx, 0x3D4
    mov al, 0x0C
    out dx, al
    mov dx, 0x3D5
    mov al, 0x00
    out dx, al
    mov dx, 0x3D4
    mov al, 0x0D
    out dx, al
    mov dx, 0x3D5
    mov al, 0x00
    out dx, al

    ; Set cursor to (0,0)
    mov dx, 0x3D4
    mov al, 0x0E
    out dx, al
    mov dx, 0x3D5
    mov al, 0x00
    out dx, al
    mov dx, 0x3D4
    mov al, 0x0F
    out dx, al
    mov dx, 0x3D5
    mov al, 0x00
    out dx, al

    ; Clear BDA cursor position
    mov ax, 0x0040
    mov ds, ax
    mov word [0x0050], 0x0000   ; row=0, col=0

    ; ── DEBUG: Send '1' to COM1 after VGA init ──
    mov dx, 0x3F8
    mov al, 0x31
    out dx, al

    ; Clear VGA screen (write 2000 chars = 80x25)
    push 0xB800
    pop es
    xor di, di
    mov ax, 0x0720       ; space with light gray attribute
    mov cx, 2000
    cld
    rep stosw

    ; ── DEBUG: Send '2' to COM1 after VGA clear ──
    mov dx, 0x3F8
    mov al, 0x32
    out dx, al

    ; Set DS back to BIOS segment
    mov ax, 0xF000
    mov ds, ax

    ; ── Print banner via serial (skip INT 10h for now) ──
    sti
    mov si, msg_banner
    call serial_print_bios

    ; ── DEBUG: Send '3' to COM1 after banner print ──
    mov dx, 0x3F8
    mov al, 0x33
    out dx, al

    ; ── Copy bootloader from ROM to 0x0000:0x7C00 ──
    ; Source: embedded_bootloader at offset boot_code in ROM
    ; Destination: 0x0000:0x7C00
    mov ax, 0xF000
    mov ds, ax
    mov si, boot_code
    xor ax, ax
    mov es, ax
    mov di, 0x7C00
    mov cx, 256            ; 256 words = 512 bytes
    cld
    rep movsw

    ; ── DEBUG: Send '4' to COM1 after bootloader copy ──
    mov dx, 0x3F8
    mov al, 0x34
    out dx, al

    ; ── Copy kernel from ROM to 0x0000:0x1000 ──
    ; Source: embedded_kernel at offset kernel_code in ROM
    ; Destination: 0x0000:0x1000
    mov ax, 0xF000
    mov ds, ax
    mov si, kernel_code
    xor ax, ax
    mov es, ax
    mov di, 0x1000
    mov cx, 2048           ; 2048 words = 4096 bytes
    cld
    rep movsw

    ; ── DEBUG: Send '5' to COM1 after kernel copy ──
    mov dx, 0x3F8
    mov al, 0x35
    out dx, al

    ; ── Jump to bootloader ──
    jmp 0x0000:0x7C00

; ══════════════════════════════════════════════════════════════════════════
; INT 10h — Video Services
; ══════════════════════════════════════════════════════════════════════════

int10h:
    pusha
    push ds
    push es

    xor bx, bx
    mov ds, bx

    cmp ah, 0x00
    je .set_mode
    cmp ah, 0x03
    je .get_cursor
    cmp ah, 0x0E
    je .tty
    jmp .done

.set_mode:
    ; Store mode in BDA
    mov byte [0x0449], al

    ; ── Full VGA hardware init for 80x25 color text mode ──

    ; Miscellaneous Output Register (color mode, 80x25)
    mov dx, 0x3C2
    mov al, 0x63
    out dx, al

    ; Sequencer: reset then enable
    mov dx, 0x3C4
    mov al, 0x00            ; Reset register
    out dx, al
    mov dx, 0x3C5
    mov al, 0x03            ; End reset
    out dx, al

    ; Clocking Mode: enable display
    mov dx, 0x3C4
    mov al, 0x01            ; Clocking Mode register
    out dx, al
    mov dx, 0x3C5
    mov al, 0x00            ; Normal operation
    out dx, al

    ; Map Mask: planes 0+1
    mov dx, 0x3C4
    mov al, 0x02
    out dx, al
    mov dx, 0x3C5
    mov al, 0x03
    out dx, al

    ; CRTC: unlock vertical retrace
    mov dx, 0x3D4
    mov al, 0x03
    out dx, al
    mov dx, 0x3D5
    mov al, 0x80
    out dx, al

    ; CRTC: Start Address = 0x0000
    mov dx, 0x3D4
    mov al, 0x0C            ; Start Address High
    out dx, al
    mov dx, 0x3D5
    mov al, 0x00
    out dx, al
    mov dx, 0x3D4
    mov al, 0x0D            ; Start Address Low
    out dx, al
    mov dx, 0x3D5
    mov al, 0x00
    out dx, al

    ; CRTC: Cursor = (0,0)
    mov dx, 0x3D4
    mov al, 0x0E            ; Cursor Location High
    out dx, al
    mov dx, 0x3D5
    mov al, 0x00
    out dx, al
    mov dx, 0x3D4
    mov al, 0x0F            ; Cursor Location Low
    out dx, al
    mov dx, 0x3D5
    mov al, 0x00
    out dx, al

    ; Graphics Controller: misc = 0x0D
    mov dx, 0x3CE
    mov al, 0x06
    out dx, al
    mov dx, 0x3CF
    mov al, 0x0D
    out dx, al

    ; Attribute Controller: enable display output
    mov dx, 0x3DA           ; Read Input Status to reset AC flip-flop
    in al, dx
    mov dx, 0x3C0
    mov al, 0x20            ; Bit 5 = video enabled
    out dx, al

    ; Reset BDA cursor position to (0,0)
    push ds
    mov bx, 0x0040
    mov ds, bx
    mov word [0x0050], 0x0000
    pop ds

    jmp .done

.get_cursor:
    mov dh, byte [0x0450]
    mov dl, byte [0x0451]
    mov cx, 0x0700
    jmp .done

.tty:
    cmp al, 0x0D
    je .tty_cr
    cmp al, 0x0A
    je .tty_lf
    cmp al, 0x08
    je .tty_bs

    push 0xB800
    pop es
    xor bx, bx
    mov bl, byte [0x0450]
    mov bh, 80
    push ax
    mov ax, bx
    mul bh
    xor bx, bx
    mov bl, byte [0x0451]
    add ax, bx
    shl ax, 1
    mov di, ax
    pop ax
    mov ah, 0x07
    stosw

    inc byte [0x0451]
    cmp byte [0x0451], 80
    jl .done
    mov byte [0x0451], 0
    inc byte [0x0450]
    jmp .done

.tty_cr:
    mov byte [0x0451], 0
    jmp .done

.tty_lf:
    inc byte [0x0450]
    jmp .done

.tty_bs:
    cmp byte [0x0451], 0
    je .done
    dec byte [0x0451]
    jmp .done

.done:
    pop es
    pop ds
    popa
    iret

; ══════════════════════════════════════════════════════════════════════════
; INT 13h — Disk Services
; ══════════════════════════════════════════════════════════════════════════

int13h:
    pusha
    push ds
    push es

    xor bx, bx
    mov ds, bx

    cmp ah, 0x00
    je .reset
    cmp ah, 0x02
    je .read

    stc
    mov ah, 0x01
    jmp .done

.reset:
    clc
    mov ah, 0x00
    jmp .done

.read:
    ; Copy embedded kernel from ROM to caller buffer ES:BX
    ; Source: kernel_code at ROM offset (after boot_code + 512)
    cmp dl, 0x00
    jne .err
    cmp al, 0
    je .err

    push es
    push bx

    mov ax, 0xF000
    mov ds, ax
    mov si, kernel_code

    pop bx
    pop es

    ; Copy CX sectors (but we only support up to our kernel size)
    xor ch, ch
    shl cx, 8            ; CX = sectors * 512 / 2 (words)
    jcxz .read_ok
    rep movsw

.read_ok:
    clc
    mov ah, 0x00
    jmp .done

.err:
    stc
    mov ah, 0x01

.done:
    pop es
    pop ds
    popa
    iret

; ══════════════════════════════════════════════════════════════════════════
; Helper: print null-terminated string via INT 10h AH=0Eh
; ══════════════════════════════════════════════════════════════════════════

bios_print:
    pusha
.loop:
    lodsb
    or al, al
    jz .end
    mov ah, 0x0E
    mov bx, 0x0007
    int 0x10
    jmp .loop
.end:
    popa
    ret

; ══════════════════════════════════════════════════════════════════════════
; Helper: print null-terminated string to COM1 (16-bit real mode)
; ══════════════════════════════════════════════════════════════════════════

serial_print_bios:
    pusha
.loop:
    lodsb
    or al, al
    jz .end
    call serial_write_bios
    jmp .loop
.end:
    popa
    ret

serial_write_bios:
    ; Write AL to COM1 (16-bit real mode)
    push dx
    push ax
    mov dx, 0x3FD           ; Line status register
.wait:
    in al, dx
    test al, 0x20           ; Bit 5 = transmitter holding register empty
    jz .wait
    pop ax
    mov dx, 0x3F8           ; Data register
    out dx, al
    pop dx
    ret

; ══════════════════════════════════════════════════════════════════════════
; Data
; ══════════════════════════════════════════════════════════════════════════

msg_banner: db "AI Compteur BIOS v0.1", 0x0D, 0x0A
            db "Loading bootloader...", 0x0D, 0x0A, 0

; ══════════════════════════════════════════════════════════════════════════
; Embedded Bootloader (512 bytes, padded with zeros)
; build_bios() overlays the real bootloader binary here
; ══════════════════════════════════════════════════════════════════════════

boot_code:
    times 510 db 0
    dw 0xAA55

; ══════════════════════════════════════════════════════════════════════════
; Embedded Kernel (up to 4KB, padded with zeros)
; build_bios() overlays the real kernel binary here
; ══════════════════════════════════════════════════════════════════════════

kernel_code:
    times 4096 db 0
"""


def build_bios(output_path: str = "bios.bin") -> str:
    """Build a 64KB BIOS ROM image.

    Assembles the BIOS code using X86Assembler, overlays the real
    bootloader and kernel binaries, pads to 64KB, and writes the
    reset vector JMP at offset 0xFFF0.
    """
    from .vm import X86Assembler

    asm = X86Assembler()
    bios_code = asm.assemble(X86_BIOS_ASM)

    # Assemble bootloader and kernel
    bootloader = asm.assemble(X86_BOOTLOADER_ASM)
    kernel = asm.assemble(X86_KERNEL_ASM)

    # BIOS ROM is 64KB (65536 bytes)
    BIOS_SIZE = 65536
    rom = bytearray(BIOS_SIZE)

    # Write BIOS code at offset 0x0000
    rom[:len(bios_code)] = bios_code

    # Find boot_code and kernel_code offsets in the assembled BIOS
    # They are at fixed positions: boot_code is right after the data section,
    # kernel_code is right after boot_code (512 bytes later)
    # The assembler places them sequentially, so we find them by scanning
    # for the pattern: boot_code area starts after msg_banner string ends
    # Actually, let's compute: bios_code includes the boot_code label
    # which is assembled as times 510 db 0 + dw 0xAA55
    # The boot_code offset is len(bios_code) - 4096 - 512 (kernel + bootloader)
    boot_offset = len(bios_code) - 4096 - 512
    kernel_offset = len(bios_code) - 4096

    # Overlay real bootloader
    if len(bootloader) <= 512:
        rom[boot_offset:boot_offset + len(bootloader)] = bootloader
    else:
        rom[boot_offset:boot_offset + 512] = bootloader[:512]

    # Overlay real kernel
    kernel_size = min(len(kernel), 4096)
    rom[kernel_offset:kernel_offset + kernel_size] = kernel[:kernel_size]

    # Reset vector at offset 0xFFF0:
    # JMP FAR 0xF000:0x0000 (EA 00 00 00 F0)
    reset_offset = 0xFFF0
    rom[reset_offset] = 0xEA      # far JMP opcode
    rom[reset_offset + 1] = 0x00  # offset low
    rom[reset_offset + 2] = 0x00  # offset high
    rom[reset_offset + 3] = 0x00  # segment low
    rom[reset_offset + 4] = 0xF0  # segment high

    with open(output_path, 'wb') as f:
        f.write(rom)

    return output_path
