"""Built-in x86-32 assembly programs for the VM.

Real, runnable programs for ``X86VirtualSystem``.  Each program uses the
VM syscall convention:  EAX=syscall number, EBX/ECX/EDX=args, ``INT 0x80``.

Syscall numbers (X86SyscallHandler):
    SYS_READ=0, SYS_EXIT=1, SYS_WRITE=3,
    SYS_TRAIN_START=28, SYS_TRAIN_STATUS=29, SYS_TRAIN_GET_RESULT=30

Queries are resolved programmatically from syscall-5 constants in
``domains.shell.vm`` at import time, so the registry never hardcodes a
number the VM does not itself define.
"""

from __future__ import annotations

import sys
from pathlib import Path

from domains.shared import find_repo_root

_CORE_DIR = str(find_repo_root(Path(__file__).resolve()) / "packages" / "core-py")
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

from domains.shell.vm import (  # noqa: E402
    X86SyscallHandler,
)


def _syscall_num(name: str) -> int:
    """Look up a syscall number by attribute name on the handler class."""
    value = getattr(X86SyscallHandler, name)
    if not isinstance(value, int):
        raise RuntimeError(f"syscall {name} is not an int constant")
    return value


# Resolve real syscall numbers (keeps assembly in lockstep with the VM).
_NR_READ = _syscall_num("SYS_READ")
_NR_EXIT = _syscall_num("SYS_EXIT")
_NR_WRITE = _syscall_num("SYS_WRITE")
_NR_TRAIN_START = _syscall_num("SYS_TRAIN_START")
_NR_TRAIN_STATUS = _syscall_num("SYS_TRAIN_STATUS")
_NR_TRAIN_GET_RESULT = _syscall_num("SYS_TRAIN_GET_RESULT")


# ── Shared subroutines (spliced into each program) ───────────────────────────

_PRINT_STR = f"""
; Print null-terminated string pointed to by ESI.
; Label namespace is global in X86Assembler — ps_len/ps_done are unique.
print_str:
    pusha
    push esi
    xor edx, edx
ps_len:
    lodsb
    test al, al
    jz ps_done
    inc edx
    jmp ps_len
ps_done:
    pop ecx
    mov eax, {_NR_WRITE}
    mov ebx, 1
    int 0x80
    popa
    ret
"""

_PRINT_NUM = """
; Print unsigned EAX as decimal.  EDI is the buffer pointer:  [edi] is a
; register-indirect operand, which the X86Assembler encodes correctly,
; whereas [ebp] is reserved for the [disp32] addressing form in x86-32.
;
; The scratch buffer (numbuf/numbuf_end) is declared in each program's
; preamble (right after `jmp start`), NOT here.  A buffer spliced at the
; very end of the loaded image sits at the top edge of a 1 MB VM
; (org=0x100000, memory_size=0x100000), so print_str's trailing scan
; reads one byte past addressable memory and faults.
print_num:
    pusha
    mov edi, numbuf_end
    mov ecx, 10
pn_loop:
    xor edx, edx
    div ecx
    add dl, '0'
    dec edi
    mov [edi], dl
    test eax, eax
    jnz pn_loop
    mov esi, edi
    call print_str
    popa
    ret
"""

_SCRATCH = """
; numbuf_end points at a zero terminator, so print_str's forward scan
; stops right after the digits.  print_num writes digits at
; numbuf_end-1 downward, leaving the terminator intact.
numbuf: times 11 db 0
numbuf_end: db 0
"""

_EXIT = f"""
    mov eax, {_NR_EXIT}
    xor ebx, ebx
    int 0x80
    hlt
"""


def _hello() -> str:
    return f"""; hello — print a greeting to stdout.
[BITS 32]
[ORG 0x100000]
    jmp start
msg_hello: db "Hello, VM!", 10, 0
start:
    mov esi, msg_hello
    call print_str
{_EXIT}
{_PRINT_STR}
"""


def _count() -> str:
    return f"""; count — print 0..9 separated by spaces.
[BITS 32]
[ORG 0x100000]
    jmp start
msg_space: db " ", 0
msg_nl: db 10, 0
{_SCRATCH}
start:
    mov eax, 0
cnt_loop:
    cmp eax, 10
    jge cnt_done
    call print_num
    mov esi, msg_space
    call print_str
    inc eax
    jmp cnt_loop
cnt_done:
    mov esi, msg_nl
    call print_str
{_EXIT}
{_PRINT_STR}
{_PRINT_NUM}
"""


def _fib() -> str:
    return f"""; fib — print the first 10 Fibonacci numbers.
[BITS 32]
[ORG 0x100000]
    jmp start
msg_space: db " ", 0
msg_nl: db 10, 0
{_SCRATCH}
start:
    mov eax, 0          ; a
    mov ebx, 1          ; b
    mov ecx, 10         ; iterations
fib_loop:
    push ecx
    call print_num
    mov esi, msg_space
    call print_str
    pop ecx
    mov edx, eax
    mov eax, ebx        ; a = b
    add ebx, edx        ; b = a + b
    dec ecx
    jnz fib_loop
    mov esi, msg_nl
    call print_str
{_EXIT}
{_PRINT_STR}
{_PRINT_NUM}
"""


def _sort() -> str:
    return f"""; sort — bubble sort an 8-element array, print sorted values.
[BITS 32]
[ORG 0x100000]
    jmp start
msg_space: db " ", 0
msg_nl: db 10, 0
arr: times 8 dd 0
{_SCRATCH}
start:
    ; initialize array: 8 3 6 1 7 2 5 4
    mov dword [arr], 8
    mov dword [arr+4], 3
    mov dword [arr+8], 6
    mov dword [arr+12], 1
    mov dword [arr+16], 7
    mov dword [arr+20], 2
    mov dword [arr+24], 5
    mov dword [arr+28], 4
    mov esi, 7          ; pass counter (n-1)
srt_outer:
    mov edi, 0          ; swapped flag
    mov ecx, 0          ; index
srt_inner:
    mov eax, ecx
    shl eax, 2
    mov edx, [arr+eax]
    cmp edx, [arr+eax+4]    ; EDX vs arr[i+1]
    jle srt_no_swap
    mov ebx, [arr+eax+4]
    mov [arr+eax], ebx
    mov [arr+eax+4], edx
    mov edi, 1
srt_no_swap:
    inc ecx
    cmp ecx, esi
    jl srt_inner
    test edi, edi
    jz srt_done
    dec esi
    cmp esi, 0
    jg srt_outer
srt_done:
    mov ecx, 0
srt_print:
    mov eax, ecx
    shl eax, 2
    mov eax, [arr+eax]
    call print_num
    mov esi, msg_space
    call print_str
    inc ecx
    cmp ecx, 8
    jl srt_print
    mov esi, msg_nl
    call print_str
{_EXIT}
{_PRINT_STR}
{_PRINT_NUM}
"""


def _vga_color() -> str:
    return f"""; vga_color — fill VGA text buffer with colored stripes.
; 80x25 text mode, attribute bytes cycle for a rainbow.
[BITS 32]
[ORG 0x100000]
    jmp start
start:
    mov edi, 0xB8000
    mov eax, 2000       ; cells
    mov bl, 4           ; color index starts at red
vga_loop:
    test eax, eax
    jz vga_done
    mov byte [edi], ' '
    mov byte [edi+1], bl
    add edi, 2
    inc bl
    dec eax
    jmp vga_loop
vga_done:
{_EXIT}
"""


def _primes() -> str:
    return f"""; primes — Sieve of Eratosthenes up to 50, print primes.
[BITS 32]
[ORG 0x100000]
    jmp start
msg_space: db " ", 0
msg_nl: db 10, 0
sieve: times 51 db 0
{_SCRATCH}
start:
    ; sieve[i] = 1 initially; mark 0 and 1 as non-prime
    mov byte [sieve+0], 0
    mov byte [sieve+1], 0
    mov ecx, 2
prm_init:
    cmp ecx, 51
    jge prm_scan
    mov byte [sieve+ecx], 1
    inc ecx
    jmp prm_init
prm_scan:
    mov ecx, 2
prm_outer:
    cmp ecx, 50
    jg prm_print
    cmp byte [sieve+ecx], 1 ; sieve[i] set?
    jne prm_next_i
    mov eax, ecx
    add eax, ecx        ; first multiple = 2p
prm_mark:
    cmp eax, 51
    jge prm_next_i
    mov byte [sieve+eax], 0
    add eax, ecx
    jmp prm_mark
prm_next_i:
    inc ecx
    jmp prm_outer
prm_print:
    mov ecx, 2
prm_loop:
    cmp ecx, 51
    jge prm_done
    cmp byte [sieve+ecx], 1 ; sieve[i] set?
    jne prm_skip
    mov eax, ecx
    call print_num
    mov esi, msg_space
    call print_str
prm_skip:
    inc ecx
    jmp prm_loop
prm_done:
    mov esi, msg_nl
    call print_str
{_EXIT}
{_PRINT_STR}
{_PRINT_NUM}
"""


def _calculator() -> str:
    return f"""; calculator — compute 7*8+5 and print the result.
[BITS 32]
[ORG 0x100000]
    jmp start
msg_nl: db 10, 0
{_SCRATCH}
start:
    mov eax, 7
    mov ebx, 8
    imul eax, ebx       ; EAX = 7*8
    add eax, 5
    call print_num
    mov esi, msg_nl
    call print_str
{_EXIT}
{_PRINT_STR}
{_PRINT_NUM}
"""


def _factorial() -> str:
    return f"""; factorial — compute 6! and print it.
[BITS 32]
[ORG 0x100000]
    jmp start
msg_nl: db 10, 0
{_SCRATCH}
start:
    mov eax, 1
    mov ecx, 6
fac_loop:
    imul eax, ecx       ; EAX *= ECX
    dec ecx
    jnz fac_loop
    call print_num
    mov esi, msg_nl
    call print_str
{_EXIT}
{_PRINT_STR}
{_PRINT_NUM}
"""


def _guess() -> str:
    return f"""; guess — read a digit via SYS_READ (stdin = keyboard buffer)
; and report whether it equals a hidden value.
[BITS 32]
[ORG 0x100000]
    jmp start
msg_hint: db "Enter a digit (0-9): ", 0
msg_right: db "Right!", 10, 0
msg_wrong: db "Wrong!", 10, 0
buf: times 4 db 0
start:
    mov esi, msg_hint
    call print_str
    mov eax, {_NR_READ}
    mov ebx, 0
    mov ecx, buf
    mov edx, 1
    int 0x80
    mov al, [buf]
    sub al, '0'
    cmp al, 7           ; the hidden value
    jne .wrong
    mov esi, msg_right
    call print_str
{_EXIT}
.wrong:
    mov esi, msg_wrong
    call print_str
{_EXIT}
{_PRINT_STR}
"""


def _rainbow() -> str:
    return f"""; rainbow — print "HELLO VM!" to VGA with per-char colors.
[BITS 32]
[ORG 0x100000]
    jmp start
msg_rainbow: db "HELLO VM!", 0
start:
    mov esi, msg_rainbow
    mov edi, 0xB8000
    mov bl, 9
rbw_loop:
    lodsb
    test al, al
    jz rbw_done
    mov byte [edi], al
    mov byte [edi+1], bl
    add edi, 2
    inc bl
    jmp rbw_loop
rbw_done:
{_EXIT}
"""


def _train() -> str:
    return f"""; train — launch a training job via SYS_TRAIN_START.
; Requires the ADMIN role; bridge proxies to POST /training/start.
[BITS 32]
[ORG 0x100000]
    jmp start
cfg: db '{{"dataset":"shakespeare","epochs":3}}', 0
msg_nl: db 10, 0
{_SCRATCH}
start:
    mov eax, {_NR_TRAIN_START}
    mov ebx, cfg
    int 0x80            ; EAX = job_id (>=1) or -1
    call print_num
    mov esi, msg_nl
    call print_str
{_EXIT}
{_PRINT_STR}
{_PRINT_NUM}
"""


def _train_status() -> str:
    return f"""; train-status — poll a training job (id 1) via SYS_TRAIN_STATUS.
[BITS 32]
[ORG 0x100000]
    jmp start
msg_nl: db 10, 0
{_SCRATCH}
start:
    mov eax, {_NR_TRAIN_STATUS}
    mov ebx, 1          ; job_id from a previous SYS_TRAIN_START
    int 0x80            ; EAX: 0=running 1=completed 2=failed -1=not found
    call print_num
    mov esi, msg_nl
    call print_str
{_EXIT}
{_PRINT_STR}
{_PRINT_NUM}
"""


# Each name maps to a builder that splices in the real syscall numbers.
BUILTIN_PROGRAMS: dict[str, dict[str, str]] = {
    "hello": {"description": "Print 'Hello, VM!' to stdout", "program": _hello},
    "count": {"description": "Count 0 to 9 via sys_write", "program": _count},
    "fib": {"description": "Fibonacci sequence (first 10)", "program": _fib},
    "sort": {"description": "Bubble sort 8-element array", "program": _sort},
    "vga_color": {"description": "Rainbow stripe pattern (colored VGA)", "program": _vga_color},
    "primes": {"description": "Sieve of Eratosthenes — primes up to 50", "program": _primes},
    "calculator": {"description": "Compute 7*8+5, display the result", "program": _calculator},
    "factorial": {"description": "Compute 6! = 720, display the result", "program": _factorial},
    "guess": {"description": "Number guessing game (keyboard input)", "program": _guess},
    "rainbow": {"description": "Rainbow colored 'HELLO VM!' text (VGA)", "program": _rainbow},
    "train": {"description": "Launch a training job via SYS_TRAIN_START (requires ADMIN role)", "program": _train},
    "train-status": {"description": "Poll a training job via SYS_TRAIN_STATUS (requires ADMIN role)", "program": _train_status},
}


def get_builtin(code: str) -> str:
    """Resolve a builtin program's assembly source by name."""
    entry = BUILTIN_PROGRAMS.get(code)
    if entry is None:
        raise KeyError(code)
    return entry["program"]()