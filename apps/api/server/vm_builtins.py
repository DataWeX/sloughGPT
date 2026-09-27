"""Built-in x86-32 assembly programs for the VM.

Real, runnable programs for ``X86VirtualSystem``.  Each program uses the
VM syscall convention:  EAX=syscall number, EBX/ECX/EDX=args, ``INT 0x80``.

Syscall numbers (X86SyscallHandler):
    SYS_READ=0, SYS_EXIT=1, SYS_WRITE=3,
    SYS_TRAIN_START=28, SYS_TRAIN_STATUS=29, SYS_TRAIN_GET_RESULT=30

Queries are resolved programmatically from syscall-5 constants in
``domain.shell._internal.vm`` at import time, so the registry never hardcodes a
number the VM does not itself define.
"""

from __future__ import annotations

import sys
from pathlib import Path

from domain.shared import find_repo_root

_CORE_DIR = str(find_repo_root(Path(__file__).resolve()) / "packages" / "core-py")
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

from domain.shell._internal.vm import (  # noqa: E402
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
_NR_OPEN = _syscall_num("SYS_OPEN")
_NR_CLOSE = _syscall_num("SYS_CLOSE")
_NR_READDIR = _syscall_num("SYS_READDIR")
_NR_UNAME = _syscall_num("SYS_UNAME")
_NR_GETPID = _syscall_num("SYS_GETPID")
_NR_GETROLE = _syscall_num("SYS_GETROLE")


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


def _shell() -> str:
    return f"""; shell — interactive console REPL for the browser VM session.
; Line-buffered stdin read loop: prompts, echoes typing, handles
; backspace, then dispatches one of: help, ls, cat <file>, uname,
; pid, echo <text>, train, train-status, train-result, about,
; clear, halt.  Spins on empty keyboard
; buffer (SYS_READ returns 0) until the session pump feeds keys
; via transfer_key().
[BITS 32]
[ORG 0x100000]
    jmp start
prompt: db 10, "sloughvm> ", 0
msg_nl: db 10, 0
msg_sp: db " ", 0
msg_help: db "commands: help, ls, cat <file>, uname, pid, echo <text>, train, train-status, train-result, about, clear, halt", 10, 0
msg_about: db "SloughGPT own-VM console (X86VirtualSystem)", 10, 0
msg_unknown: db "unknown command - type help", 10, 0
msg_clear: db 27, "[2J", 27, "[H", 0
msg_bs: db 8, 32, 8, 0
msg_ls_empty: db "(no files)", 10, 0
msg_cat_usage: db "usage: cat <file>", 10, 0
msg_no_file: db "cat: no such file", 10, 0
msg_train_started: db "started job ", 0
msg_train_fail: db "train: could not start job", 10, 0
msg_denied: db "permission denied (ADMIN role required)", 10, 0
msg_no_job: db "no training job yet - run train first", 10, 0
msg_status_running: db "status: running", 10, 0
msg_status_completed: db "status: completed", 10, 0
msg_status_failed: db "status: failed", 10, 0
msg_status_notfound: db "status: job not found", 10, 0
msg_no_result: db "no result yet", 10, 0
cmd_help: db "help", 0
cmd_echo: db "echo", 0
cmd_about: db "about", 0
cmd_clear: db "clear", 0
cmd_halt: db "halt", 0
cmd_ls: db "ls", 0
cmd_uname: db "uname", 0
cmd_pid: db "pid", 0
cmd_cat: db "cat", 0
cmd_train: db "train", 0
cmd_train_status: db "train-status", 0
cmd_train_result: db "train-result", 0
chbuf: times 4 db 0
llen: times 1 db 0
line: times 80 db 0
dirents: times 320 db 0
catbuf: times 201 db 0
unamebuf: times 325 db 0
catfd: times 4 db 0
lastjob: times 4 db 0
resultbuf: times 201 db 0
cfg: db '{{"dataset":"shakespeare","epochs":3}}', 0
{_SCRATCH}
start:
repl:
    xor eax, eax
    mov [llen], al
    mov esi, prompt
    call print_str
read_loop:
    mov eax, {_NR_READ}
    mov ebx, 0
    mov ecx, chbuf
    mov edx, 1
    int 0x80
    test eax, eax
    jle read_loop
    mov al, [chbuf]
    cmp al, 10
    je line_done
    cmp al, 13
    je line_done
    cmp al, 8
    je do_bs
    cmp al, 127
    je do_bs
    xor ecx, ecx
    mov cl, [llen]
    cmp cl, 78
    jae read_loop
    mov edi, line
    add edi, ecx
    mov [edi], al
    inc ecx
    mov [llen], cl
    mov eax, {_NR_WRITE}
    mov ebx, 1
    mov ecx, chbuf
    mov edx, 1
    int 0x80
    jmp read_loop
do_bs:
    xor ecx, ecx
    mov cl, [llen]
    test cl, cl
    jz read_loop
    dec ecx
    mov [llen], cl
    mov edi, line
    add edi, ecx
    xor eax, eax
    mov [edi], al
    mov esi, msg_bs
    call print_str
    jmp read_loop
line_done:
    xor ecx, ecx
    mov cl, [llen]
    xor eax, eax
    mov edi, line
    add edi, ecx
    mov [edi], al
    mov esi, msg_nl
    call print_str
    test cl, cl
    jz repl
    mov esi, line
    mov edi, cmd_help
    call strcmp
    jz do_help
    mov esi, line
    mov edi, cmd_about
    call strcmp
    jz do_about
    mov esi, line
    mov edi, cmd_clear
    call strcmp
    jz do_clear
    mov esi, line
    mov edi, cmd_halt
    call strcmp
    jz do_halt
    mov esi, line
    mov edi, cmd_ls
    call strcmp
    jz do_ls
    mov esi, line
    mov edi, cmd_uname
    call strcmp
    jz do_uname
    mov esi, line
    mov edi, cmd_pid
    call strcmp
    jz do_pid
    mov esi, line
    mov edi, cmd_train
    call strcmp
    jz do_train
    mov esi, line
    mov edi, cmd_train_status
    call strcmp
    jz do_train_status
    mov esi, line
    mov edi, cmd_train_result
    call strcmp
    jz do_train_result
    mov esi, line
    mov edi, cmd_cat
    call prefix_match
    test eax, eax
    jnz do_cat
    mov esi, line
    mov edi, cmd_echo
    call prefix_match
    test eax, eax
    jnz do_echo
    mov esi, msg_unknown
    call print_str
    jmp repl
do_help:
    mov esi, msg_help
    call print_str
    jmp repl
do_about:
    mov esi, msg_about
    call print_str
    jmp repl
do_clear:
    mov esi, msg_clear
    call print_str
    jmp repl
do_halt:
{_EXIT}
do_ls:
    mov eax, {_NR_READDIR}
    mov ebx, dirents
    mov ecx, 10
    int 0x80
    test eax, eax
    jz ls_empty
    mov esi, dirents
    mov ecx, eax
ls_loop:
    test ecx, ecx
    jz ls_done
    push esi
    call print_str
    pop esi
    push esi
    mov esi, msg_nl
    call print_str
    pop esi
    add esi, 32
    dec ecx
    jmp ls_loop
ls_done:
    jmp repl
ls_empty:
    mov esi, msg_ls_empty
    call print_str
    jmp repl
do_uname:
    mov eax, {_NR_UNAME}
    mov ebx, unamebuf
    int 0x80
    mov esi, unamebuf
    call print_str
    call print_sp
    mov esi, unamebuf
    add esi, 65
    call print_str
    call print_sp
    mov esi, unamebuf
    add esi, 130
    call print_str
    call print_sp
    mov esi, unamebuf
    add esi, 195
    call print_str
    call print_sp
    mov esi, unamebuf
    add esi, 260
    call print_str
    mov esi, msg_nl
    call print_str
    jmp repl
do_pid:
    mov eax, {_NR_GETPID}
    int 0x80
    call print_num
    mov esi, msg_nl
    call print_str
    jmp repl
do_train:
    mov eax, {_NR_TRAIN_START}
    mov ebx, cfg
    int 0x80            ; EAX = job_id (>=1), -1 = error, -2 = denied
    test eax, eax
    js train_err
    mov [lastjob], al
    mov esi, msg_train_started
    call print_str
    call print_num
    mov esi, msg_nl
    call print_str
    jmp repl
train_err:
    cmp eax, -2
    je denied_out
    mov esi, msg_train_fail
    call print_str
    jmp repl
do_train_status:
    mov eax, {_NR_GETROLE}
    int 0x80
    test eax, eax
    jz denied_out        ; role 0 (USER) -> ADMIN required
    xor ebx, ebx
    mov bl, [lastjob]
    test ebx, ebx
    jz no_job_out
    mov eax, {_NR_TRAIN_STATUS}
    int 0x80            ; EAX: 0 running 1 completed 2 failed -1 not found -2 denied
    test eax, eax
    js st_err
    cmp eax, 0
    je st_running
    cmp eax, 1
    je st_completed
    mov esi, msg_status_failed
    call print_str
    jmp repl
st_running:
    mov esi, msg_status_running
    call print_str
    jmp repl
st_completed:
    mov esi, msg_status_completed
    call print_str
    jmp repl
st_err:
    cmp eax, -2
    je denied_out
    mov esi, msg_status_notfound
    call print_str
    jmp repl
do_train_result:
    mov eax, {_NR_GETROLE}
    int 0x80
    test eax, eax
    jz denied_out        ; role 0 (USER) -> ADMIN required
    xor ebx, ebx
    mov bl, [lastjob]
    test ebx, ebx
    jz no_job_out
    mov eax, {_NR_TRAIN_GET_RESULT}
    mov ecx, resultbuf
    mov edx, 200
    int 0x80            ; EAX = bytes written, 0 = not ready, -2 = denied
    test eax, eax
    jz res_none
    js res_err
    mov esi, resultbuf
    call print_str
    mov esi, msg_nl
    call print_str
    jmp repl
res_err:
    cmp eax, -2
    je denied_out
res_none:
    mov esi, msg_no_result
    call print_str
    jmp repl
no_job_out:
    mov esi, msg_no_job
    call print_str
    jmp repl
denied_out:
    mov esi, msg_denied
    call print_str
    jmp repl
do_cat:
    ; prefix matched: line must be exactly "cat" or "cat <args>"
    mov esi, line
    add esi, 3
    mov al, [esi]
    test al, al
    jz cat_usage
    cmp al, ' '
    jne do_unknown
cat_skip:
    inc esi
    mov al, [esi]
    cmp al, ' '
    je cat_skip
    test al, al
    jz cat_usage
    mov eax, {_NR_OPEN}
    mov ebx, esi
    mov ecx, 0
    int 0x80
    cmp eax, 0
    jl cat_missing
    mov [catfd], al
    mov ebx, eax
    mov eax, {_NR_READ}
    mov ecx, catbuf
    mov edx, 200
    int 0x80
    cmp eax, 0
    jl cat_read_err
    mov edi, catbuf
    add edi, eax
    xor eax, eax
    mov [edi], al
    mov eax, {_NR_CLOSE}
    xor ebx, ebx
    mov bl, [catfd]
    int 0x80
    mov esi, catbuf
    call print_str
    mov esi, msg_nl
    call print_str
    jmp repl
cat_usage:
    mov esi, msg_cat_usage
    call print_str
    jmp repl
cat_missing:
    mov esi, msg_no_file
    call print_str
    jmp repl
cat_read_err:
    mov esi, msg_no_file
    call print_str
    jmp repl
do_unknown:
    mov esi, msg_unknown
    call print_str
    jmp repl
do_echo:
    mov esi, line
    add esi, 4
echo_skip:
    mov al, [esi]
    cmp al, ' '
    jne echo_pr
    inc esi
    jmp echo_skip
echo_pr:
    call print_str
    mov esi, msg_nl
    call print_str
    jmp repl
strcmp:
    mov al, [esi]
    mov bl, [edi]
    cmp al, bl
    jne sc_ne
    test al, al
    jz sc_eq
    inc esi
    inc edi
    jmp strcmp
sc_ne:
    mov eax, 1
    ret
sc_eq:
    xor eax, eax
    ret
prefix_match:
    mov al, [edi]
    test al, al
    jz pm_yes
    mov bl, [esi]
    cmp al, bl
    jne pm_no
    inc esi
    inc edi
    jmp prefix_match
pm_yes:
    mov eax, 1
    ret
pm_no:
    xor eax, eax
    ret
print_sp:
    push esi
    mov esi, msg_sp
    call print_str
    pop esi
    ret
{_PRINT_NUM}
{_PRINT_STR}
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
    "train": {
        "description": "Launch a training job via SYS_TRAIN_START (requires ADMIN role)",
        "program": _train,
    },
    "train-status": {
        "description": "Poll a training job via SYS_TRAIN_STATUS (requires ADMIN role)",
        "program": _train_status,
    },
    "shell": {
        "description": "Interactive console REPL (line-buffered commands over stdin)",
        "program": _shell,
    },
}


def get_builtin(code: str) -> str:
    """Resolve a builtin program's assembly source by name."""
    entry = BUILTIN_PROGRAMS.get(code)
    if entry is None:
        raise KeyError(code)
    return entry["program"]()
