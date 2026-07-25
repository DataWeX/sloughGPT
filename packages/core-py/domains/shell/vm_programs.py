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
