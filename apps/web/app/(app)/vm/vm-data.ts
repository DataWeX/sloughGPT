// Pure constants, helpers, and sample programs for the VM console page.
// Extracted from page.tsx by the page-decomposition refactor — no React, no
// controllers; the x86 sample programs below are data, not logic.

export const DEFAULT_MAX_STEPS = 5000
export const MAX_STEPS_LIMIT = 1_000_000

export const DEFAULT_TRAIN_CONFIG = {
  dataset: 'shakespeare',
  epochs: 1,
  lr: 0.001,
  batch_size: 32,
  n_layer: 4,
  n_head: 4,
  embed_dim: 128,
}

export type TrainConfig = typeof DEFAULT_TRAIN_CONFIG

export function buildTrainSource(config: TrainConfig): string {
  const json = JSON.stringify(config)
  return `[BITS 32]

; Launch a training job through the x86 training bridge.
; SYS_TRAIN_START: EAX=28, EBX=config JSON addr.
; Returns EAX = job_id (>=1), or -2 when permission denied.
; Requires the ADMIN role (role selector above).
; The Training card on the right polls the job to completion, shows the
; final result, and offers a Stop button for running jobs.

MOV EBX, config
MOV EAX, 28
INT 0x80
HLT

config: db '${json}', 0`
}

export function clampSteps(n: number): number {
  return Number.isFinite(n)
    ? Math.min(MAX_STEPS_LIMIT, Math.max(1, Math.floor(n)))
    : DEFAULT_MAX_STEPS
}

export function clampTrainConfig(config: TrainConfig): TrainConfig {
  return {
    ...config,
    dataset: config.dataset.trim() || DEFAULT_TRAIN_CONFIG.dataset,
    epochs: Number.isFinite(config.epochs)
      ? Math.max(1, Math.floor(config.epochs))
      : DEFAULT_TRAIN_CONFIG.epochs,
    lr: Number.isFinite(config.lr) && config.lr > 0 ? config.lr : DEFAULT_TRAIN_CONFIG.lr,
    batch_size: Number.isFinite(config.batch_size)
      ? Math.max(1, Math.floor(config.batch_size))
      : DEFAULT_TRAIN_CONFIG.batch_size,
    n_layer: Number.isFinite(config.n_layer)
      ? Math.max(1, Math.floor(config.n_layer))
      : DEFAULT_TRAIN_CONFIG.n_layer,
    n_head: Number.isFinite(config.n_head)
      ? Math.max(1, Math.floor(config.n_head))
      : DEFAULT_TRAIN_CONFIG.n_head,
    embed_dim: Number.isFinite(config.embed_dim)
      ? Math.max(1, Math.floor(config.embed_dim))
      : DEFAULT_TRAIN_CONFIG.embed_dim,
  }
}

export const num = (v: string) => (v.trim() === '' ? NaN : Number(v))

export function trainConfigHints(config: TrainConfig): { label: string; message: string }[] {
  const hints: { label: string; message: string }[] = []
  if (!config.dataset.trim()) {
    hints.push({ label: 'Dataset', message: `using default "${DEFAULT_TRAIN_CONFIG.dataset}"` })
  }
  if (!Number.isFinite(config.epochs)) {
    hints.push({ label: 'Epochs', message: `using default ${DEFAULT_TRAIN_CONFIG.epochs}` })
  }
  if (!Number.isFinite(config.lr) || config.lr <= 0) {
    hints.push({ label: 'Learning rate', message: `using default ${DEFAULT_TRAIN_CONFIG.lr}` })
  }
  if (!Number.isFinite(config.batch_size)) {
    hints.push({ label: 'Batch size', message: `using default ${DEFAULT_TRAIN_CONFIG.batch_size}` })
  }
  if (!Number.isFinite(config.n_layer)) {
    hints.push({ label: 'Layers', message: `using default ${DEFAULT_TRAIN_CONFIG.n_layer}` })
  }
  if (!Number.isFinite(config.n_head)) {
    hints.push({ label: 'Heads', message: `using default ${DEFAULT_TRAIN_CONFIG.n_head}` })
  }
  if (!Number.isFinite(config.embed_dim)) {
    hints.push({ label: 'Embed size', message: `using default ${DEFAULT_TRAIN_CONFIG.embed_dim}` })
  }
  return hints
}

export const DEFAULT_PROGRAMS: Record<string, string> = {
  hello: `[BITS 32]

; Write "Hello, VM!" to VGA text buffer
MOV ESI, msg
MOV EDI, 0xB8000
MOV AH, 0x07

.loop:
LODSB
OR AL, AL
JZ .done
STOSW
JMP .loop

.done:
HLT

msg: db 'Hello, VM!', 0`,

  count: `[BITS 32]

; Count 0-9, write digits to VGA
MOV ECX, 0
MOV EDI, 0xB8000
MOV AH, 0x0B

.loop:
CMP ECX, 10
JGE .done
MOV AL, CL
ADD AL, 48
STOSW
INC ECX
JMP .loop

.done:
HLT`,

  fib: `[BITS 32]

; Fibonacci: compute and display first 10 numbers on VGA
MOV EAX, 0
MOV EBX, 1
MOV ECX, 0
MOV EDI, 0xB8000

.loop:
CMP ECX, 10
JGE .done

; Convert EAX (fib value) to ASCII
PUSH EAX
PUSH EBX
PUSH ECX
MOV EBX, 10
XOR ECX, 0

.digit_loop:
XOR EDX, EDX
DIV EBX
ADD DL, 48
PUSH EDX
INC ECX
TEST EAX, EAX
JNZ .digit_loop

; Write digits to VGA
MOV AH, 0x0A
.write_loop:
POP EDX
MOV AL, DL
STOSW
LOOP .write_loop

; Write space
MOV WORD [ES:EDI], 0x0A20
ADD EDI, 2

POP ECX
POP EBX
POP EAX

; next = fib(n-2) + fib(n-1)
MOV EDX, EAX
ADD EDX, EBX
MOV EAX, EBX
MOV EBX, EDX

INC ECX
JMP .loop

.done:
HLT`,

  sort: `[BITS 32]

; Bubble sort 8 bytes, display sorted array on VGA
MOV ECX, 7

.outer:
MOV EDX, 0
MOV BYTE [swapped], 0

.inner:
MOV AL, [arr + EDX]
MOV BL, [arr + EDX + 1]
CMP AL, BL
JLE .no_swap
MOV [arr + EDX], BL
MOV [arr + EDX + 1], AL
MOV BYTE [swapped], 1

.no_swap:
INC EDX
CMP EDX, ECX
JL .inner

CMP BYTE [swapped], 0
JZ .done
DEC ECX
JMP .outer

.done:
; Display sorted array on VGA
MOV ESI, arr
MOV EDI, 0xB8000
MOV AH, 0x0E

MOV ECX, 8
.disp_loop:
LODSB
ADD AL, 48
STOSW
LOOP .disp_loop

HLT

arr: db 5, 3, 8, 1, 9, 2, 7, 4
swapped: db 0`,

  vga_color: `[BITS 32]

; Rainbow stripe pattern on VGA
MOV EDI, 0xB8000
MOV ECX, 25
MOV BL, 1

.row_loop:
PUSH ECX
MOV ECX, 80

.col_loop:
MOV AL, '*'
MOV AH, BL
STOSW
LOOP .col_loop

POP ECX
INC BL
CMP BL, 16
JL .no_wrap
MOV BL, 1

.no_wrap:
LOOP .row_loop
HLT`,

  rainbow: `[BITS 32]

; Rainbow text "HELLO VM!" on VGA
MOV ESI, msg
MOV EDI, 0xB8000
ADD EDI, 160

MOV ECX, 9
MOV EBX, colors

.loop:
LODSB
MOV AH, [EBX]
INC EBX
STOSW
LOOP .loop

HLT

msg: db 'HELLO VM!'
colors: db 4, 14, 2, 1, 5, 6, 3, 11, 4`,

  primes: `[BITS 32]

; Sieve of Eratosthenes — find primes up to 50, display on VGA
MOV EDI, sieve
MOV ECX, 51
MOV AL, 1
REP STOSB

MOV ESI, 2

.sieve_loop:
CMP ESI, 51
JGE .sieve_done
MOV AL, [sieve + ESI]
TEST AL, AL
JZ .next
MOV EDI, ESI
ADD EDI, ESI

.mark_loop:
CMP EDI, 51
JGE .next
MOV BYTE [sieve + EDI], 0
ADD EDI, ESI
JMP .mark_loop

.next:
INC ESI
JMP .sieve_loop

.sieve_done:
; Display primes on VGA
MOV ESI, 2
MOV EDI, 0xB8000
MOV AH, 0x0B

.print_loop:
CMP ESI, 51
JGE .done
MOV AL, [sieve + ESI]
TEST AL, AL
JZ .skip
MOV AL, SIL
ADD AL, 48
STOSW
; space
MOV WORD [ES:EDI], 0x0B20
ADD EDI, 2

.skip:
INC ESI
JMP .print_loop

.done:
HLT

sieve: times 51 db 0`,

  calculator: `[BITS 32]

; Compute 7 * 8 + 5 = 61, display on VGA
MOV EAX, 7
MOV EBX, 8
MUL EBX
ADD EAX, 5

; Convert EAX to decimal (2 digits)
MOV EBX, 10
XOR EDX, EDX
DIV EBX
ADD AL, 48
ADD DL, 48
MOV [result], AL
MOV [result+1], DL
MOV BYTE [result+2], 0

; Write to VGA
MOV ESI, result
MOV EDI, 0xB8000
MOV AH, 0x0A

.loop:
LODSB
OR AL, AL
JZ .done
STOSW
JMP .loop

.done:
HLT

result: times 4 db 0`,

  factorial: `[BITS 32]

; Compute 6! = 720, display on VGA
MOV EAX, 1
MOV ECX, 6

.loop:
MUL ECX
DEC ECX
JNZ .loop

; EAX = 720, convert to decimal
MOV EBX, 10
MOV ECX, 0

.digit_loop:
XOR EDX, EDX
DIV EBX
ADD DL, 48
PUSH EDX
INC ECX
TEST EAX, EAX
JNZ .digit_loop

; Pop digits to VGA
MOV EDI, 0xB8000
MOV AH, 0x0E

.write_loop:
POP EDX
MOV AL, DL
STOSW
LOOP .write_loop

HLT`,

  guess: `[BITS 32]

; Number guessing game (pre-set answer = 5)
MOV ESI, prompt
MOV EDI, 0xB8000
MOV AH, 0x0A

.prompt_loop:
LODSB
OR AL, AL
JZ .done
STOSW
JMP .prompt_loop

.done:
HLT

prompt: db 'Answer is 5!', 0`,

  train: `[BITS 32]

; Launch a training job through the x86 training bridge.
; SYS_TRAIN_START: EAX=28, EBX=config JSON addr.
; Returns EAX = job_id (>=1), or -2 when permission denied.
; Requires the ADMIN role (role selector above).
; The Training card on the right polls the job to completion, shows the
; final result, and offers a Stop button for running jobs.

MOV EBX, config
MOV EAX, 28
INT 0x80
HLT

config: db '{"dataset":"shakespeare","epochs":1}', 0`,

  'train-status': `[BITS 32]

; Inspect a training job via the x86 training bridge.
; Requires the ADMIN role (role selector above).
; 1) SYS_TRAIN_STATUS: EAX=29, EBX=job_id
;    EAX: 0 running | 1 completed | 2 failed | -1 not found
; 2) SYS_TRAIN_GET_RESULT: EAX=30, EBX=job_id, ECX=buf, EDX=size
;    Writes result JSON into buf, returns bytes written in EAX.
; Both results are stored to guest memory; the returned JSON is also
; surfaced in the Training result card on the right.

MOV EBX, 1
MOV EAX, 29
INT 0x80
MOV [0x5000], EAX

MOV EBX, 1
MOV EAX, 30
MOV ECX, 0x90000
MOV EDX, 256
INT 0x80
MOV [0x5004], EAX
HLT`,
}
