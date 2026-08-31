---
id: 20260814_002906_x86-assembler-bugfix-imulmovcmp-encodings
title: X86 assembler bugfix: imul/mov/cmp encodings
status: done
tags: vm,assembler
created: 2026-08-14T00:29:06.142044+00:00
---

X86 assembler bugfix: imul/mov/cmp encodings

Fixed four X86Assembler root-cause bugs in packages/core-py/domains/shell/vm.py that produced wrong machine code for VM builtin programs (sort, primes, calculator, factorial):

1. 2-operand IMUL r32, r/m32 (0F AF /r) — previously routed to unary emitter, emitted F7 /5 (EAX=EAX*EAX). Added _emit_imul + dispatch branch.
2. MOV [mem], size imm with size word AFTER bracket (mov [arr], dword 8) — silently emitted nothing. Added branch handling byte/word/dword.
3. ALU reg, [mem] (cmp edx, [arr+eax+4]) — no branch, src parsed as immediate. Added reg,[mem] direction using 03+op*8 opcodes (3B for cmp).
4. ALU [mem], imm with displacement — inline modrm logic hardcoded mod=00, corrupting the byte stream (e.g. cmp byte [sieve+ecx], 1 -> 83 39 garbage). Refactored shared _mem_operand helper (returns mod_bits, rm_field, disp_bytes) used by _emit_modrm_mem, _emit_alu, _emit_mov, _emit_imul. Added byte (80 /digit ib) and word (66 83/81) sized forms.

Also:
- Updated pinned emission tests: add [ebx+4], eax -> 014304 (was corrupt 010304); add byte/word [ebx], 5 -> 800305 / 66830305 (were wrong 83 forms).
- Added 6 new emission tests covering imul 2-op, mov [mem] imm, cmp reg,[mem], disp mod-bits, byte mem-imm.
- Reverted workaround edits in apps/api/server/vm_builtins.py back to idiomatic assembly (imul eax, ecx; cmp edx, [arr+eax+4]; cmp byte [sieve+ecx], 1) to prove the assembler fix handles them.

Verification: all 13 builtins run correctly (sort '1 2 3 4 5 6 7 8', primes '2 3 5 7 11 13 17 19 23 29 31 37 41 43 47', calculator 61, factorial 720). Full VM suite 1932 tests pass. train/train-status output 4294967295 (-1) expected when API server is down.