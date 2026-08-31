---
id: 20260814_011423_x86-vm-assembler-fix-shiftincdec-memory-encodings
title: x86 VM assembler: fix shift/inc/dec memory encodings
status: done
tags: shell,vm,assembler
created: 2026-08-14T01:14:23.687972+00:00
---

x86 VM assembler: fix shift/inc/dec memory encodings

Also added missing 8-bit register inc/dec path to _emit_inc_dec (FE /0,/1 register form): 'inc bl' -> fec3, 'dec dl' -> feca, 'inc al' -> fec0. Previously emitted nothing. Runtime 0xFE handler already supported registers. New pins added to test_inc_dec (test_inc_dec / test_shift_imm_and_cl area). 938 VM tests + 1746 shell/VM tests pass; hello/count/counter builtins all exit 0.