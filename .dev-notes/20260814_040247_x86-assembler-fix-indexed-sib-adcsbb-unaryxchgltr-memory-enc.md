---
id: 20260814_040247_x86-assembler-fix-indexed-sib-adcsbb-unaryxchgltr-memory-enc
title: X86 assembler: fix indexed SIB, adc/sbb, unary/xchg/ltr memory encodings
status: done
tags: shell,vm,assembler
created: 2026-08-14T04:02:47.420097+00:00
---

X86 assembler: fix indexed SIB, adc/sbb, unary/xchg/ltr memory encodings

VMEngine error handling overhaul: 391 tests pass (107 emission + 143 exec + 141 engine). Fixes: (1) X86CPU.step() re-raises InsFault/MemFault instead of swallowing, only catches unexpected exceptions. (2) Stack overflow/underflow protection on _push32/_pop32 and 16-bit PUSH/POP. (3) MemFault raised on out-of-bounds memory access (_read8/_write8/_read32/_write32). (4) VMEngine.step() preserves original exception type and message in FaultEvent. (5) Unknown opcode raises InsFault instead of silent skip. (6) Breakpoint hits in run() now set exit_reason='breakpoint' (was misclassified as 'fault'). (7) VMEngine.step() checks breakpoints on single-step too. (8) HLT in 16-bit mode raises Halt (was InsFault). (9) read_byte/write_byte have bounds checks. (10) 14 new error handling tests added.