---
id: 20260805_063112_vm-branch-coverage-push-42-76
title: VM branch coverage push: 42% → 76%
status: done
tags: vm,testing,coverage
created: 2026-08-05T06:31:12.715161+00:00
---

VM branch coverage push: 42% → 76%

Completed: vm.py branch coverage push 42% → 76%

- 208 new tests added (840 → 1048 total in test_vm_os_layer.py)
- All 1048 tests pass, 98 assembler emission tests pass
- Fixed: FLAG_DF import, CPU register indices (regs[4]=ESP), addresses within 1MB, raw bytes for unsupported opcodes (CMPSW/SCASW/SAHF/LAHF/CDQ), InsFault caught by step() returns False, ROR carry bit
- Remaining uncovered: L4541-4658 (MOV instructions), L3168-3237 (assembler memory operand encoding), L6686-6748 (_sys_exec), L6502-6551 (syscall dispatch), L540-586 (VGADevice)