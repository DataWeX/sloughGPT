---
id: 20260804_074953_vmpy-os-layer-test-coverage-944-96-268-112-missed
title: vm.py OS-layer test coverage: 94.4% -> 96% (268 -> 112 missed)
status: done
tags: core-py,vm,coverage,tests
created: 2026-08-04T07:49:53.971641+00:00
---

vm.py OS-layer test coverage: 94.4% -> 96% (268 -> 112 missed)

vm.py coverage 94.4% -> 96% (268 -> 112 missed stmts, 130 branch partials). Added 62 new tests across 9 test classes in tests/test_vm_os_layer.py.

New test classes:
- TestPITDeviceIO (13): write_command channel latch, read_counter latch, write_counter, pit_tick
- TestX86VirtualSystemExtended (8): run_break, reset, fire_irq, status, properties, fs preservation
- TestDiskProgramLoader (6): list_programs, load_source, save/load roundtrip, run with stdout
- TestCPUTrace (3): format_trace integer/ndarray regs, get_trace
- TestX86AssemblerMovRegMem (2): MOV EAX direct address, MOV EAX imm hex
- TestCPUErrorPaths (16): _reg/_check_arity/_truthy/_parse_tensor error paths, CPU step edge cases
- TestAssemblerLabelParsing (3): standalone label parsing
- TestX86SyscallExit (2): sys_exit via INT 0x80, keyboard handler registration
- TestX86CPUInstructions (9): PUSHAD/POPAD, RET, INT, INC/DEC reg/mem, JMP near

Key findings:
- Lines 1662-1663 (Assembler standalone label) are dead code for Assembler class (caught by earlier branch)
- Lines 3133-3150 (MOV reg,[imm]) are unreachable in X86Assembler (earlier branch catches it)
- Lines 4323-4339 (PUSHAD/POPAD) now covered
- Lines 6436-6437 (_sys_exit) now covered
- Remaining 112 missed lines are scattered CPU edge cases, assembler _estimate_* methods, and dead-code paths

Coverage recipe: run test files individually with coverage --append, then report. Full suite hangs on 7189-line file.