---
id: 20260815_100000_eax-moffs-landed-and-vm-suite-green
title: EAX/AX moffs landed; full VM test suite green
status: done
tags: shell,vm,assembler,memory-fault
created: 2026-08-15T10:00:00+00:00
---

EAX/AX moffs landed; full VM test suite green

All 14 test_vm*.py files pass: 2607 passed, EXIT=0. EAX moffs A1/A3 disp32 landed (test pins a134120000 / 66a334120000 / 66a134120000 confirmed intact). Root-caused the mass test_vm_os_layer.py failures to commit a28cf10a's strict MemFault bounds checks: tests grow cpu._mem past _mem_size (1MB default) by slice-assigning code at 0x100000, so _read8(0x100000) faulted. Fix: runtime fault checks now compare against len(self._mem) (actual backing store) instead of _mem_size (configured capacity) in _read8/_write8/_write16/_read32/_write32/_push32/_pop32, step() trace fallback, and 16-bit PUSH/POP stack checks. load() intentionally stays on _mem_size (enforced sandbox API). Also fixed 5 stale tests in test_vm_os_layer.py to conform to the strict contract (step()/run() raise InsFault/MemFault, per test_vm.py:389-440): TestX86REPExecution::test_rep_stosb, TestX86ShiftOps::test_ror_eax/test_sar_eax/test_shl_8bit_cl switched from run() to step() (they ran off the end of code into zero memory); TestCPUInstructionDecode::test_div_8_overflow now wraps step() in pytest.raises(InsFault) instead of asserting step() returns False. Test helpers at 7516/8715 upgraded to X86CPU(memory_size=2*1024*1024); length pins at 9626/9630 >= 6 to >= 5 for 5-byte moffs. NOTE: new test_vm*.py file set (agent restructured): test_vm.py, test_vm_os_layer.py, test_vm_assembler_emission.py, test_vm_cpu_exec.py, test_vm_devices_coverage.py, test_vm_devices_layer.py, test_vm_engine.py, test_vm_permissions_more.py, test_vm_programs.py, test_vm_rbac.py, test_vm_router.py, test_vm_runner_ops.py, test_vm_syscalls_shell.py, test_vm_training_bridge_more.py.