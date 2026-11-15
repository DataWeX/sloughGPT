---
id: 20261108_140706_x86assembler-times-n-insn-with-operands-emits-0x00-bytes-ins
status: done
board: 20261108_140706_x86assembler-times-n-insn-with-operands-emits-0x00-bytes-ins
title: "X86Assembler: times N <insn-with-operands> emits 0x00 bytes instead of code"
tags: ["bug", "vm", "assembler"]
sprint: ""
gh: null
---

# X86Assembler: times N <insn-with-operands> emits 0x00 bytes instead of code

SILENT wrong code generation in X86Assembler (domain/shell/_internal/vm.py): `times N <mnemonic with operands>` emits one 0x00 byte per repetition instead of the instruction. Repro (verified 2026-11-08 on a014d5382):
  times 5 mov eax, 42  -> 0000000000f4   (should be b82a000000 x5)
  times 5 add eax, 1   -> 0000000000f4   (should be 83c001 x5)
  times 5 push eax     -> 0000000000f4   (should be 50 x5)
  times 5 nop          -> 9090909090f4   (zero-operand form OK)
Plain (non-times) `add eax, 1` assembles correctly (83c001), so the defect is in the times path: parse at ~line 3968, _estimate_times_size ~3988, _emit_times ~4120 — the repetition loop drops operands (placeholder zeros). Downstream it executes as `add [eax],al` garbage: a hlt swallowed as a ModRM byte makes runs fly to max_steps (observed 100000 steps, eip past the program). Escaped detection: tests/core-py/test_vm_assembler_emission.py test_times() only covers db/dw/nop, and tests/core-py/test_vm_benchmark.py is assertion-free timing (its times-based workloads were measuring garbage). Fix: emit each repetition through the full single-instruction emitter (or emit once and repeat the byte slice); then regression-test times+mov/add/ sub/push byte-exact AND assert executed steps in the benchmark workloads (label-loop reference: 2001 ALU / 1001 MEM steps). Discovered by the vm-bench lane (scripts/benchmark_vm_memory.py, AGENT_SYNC a014d5382).
