---
id: 20261108_143743_assembler-times-n-insn-with-operands-emits-0x00-bytes-instea
status: done
board: 20261108_143743_assembler-times-n-insn-with-operands-emits-0x00-bytes-instea
title: "Assembler: `times N <insn-with-operands>` emits 0x00 bytes instead of the instruction"
tags: [assembler, vm, x86, shell]
sprint: ""
gh: null
---

# Assembler: `times N <insn-with-operands>` emits 0x00 bytes

Session 2026-11-10 (assembler-times lane, built directly after Gen hit the free-model limit; batch approved by user 2026-11-09).

## Problem

`_emit_times`' else-branch fed the whole operand string to `_parse_imm`, which silently returns 0:
- `times 5 mov eax,42` → `0000000000` (should be `B8 2A000000` ×5)
- `times 5 push eax` → `00`×5 (should be `50`×5)
- `times 5 add eax,1` → `00`×5 (should be `83C001` ×5)

Found by `scripts/benchmark_vm_memory.py` (`a014d5382`). Worst failure mode: a swallowed `0xF4` hlt consumed as ModRM → execution runs off into zeros to `max_steps`. Zero-operand (`times 5 nop`) and db/dw/dd forms already worked (`302a7ab30`, on main).

## Fix (branch `fix/assembler-times-operand-emission`, from origin/main a163f2ba0)

- else-branch: assemble once via `_emit_instruction`, repeat the bytes ×N (nasm semantics; `count<=0` emits nothing).
- new `_is_times_value()`: literals/expressions/`$`/label-const refs keep the old byte-value path; an identifier that is not a label/const is a mnemonic.

## Evidence

- Pushed: `225fc8448` (fix) + `613ba7082` (domain rehash) + `28d37d45d` (AGENT_SYNC) — local == remote, 0 ahead. NOT merged (user sanction required).
- TDD: RED watched (card's 3 evidence cases + disp8 memory form + times-0 guard) → GREEN.
- Gates: `test_vm_assembler_emission.py` 146 passed; vm family **2856 passed / 1 skipped**; ruff check+format clean; post-commit domain rehash (`shell f7ae25e2 → b4a5cfac`) + `--check` + `test_domain_baseline` 8/8.

## Not touched

`3cad61702` vm.py→12-modules split (NOT on main — fix kept in facade file); `tests/core-py/test_results/*.json`; notification surfaces (frozen).

## Session trap recorded

JS `String.replace` replacement containing `` $` `` (from a `\$\`` docstring escape) silently injected "text before match" into vm.py → collection broke; callback form fixes it.
