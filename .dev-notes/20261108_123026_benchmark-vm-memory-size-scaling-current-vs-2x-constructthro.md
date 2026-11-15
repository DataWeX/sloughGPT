---
id: 20261108_123026_benchmark-vm-memory-size-scaling-current-vs-2x-constructthro
status: done
board: 20261108_123026_benchmark-vm-memory-size-scaling-current-vs-2x-constructthro
title: "Benchmark VM memory-size scaling (current vs 2x) — construct/throughput/footprint sweep"
tags: ["benchmark", "vm"]
sprint: ""
gh: null
---

# Benchmark VM memory-size scaling (current vs 2x) — construct/throughput/footprint sweep

New scripts/benchmark_vm_memory.py: sweep memory_size 0.5x/1x/2x/4x of the two defaults (X86CPU 1 MiB, X86VirtualSystem 4 MiB) and measure construct ms (bytearray zeroing — expect linear), run ns/op for an ALU and a stack/memory workload (VM is dispatch-bound — expect FLAT), and tracemalloc footprint (expect 2x RAM). Informational: timing never gates; structural checks (identical step counts across sizes, footprint linearity) do. History in scripts/benchmark_vm_memory.json. LANDED a014d5382: scripts/benchmark_vm_memory.py sweeps memory_size 0.5x/1x/2x/4x of both defaults (X86CPU 1 MiB = API default, X86VirtualSystem 4 MiB). Answer to current-vs-2x: construct linear (+72..108% at 2x), run FLAT (CPU alu -0.3% / mem -1.4%, system +0.1% — a +9.3% blip failed to reproduce = host noise), footprint exact (+1.00/+4.00 MiB; tracemalloc current, not peak — peak double-counts bytearray's transient buffer). 6/6 structural checks gate (steps identical across sizes, footprint linearity, spawn ok); timing never gates. History: scripts/benchmark_vm_memory.json (2 valid records). FOUND EN ROUTE (needs own card, not fixed here): assembler defect — times N <insn-with-operands> emits one 0x00 byte per repetition (times 5 mov eax,42 -> 0000000000); test_times() only covers db/dw/nop and test_vm_benchmark.py is assertion-free timing, so nothing caught it — benchmark workloads were byte-verified and replaced with label loops (2001/1001 steps at every size, both paths). AGENT_SYNC entry rode the push (conflict with route-map 7/7 resolved keeping both).
