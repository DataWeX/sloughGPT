---
id: 20261103_121700_slonet-gap-benchmark-can-it-power-x-today-inference-toks-tra
status: done
board: 20261103_121700_slonet-gap-benchmark-can-it-power-x-today-inference-toks-tra
title: "SloNet gap benchmark — can it power X today? (inference tok/s, train tok/s, eval score @ size S)"
tags: ["sloNet", "benchmark"]
sprint: ""
gh: null
---

# SloNet gap benchmark — can it power X today? (inference tok/s, train tok/s, eval score @ size S)

Gap diagnosis (user Q: 'can SloNet power even a computer on a fully self-made stack'). ONE number before any build: run/extend benchmark_slonet.py + benchmark_slonet_training.py + domain/training/_internal/lm_eval_char.py into a single report: SloNet @ size S -> inference tokens/sec, training tokens/sec, eval score, on CPU. Target context: the v86 browser VM is an EMULATED x86 — its throughput budget is far harsher than bare metal, so state the SLO for that target explicitly. No model changes. Cards B/C blocked on this number. User ruled 2026-11-03: benchmark-first sequencing.
