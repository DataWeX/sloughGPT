---
id: 20261101_144012_owned-vs-bootstrap-quality-benchmark-slonet-vs-qwen25-05b-on
status: todo
board: 20261101_144012_owned-vs-bootstrap-quality-benchmark-slonet-vs-qwen25-05b-on
title: "Owned-vs-bootstrap quality benchmark: SloNet vs Qwen2.5-0.5B on one eval set"
tags: ["roadmap", "decision", "model", "area:models", "track:next"]
sprint: ""
gh: null
---

# Owned-vs-bootstrap quality benchmark: SloNet vs Qwen2.5-0.5B on one eval set

Committed in the model-candidates decision session 2026-11-01 (card 20261101_140449). Turns 'borrowed weights are bootstrap, not destination' from faith into measurement: owned SloNet vs Qwen2.5-0.5B on one eval set; informs when to flip the SLO_BOOTSTRAP_HF default gate (goal 12). benchmark_owned_path.py records weight-identity only — the quality instrument is missing. Close with: recorded run via benchmark_results.py record --kind training.
