---
id: 20261103_104018_perfsouls-soul-config-is-41-mb-because-it-embeds-adam-optimi
status: todo
board: 20261103_104018_perfsouls-soul-config-is-41-mb-because-it-embeds-adam-optimi
title: "perf(souls): soul config is 41 MB because it embeds Adam optimizer state — drop it at write time"
tags: ["inference", "perf", "souls", "area:models"]
sprint: ""
gh: null
---

# perf(souls): soul config is 41 MB because it embeds Adam optimizer state — drop it at write time

A cold soul index still costs up to ~2.2 s of event-loop latency, once.

Measured 2026-11-03: one .soul config is 41 MB of JSON and json.loads(s) is
~2.2 s (read 0.20 s, parse 2.21 s). The parse does not release the GIL, so no
thread placement helps — the loop stalls for the duration regardless of
warm_soul_index_async(). Today only the FIRST boot after a soul changes pays
it (the index then makes it ~12 ms), and _sync_soul_traits() now runs inside
the PGQ model-load worker so the stall cannot block /health or the model load
ordering — but it is still a visible latency cliff for in-flight requests.

Root cause is not the reader, it is the writer: `metadata.training_state`
carries the full Adam optimizer state (exp_avg / exp_avg_sq moments) as JSON
floats, ~41 MB of text for a 5 MB model. Inventory only needs name/description/
traits/personality.

Fix (soul-write time, one owner): when saving a .soul, store
metadata.training_state.optimizer out-of-band (sidecar file next to the soul,
e.g. .soul.optim.npz) or strip it from the config header behind an explicit
`soul save --keep-optimizer` flag. Resume-from-checkpoint must keep reading
it, so the loader grows one fallback: config.optimizer if present, else the
sidecar. Do NOT migrate existing artifacts in place on a 99%-full disk —
readers must tolerate both layouts, and a soul is re-written on its next save.

Acceptance: a newly saved soul's config header is < 1 MB; parse time for the
cold scan drops under 50 ms for 5 souls; an existing 41 MB soul keeps loading
unchanged (fallback path); a soul saved with --keep-optimizer still resumes
training byte-identically. Benchmark: scripts/benchmark_soul_index.py
(cold_seconds) before/after on a re-saved copy; and a resume-equivalence test
that trains 1 step from a checkpoint saved both ways and compares weights.
