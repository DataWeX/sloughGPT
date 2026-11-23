---
id: 20261104_053224_perfstartup-resident-model-materialisation-peaks-74-gb-for-a
status: todo
board: 20261104_053224_perfstartup-resident-model-materialisation-peaks-74-gb-for-a
title: "perf(startup): resident-model materialisation peaks ~7.4 GB for a 2.5 GB artifact"
tags: ["perf", "model", "startup", "area:models"]
sprint: ""
gh: null
---

# perf(startup): resident-model materialisation peaks ~7.4 GB for a 2.5 GB artifact

`scripts/benchmark_lazy_autoload.py` (added with card 20261101_144013 scope B): eager from_slnc peaks 7319 MB and lazy+materialise peaks 7378 MB on a 2.5 GB int8 `.slnc`; resident after GC is ~1.2 GB. Scope B buys time to ready, not memory — the ~3x transient amplification during materialisation remains.
Not yet attributed: candidates worth measuring before choosing a fix are the read buffer plus the quantized-originals window (the call passes free_quantized_originals=True) and per-tensor copies during load.
Target: peak ≈ artifact size (mmap or stream), no regression in time_to_ready_s. Gate: benchmark_lazy_autoload.py --json peak_rss_mb drops materially against today's 7378 MB.
