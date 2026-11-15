---
id: 20261107_113837_training-throughput-14-stepss-vs-documented-31-trainer-threa
status: todo
board: 20261107_113837_training-throughput-14-stepss-vs-documented-31-trainer-threa
title: "training throughput 1.4 steps/s vs documented 31+ (trainer thread ~18% CPU)"
tags: ["training", "performance", "area:training"]
sprint: ""
gh: null
---

# training throughput 1.4 steps/s vs documented 31+ (trainer thread ~18% CPU)

Evidence job_cbc6162a: device=cpu (by design, SloNet pure numpy), @_gc_parked active on train() so the documented GC fix applies, yet measured 1.4 steps/s while train_pipeline.py:864 documents 29 vs 31+ steps/s with GC parked. API process averages ~18% CPU during the run -> thread mostly waiting, not computing. Known incident class: trace.py documents job_c87a2dd6 (0.4 steps/s, GPU idle, silent CPU fallback). Overhead visible in logs: eval every 100 steps (~2.5s) + checkpoint .soul save roughly every 60-90s (16 saves/28min). Use StepTrace tripwires (device, gpu_matmuls, cpu_matmuls, fallbacks) to profile where the time goes.
