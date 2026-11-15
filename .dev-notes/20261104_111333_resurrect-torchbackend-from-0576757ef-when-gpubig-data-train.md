---
id: 20261104_111333_resurrect-torchbackend-from-0576757ef-when-gpubig-data-train
status: todo
board: 20261104_111333_resurrect-torchbackend-from-0576757ef-when-gpubig-data-train
title: "Resurrect _TorchBackend from 0576757ef when GPU/big-data training becomes a real journey"
tags: ["training", "gpu", "torch", "follow-up", "area:training"]
sprint: ""
gh: null
---

# Resurrect _TorchBackend from 0576757ef when GPU/big-data training becomes a real journey

Main deleted _TorchBackend (~250 lines, domain/slolib/_internal/gpu.py) in 0576757ef: torch is a multi-GB wheel nobody was installing, and the arm was declared-but-unexecuted (dishonest seam). Replaced by a loud refusal — _REMOVED_BACKENDS makes select_backend('torch')/device='torch' fail with the reason; the seam is now cpu|cuda|metal|opencl only. infra/docker/Dockerfile.cli agrees (torch intentionally NOT installed). RULING (user, 2026-11-04): keep the deletion — the code is parked, not lost: git show 0576757ef^:domain/slolib/_internal/gpu.py restores it in one command, and _REMOVED_BACKENDS is the signpost for anyone who types 'torch' later. RESURRECTION TRIGGER: only when GPU/big-data training at volume lands as a real journey (big infra, data work) — resurrect from history THEN, likely evolved: torch APIs, SloughGPTTrainer and the device-declaration seam (ef8c88bce) will all have moved. Reverting now = re-porting against a seam that changed twice the same day.
