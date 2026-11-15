---
id: 20261113_094108_fix-native-engine-toolchain-false-sentinel-guards-ubuntu-260
status: done
board: 20261113_094108_fix-native-engine-toolchain-false-sentinel-guards-ubuntu-260
title: "fix native engine toolchain + False-sentinel guards: ubuntu-26.04 runner gcc breaks CI test-core-py (150 red on main, blocks all PR lanes)"
tags: ["area:kernels"]
sprint: ""
gh: null
---

# fix native engine toolchain + False-sentinel guards: ubuntu-26.04 runner gcc breaks CI test-core-py (150 red on main, blocks all PR lanes)

Root cause (forensics 2026-11-13): GitHub flipped ubuntu-latest to Image ubuntu-26.04 (20261101.186). The3 C kernels (tensor_kernels.c/trace_core.c/scene_build.c) deliberately omit #include <math.h> (tensor header even documents gcc 13 assumption) — old gcc treated implicit tanh/sqrtf as warnings, new gcc errors → compile_cpu returncode!=0 → engines unavailable. Second bug amplifies: _load_engines() returns None|engine|False (False = cached failure) but consumers trace() and test_cycles_native._native_available only guard None → False leaks → AttributeError 'bool' has no attribute cpu → collection ERROR +150-test cascade (test_cycles*/world_stream/vm_bus/render_neural/tensor*). Exactly reproduces on main CI run43383838040 (efad52746) and PR #311 attempts ×2. Fix: add math.h ×3 C files + if not eng guards ×2 (precedent: _native_build.py slot-or-None idiom). Blocks: #311/#54/#91/#55/#254 test-core-py.
