---
title: Backend cleanup: remove NumpyBackend and ONNXBackend dead classes
created: 2026-07-29T01:50:08.058429+00:00
updated: 2026-07-29T01:50:08.058429+00:00
tags: infrastructure, model-server, cleanup
status: done
---

Removed NumpyBackend (~90 lines) and ONNXBackend (~70 lines) backend classes — both were never instantiated in production (only in test fixtures). Simplified _select_backend to GuardBackend > LocalBackend only. Removed numpy_engine/onnx_engine params from ModelServer.__init__. Deleted test_onnx_backend.py (6 tests), removed TestNumpyBackend and TestModelServerNumpy from test_numpy_engine.py (7 tests). All 44 server integration + 23 queue tests pass.
