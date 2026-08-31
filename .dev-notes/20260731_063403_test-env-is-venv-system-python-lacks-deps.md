---
id: 20260731_063403_test-env-is-venv-system-python-lacks-deps
title: Test env is .venv — system python lacks deps
status: done
tags: tests,environment,workflow
created: 2026-07-31T06:34:03.550213+00:00
---

Test env is .venv — system python lacks deps

The repo .venv (Python 3.12.3) has pydantic 2.13.4, fastapi 0.139.2, psutil 7.2.2, starlette 1.3.1, pytest 9.1.1. System /usr/bin/python3 lacks these, so regression runs must use .venv/bin/python (via PYTHONPATH=packages/core-py). Final regression (124 modules, per-file, venv): 108 PASS / 0 FAIL / 4 ERROR (weight-only: morph_tokenizer, numpy_engine, point_library, safetensors_loader) / 11 slow-module deselections (pytest.ini -m 'not slow'). All source/test bugs fixed; remaining errors need gpt2 weight download (~500MB, bandwidth-gated).