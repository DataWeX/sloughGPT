---
id: 20260824_130653_backend-error-classification-across-routers
title: Backend error classification across routers
status: done
tags: backend
created: 2026-08-24T13:06:53.962359+00:00
---

Backend error classification across routers

Replaced raise_error(str(e)) with classify_and_raise(e) in experiments.py (5 locations), collections.py (5 locations), world_render.py (4 locations). Added Field constraints to ExperimentCreate.name.