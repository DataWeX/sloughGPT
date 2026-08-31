---
id: 20260729_095516_eventbus-fixes-slonetprovider-temperaturecancel-fixes
title: EventBus fixes + SloNetProvider temperature/cancel fixes
status: done
tags: event-bus,slonet,tests
created: 2026-07-29T09:55:16.933902+00:00
---

EventBus fixes + SloNetProvider temperature/cancel fixes


Fixed: temperature or 0.7 falsy bug, cancel_event forwarding in SloNetServer/SloNetChatProvider, 7 EventBus bugs (callable validation, clear(*) wildcards, emit_sync async skip, docstring, inline import warnings), psutil ImportError crash in kernel_npu.py. Added: 11 EventBus tests (32 total), 6 ModelMetrics edge-case tests. Fixed: pytest asyncio fixture_loop_scope deprecation warning in both pytest.ini files. Fixed: GPT-2 weight transposition bug (transpose_weights=False→True for Conv1D in arch_config.py). Fixed: cancel_event not forwarded from SloNetServer.generate() to _generate_sync(). Added: EventBus events for model registration/unregistration in model_registry.py (model.registered, model.unregistered).