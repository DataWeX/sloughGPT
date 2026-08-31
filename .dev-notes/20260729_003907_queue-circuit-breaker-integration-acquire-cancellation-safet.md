---
id: 20260729_003907_queue-circuit-breaker-integration-acquire-cancellation-safet
title: Queue-circuit-breaker integration + acquire cancellation safety
status: done
tags: infrastructure,scheduler,circuit-breaker
created: 2026-07-29T00:39:07.594688+00:00
---

Queue-circuit-breaker integration + acquire cancellation safety


Queue-full errors in ModelServer.generate() and generate_stream() now trip the circuit breaker (failure_threshold consecutive rejections opens the breaker). acquire() now safely handles task cancellation: if the marker is still in the heap, it's removed; if the worker already popped it, the slot is released. 4 new tests — all 67 tests pass (1 pre-existing warmup flake).