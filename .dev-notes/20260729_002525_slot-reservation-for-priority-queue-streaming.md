---
id: 20260729_002525_slot-reservation-for-priority-queue-streaming
title: Slot reservation for priority queue streaming
status: done
tags: infrastructure,scheduler
created: 2026-07-29T00:25:25.046493+00:00
---

Slot reservation for priority queue streaming


Added acquire()/release() to PriorityRequestQueue for long-lived work (streaming). Worker handles reservation markers (coro=None). Wired ModelServer.generate_stream() through the queue. 5 new tests — all 21 priority queue tests pass. 42/43 server integration tests pass (1 pre-existing warmup timing flake).