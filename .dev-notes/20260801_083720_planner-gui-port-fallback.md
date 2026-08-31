---
id: 20260801_083720_planner-gui-port-fallback
title: planner gui port fallback
status: done
tags: planner,gui
created: 2026-08-01T08:37:20.990796+00:00
---

planner gui port fallback

planner gui now steps past occupied ports: _bind_server() tries the requested port then port+1..+attempts-1 on EADDRINUSE, prints 'port X in use, moved to Y'. --port 0 keeps ephemeral kernel assignment. Live-verified against occupied 8787 -> served on 8788. Module docstring + README updated. 2 new gui tests (step-past-occupied, ephemeral). 62 planner tests pass.