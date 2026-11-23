---
id: 20261104_062921_kanban-syncservice-runs-with-timeoutstartsecinfinity-and-flo
status: done
board: 20261104_062921_kanban-syncservice-runs-with-timeoutstartsecinfinity-and-flo
title: "kanban-sync.service runs with TimeoutStartSec=infinity, and flock -n makes every later tick silently skip"
tags: ["kanban", "systemd", "observability", "producer"]
sprint: ""
gh: null
---

# kanban-sync.service runs with TimeoutStartSec=infinity, and flock -n makes every later tick silently skip

Measured 2026-11-04: systemctl --user show kanban-sync.service -p TimeoutStartUSec => infinity. One 'git fetch origin main' that stalled at 17:36:03 on Nov 3 held the single-writer flock (/run/user/1000/kanban-sync.lock) for ~12h56m; because scripts/kanban_sync.sh uses 'flock -n' and exits 0 with 'another run holds the lock — skipping', all ~150 subsequent timer ticks succeeded cosmetically while producing nothing. The board froze with no failure signal anywhere except the (also misleading, see sibling card) freshness check. The stalled run eventually completed on its own (Result=success around 06:33, then 07:02 'pushed (2 paths)') — nothing had to be killed. Fix: add TimeoutStartSec (e.g. 120s) to infra/systemd-user/kanban-sync.service and the installed copy, and give the lock-skip path a staleness signal (emit which PID holds it, and fail when the holder is older than N minutes) so a frozen producer is loud instead of invisible. Unresolved: I could not confirm WHY that ssh stalled — no ControlMaster config and no cm-* sockets, and the /proc agent-socket probe lost its target once the process exited.
