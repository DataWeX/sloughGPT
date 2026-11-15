---
id: 20261101_152856_graft-dev-ports-grab-portsh-replaces-devrestart-blind-kill-9
status: done
board: 20261101_152856_graft-dev-ports-grab-portsh-replaces-devrestart-blind-kill-9
title: "Graft dev ports — grab-port.sh replaces dev:restart blind kill -9 (identity-checked via X-Slo-Web-Tree)"
tags: []
sprint: ""
gh: null
---

# Graft dev ports — grab-port.sh replaces dev:restart blind kill -9 (identity-checked via X-Slo-Web-Tree)

User ruling: 'should be like grafting' — ports grafted, not murdered. Built scripts/grab-port.sh: lsof holder -> curl X-Slo-Web-Tree vs this checkout's sha256(realpath(apps/web))[:16] -> match: SIGTERM + GRAB_PORT_DRAIN 5s poll + KILL escalation (mirrors dev-stack.sh family trait); mismatch/absent: identify pid+cmd+tree ids, exit 1, zero signals. dev:restart rewired (package.json). AGENTS.md Core Rules graft rule added. Root defect: header had ZERO consumers (declaration-without-enforcement). Tests 3/3: free exit 0; foreign :5173 (pid 1603) refused+identified; our-tree disposable :59173 grafted TERM. Scope: mid-train resumability (JobStore checkpoint) = separate card; e2e:vite 5173 hardcode = follow-up.
