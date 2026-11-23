---
id: 20261101_052821_decompose-vm-pagetsx-stateful-hook-dumb-components-1657l
status: done
board: 20261101_052821_decompose-vm-pagetsx-stateful-hook-dumb-components-1657l
title: "Decompose vm page.tsx — stateful hook + dumb components (1657L)"
tags: []
sprint: ""
gh: null
---

# Decompose vm page.tsx — stateful hook + dumb components (1657L)

Second target of the oversized-page decompose series (template card 20261031_173606 done: consciousness/dashboard 874L->80L). Target: apps/web/app/(app)/vm/page.tsx (1657L — largest of the series; remaining: agents 1646, souls 1565, knowledge 1359, developer 1237). Colocated page.test.tsx is 987L and must stay green. error.tsx (13L) + loading.tsx (10L) untouched. OWNERSHIP CHECK: feat/cave-game-vm + feat/vm-asm-intake touch page.tsx but are 24 days stale, 1488 behind origin/main, unmerged — abandoned, not live owners; they need rebase regardless. PATTERN (from dashboard template): (1) vm-data.ts — types + formatters (pure); (2) useVmPage.ts (or split: useVmSession/useVmConsole/etc per state cluster) — all useState/useEffect/controllers/handlers verbatim; (3) components/ — dumb presentational sections (console/log pane, control buttons, file/asm intake, status/KPI, etc — actual split decided after reading the file); (4) page.tsx -> thin composition, JSX moved class-for-class. Behavior byte-identical, colocated test untouched+green. GATES: tsc, eslint 0 errors (record warning delta), vitest vm closure, audit greps (no hardcoded hues). SOP: branch refactor/vm-page-decompose from origin/main, benchmark n/a (UI refactor), push HEAD:main with AGENT_SYNC entry, card done with evidence.
