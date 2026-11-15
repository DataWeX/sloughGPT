---
id: 20261101_115301_fix-consciousness-dashboard-blank-render-getqualia-envelope
status: done
board: 20261101_115301_fix-consciousness-dashboard-blank-render-getqualia-envelope
title: "Fix consciousness dashboard blank render — getQualia envelope stored as flat map crashes CurrentStateFold"
tags: []
sprint: ""
gh: null
---

# Fix consciousness dashboard blank render — getQualia envelope stored as flat map crashes CurrentStateFold

Audit finding (card 20261101_104216, finding 3): /consciousness/dashboard renders BLANK — journey blocker. Root cause: consciousnessController.getQualia() is typed Promise<Record<string,number>> but the backend returns {current:{valence...},narrative,history_count} (apiGet unwraps envelope.data at apps/lib/request.ts:255). useConsciousnessDashboard stores the whole object as currentQualia; CurrentStateFold (components/DashboardFolds.tsx:94) calls (v as number).toFixed(2) on the nested 'current' object -> TypeError crash-loop on every live refresh -> blank page. PRE-EXISTING: identical line in pre-decomposition page.tsx:166 (decompose refactor moved it verbatim). Test mock (page.test.tsx:86) replicated the wrong flat shape so tests stayed green. FIX: (1) controller getQualia return type -> envelope shape; (2) hook: setCurrentQualia(value.current ?? {}); (3) test mock -> real backend shape + assert qualia bars render; (4) fold: Number.isFinite guard so a future shape drift degrades to bars-skipped, never a blank page. Consumer check: getQualia() used only by this hook. Gates: tsc, eslint, vitest dashboard closure, then RE-SCREENSHOT to prove the page renders (audit methodology). benchmark n/a (bugfix).
