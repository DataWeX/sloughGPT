---
id: 20261108_134212_off-critical-path-policy-for-heavy-visuals-and-heavy-compute
status: todo
board: 20261108_134212_off-critical-path-policy-for-heavy-visuals-and-heavy-compute
title: "Off-critical-path policy for heavy visuals and heavy compute"
tags: ["frontend", "perf", "decision", "area:web"]
sprint: ""
gh: null
---

# Off-critical-path policy for heavy visuals and heavy compute

Source: fungolabs launchpad analysis (2026-11-08). Their FHE encryption runs in a versioned Web Worker (encrypt.worker.3.7.0.js) with batchRpcCalls:true and a 300s explicit budget and terminate() teardown — while their decorative 608KB three.js hero loads eagerly and makes Lighthouse return PAGE_HUNG + CLS 0.832. The lesson: they knew how to offload and spent it on the important half only. PLAN: (1) inventory sloughGPT surfaces that run heavy work during first paint (canvas/WebGL/terminal renderers, large charts, animation loops). (2) Adopt the policy: expensive visuals gate on intersection + requestIdleCallback with a static poster first; heavy compute goes to a Web Worker with an explicit long timeout and teardown. (3) Record the rule in docs/PRODUCT_ENGINEERING.md so the designer/frontend agents enforce it. VERIFY: no new above-the-fold rAF loop on route entry (test or lint heuristic) + Lighthouse/fps check on the touched page stays audit-able.
