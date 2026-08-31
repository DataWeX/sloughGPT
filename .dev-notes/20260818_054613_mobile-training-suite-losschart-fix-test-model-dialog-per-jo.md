---
id: 20260818_054613_mobile-training-suite-losschart-fix-test-model-dialog-per-jo
title: Mobile training suite: LossChart fix, Test Model dialog, per-job stop
status: done
tags: mobile,training,status:done
created: 2026-08-18T05:46:13.469046+00:00
---

Mobile training suite: LossChart fix, Test Model dialog, per-job stop

Fixed 3 issues in the mobile training suite:

1. **LossChart scope bug** — Moved LossChart inside TrainingScreen component so it has access to  from useColors() hook. Was referencing module-scope  which didn't exist. Converted to useCallback.

2. **Test Model dialog** — Added POST /inference/generate integration. After training completes (distill or fine-tune), a 'Test Model' button opens a modal where users type a prompt and see the model's response. Shows loading state and error handling.

3. **Per-job stop for fine-tune** — training-store.ts stop() now calls POST /training/jobs/{id}/stop when a hfJobId exists. Previously only cleared local state, leaving the server job running.

Verification: npx tsc --noEmit passes (only pre-existing KnowledgeScreen errors). 838/838 mobile tests pass (91 training tests). 1 pre-existing failure in KnowledgeScreen test.