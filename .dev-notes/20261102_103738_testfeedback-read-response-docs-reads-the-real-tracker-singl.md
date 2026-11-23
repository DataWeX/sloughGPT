---
id: 20261102_103738_testfeedback-read-response-docs-reads-the-real-tracker-singl
status: done
board: 20261102_103738_testfeedback-read-response-docs-reads-the-real-tracker-singl
title: "test(feedback): read_response_docs reads the real tracker singleton, so another file's docs leak into the 'empty journal' test"
tags: ["testing", "feedback", "ci"]
sprint: ""
gh: null
---

# test(feedback): read_response_docs reads the real tracker singleton, so another file's docs leak into the 'empty journal' test

FOUND 2026-11-02 ~10:30 in CI/CD run 41422832686 (8ff6f3432), job test-core-py.

RED: TestReadResponseDocs::test_canonical_dir_with_empty_journal_returns_none —
`assert [...docs...] is None`, and the leaked docs are canned fixture text
('hi', 'capital of France?' -> 'The capital of France is Paris.').

THE MECHANISM: the test monkeypatches `_canonical_log_dir()` to a fresh tmp dir but
then calls read_response_docs(), which resolves the singleton via
`get_response_tracker()`. That singleton is process-global and lazily built on the
REAL canonical dir; whichever test file logged a response first leaves a populated
collection in it. The canonical-dir gate then passes (tmp == patched canonical), the
singleton hands back the other file's docs, and the 'empty journal' assertion sees
them. Passes in isolation (183/183 with test_slolib_gpu), fails only in the full run
— classic shared-mutable-singleton order dependency.

FIX: patch `_response_tracker` to a fresh ResponseTracker(log_dir=tmp) alongside
`_canonical_log_dir`, exactly like the sibling test_canonical_dir_returns_docs does.
The test's subject is 'empty collection -> None', which needs an empty collection,
not 'whatever the process happens to have cached'.

WHY IT BLOCKS: same as the gpu card — CI/CD red on main, deploy gated.
