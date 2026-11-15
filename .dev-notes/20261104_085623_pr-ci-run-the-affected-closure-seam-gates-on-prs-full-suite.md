---
id: 20261104_085623_pr-ci-run-the-affected-closure-seam-gates-on-prs-full-suite
status: todo
board: 20261104_085623_pr-ci-run-the-affected-closure-seam-gates-on-prs-full-suite
title: "PR CI: run the affected closure + seam gates on PRs, full suite only on merge to main (align with Test Rule clause 3)"
tags: ["tests", "ci", "area:tooling"]
sprint: ""
gh: null
---

# PR CI: run the affected closure + seam gates on PRs, full suite only on merge to main (align with Test Rule clause 3)

Goal: Test Rule clause 3 says the local/PR loop runs only the affected closure and merge-to-main runs the full suite as the backstop — but PR CI pays the full matrix anyway (measured 2026-11-04: test-core-py 31m52s + E2E 26m27s + test-web 10m20s per PR, every push). Bring CI in line with the rule.\n\nScope (design decisions for intake/decomposition):\n- PR jobs: affected-closure selection (git diff base → changed files → pytest-testmon selection or a conservative mapping) + the seam gates which always run (contract, board integrity, feature tags, lint, command parity) — they're seconds.\n- Merge-to-main: full suite unchanged (the backstop — never weakened).\n- Required checks: test-core-py is a required check NAME; policy must decide whether the required check becomes 'affected+seam' on PRs.\n- Trade-off to accept explicitly: main can go briefly red post-merge instead of pre-merge; the backstop catches it and the offender fixes forward fast.\n\nPrecedent: kanban-sync paths-ignore (docs/.kanban never trigger CI) already proved the 'right-size the trigger' direction.\nNot in scope: weakening the merge-to-main full run.
