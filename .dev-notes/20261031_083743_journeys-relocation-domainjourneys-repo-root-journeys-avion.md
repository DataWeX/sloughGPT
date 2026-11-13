---
id: 20261031_083743_journeys-relocation-domainjourneys-repo-root-journeys-avion
status: done
board: 20261031_083743_journeys-relocation-domainjourneys-repo-root-journeys-avion
title: "journeys relocation: domain/journeys -> repo-root journeys/ (avion harness out of domain)"
tags: [journeys, domain-hygiene, avion, mole]
sprint: domain-hygiene
gh: null
---
# Journeys relocation: `domain/journeys` → repo-root `journeys/`

## Ruling (from planning session)

- Target = **repo-root `journeys/`**, NOT `packages/uxjourneys` — that
  branch idea was vetoed by user decision `433ccdb9` (harness deleted;
  `tests/test_substrate_record_completeness.py` still carries the leftover
  root — left as-is, gate's cleanup is not this card's scope).
- **Why repo-root:** mole's journey probe does
  `importlib.import_module("<pkg>.journeys.runner")` from inside the running
  API server → target must inherit `domain/`'s importability (repo root on
  PYTHONPATH). `journeys` sits with `apps/`, `scripts/`, `tests/` — same seam.
- `python -m journeys` keeps working (repo root on path via `scripts/python`).

## Landed (pushed `075dc7c49`, branch `chore/journeys-out-of-domain`)

- `git mv domain/journeys journeys` (5 modules) + **50 lines across 21
  files** rewritten (`domain.journeys`→`journeys`, `domain/journeys`→
  `journeys`): harness self-refs, mole 3 files (2 importlib targets +
  comments), `scripts/{journey_report,site_doctor_probe}.py`, 6 tests,
  `docs/{TESTING,FEATURES,UX_FLOWS}.md`, 4 mobile TSX comments.
- `test_code_hygiene`: `SCAN_ROOTS` += `"journeys"`; its `PRINT_ALLOWLIST`
  key moved to `journeys/runner.py` (hygiene follows the folder).
- `test_domain_print_allowlist` (domain-only): dead key dropped.
- `test_substrate_record_completeness`: roots tuple now `"journeys"`.
- AGENT_SYNC entry rode the push (past entries untouched — history kept).

## Gates

- 6 gate files **85 passed** (1 slow deselected, 6 xfailed) — twice (pre-
  and post-rebase over 3 incoming commits).
- ruff check+format clean; `from journeys import FLOWS` = 13/17 flows;
  `python -m journeys --list` ok; `importlib.import_module("journeys.runner")`
  ok; `scripts/journey_report.py --help` ok.
- Remaining-ref grep: **zero** `domain.journeys`/`domain/journeys` outside
  `.dev-notes/*.md` (journal mirror — history keeps its wording) and past
  AGENT_SYNC entries.

## Pitfalls

- Pattern must be `domain/journeys`/`domain.journeys` — `flows.py` also
  contains `domain/infrastructure/...` (different path, untouched).
- Mobile TSX changes are comment-only → vitest/tsc not run (config-only
  precedent).
