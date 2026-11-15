---
id: 20261031_083744_domain-baseline-gate-catalogue-39-domain-dirs-ratchet-test-p
status: done
board: 20261031_083744_domain-baseline-gate-catalogue-39-domain-dirs-ratchet-test-p
title: "domain-baseline gate: catalogue 39 domain dirs + ratchet test (page-rule pattern)"
tags: [domain-hygiene, ccgt, baseline, ratchet, gate]
sprint: domain-hygiene
gh: null
---
# domain-baseline gate: CCGT catalogue + shrink-only ratchet (landed `a98510145`)

## Ruling (user, this card)

- "add metadata and hash to the board" + consolidate known dupes into CCGT
  + explicit pick **"1" = directory git-tree hash + staged plan**: catalogue
  now; physical merges ride their own cards; 4-parent re-root evaluated
  later (NOT now).
- CCGT = **coverage, not partition** (facets multi-label; e.g. inference =
  core + training) — `docs/PRODUCT_ENGINEERING.md:28`.
- `domain/mobile` stays as-is (feature-class + misleading-name note).

## Landed (branch `feat/domain-baseline-gate` → `HEAD:main` = `a98510145`)

- `domain/domain-baseline.json` — all **37** `domain/*/` dirs (title says
  39; actual = 37 — journeys already moved to repo-root by card
  083743, `__pycache__` pruned): `{class, facets[], consumers[], note,
  debt, tree_hash}` + `metadata{schema, generated, recorded_commit,
  hash-provenance, ratchet_rule}` + `ratchet{misfiled:0, projection:0,
  debt:6}` (6 = api/infrastructure/ops overlap + cognition/cognitive dup
  pair + mobile name).
- `tests/core-py/test_domain_baseline.py` — 8 tests: completeness both
  directions, schema/evidence, shrink-only counts, new-dirs-cannot-be-
  misfiled, **hash drift** vs `git rev-parse HEAD:domain/<name>`.
- `scripts/domain_baseline_rehash.py` — hashes-only rewrite (humans move
  classification); `--check` = exit 1 on drift.
- AGENT_SYNC entry rode the push (union-merged over two concurrent
  conflicts).

## Gates

- 8 passed (pre- and post-rebase); ruff check+format clean; `--check` = no
  drift at landing.
- **Alarm proven by corruption test**: mutated one hash → drift test red →
  restored byte-identical.
- **Rehash loop exercised for real**: incoming image-facade commit touched
  `agents`/`inference` contents → drift (2) → rehashed → amend → green →
  push.

## Pitfalls

- After ANY rebase that brings `domain/` changes, rehash before push or CI
  reds by design.
- Disk hit 0 bytes mid-gate (pytest cannot mkdtemp) — freed playwright +
  Cypress binary caches (~5G, regenerable; apps/web has cypress dep — CI
  installs its own).

## Follow-ups (carded)

② cognition↔cognitive merge (orphan: zero external consumers);
③ api/ops/infrastructure overlap resolution;
④ 4-parent CCGT re-root decision after ② ③ shrink.
