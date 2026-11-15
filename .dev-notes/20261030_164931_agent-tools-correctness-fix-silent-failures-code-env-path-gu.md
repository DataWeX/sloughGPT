---
id: 20261030_164931_agent-tools-correctness-fix-silent-failures-code-env-path-gu
status: done
board: 20261030_164931_agent-tools-correctness-fix-silent-failures-code-env-path-gu
title: "agent tools correctness: fix silent failures, code env, path guard"
tags: [agents, tools, security, quality]
sprint: domain-hygiene
gh: null
---
# Agent tools correctness (user: "build the tools correctly")

## Audit + fixes — all 8 tool implementations reviewed, landed `0edfa9980`

- **code exec**: `sys.executable` (was bare `python3` — vanished under
  conda/env resolution); child **killed on wait_for-cancel** (was orphaned).
- **path guard**: new `_path_violation()` on file_read/file_search/
  data_analysis/image_analysis — sensitive dirs (~/.ssh, .gnupg, .aws,
  .kube, .docker), `.env*`/`id_*` basenames, and
  `SecurityConfig.allowed_directories` **which existed but was never
  enforced** on reads.
- **web search**: `follow_redirects=True`, `raise_for_status`, per-result
  unwrapped `url` via `_unwrap_ddg`.
- **data_analysis**: truncation flag; jsonl bad-tail-line skip.
- **knowledge**: fallback token-subset match (punctuation/line-wrap safe).
- **image_analysis**: honest `success: False` (was false success).
- **test_real_grep**: asserts an actual hit (was green on zero matches).

## Also on this branch (same push): model-quality pass `6e0296c8d`

Keyword rescue on step-0 (`[]` plans), `ast.literal_eval` fallback,
refusal+degeneracy synthesis guards (module-level `REFUSAL_OPENERS`),
few-shot planning prompt, grep `--max-count=0` fix + excludes.

## Gates / evidence

- Agent suite **661 passed** post-rebase over 15 incoming commits
  (was 471 pre-rebase in the narrower set); ruff check+format clean.
- Fakes updated (`_FakeAnchor`, kwargs AsyncClient, raise_for_status).
- Pushed `ad6c06e69..0edfa9980` + AGENT_SYNC entry (follow-up commit).
- Live smoke: keyword rescue → file_search 20 files; synthesis still weak
  at times — **real lever = SFT tool-use training via SloughGPTTrainer
  (card pending user go)**.
