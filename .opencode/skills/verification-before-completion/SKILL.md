---
name: verification-before-completion
description: >
  Use before claiming any card, task, fix, or branch is done. Never trust a pass,
  a zero, or a subagent's word you did not cause yourself. Covers the false-clean
  traps, the verification incantation, proving a failure is pre-existing before
  blaming your change, mining output by root cause, and the merge-when-green rule.
---

# Verification Before Completion

## Overview

A claim of "done" is a claim about reality, so it needs evidence you produced
yourself, on the code as it stands right now. Every other form of confidence in
this repo has a body count: a card closed on a subagent's word, a sweep that
printed nothing and was reported as "0 issues", a "fixed" branch that was never
green on the machine that merged it.

> **A card is not done because something said so. It is done when you re-ran the
> proof and read the output.**

## Never trust a zero you did not cause

Real incident, this repo: `rg` (ripgrep) is **not installed**. A sweep ran
`rg ...`, printed nothing, read as _"0 issues found"_, and was reported as a real
result. The absence of an error is not the presence of a pass.

Three false-clean shapes to assume until disproven:

| Shape             | What happened                                     | Counter                                                     |
| ----------------- | ------------------------------------------------- | ----------------------------------------------------------- |
| Command not found | Sweep printed nothing                             | `command -v <tool>` first, or use the `grep` equivalent     |
| Piped exit code   | `cmd \| tail` hides the real status               | Redirect to a file, read `$?`, never pipe for the exit code |
| Empty collection  | Import error killed the run; zero tests collected | Assert the collected count is what you expected             |

**Confirm the tool ran.** After any sweep, deliberately search for a string you
know exists (`__init__`, `def `) and expect hits. A clean you cannot distinguish
from a broken tool is not a clean.

## Re-run it yourself

The verification incantation, from `AGENTS.md`:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/packages/downcraft:$PWD/packages/core-py:$PWD/apps/api/server" \
  /home/mana/miniconda3/envs/sloughgpt/bin/python -m pytest <paths> -q -p no:randomly
```

- `PYTHONNOUSERSITE=1` is mandatory: a stray `tests` package in
  `~/.local/lib/python3.12/site-packages/` shadows the repo's `tests/` and kills
  collection entirely — with `pytest` itself, so it dies as an import error, not a failure.
- Use the pinned conda interpreter. Never bare `python`, `python3`, or `pip`.
- `pytest.ini` defaults to `-m "not slow"`; server/e2e/integration files need `-m slow` and a live API.
- Lint gate: `ruff check <paths>` **and** `ruff format --check <paths>`.

Also verify the _shape_ of the commit, not just its tests:

```bash
git show --stat <sha>      # only the card's owned paths?
```

A commit that sweeps unrelated files is a defect even when green. Pathspec limits
**files, not hunks** — `git commit -- <path>` commits that file's whole
working-tree state, so a foreign uncommitted edit rides along under your message.

## Prove pre-existing before blaming your change

When a sweep surfaces failures you did not expect, do not guess:

```bash
git worktree add --detach /tmp/opencode/wt-head-check HEAD
cd /tmp/opencode/wt-head-check && <verification command> > /tmp/opencode/head_run.txt 2>&1
grep '^FAILED' /tmp/opencode/head_run.txt | sort > /tmp/opencode/head_failed.txt
# repeat in your tree, then:
diff /tmp/opencode/head_failed.txt /tmp/opencode/mine_failed.txt
```

Identical sets → pre-existing; quote that sha and that diff in the commit. A diff
→ your regression. Remove the probe worktree when done.

Known pre-existing debt that is **not yours to chase** (verify, don't re-fix):
`test_truth_context.py`, `test_feedback_dataclasses.py` + `test_feedback_router.py`.

## Mine by root cause, never by count

A run is evidence about the system, not a verdict on it. Two obligations, both
mandatory:

1. **Read the latest results before building anything new.** A red suite or a
   regressed benchmark changes what should be built next. Building on top of
   unread results is how regressions get buried.
2. **Group failures by root cause.** `"GPU/hardware 181, LoRA/HF 150"` is an
   answer; `"546 failed"` is not. Name the missing prerequisite or the actual bug
   behind each bucket.

Then turn each finding into a concrete next-feature candidate and feed it into
the next unit of work. A report that says only whether tests passed is incomplete.

## Merge when green — and only then

From `AGENTS.md` SOP: run the relevant benchmark after any training, inference,
quantization, or model-architecture change (`scripts/benchmark_*.py`). If it
regresses or fails, fix it or delete the branch — `git checkout main -- <files>`
or `git branch -D feat/<name>`. Never merge a broken implementation.

```
git checkout main && git merge feat/<name>     # only after tests + benchmarks pass
```

## The checklist

Before any "done":

- [ ] I ran the verification myself, on the current tree.
- [ ] The tool actually ran (expected-hit probe, or exit code read from a file).
- [ ] The collected/parsed count is nonzero and plausible.
- [ ] Failures are grouped by root cause, with the prerequisite named.
- [ ] Unexpected failures proven pre-existing by detached-worktree diff.
- [ ] `git show --stat` shows only owned paths; insertion count matches my edits.
- [ ] Lint **and** format gates clean.
- [ ] Benchmark run and recorded where the change touches training/inference.
- [ ] The landed sha is quoted on the card before it moves to `done`.
