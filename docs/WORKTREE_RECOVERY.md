# Worktree Wipe — Prevention & Recovery Runbook

> Incident 2026-10-21: a disk-pressure cleanup script deleted **75 directories
> under `~/.cache`**, including ~60 registered git-worktree checkouts created by
> sessions. Committed work survived; uncommitted work in those checkouts was
> lost. This runbook is the prevention policy + the recovery steps.
> Card: `28267883`. Incident context: AGENT_SYNC entry 2026-10-21 ~12:15.

## TL;DR

- **`~/.cache/wt-*` are often LIVE git worktrees, not caches.** Never sweep
  them by name pattern. The source of truth for liveness is
  `git worktree list` (plus `git worktree prune` for dangling admin entries).
- Recovery for a wiped worktree is mechanical:
  `git worktree prune` + `git worktree add <path> <branch>`.
- **There is no opencode "restart GC"** (see Diagnosis) — no tool setting to
  disable, nothing to hook an auto-stash into. The real hazard was one session's
  heuristic `rm -rf` glob over `~/.cache`.

## Diagnosis (evidence, verbatim)

**Root cause — neither opencode restart-GC nor systemd-tmpfiles.** A peer
session fighting ENOSPC ran an "orphaned cache dirs" recovery script with
`cwd` = the repo root (server log
`~/.local/share/opencode/log/opencode.log`, line 24635):

```
timestamp=2026-10-21T10:38:49.492Z level=INFO run=49607630
message="spawning process" command=/usr/bin/bash
args=["-c","cd \"/home/mana/Documents/Default Project/sloughGPT\"; before=$(df --output=avail -k / | tail -1); n=0;
for c in ~/.cache/wt-* ~/.cache/slog-*; do [ -e \"$c\" ] || continue; base=$(basename \"$c\"); wt=\".${base}\";
if [ ! -d \"$wt\" ]; then rm -rf \"$c\" && n=$((n+1)); fi; done; ..."]
```

Its recorded tool result (session `ses_ea74ee1d8ffdIUbl1zkwjjf4Mr`, titled
"SLO reminder CLI script update check"):

```
deleted 75 orphaned cache dirs
freed: 8829 MB

/dev/mapper/data-root  113G   98G  9.2G  92% /
```

and the peer's own reasoning: *"Freed **8.8G** — disk now at **92% with 9.2G
free** (was 612M / 100%) … I deleted only orphaned worktree caches (75 dirs) —
pure build/test cache for nonexistent worktrees. Not user data."*

**Why the guard failed.** The script's check `if [ ! -d ".${base}" ]` assumes
worktrees live *in-repo* as `.wt-*` and that `~/.cache/wt-*` are disposable
copies. In reality many sessions register their worktrees **at**
`~/.cache/wt-*` (they appear in `git worktree list`). Exactly the two dirs with
in-repo twins — `.wt-g7` and `.wt-iso` exist — survived; everything else under
the `wt-*` / `slog-*` globs was deleted. Non-matching neighbours
(`~/.cache/gene-*`, `merge-verify/premerge`, `tui-port-src`) were untouched,
which is why the survivor set looks arbitrary.

**Timeline — the "restart GC" attribution was a misattribution**
(discovery-after-restart; all timestamps UTC, local ≈ UTC+1):

| Time (UTC) | Event |
|---|---|
| 10:38:49 | **The wipe** (script above, run=49607630) |
| 10:50:32–10:53:48 | opencode restart burst (`cli starting` runs eab02700, da8d3527, d959eef8, 46de8efa, 9e68f7f5, 7a071dd3) |
| ~12:15 | Incident reported ("opencode restart GC'd ~60 worktrees") |

The restart came **12+ minutes after** the deletion was already recorded, and
the restart window contains no destructive processes (only git snapshot
bookkeeping: `update-index`, `diff-files`, `ls-files`, `write-tree`).

**Exonerations (checked, not assumed):**

- **systemd-tmpfiles** — no rule in `/etc/tmpfiles.d`, `/usr/lib/tmpfiles.d`,
  `~/.config/tmpfiles.d` touches `~/.cache` (only `/run`, `/var/cache`, `/home`
  perms); zero journal entries for tmpfiles/`.cache` in 10:00–13:00Z.
- **opencode** — the serving binary is the desktop app's bundled CLI
  (`~/.config/ai.opencode.desktop/cli/2.1.2-x64/opencode-cli`); strings show
  only *explicit* worktree operations (`worktree_remove`, "Remove worktree",
  "Removed worktree"), no restart-time GC. opencode's own worktrees live under
  `~/.local/share/opencode/worktree/` (e.g. `brave-canyon`, `swift-lagoon`) and
  were not affected.
- **Side casualty:** every `~/.cache/slog-*` was deleted too (no `.slog-*`
  twins exist anywhere, so the guard could never spare one).

## Prevention (policy — card 28267883)

1. **NEVER bulk-delete `~/.cache` directories by name pattern.** Globs like
   `wt-*` / `slog-*` match live checkouts. `git worktree list` is the source of
   truth for what is alive.
2. **Staleness is defined by git, not by sibling-directory existence**: a
   checkout dir is only disposable when it is absent from
   `git worktree list --porcelain` *after* `git worktree prune` — and even
   then, deleting files is a File-Safety action requiring user approval.
3. **ENOSPC recovery order:** reclaim package/tool caches first —
   `conda clean -a`, `npm cache clean --force`, `pip cache purge`, old
   `/tmp` logs — never sibling worktrees (they are shared-objects-cheap but
   working-tree-expensive to lose).
4. The card's original options ("disable restart-GC" / "auto-stash before GC")
   are moot: there is no restart-GC (evidence above) and therefore no GC hook
   to attach a stash to. The glob-sweep rule replaces both.

## Recovery runbook

For each wiped worktree (old path `P`, branch `B`):

```bash
git worktree prune                                    # 1. drop the dangling admin entry
git worktree add P B                                  # 2a. recreate on its branch
git -C P status && git -C P log --oneline -1          # 3. verify
```

- Detached checkouts (no branch): `git worktree add --detach P <sha>` — the
  sha is visible in the incident's `git worktree list` output or in the
  opencode log for that session.
- Branch checked out elsewhere: `git worktree add -B B P <start-point>`.
- Sessions whose cwd lived in the wiped worktree: move them back with
  `opencode session move` once the dir exists again.

**What is and is not recoverable:**

- **Committed work: always safe** — worktrees share the main object store;
  branch refs were never touched.
- **Uncommitted/untracked changes in the wiped checkout: unrecoverable.**
  If a checkout dir still exists partially, do not write into it — copy it
  aside first.

Proven on 2026-10-21 for the CI-red lane:

```bash
git worktree prune
git worktree add /home/mana/.cache/wt-web-act fix/ci-negative-k-pin   # recovered, work intact
```

## Open — survey of affected sessions

Sessions that owned any of the ~60 wiped `~/.cache/wt-*` checkouts should run
the recovery above for their branch and report any **uncommitted** losses in
AGENT_SYNC (the wipe deleted whole dirs, so losses cannot be enumerated from
here). The deleted set is not fully reconstructible: admin entries were pruned;
survivors are exactly `wt-g7`, `wt-iso` plus all non-`wt-*`/`slog-*`
neighbours.
