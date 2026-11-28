"""Worktree lifecycle command — `sloughgpt wt new|ls|done`.

`sloughgpt tree` answers *which tree am I running, and is it stale*. This
answers the other half: **who owns a tree, is its branch merged, and may it be
removed** — so a lane is created with one command and reaped with one command
instead of accumulating as a directory nobody dares delete.

The failure it exists for is measured, not hypothetical: 65 registered
worktrees, 9 carrying their own `node_modules`, and a primary checkout whose
working tree had silently reverted a landed `done` commit for three hours.

    sloughgpt wt new 20261128_103632   # create the lane, claim it
    sloughgpt wt ls                    # what is live, dead, and why
    sloughgpt wt done                  # reap this lane (card must be done)
    sloughgpt wt done --all            # reap every provably dead lane
"""

from __future__ import annotations

from pathlib import Path

from core import tree as tree_mod
from core import worktree as wt_mod

from domain.logging import get_global

log = get_global()


def _state_tag(item: wt_mod.Lifecycle) -> str:
    """ok for a healthy lane, warn for a leak the user should look at."""
    return "warn" if item.state == "dead" else "ok"


def cmd_wt_new(args) -> int:
    """Create the lane for a card and claim it."""
    card = getattr(args, "card", None)
    title = getattr(args, "title", None)
    if not card and not title:
        log.error("wt new needs a card id or --title (the slug is derived from it)")
        return 2
    if title is None:
        title = wt_mod.card_title(card) if card else None
    if not title:
        log.error(f"card {card} is not on the board — pass --title to name the lane")
        return 2

    try:
        path, branch = wt_mod.create(
            card or "",
            title,
            base=getattr(args, "base", None) or "origin/main",
            root=getattr(args, "root", None),
        )
    except wt_mod.WorktreeError as exc:
        log.error(str(exc))
        return 1

    log.success(f"Lane {path.name} -> {branch}")
    log.info(f"cd {path}")
    log.info("Claim recorded. Reap it with `sloughgpt wt done` once the card is done.")
    return 0


def cmd_wt_ls(args) -> int:
    """List every worktree with its lifecycle state."""
    items = wt_mod.classify(
        session=None,
        root=getattr(args, "root", None),
    )
    if not items:
        log.info("no worktrees")
        return 0
    order = {"live": 0, "dead": 1, "idle": 2}
    log.header("Worktrees")
    for item in sorted(items, key=lambda i: (order.get(i.state, 3), i.name)):
        detail = item.info.where()
        if item.info.dirty:
            detail += ", dirty"
        if item.info.behind:
            detail += f", {item.info.behind} behind"
        reason = item.reason()
        if item.state == "dead":
            detail += "  [reapable]"
        elif reason:
            detail += f"  [{reason}]"
        log.status(item.name, detail, _state_tag(item))

    dead = [item for item in items if item.state == "dead"]
    if dead:
        log.info(f"{len(dead)} reapable — `sloughgpt wt done --all` removes them")
    return 0


def cmd_wt_done(args) -> int:
    """Reap a finished lane (or every provably dead one with --all)."""
    root = getattr(args, "root", None)
    force = bool(getattr(args, "force", False))

    if getattr(args, "all", False):
        dead = wt_mod.dead_trees(root=root)
        if not dead:
            log.info("nothing reapable — every tree is live, dirty, or unmerged")
            return 0
        failures = 0
        for item in dead:
            try:
                log.success(wt_mod.reap(item.info.path, force=False, root=root))
            except wt_mod.WorktreeError as exc:
                log.warning(str(exc))
                failures += 1
        if failures:
            log.info(f"{len(dead) - failures} reaped, {failures} refused")
            return 1
        log.success(f"{len(dead)} lane(s) reaped")
        return 0

    target = getattr(args, "path", None)
    if target is None:
        target = tree_mod.resolve_tree()
    try:
        log.success(wt_mod.reap(Path(target), force=force, root=root))
    except wt_mod.WorktreeError as exc:
        log.error(str(exc))
        return 1
    return 0


def cmd_wt(args) -> int:
    """Dispatch `sloughgpt wt <subcommand>`."""
    sub = getattr(args, "subcommand", None)
    if sub == "new":
        return cmd_wt_new(args)
    if sub in ("ls", "list"):
        return cmd_wt_ls(args)
    if sub == "done":
        return cmd_wt_done(args)
    log.error("wt: pick a subcommand — new | ls | done")
    return 2
