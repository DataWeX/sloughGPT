"""Tree command — which checkout this CLI runs from, and switching it.

`sloughgpt tree` is the CLI's answer to the incident in `core/tree.py`'s
docstring: the launcher is tree-bound, so whatever it points at is the code
being run, and that is invisible from the command line. This command makes it
visible (`--resolve` for scripts, `--check`/`--list` for humans) and switchable
(`--set latest`, `--set PATH`) behind the checks that stop it from stealing a
checkout another session owns.
"""

from pathlib import Path

from core import tree as tree_mod

from domain.logging import get_global

log = get_global()


def _summary(info: tree_mod.TreeInfo) -> str:
    """One line: where, what it is, how far behind."""
    return tree_mod.summary_line(info)


def _print_check(info: tree_mod.TreeInfo) -> None:
    log.header("Tree")
    log.status("Running", _summary(info), "ok" if info.current else "warn")
    owner = tree_mod.active_claims().get(info.path)
    if owner:
        log.status("Claimed", owner, "warn")
    reason = info.stale_reason()
    if reason:
        log.warning(reason)
        log.info("Switch with: sloughgpt tree --set latest")
    else:
        log.info("This is the code the launcher runs (SLO_TREE overrides it).")
    if info.rolled_back:
        log.warning(info.rollback_reason())
        for finding in info.rolled_back[:10]:
            log.info(f"  {finding.path}")
        if len(info.rolled_back) > 10:
            log.info(f"  … and {len(info.rolled_back) - 10} more")
        log.info("Back up local work, then: git restore --source=HEAD --worktree")


def _print_list() -> None:
    log.header("Worktrees")
    claims = tree_mod.active_claims()
    infos = tree_mod.describe_all(rollback=True)
    for info in sorted(infos, key=lambda i: (not i.current, i.name)):
        trend = "current" if info.current else f"{info.behind} behind"
        if info.dirty:
            trend += ", dirty"
        if info.rolled_back:
            trend += f", rolled back ({len(info.rolled_back)})"
        owner = claims.get(info.path)
        if owner:
            trend += f", claimed {owner}"
        log.status(info.name, f"{info.where()}  {trend}", "ok" if info.current else "warn")


def _resolve_latest() -> Path:
    """The repository's main checkout, when it is clean and current.

    Deliberately not "the newest worktree": dozens of clean lane worktrees are
    in flight on this box, and adopting one would put two sessions on it. The
    incident was the *main* checkout being stale, so that is what gets fixed.
    """
    primary = tree_mod.main_checkout()
    if primary is None:
        raise ValueError("not inside a git repository — nothing to adopt")
    best = tree_mod.latest_tree(tree_mod.describe_all(), primary=primary)
    if best is None:
        raise ValueError(
            f"the main checkout {primary} is not clean and current — commit, "
            "`git pull --ff-only`, or pass an explicit path"
        )
    return best.path


def _do_set(target: str) -> int:
    try:
        chosen = _resolve_latest() if target == "latest" else Path(target)
        adopted = tree_mod.adopted_tree(chosen)
        old, new = tree_mod.repoint_launcher(adopted)
    except ValueError as exc:
        log.error(str(exc))
        return 1
    if old == new:
        log.info(f"Already running {new}")
        return 0
    log.success(f"Launcher: {old} -> {new}")
    log.info("New code takes effect on the next `sloughgpt` invocation.")
    return 0


def cmd_tree(args):
    """Show or switch which checkout the launcher runs."""
    target = getattr(args, "set", None)

    if target is not None:
        raise SystemExit(_do_set(target))

    if getattr(args, "resolve", False):
        print(tree_mod.resolve_tree())
        return

    if getattr(args, "list", False):
        _print_list()
        return

    _print_check(tree_mod.tree_info(tree_mod.resolve_tree(), dirty=True, rollback=True))
