"""Worktree lifecycle — create one for a card, reap it when the card closes.

The failure this exists for (2026-11-28): **65 registered worktrees**, 9 of them
carrying their own ``node_modules``, 30 claim events (2 live, 28 expired), and a
primary checkout whose working tree had silently reverted a landed ``done``
commit for three hours. Worktrees are cheap to create and nothing reaped them,
so the fleet grew faster than the cards.

``core.tree`` already answers *which tree am I running and is it stale*. This is
the other half: **who owns a tree, is its branch merged, and may it be removed.**

Rules, and why:

* **A worktree exists for the life of one card.** Creation claims it; reaping
  releases it. A tree with no card is a leak, not a workspace.
* **Reaping is prove-or-refuse.** Clean tree, branch merged into ``origin/main``,
  no live claim by another session, never the primary checkout, never a
  protected path. ``--force`` is the deliberate override and says so in output.
* **Claims stay advisory.** An expired claim is free; a live one stops the
  *default* reap, not the user — same contract as ``core.tree.claim_blocker``.

Derived-state seeding (symlinked ``node_modules``, shared caches) deliberately
does not live here: it is a separate change unit on its own surface, and plugs
into :func:`create` later rather than widening it now.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from core import tree as tree_mod

#: Worktrees a rule must never reap. Two classes, one gate:
#:
#: * **user data** — ``.wt-fbdrv`` (feedback/uploads/gallery journals) and
#:   ``.wt-voice`` (pre-Next-removal agent workspace), both held by AGENTS.md;
#: * **machine-managed infrastructure** — ``kanban-sync-wt`` is the single
#:   detached mirror worktree ``scripts/kanban_sync.sh`` exports and pushes
#:   from. It is clean, detached and merged, so every other rule calls it
#:   dead; reaping it breaks the board producer until it is recreated. Found
#:   by running ``wt ls`` on the real repo, not by a test.
#:
#: ``KANBAN_SYNC_WT`` can point the sync script at another path; that path is
#: not protected by name, so it must never be a lane-shaped ``feat/`` tree.
PROTECTED_NAMES = (".wt-fbdrv", ".wt-voice", "kanban-sync-wt")

#: Branch prefix a created lane gets; reaping refuses to delete anything else.
LANE_PREFIX = "feat/"


class WorktreeError(RuntimeError):
    """A lifecycle action refused. The message is the human reason."""


@dataclass(frozen=True)
class Lifecycle:
    """One worktree's lifecycle state: what it is, who owns it, may it go."""

    info: tree_mod.TreeInfo
    #: Session id holding a live claim, or ``None`` when the tree is free.
    claim: str | None
    #: Is this worktree's HEAD reachable from ``origin/main``?
    merged: bool
    #: The repository's primary checkout — never reaped, never "dead".
    primary: bool

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def protected(self) -> bool:
        return self.name in PROTECTED_NAMES

    @property
    def state(self) -> str:
        """``live`` (claimed) / ``dead`` (reapable) / ``idle`` (anything else)."""
        if self.claim:
            return "live"
        if self.primary or self.info.on_main or self.protected:
            return "idle"
        if self.merged and self.info.dirty is False:
            return "dead"
        return "idle"

    def reason(self) -> str:
        """Why this tree is not reapable, in one line (empty when it is)."""
        if self.primary:
            return "primary checkout"
        if self.info.on_main:
            return "on main"
        if self.protected:
            return "protected (never reaped by a rule)"
        if self.claim:
            return f"claimed by {self.claim}"
        if self.info.dirty:
            return "uncommitted changes"
        if not self.merged:
            return "branch not merged into origin/main"
        return ""


def _run(tree: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command in ``tree``; raise :class:`WorktreeError` on failure."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(tree), *args],
            capture_output=True,
            text=True,
            timeout=tree_mod.GIT_TIMEOUT,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise WorktreeError(f"git {' '.join(args)} failed: {exc}") from exc
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise WorktreeError(detail[0] if detail else f"git {' '.join(args)} failed")
    return proc


def _is_ancestor(tree: Path, commit: str, base: str) -> bool:
    """Is ``commit`` reachable from ``base``? Unknown refs read as "not merged"."""
    proc = _run(tree, "merge-base", "--is-ancestor", commit, base, check=False)
    return proc.returncode == 0


def _branch_exists(tree: Path, branch: str) -> bool:
    proc = _run(tree, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}", check=False)
    return proc.returncode == 0


def slug_for(title: str) -> str:
    """The lane slug for a card title — one spelling for path and branch."""
    try:
        from grib.ids import slugify
    except ImportError:  # pragma: no cover - grib is a workspace package
        slugify = None
    if slugify is not None:
        slug = slugify(title, max_len=40)
        if slug:
            return slug
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in title.lower())
    return "-".join(part for part in cleaned.split("-") if part)[:40] or "card"


def _claim_event(event: str, *, card_id: str | None, path: Path, branch: str, session: str) -> None:
    """Append a claim/release event through grib's own fold.

    In-process by design (same process -> function call): the claims store is a
    file both sides read, so an endpoint would be the expensive rung for a
    single append. A missing grib means "no claim", never "refuse the action".
    """
    try:
        from grib.claims import append_event
        from grib.config import default_board_dir
    except ImportError:
        return
    try:
        append_event(
            default_board_dir(),
            event=event,
            session=session,
            card_id=card_id,
            worktree=str(path),
            branch=branch,
        )
    except OSError:
        return


def create(
    card_id: str,
    title: str,
    *,
    base: str = "origin/main",
    root: Path | str | None = None,
    session: str | None = None,
    board_dir: Path | str | None = None,
) -> tuple[Path, str]:
    """Create the lane worktree for ``card_id`` and claim it. Returns (path, branch).

    ``root`` defaults to the repository's primary checkout, so every lane lands
    beside it rather than inside whichever tree happened to invoke the command.
    """
    root_path = (
        Path(root) if root is not None else (tree_mod.main_checkout() or tree_mod.package_tree())
    )
    root_path = root_path.resolve()
    if _run(root_path, "rev-parse", "--show-toplevel", check=False).returncode != 0:
        raise WorktreeError(f"{root_path} is not a git repository")

    slug = slug_for(title)
    path = root_path / f".wt-{slug}"
    branch = f"{LANE_PREFIX}{slug}"

    if path.exists():
        raise WorktreeError(f"{path} already exists — reap it with `sloughgpt wt done {path}`")
    if _branch_exists(root_path, branch):
        raise WorktreeError(f"branch {branch} already exists — delete it or pick another title")
    if (
        not _is_ancestor(root_path, base, base)
        and _run(root_path, "rev-parse", "--verify", "--quiet", base, check=False).returncode != 0
    ):
        raise WorktreeError(f"base {base} does not resolve to a commit")

    _run(root_path, "worktree", "add", str(path), "-b", branch, base)
    _claim_event(
        "claim",
        card_id=card_id,
        path=path,
        branch=branch,
        session=session or _session(),
        board_dir=board_dir,
    )
    return path, branch


def classify(
    paths: list[Path] | None = None,
    *,
    session: str | None = None,
    dirty: bool = True,
    root: Path | str | None = None,
) -> list[Lifecycle]:
    """Lifecycle state for every worktree (or just ``paths``).

    ``root`` names the repository to read (default: the tree this code runs
    from). Tests point it at a throwaway repo; the CLI leaves it alone.
    """
    primary = tree_mod.main_checkout(root)
    primary = primary.resolve() if primary else None
    claims = tree_mod.active_claims()
    targets = list(paths) if paths is not None else [wt.path for wt in tree_mod.worktrees(root)]
    infos = tree_mod.describe_all(targets, dirty=dirty)
    out: list[Lifecycle] = []
    for info in infos:
        claim = claims.get(info.path)
        if claim and claim == session:
            claim = None  # our own claim is not a blocker
        out.append(
            Lifecycle(
                info=info,
                claim=claim,
                merged=info.on_main or _is_ancestor(info.path, "HEAD", "origin/main"),
                primary=primary is not None and info.path == primary,
            )
        )
    return out


def dead_trees(session: str | None = None, *, root: Path | str | None = None) -> list[Lifecycle]:
    """Worktrees that are provably finished: merged, clean, unclaimed."""
    return [item for item in classify(session=session, root=root) if item.state == "dead"]


def reap(
    path: Path | str,
    *,
    force: bool = False,
    session: str | None = None,
    board_dir: Path | str | None = None,
    root: Path | str | None = None,
) -> str:
    """Remove a finished worktree, delete its branch, release its claim.

    Raises :class:`WorktreeError` with the reason when it is not safe to remove.
    """
    target = Path(path).expanduser().resolve()
    session = session or _session()

    primary = tree_mod.main_checkout(root)
    if primary is not None and target == primary.resolve():
        raise WorktreeError(f"{target} is the primary checkout — never reaped")
    if target == tree_mod.resolve_tree().resolve():
        raise WorktreeError(f"{target} is the tree you are running from — cd elsewhere first")
    if target.name in PROTECTED_NAMES and not force:
        raise WorktreeError(
            f"{target.name} is protected — user data or machine-managed "
            "(AGENTS.md); pass --force only if you mean it"
        )

    known = {wt.path.resolve() for wt in tree_mod.worktrees(root)}
    if target not in known:
        raise WorktreeError(f"{target} is not a registered worktree of this repository")

    info = tree_mod.tree_info(target, dirty=True)
    owner = tree_mod.active_claims().get(target)
    merged = _is_ancestor(target, "HEAD", "origin/main")

    blockers: list[str] = []
    if info.dirty and not force:
        blockers.append("uncommitted changes (commit or stash, or --force to discard)")
    if not merged and not force:
        blockers.append("branch is not merged into origin/main (--force to discard)")
    if owner and owner != session and not force:
        blockers.append(f"claimed by another live session ({owner})")
    # A worktree exists for the life of one card: reaping before the card is
    # closed deletes the workspace of work that has not landed. ``card_column``
    # answers ``None`` when grib is absent — an unknown card never blocks.
    card = claimed_card(target)
    if card and not force:
        column = card_column(card)
        if column is not None and column != "done":
            blockers.append(f"card {card} is '{column}', not done (close it, or --force)")
    if blockers:
        raise WorktreeError(f"refusing to reap {target.name}: " + "; ".join(blockers))

    branch = info.branch
    _run(primary or target, "worktree", "remove", str(target), *(["--force"] if force else []))
    if branch and branch not in tree_mod.MAIN_BRANCHES and branch.startswith(LANE_PREFIX):
        _run(primary or target, "branch", "-D" if force else "-d", branch, check=False)
    _claim_event(
        "release",
        card_id=None,
        path=target,
        branch=branch,
        session=session,
        board_dir=board_dir,
    )
    removed = f"removed {target.name}"
    return f"{removed}{' (forced)' if force else ''}, branch {branch} deleted, claim released"


def card_column(card_id: str) -> str | None:
    """The card's board column, or ``None`` when grib cannot answer."""
    try:
        from grib.store import get_store
    except ImportError:  # pragma: no cover - grib is a workspace package
        return None
    try:
        card = get_store().get_card(card_id)
    except (OSError, ValueError):
        return None
    return card.column if card else None


def card_title(card_id: str) -> str | None:
    """The card's title, or ``None`` when grib cannot answer.

    The lane slug is derived from the title, so `wt new <card>` can name the
    branch without the caller restating what the card already says.
    """
    try:
        from grib.store import get_store
    except ImportError:  # pragma: no cover - grib is a workspace package
        return None
    try:
        card = get_store().get_card(card_id)
    except (OSError, ValueError):
        return None
    return card.title if card else None


def claimed_card(path: Path | str) -> str | None:
    """The card a worktree's claim names, if any session ever claimed it."""
    target = Path(path).expanduser().resolve()
    try:
        from grib.claims import fold
        from grib.config import default_board_dir
    except ImportError:  # pragma: no cover
        return None
    try:
        rows = fold(default_board_dir())
    except (OSError, ValueError):
        return None
    for row in rows:
        worktree = row.get("worktree")
        if worktree and Path(worktree).resolve() == target and row.get("card_id"):
            return str(row["card_id"])
    return None


def _session() -> str:
    return os.environ.get("OPENCODE_SESSION_ID") or "local"
