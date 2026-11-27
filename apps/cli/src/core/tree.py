"""Tree identity — which checkout this CLI runs from, and whether it is stale.

The failure this exists for (2026-11-27): ``~/.local/bin/sloughgpt`` pointed at
the main checkout, which was parked on a lane branch hundreds of commits behind
``origin/main``. Every command ran month-old code and the only way to notice was
to read the source. The web side already answers the same question with the
``X-Slo-Web-Tree`` header; this is the CLI's half.

Rules, and why:

* **``SLO_TREE`` wins when it names a real checkout, otherwise this tree.** No
  cwd probing and no auto-picking another worktree on launch: with dozens of
  worktrees (several owned by other sessions) silently running one of them is
  worse than a stale binary that says it is stale.
* **Staleness is read from local refs only** — ``HEAD..origin/main``. A launch
  must never fetch: it would block on the network and would rewrite refs in a
  tree somebody else may be using.
* **A branch name is not staleness.** Lane branches are legitimately ahead;
  being *behind* ``origin/main`` is the signal.

The counter is exposed as ``sloughgpt tree`` (plus one line on stderr at
startup), so the honest answer to "which code am I running" is one command,
not an archaeology session.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

GIT_TIMEOUT = 5
MAIN_BRANCHES = ("main", "master")

#: Printed at most once per process — ``main()`` and the welcome banner both ask.
_notice_cache: dict[str, str | None] = {}


def _git(tree: Path, *args: str) -> str | None:
    """Run a git command in ``tree``; ``None`` when git or the command fails."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(tree), *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _count(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def looks_like_tree(path: Path | str) -> bool:
    """A runnable checkout: the launcher, both top-level source dirs."""
    path = Path(path).expanduser()
    return (
        (path / "cli.py").is_file()
        and (path / "apps").is_dir()
        and (path / "packages").is_dir()
        and (path / "sloughgpt").is_file()
    )


def package_tree() -> Path:
    """The checkout this code lives in (``apps/cli/src/core/tree.py`` -> root).

    ``find_repo_root`` is the repository's existing answer to "where is the
    root" (walk up to ``apps/`` + ``packages/``); this must agree with it or a
    checkout would have two different ideas of itself.
    """
    from domain.shared._internal.paths import find_repo_root

    return find_repo_root(Path(__file__).resolve().parent)


def resolve_tree(env: dict[str, str] | None = None) -> Path:
    """The tree to run: ``SLO_TREE`` when valid, else the tree running this code."""
    env = os.environ if env is None else env
    override = (env.get("SLO_TREE") or "").strip()
    if override:
        candidate = Path(override).expanduser()
        if looks_like_tree(candidate):
            return candidate.resolve()
    return package_tree()


#: Dev-server header apps/web/vite.config.ts emits to identify its checkout.
WEB_TREE_HEADER = "X-Slo-Web-Tree"


def web_tree_id(root: Path | str | None = None) -> str:
    """Identity token of a checkout's web app — mirrors ``apps/web/vite.config.ts``.

    The dev server serves ``sha256(realpath(apps/web))[:16]`` as
    ``X-Slo-Web-Tree``; Node's ``realpathSync`` and Python's ``realpath``
    resolve symlinks the same way, so the two sides agree byte-for-byte. This
    is how the CLI can tell *its own* Vite from a foreign listener on the same
    port — port-open is not readiness.
    """
    import hashlib

    base = Path(root) if root is not None else package_tree()
    web_dir = os.path.realpath(str(base / "apps" / "web"))
    return hashlib.sha256(web_dir.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class TreeInfo:
    """What a checkout is, right now, from local refs."""

    path: Path
    branch: str
    head: str
    behind: int | None
    ahead: int | None
    #: ``None`` when not checked — the startup path skips ``git status``
    #: (96 ms on this repo) because it only needs the ref arithmetic.
    dirty: bool | None = None

    @property
    def name(self) -> str:
        return self.path.name or str(self.path)

    @property
    def current(self) -> bool:
        """Caught up with ``origin/main`` (or no ``origin/main`` to compare)."""
        return self.behind in (None, 0)

    @property
    def on_main(self) -> bool:
        return self.branch in MAIN_BRANCHES

    def where(self) -> str:
        """``branch @ head`` — what the tree is, not where it is."""
        if self.branch in ("", "HEAD"):
            return f"detached @ {self.head}"
        return f"{self.branch} @ {self.head}"

    def stale_reason(self) -> str | None:
        """One line explaining why this tree should not be run, or ``None``."""
        if self.behind is None or self.behind == 0:
            return None
        return f"{self.name} is {self.behind} commit(s) behind origin/main ({self.where()})"


def tree_info(path: Path | str, *, dirty: bool = False) -> TreeInfo:
    """Describe ``path`` using local refs only — never fetches."""
    path = Path(path).expanduser()
    # Two calls on purpose: `rev-parse --short --abbrev-ref HEAD` prints only
    # the branch, so a combined call silently reports the branch as the hash.
    head = _git(path, "rev-parse", "--short", "HEAD") or "?"
    branch = _git(path, "rev-parse", "--abbrev-ref", "HEAD") or "HEAD"

    behind = ahead = None
    counts = _git(path, "rev-list", "--count", "--left-right", "origin/main...HEAD")
    if counts:
        parts = counts.split()
        if len(parts) == 2:
            behind, ahead = _count(parts[0]), _count(parts[1])

    is_dirty = bool(_git(path, "status", "--porcelain")) if dirty else None
    return TreeInfo(
        path=path.resolve(), branch=branch, head=head, behind=behind, ahead=ahead, dirty=is_dirty
    )


@dataclass(frozen=True)
class Worktree:
    """One entry from ``git worktree list --porcelain``."""

    path: Path
    head: str
    branch: str


def worktrees(start: Path | str | None = None) -> list[Worktree]:
    """Every worktree of the repository containing ``start`` (default: this tree)."""
    start = Path(start) if start is not None else package_tree()
    out = _git(start, "worktree", "list", "--porcelain") or ""
    found: list[Worktree] = []
    current: dict[str, str] = {}
    for line in out.splitlines() + [""]:
        if not line.strip():
            if "worktree" in current:
                found.append(
                    Worktree(
                        path=Path(current["worktree"]),
                        head=current.get("HEAD", "?")[:7],
                        branch=current.get("branch", "").replace("refs/heads/", "") or "detached",
                    )
                )
            current = {}
            continue
        key, _, value = line.partition(" ")
        current[key] = value
    return found


def describe_all(
    paths: list[Path] | None = None, *, dirty: bool = True, workers: int = 8
) -> list[TreeInfo]:
    """``tree_info`` for many worktrees at once.

    Each tree costs ~350 ms of git (``status`` is the expensive one), so a
    serial walk over 40 worktrees took 14 s — long enough that the listing felt
    broken. These are independent read-only subprocesses; eight at a time keeps
    the answer under two seconds.
    """
    from concurrent.futures import ThreadPoolExecutor

    targets = list(paths) if paths is not None else [wt.path for wt in worktrees()]
    if not targets:
        return []
    if len(targets) == 1:
        return [tree_info(targets[0], dirty=dirty)]
    with ThreadPoolExecutor(max_workers=min(workers, len(targets))) as pool:
        return list(pool.map(lambda p: tree_info(p, dirty=dirty), targets))


def main_checkout(start: Path | str | None = None) -> Path | None:
    """The repository's primary checkout — first entry of ``git worktree list``."""
    found = worktrees(start)
    return found[0].path.resolve() if found else None


def latest_tree(candidates: list[TreeInfo], primary: Path | None = None) -> TreeInfo | None:
    """The tree to run: caught up with origin/main, clean — never a guess.

    ``primary`` is the repository's main checkout. When it is given, only that
    one may be returned: every other entry is somebody's lane worktree, and
    silently adopting one would put two sessions on it (the incident lane
    claims exist for). When it is omitted, the newest clean, current tree wins,
    preferring the main branch — used for reporting, never for adoption.
    """
    usable = [info for info in candidates if info.current and info.dirty is False]
    if not usable:
        return None
    if primary is not None:
        target = Path(primary).resolve()
        return next((info for info in usable if info.path == target), None)

    def score(info: TreeInfo) -> tuple[int, int]:
        return (1 if info.on_main else 0, _commit_time(info.path))

    return max(usable, key=score)


def _commit_time(path: Path) -> int:
    value = _git(path, "log", "-1", "--format=%ct")
    try:
        return int(value) if value else 0
    except ValueError:
        return 0


def summary_line(info: TreeInfo, *, path: bool = True) -> str:
    """``path  branch @ head  (current | N behind)`` — the one-line identity.

    ``path=False`` swaps the full path for the checkout's directory name, which
    is what a banner has room for; ``sloughgpt tree`` keeps the full path.
    """
    if info.behind is None:
        trend = "no origin/main"
    elif info.behind == 0:
        trend = "current"
    else:
        trend = f"{info.behind} behind"
    if info.ahead:
        trend += f", {info.ahead} ahead"
    if info.dirty:
        trend += ", dirty"
    where = str(info.path) if path else info.name
    return f"{where}  {info.where()}  ({trend})"


def stale_notice(info: TreeInfo | None = None, env: dict[str, str] | None = None) -> str | None:
    """The startup warning line, or ``None`` when there is nothing to say.

    Cached per process: the entry point and the welcome banner both ask, and
    neither should pay for the git calls twice.
    """
    env = os.environ if env is None else env
    key = (env.get("SLO_TREE") or "") + "|" + (env.get("SLO_NO_TREE_WARN") or "")
    if key in _notice_cache:
        return _notice_cache[key]
    if env.get("SLO_NO_TREE_WARN") == "1":
        _notice_cache[key] = None
        return None
    info = info or tree_info(resolve_tree(env))
    reason = info.stale_reason()
    notice = f"! {reason} — run `sloughgpt tree --set latest` to switch" if reason else None
    _notice_cache[key] = notice
    return notice


def active_claims() -> dict[Path, str]:
    """Worktrees another live session claims: ``{path: session}``.

    Reads the canonical claims store through ``grib``'s own fold (TTL and
    release semantics live there). A missing store or a broken line means "no
    claims", never "refuse everything" — the check is advisory.
    """
    try:
        from grib import claims as grib_claims
        from grib.config import default_board_dir
    except ImportError:
        return {}
    try:
        rows = grib_claims.fold(default_board_dir())
    except (OSError, ValueError):
        return {}
    out: dict[Path, str] = {}
    for row in rows:
        if row.get("status") != "active" or not row.get("worktree"):
            continue
        try:
            out[Path(row["worktree"]).resolve()] = str(row.get("session") or "")
        except OSError:
            continue
    return out


def claim_blocker(target: Path, session: str | None = None) -> str | None:
    """Why ``target`` must not be adopted, or ``None`` when it is free.

    Another session's worktree may be mid-rebase or mid-test with an untouched
    but real working state; switching the launcher onto it would put two
    sessions on one checkout — the incident lane claims exist to prevent.
    """
    session = session or os.environ.get("OPENCODE_SESSION_ID") or ""
    owner = active_claims().get(target.resolve())
    if owner and owner != session:
        return f"{target} is claimed by another session ({owner})"
    return None


def launcher_path() -> Path | None:
    """The ``sloughgpt`` executable on PATH, or ``None``."""
    found = shutil.which("sloughgpt")
    return Path(found) if found else None


def adopted_tree(target: Path | str, *, session: str | None = None) -> Path:
    """Validate ``target`` as a tree to run and return it resolved.

    Raises :class:`ValueError` with a human reason — every rejection here is
    something the user can see and fix, so the message is the feature.
    """
    target = Path(target).expanduser()
    if not looks_like_tree(target):
        raise ValueError(f"{target} is not a sloughgpt checkout (no cli.py/apps/packages)")
    info = tree_info(target, dirty=True)
    if info.behind is None:
        raise ValueError(f"{target} has no origin/main to compare against")
    if info.behind:
        raise ValueError(
            f"{target} is {info.behind} commit(s) behind origin/main — not adopting it"
        )
    if info.dirty:
        raise ValueError(f"{target} has uncommitted changes — commit or stash them first")
    blocker = claim_blocker(info.path, session)
    if blocker:
        raise ValueError(blocker)
    return info.path


def repoint_launcher(target: Path, launcher: Path | None = None) -> tuple[Path, Path]:
    """Point the ``sloughgpt`` launcher at ``target``'s own launcher.

    The launcher is tree-bound by construction (``$DIR/scripts/python
    $DIR/cli.py``), so the symlink target *is* the code that runs — which is
    exactly why it must not be repointed without the checks in
    :func:`adopted_tree`. Writes a temp link and renames it over the old one:
    readers see either the old or the new target, never a missing link.

    Returns ``(old, new)`` where both are tree paths.
    """
    launcher = launcher or launcher_path()
    if launcher is None:
        raise ValueError("sloughgpt is not on PATH — cannot find the launcher to repoint")
    if not launcher.is_symlink():
        raise ValueError(f"{launcher} is not a symlink — repoint it by hand, or set SLO_TREE")
    old_target = launcher.resolve()
    new_target = target / "sloughgpt"
    if old_target == new_target.resolve():
        return old_target.parent, target
    tmp = launcher.with_name(launcher.name + f".new-{os.getpid()}")
    try:
        os.symlink(new_target, tmp)
        os.replace(tmp, launcher)
    finally:
        if tmp.exists() or tmp.is_symlink():
            try:
                tmp.unlink()
            except OSError:
                pass
    return old_target.parent, target


def temp_link_dir() -> Path:
    """Where tests put throwaway launchers (kept out of the real PATH)."""
    return Path(tempfile.mkdtemp(prefix="slo-tree-"))
