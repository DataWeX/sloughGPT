"""
app_planner.verify — three-way diff: journal note <-> board card <-> git.

The note card (markdown + YAML frontmatter) is the source of truth for a
task's metadata. `verify` reports any drift between:

  1. ``status:`` in the note frontmatter  <->  ``column`` of its board card
     (via ``config.STATUS_TO_COLUMN``);
  2. the ``board:`` frontmatter uuid      <->  a card id in ``board.jsonl``
     (falls back to matching the card by the note's title);
  3. the ``landed:`` frontmatter shas     <->  real commits in a git repo —
     each sha must exist and be an **ancestor of HEAD** (i.e. actually
     landed on the checked-out line), otherwise the note claims a landing
     that the repo does not have.

Findings are returned as data (:class:`Finding`); the CLI renders them and
exits 1 when any finding is not ok, so it can gate CI or pre-commit hooks.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class Finding:
    """One diff result: ``kind`` in {status, board, git}, ``ok`` = no drift."""

    kind: str
    ok: bool
    detail: str

    def __str__(self) -> str:
        mark = "✓" if self.ok else "✗"
        return f"{mark} [{self.kind}] {self.detail}"


def parse_frontmatter(path: Path) -> dict[str, str]:
    """Raw frontmatter of a journal note.

    Unlike ``core.Note.from_markdown`` this keeps *every* key (``board``,
    ``landed``, …) — verify exists precisely for the keys the Note dataclass
    does not model. Falls back to line parsing when PyYAML is unavailable.
    """
    text = Path(path).read_text()
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}
    meta: dict[str, str] = {}
    block: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        block.append(line)
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()
    try:
        import yaml

        # Parse only the frontmatter block: feeding the whole file to
        # safe_load sees the closing ``---`` as a second document, throws
        # on every note, and silently leaves the raw (still-quoted) values
        # above — which made quoted titles never match a card title.
        # Only YAML string values are adopted: quoting is resolved there,
        # while non-scalar parses (dates, underscore ints, lists) keep the
        # byte-exact raw line value verify compares against.
        parsed = yaml.safe_load("\n".join(block)) or {}
        if isinstance(parsed, dict):
            meta.update({str(k): v for k, v in parsed.items() if isinstance(v, str)})
    except Exception:
        pass
    return meta


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30
    )


def _landed_groups(landed: str) -> list[tuple[str, list[str]]]:
    """Parse ``shas (label), shas (label)`` into ``(label, shas)`` groups.

    Each parenthesized label applies to the sha segment **before** it, e.g.
    ``5962f1742 (main/api-std), b1b39a7e5 (feat/x)`` — a label after the last
    group's closing paren governs that group; a trailing unlabeled segment is
    treated as mainline. Labels starting with ``main`` (or no label) require
    ancestor-of-HEAD; any other label is provenance only — the sha must
    **exist** in the repo but is allowed to live on an unmerged/rewritten
    branch (squash landings never make the branch sha an ancestor).
    """
    groups: list[tuple[str, list[str]]] = []
    label_re = re.compile(r"\(([^)]*)\)")
    prev_end = 0
    for m in label_re.finditer(landed):
        seg = landed[prev_end : m.start()]
        shas = re.findall(r"\b[0-9a-f]{7,40}\b", seg.lower())
        if shas:
            groups.append((m.group(1).strip(), shas))
        prev_end = m.end()
    tail = landed[prev_end:]
    tail_shas = re.findall(r"\b[0-9a-f]{7,40}\b", tail.lower())
    if tail_shas:
        # a trailing unlabeled segment (e.g. "A (main/x), B") is mainline
        groups.append(("", tail_shas))
    return groups


def _is_mainline(label: str) -> bool:
    return not label or label.lower().startswith("main")


def _load_cards(board_file: Path) -> list[dict]:
    if not Path(board_file).exists():
        return []
    cards: list[dict] = []
    for raw in Path(board_file).read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "id" in obj and "title" in obj:
            cards.append(obj)
    return cards


def verify_note(note_path: Path, board_file: Path, repo: Path | None = None) -> list[Finding]:
    """Diff *note_path* against its board card and (if ``landed:`` is set)
    against the git history of *repo* (default: current directory)."""
    note_path = Path(note_path)
    board_file = Path(board_file)
    repo = Path(repo) if repo else Path.cwd()

    meta = parse_frontmatter(note_path)
    status = (meta.get("status") or "").lower()
    title = meta.get("title") or note_path.stem
    board_uuid = meta.get("board", "").strip()
    landed = meta.get("landed", "")
    cards = _load_cards(board_file)

    findings: list[Finding] = []

    # 1+2: locate the card (uuid first, title fallback) and diff status/column
    card = next((c for c in cards if board_uuid and c["id"] == board_uuid), None)
    if card is None:
        card = next((c for c in cards if c["title"] == title), None)

    if card is None:
        findings.append(
            Finding(
                "board",
                False,
                f"no board card for note '{title}'"
                + (f" (board: {board_uuid})" if board_uuid else ""),
            )
        )
    else:
        if board_uuid and card["id"] != board_uuid:
            findings.append(
                Finding("board", False, f"frontmatter board: {board_uuid} != card id {card['id']}")
            )
        else:
            findings.append(
                Finding("board", True, f"note links card {card['id']} ('{card['title']}')")
            )

        want_col = config.STATUS_TO_COLUMN.get(status, "todo")
        have_col = card.get("column", "")
        if have_col == want_col:
            findings.append(
                Finding("status", True, f"note status '{status}' <-> column '{have_col}'")
            )
        else:
            findings.append(
                Finding(
                    "status",
                    False,
                    f"note status '{status}' maps to column '{want_col}', but card is '{have_col}'",
                )
            )

    # 3: landed shas must exist; mainline shas must also be ancestors of HEAD
    groups = _landed_groups(landed)
    if not groups:
        return findings
    if not (repo / ".git").exists() and _git(repo, "rev-parse", "--git-dir").returncode != 0:
        findings.append(Finding("git", False, f"repo not found: {repo}"))
        return findings
    for label, shas in groups:
        mainline = _is_mainline(label)
        for sha in shas:
            if _git(repo, "cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
                findings.append(Finding("git", False, f"{sha}: not found in repo {repo}"))
            elif mainline:
                if _git(repo, "merge-base", "--is-ancestor", sha, "HEAD").returncode != 0:
                    findings.append(
                        Finding(
                            "git", False, f"{sha}: exists but NOT an ancestor of HEAD (not landed)"
                        )
                    )
                else:
                    findings.append(Finding("git", True, f"{sha}: landed (ancestor of HEAD)"))
            else:
                findings.append(
                    Finding("git", True, f"{sha}: exists on branch '{label}' (provenance)")
                )
    return findings


def format_report(findings: list[Finding]) -> str:
    lines = [str(f) for f in findings]
    drift = sum(1 for f in findings if not f.ok)
    lines.append("all consistent" if drift == 0 else f"{drift} diff(s) found")
    return "\n".join(lines)
