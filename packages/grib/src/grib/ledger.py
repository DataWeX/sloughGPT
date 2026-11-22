"""Parse, query, and check the session ledger (``docs/AGENT_SYNC.md``).

The doc stays the single hand-written source — sessions prepend prose entries
at line 1 exactly as before; this module only *reads* it (no second store to
drift, the doc travels via git with the code).

Real format, measured off the doc (280 entries, 2026-10-10 → now):

- An entry starts at a line beginning with a bold ISO date
  (``**YYYY-MM-DD — …``; suffixes `` ~HH:MM`` and ``/DD`` ranges occur).
- The header's bold closes *mid-line* (``…card `done`.** prose…``) — never
  required at line end — and the entry runs until the next entry line. New
  entries are one long line; the October era wraps across many lines.
- The 23 mid-entry emphasis lines (``**Gates:**``) never start with the date
  prefix, so only the date prefix is the entry signal.

``grib ledger check`` gates shape (every bold-dated line parses, header bold
closes somewhere in the block) and order (dates non-increasing top-to-bottom).
Order is a *review* signal, not proof: a bottom-append and a backdated insert
(a session writing yesterday's date on top of today's) produce the same local
inversion — measured on the real doc, backdating is part of the convention.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

LEDGER_RELPATH = Path("docs") / "AGENT_SYNC.md"

# Entry header: bold ISO date, optional ~time or /day-range suffix, em-dash,
# then the summary prose (whose bold closes somewhere on this line or a later
# line of the block).
_HEADER_RE = re.compile(
    r"^\*\*(\d{4})-(\d{2})-(\d{2})( ~\d{1,2}:\d{2}(?::\d{2})?|/[0-9]{1,2})? — (.*)"
)
# Bold + year prefix — the ONLY entry signal (mid-entry **Gates:** never matches).
_ENTRY_PREFIX = "**20"
# Readable card ids: long slug form and the short 8_6 form (both appear).
_CARD_RE = re.compile(r"\b20\d{6}_\d{6}(?:_[0-9a-z-]+)?")
# Backticked hex shas, plus range endpoints (a..b counts both sides).
_SHA_RE = re.compile(r"([0-9a-f]{7,40})\.\.([0-9a-f]{7,40})|`([0-9a-f]{7,40})`")
_PR_RE = re.compile(r"#(\d{2,6})")


def find_doc(start: Path | None = None) -> Path | None:
    """Locate ``docs/AGENT_SYNC.md`` by walking up from *start* (default cwd).

    Repo-scoped by design: running outside a checkout reports "not found"
    instead of silently checking whichever checkout installed the package."""
    base = (start or Path.cwd()).resolve()
    for d in (base, *base.parents):
        cand = d / LEDGER_RELPATH
        if cand.is_file():
            return cand
    return None


def _parse_blocks(doc: Path) -> tuple[list[dict], list[dict]]:
    """Split the doc into entries. Returns ``(entries, malformed)``; entries
    carry a private ``text`` key (the full block) that public projections drop."""
    raw: list[tuple[int, str, str]] = []  # (line, order_date, block_text)
    heads: list[tuple[int, str]] = []  # (line, rest-of-header prose) parallel
    malformed: list[dict] = []
    cur_no: int | None = None
    cur_hdr: re.Match[str] | None = None
    cur_lines: list[str] = []

    def _flush() -> None:
        nonlocal cur_no, cur_hdr, cur_lines
        if cur_no is not None and cur_hdr is not None:
            text = "\n".join(cur_lines)
            if text.find("**", 2) < 0:
                # Bold opens at the header and never closes in the block.
                malformed.append(
                    {
                        "line": cur_no,
                        "kind": "malformed",
                        "msg": "header bold never closes: " + cur_lines[0][:90],
                    }
                )
            else:
                raw.append((cur_no, _order_date(cur_hdr), text))
                heads.append((cur_no, cur_hdr[5]))
        cur_no, cur_hdr, cur_lines = None, None, []

    for i, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith(_ENTRY_PREFIX):
            if cur_no is not None:
                cur_lines.append(line)
            continue
        m = _HEADER_RE.match(line)
        if m is not None and _valid_date(m) is None:
            m = None
        if m is None:
            _flush()
            malformed.append({"line": i, "kind": "malformed", "msg": line.strip()[:100]})
            continue
        _flush()
        cur_no, cur_hdr, cur_lines = i, m, [line]

    _flush()

    entries: list[dict] = []
    for (no, order_date, text), (_, rest) in zip(raw, heads):
        shas: list[str] = []
        for sm in _SHA_RE.finditer(text):
            for g in (sm[1], sm[2], sm[3]):
                if g:
                    shas.append(g)
        entries.append(
            {
                "line": no,
                "date": order_date,
                # Lane = header prose up to the first ":" or " — " (heuristic).
                "lane": re.split(r":| — ", rest, maxsplit=1)[0].strip(),
                "cards": list(dict.fromkeys(_CARD_RE.findall(text))),
                "shas": list(dict.fromkeys(shas)),
                "prs": list(dict.fromkeys(int(p) for p in _PR_RE.findall(text))),
                "head": rest[:160],
                "text": text,
            }
        )
    return entries, malformed


def _valid_date(m: re.Match[str]) -> date | None:
    """Resolve the header's date (a ``/DD`` suffix is a range → its end day,
    which is when the entry was written). Returns None when impossible."""
    try:
        d = date(int(m[1]), int(m[2]), int(m[3]))
        suffix = m[4] or ""
        if suffix.startswith("/"):
            d = date(d.year, d.month, int(suffix[1:]))
        return d
    except ValueError:
        return None


def _order_date(m: re.Match[str]) -> str:
    d = _valid_date(m)
    return d.isoformat() if d else ""


def _project(entries: list[dict]) -> list[dict]:
    """Drop the private full-text key — the index is a summary, not a dump."""
    return [{k: v for k, v in e.items() if k != "text"} for e in entries]


def parse(doc: str | Path) -> list[dict]:
    """JSON index of every entry: date, lane, card ids, shas, PR numbers, head."""
    entries, _ = _parse_blocks(Path(doc))
    return _project(entries)


def fingerprint(v: dict) -> str:
    """Stable identity of a violation for baseline files.

    Line numbers shift on every prepend (the doc's whole write pattern), so a
    line-keyed baseline would churn daily. Date + header head (order) and the
    message (malformed) survive prepends; they change only when the entry
    itself is edited — which is exactly when review should fire again."""
    if v["kind"] == "order":
        return f"order|{v['date']}|{v['head'][:80]}"
    return f"{v['kind']}|{v['msg'][:100]}"


def check(doc: str | Path) -> tuple[bool, list[dict]]:
    """Gate the ledger's shape and order. Violations: a bold-dated line that
    does not parse as a header (impossible date, missing em-dash separator,
    bold that never closes), and a date that increases top-to-bottom (the
    bottom-append / backdated-insert signature — review, then either move the
    entry to line 1 or accept it). Returns ``(ok, violations)``; violations
    carry ``line`` plus stable identity fields for :func:`fingerprint`."""
    entries, malformed = _parse_blocks(Path(doc))
    violations = list(malformed)
    prev: dict | None = None
    for e in entries:
        if prev is not None and e["date"] > prev["date"]:
            violations.append(
                {
                    "line": e["line"],
                    "kind": "order",
                    "date": e["date"],
                    "head": e["head"],
                    "msg": (
                        f"{e['date']} is newer than the entry above "
                        f"({prev['date']}) — review: bottom-append or "
                        "backdated insert"
                    ),
                }
            )
        prev = e
    violations.sort(key=lambda v: v["line"])
    return (not violations, violations)


def query(
    doc: str | Path,
    *,
    card: str | None = None,
    sha: str | None = None,
    lane: str | None = None,
    since: str | None = None,
    grep: str | None = None,
) -> list[dict]:
    """Entries matching ALL provided filters (substring, case-insensitive
    where the filter is prose). ``card``/``sha`` match short or long forms."""
    entries, _ = _parse_blocks(Path(doc))
    hits: list[dict] = []
    for e in entries:
        if card and not any(card in c for c in e["cards"]):
            continue
        if sha and not any(sha in s for s in e["shas"]):
            continue
        if lane and lane.casefold() not in e["lane"].casefold():
            continue
        if since and e["date"] < since:
            continue
        if grep and grep.casefold() not in e["text"].casefold():
            continue
        hits.append(e)
    return _project(hits)


def dumps(entries: list[dict]) -> str:
    """JSON projection of an index/query result (stable, indented)."""
    return json.dumps(entries, indent=2, ensure_ascii=False)
