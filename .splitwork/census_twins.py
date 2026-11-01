"""Read-only: compare duplicate-title journal records by BODY CONTENT.

Card 08baf13f + user direction 2026-10-11: collapse a duplicate group only
when the records are genuinely redundant. Same-id/same-body => safe to
collapse. Differing bodies => real conflict, must be folded and flagged, never
silently dropped.

Prints an explicit verdict per group. Touches nothing.
"""

from __future__ import annotations

import collections
import difflib
import json
from pathlib import Path

JOURNAL = Path(".dev-notes/notes.journal.jsonl")


def load() -> list[tuple[int, dict]]:
    out = []
    for i, line in enumerate(JOURNAL.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            out.append((i, json.loads(line)))
        except json.JSONDecodeError:
            continue
    return out


def main() -> None:
    recs = load()
    by_title: dict[str, list[tuple[int, dict]]] = collections.defaultdict(list)
    for i, d in recs:
        by_title[d.get("title") or ""].append((i, d))

    for title, group in by_title.items():
        if len(group) < 2:
            continue
        print("=" * 78)
        print(f"GROUP {title!r} x{len(group)}")

        ids = {d.get("id") for _, d in group}
        print(f"  unique ids      : {len(ids)} -> {sorted(i[:8] for i in ids)}")
        print(f"  same-id group   : {len(ids) == 1}")

        bodies = [(d.get("body") or "") for _, d in group]
        uniq_bodies = {b for b in bodies}
        print(f"  unique bodies   : {len(uniq_bodies)}")

        if len(uniq_bodies) == 1:
            print("  VERDICT         : REDUNDANT - bodies identical, safe to collapse to one")
            continue

        print("  VERDICT         : CONFLICT - bodies differ, must FOLD not drop")
        # show what actually differs, longest pair first
        ranked = sorted(group, key=lambda t: len(t[1].get("body") or ""), reverse=True)
        big = ranked[0][1].get("body") or ""
        for _, other in ranked[1:]:
            ob = other.get("body") or ""
            diff = list(
                difflib.unified_diff(
                    big.splitlines(), ob.splitlines(), "keep", "twin", lineterm="", n=1
                )
            )
            print(f"  --- diff vs {other.get('id', '')[:8]} ({len(diff)} lines) ---")
            for ln in diff[:24]:
                print(f"      {ln}")
        # is one a superset of the other?
        short = ranked[-1][1].get("body") or ""
        if short.strip() and all(s in big for s in short.splitlines()):
            print("  SUPERSET        : shorter body is contained in the longer -> fold is lossless")
        else:
            print("  SUPERSET        : NO - shorter body has unique lines; fold must merge both")


if __name__ == "__main__":
    main()
