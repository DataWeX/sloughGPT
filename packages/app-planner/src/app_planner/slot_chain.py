"""Kanban slot chain — per-slot parent-linked hashes (never card content).

Every card slot (``column:position``) carries its own append-only hash lineage:

    node = sha256(slot_prev || "\\n" || column:position || "\\n" || notes_hash)

* ``slot_prev``  = this slot's tip before the node (previous occupant's hash,
  ``''`` for the first node) — a reused slot can therefore never hash to the
  same value twice; the removed occupant's hash is the parent of the next one.
* ``notes_hash`` = sha256 over the card's NOTE material only (linked dev-note
  frontmatter ``hash:`` values + inline ``notes[]`` texts). Card body
  (title/description/priority/...) never enters the hash. The individual
  digests are stored on the card as ``note_hashes``.
* Every node is appended to ``slot_history.jsonl`` next to the board file —
  a removed occupant stays there as the root of its own branch (append-only,
  never recomputed).
* The board header carries ``tray_roots`` = sha256 over each tray's slot tips
  (tips derived from the journal; the journal is the source of truth).

Write path: exactly **one line** — ``PlannerStore._atomic_write`` calls
:func:`apply_board` after every board write, so add/update/move/delete/sync
re-derive the chain in the same call and slot fields are never lost (the
``Card`` dataclass stays slot-unaware; the stripped line is healed instantly).
The manual path (``scripts/kanban_slot_chain.py apply``) remains for direct
migration; ``verify_board`` is the drift gate. TS writers do not emit chains.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from .atomic import append_lines, atomic_write_text

RETIRED_FIELDS = ("chain_hash", "chain_prev", "chain_index")
SLOT_FIELDS = ("slot_position", "note_hashes", "notes_hash", "slot_prev", "slot_hash")


class SlotChainError(Exception):
    """Board cannot be chained yet (no schema header / missing meta record)."""


def sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def node_hash(prev: str, slot_key: str, notes_hash: str) -> str:
    """The only hash formula in this system: slot + notes, never card body."""
    return sha256(f"{prev}\n{slot_key}\n{notes_hash}")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ── Board file I/O (raw-preserving) ──────────────────────────────────────


def load_board(path: str | Path):
    """Parse the board JSONL.

    Returns ``(raws, objs, cards, meta_idx)`` where ``raws`` holds the exact
    original line texts (written back byte-for-byte when a line is unchanged),
    ``objs`` maps index -> parsed dict (missing for blank/malformed lines),
    ``cards`` is ``[(idx, card_dict), ...]`` and ``meta_idx`` is the header
    index (raises :class:`SlotChainError` when absent).
    """
    raws: list[str] = []
    objs: dict[int, dict] = {}
    cards: list[tuple[int, dict]] = []
    meta_idx = None
    text = Path(path).read_text(encoding="utf-8")
    for i, raw in enumerate(text.splitlines()):
        stripped = raw.strip()
        if not stripped:
            raws.append(raw)
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            raws.append(raw)  # malformed passthrough, untouched
            continue
        raws.append(raw)
        if isinstance(obj, dict):
            objs[i] = obj
            if "columns" in obj and "title" not in obj:
                meta_idx = i
            elif "title" in obj or "id" in obj:
                cards.append((i, obj))
    if meta_idx is None:
        raise SlotChainError(f"{path}: no board-meta record (with 'columns')")
    return raws, objs, cards, meta_idx


def write_board(path: str | Path, raws: list[str], objs: dict[int, dict], dirty: set[int]) -> None:
    """Rewrite only dirty lines (formatted); unchanged lines keep their bytes."""
    out = []
    for i, raw in enumerate(raws):
        if i in dirty and i in objs:
            out.append(json.dumps(objs[i], ensure_ascii=False))
        else:
            out.append(raw)
    atomic_write_text(path, "\n".join(out) + "\n")

# ── Inputs ───────────────────────────────────────────────────────────────


def canonical_positions(cards: list[tuple[int, dict]], meta_columns: list[dict]) -> dict[int, int]:
    """Column order: meta columns, then unknown non-empty columns A-Z, '' last.
    Position = 0-based index within the column in file order.

    Returns a map of *object id* of the card dict -> position.
    """
    known = [c.get("name") for c in meta_columns]
    present = {c.get("column", "") for _, c in cards}
    order = [c for c in known if c in present]
    order += sorted(c for c in present if c and c not in known)
    if "" in present:
        order.append("")
    positions: dict[int, int] = {}
    for col in order:
        idx = 0
        for _, card in cards:
            if card.get("column", "") == col:
                positions[id(card)] = idx
                idx += 1
    return positions


def dev_note_hashes(notes_home: str | Path | None = None) -> dict[str, list[str]]:
    """card_id -> sorted dev-note frontmatter hashes linked to it."""
    home = Path(notes_home) if notes_home else Path(os.path.expanduser("~/.config/dev-notes"))
    out: dict[str, list[str]] = {}
    if not home.is_dir():
        return out
    for path in sorted(glob.glob(str(home / "*.md"))):
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            continue
        m_id = re.search(r"^board(?:_id)?:\s*[\"']?(\S+?)[\"']?\s*$", text, re.M)
        m_hash = re.search(r"^hash:\s*([0-9a-f]{64})", text, re.M)
        if m_id and m_hash:
            out.setdefault(m_id.group(1), []).append(m_hash.group(1))
    for hashes in out.values():
        hashes.sort()
    return out


def note_hash_inputs(card: dict, dev: dict[str, list[str]]) -> tuple[list[str], str]:
    """Compute the card's note digests and their aggregate.

    Returns ``(inputs, notes_hash)`` where ``inputs`` = linked dev-note
    frontmatter hashes (sorted) followed by inline ``notes[]`` digests —
    stored on the card as ``note_hashes`` — and ``notes_hash`` is the legacy
    stable aggregate: sha256 over ``dev:<h>`` / ``inline:<i>:<digest>``
    parts ('' when the card has no notes). Prefixes keep dev/inline domains
    disjoint and preserve byte-parity with the existing journal.
    """
    dev_list = list(dev.get(card.get("id", ""), []))
    inline_list = [sha256(str(note)) for note in card.get("notes") or []]
    inputs = dev_list + inline_list
    if not inputs:
        return inputs, ""
    parts = [f"dev:{h}" for h in dev_list]
    parts += [f"inline:{i}:{digest}" for i, digest in enumerate(inline_list)]
    return inputs, sha256("\n".join(parts))


# ── Journal ──────────────────────────────────────────────────────────────


def journal_path(board_path: str | Path) -> Path:
    return Path(board_path).resolve().parent / "slot_history.jsonl"


def load_journal(path: str | Path) -> tuple[list[dict], dict[str, dict]]:
    """Return (nodes, tips) — tips[slot_key] = last node's dict."""
    nodes: list[dict] = []
    tips: dict[str, dict] = {}
    p = Path(path)
    if p.exists():
        for raw in p.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if raw:
                node = json.loads(raw)
                nodes.append(node)
                tips[node["slot_key"]] = node
    return nodes, tips


def compute_tray_roots(tips: dict[str, dict]) -> dict[str, str]:
    trays: dict[str, list[tuple[int, str, str]]] = {}
    for key, node in tips.items():
        col, _, pos = key.rpartition(":")
        trays.setdefault(col, []).append((int(pos), key, node["slot_hash"]))
    roots: dict[str, str] = {}
    for col, items in trays.items():
        items.sort()
        roots[col] = sha256("\n".join(f"{k}:{h}" for _, k, h in items))
    return roots


# ── Core operations ──────────────────────────────────────────────────────


def apply_board(board_path: str | Path, notes_home: str | Path | None = None) -> dict:
    """Re-derive all slot-chain metadata. Idempotent; appends to the journal.

    Quiet no-op (zero stats) when the board has no schema header — legacy
    headerless boards keep working; ``verify_board`` still flags them.

    Returns stats: ``{"cards", "nodes_created", "nodes_reused", "tips", "trays"}``.
    """
    zeros = {"cards": 0, "nodes_created": 0, "nodes_reused": 0, "tips": 0, "trays": 0}
    try:
        raws, objs, cards, meta_idx = load_board(board_path)
    except SlotChainError:
        return zeros
    meta = objs[meta_idx]
    hist = journal_path(board_path)
    _, tips = load_journal(hist)
    dev = dev_note_hashes(notes_home)
    positions = canonical_positions(cards, meta["columns"])

    dirty: set[int] = set()
    created = reused = 0
    new_nodes: list[dict] = []
    for idx, card in cards:
        pos = positions[id(card)]
        slot_key = f"{card.get('column', '')}:{pos}"
        inputs, nh = note_hash_inputs(card, dev)
        last = tips.get(slot_key)
        if last and last["card_id"] == card.get("id") and last["notes_hash"] == nh:
            node = last  # current state already journaled — idempotent
            reused += 1
        else:
            prev = last["slot_hash"] if last else ""
            node = {
                "slot_key": slot_key,
                "card_id": card.get("id", ""),
                "notes_hash": nh,
                "slot_prev": prev,
                "slot_hash": node_hash(prev, slot_key, nh),
                "ts": now(),
            }
            new_nodes.append(node)
            tips[slot_key] = node
            created += 1
        values = {
            "slot_position": pos,
            "note_hashes": inputs,
            "notes_hash": nh,
            "slot_prev": node["slot_prev"],
            "slot_hash": node["slot_hash"],
        }
        for key, value in values.items():
            if card.get(key) != value:
                card[key] = value
                dirty.add(idx)
        for retired in RETIRED_FIELDS:
            if retired in card:
                del card[retired]
                dirty.add(idx)

    roots = compute_tray_roots(tips)
    if meta.get("tray_roots") != roots:
        meta["tray_roots"] = roots
        dirty.add(meta_idx)
    hist_name = journal_path(board_path).name
    if meta.get("slot_history") != hist_name:
        meta["slot_history"] = hist_name
        dirty.add(meta_idx)

    if new_nodes:
        append_lines(
            hist,
            [json.dumps(node, ensure_ascii=False) + "\n" for node in new_nodes],
        )
    if dirty:
        write_board(board_path, raws, objs, dirty)
    return {
        "cards": len(cards),
        "nodes_created": created,
        "nodes_reused": reused,
        "tips": len(tips),
        "trays": len(roots),
    }


def verify_board(board_path: str | Path, notes_home: str | Path | None = None) -> list[str]:
    """Replay + re-derive everything; returns a list of problems (empty = ok)."""
    failures: list[str] = []
    try:
        raws, objs, cards, meta_idx = load_board(board_path)
    except SlotChainError as exc:
        return [str(exc)]
    meta = objs[meta_idx]
    nodes, tips = load_journal(journal_path(board_path))

    # 1. journal self-consistency: every node rehashes, chains link up
    per_key: dict[str, list[dict]] = {}
    for node in nodes:
        if node_hash(node["slot_prev"], node["slot_key"], node["notes_hash"]) != node["slot_hash"]:
            failures.append(f"journal: node {node['slot_key']}@{node['ts']} rehash mismatch")
        per_key.setdefault(node["slot_key"], []).append(node)
    for key, chain in per_key.items():
        if chain[0]["slot_prev"] != "":
            failures.append(f"journal: {key} first node has non-empty parent")
        for prev, node in zip(chain, chain[1:]):
            if node["slot_prev"] != prev["slot_hash"]:
                failures.append(f"journal: {key} chain broken at {node['ts']}")

    # 2. cards match the journal tips (their slot's current state)
    dev = dev_note_hashes(notes_home)
    positions = canonical_positions(cards, meta["columns"])
    for _, card in cards:
        cid = card.get("id", "?")
        pos = positions[id(card)]
        slot_key = f"{card.get('column', '')}:{pos}"
        if not all(k in card for k in SLOT_FIELDS):
            failures.append(f"{cid}: missing slot fields (run apply)")
            continue
        tip = tips.get(slot_key)
        if tip is None:
            failures.append(f"{cid}: slot {slot_key} never journaled")
            continue
        if tip["card_id"] != card.get("id"):
            failures.append(f"{cid}: occupies {slot_key} but tip belongs to {tip['card_id']}")
            continue
        if tip["notes_hash"] != card.get("notes_hash"):
            failures.append(f"{cid}: notes changed since last apply (run apply)")
            continue
        if card.get("slot_position") != pos:
            failures.append(f"{cid}: slot_position stale (run apply)")
        if card.get("slot_prev") != tip["slot_prev"] or card.get("slot_hash") != tip["slot_hash"]:
            failures.append(f"{cid}: slot chain fields stale (run apply)")
        if card.get("note_hashes") != note_hash_inputs(card, dev)[0]:
            failures.append(f"{cid}: note_hashes out of sync with linked notes")
        for retired in RETIRED_FIELDS:
            if retired in card:
                failures.append(f"{cid}: retired field {retired} present")

    # 3. header aggregates match journal-derived tips
    expect_roots = compute_tray_roots(tips)
    if meta.get("tray_roots") != expect_roots:
        failures.append("header tray_roots != journal-derived roots")
    if meta.get("slot_history") != journal_path(board_path).name:
        failures.append("header slot_history missing/wrong")
    return failures
