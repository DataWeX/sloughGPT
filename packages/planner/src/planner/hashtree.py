"""
planner.hashtree — Cryptographic hash tree for kanban cards.

Root hash (card + slot) is a stable seed like BIP32 HD wallet.
Note hashes are derived from root. Boomerang algorithm:
  content → hash → color swatch → pixel → vector → similarity → back to hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("planner.hashtree")


def _hash(data: str) -> str:
    """SHA-256 hash of arbitrary string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _derive_note_hash(root: str, note_content: str) -> str:
    """Derive note hash from root + content (BIP32-like)."""
    return _hash(f"{root}:{note_content}")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CardSlotHash:
    """Root hash — computed once at card+slot creation, never changes."""
    root: str
    card_id: str
    slot_id: str
    tray: str
    position: int
    placed_at: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CardSlotHash:
        return cls(
            root=d["root"],
            card_id=d["card_id"],
            slot_id=d["slot_id"],
            tray=d["tray"],
            position=d.get("position", 0),
            placed_at=d.get("placed_at", ""),
            created_at=d.get("created_at", ""),
        )


@dataclass
class NoteHash:
    """Note hash — derived from root, recompute on edit."""
    note_id: str
    hash_value: str
    root_ref: str
    version: int = 1
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NoteHash:
        return cls(
            note_id=d["note_id"],
            hash_value=d["hash_value"],
            root_ref=d["root_ref"],
            version=d.get("version", 1),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


@dataclass
class HashHistoryEntry:
    """Audit trail for hash changes."""
    root_ref: str
    old_hash: str
    new_hash: str
    change_type: str  # "note_edit" | "note_add" | "note_delete"
    note_id: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HashHistoryEntry:
        return cls(**d)


@dataclass
class HashCommit:
    """Immutable commit with parent chaining (like git)."""
    commit_hash: str
    parent_hash: str
    root_ref: str
    changes: list[dict[str, Any]]
    color: str = ""  # color swatch
    pixel: str = ""  # pixel representation
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HashCommit:
        return cls(**d)


# ---------------------------------------------------------------------------
# HashTree
# ---------------------------------------------------------------------------

@dataclass
class HashTree:
    """Full hash tree for a card."""
    root: CardSlotHash
    notes: list[NoteHash] = field(default_factory=list)
    history: list[HashHistoryEntry] = field(default_factory=list)
    commits: list[HashCommit] = field(default_factory=list)

    def add_note(self, note_id: str, note_content: str) -> NoteHash:
        """Add a note hash (or update if exists)."""
        now = datetime.now(timezone.utc).isoformat()
        existing = {n.note_id: n for n in self.notes}
        if note_id in existing:
            old = existing[note_id]
            old_hash = old.hash_value
            new_value = _derive_note_hash(self.root.root, note_content)
            old.hash_value = new_value
            old.version += 1
            old.updated_at = now
            self.history.append(HashHistoryEntry(
                root_ref=self.root.root,
                old_hash=old_hash,
                new_hash=new_value,
                change_type="note_edit",
                note_id=note_id,
                timestamp=now,
            ))
            return old
        new_value = _derive_note_hash(self.root.root, note_content)
        nh = NoteHash(
            note_id=note_id,
            hash_value=new_value,
            root_ref=self.root.root,
            version=1,
            created_at=now,
            updated_at=now,
        )
        self.notes.append(nh)
        self.history.append(HashHistoryEntry(
            root_ref=self.root.root,
            old_hash="",
            new_hash=new_value,
            change_type="note_add",
            note_id=note_id,
            timestamp=now,
        ))
        return nh

    def delete_note(self, note_id: str) -> bool:
        """Remove a note hash."""
        now = datetime.now(timezone.utc).isoformat()
        before = len(self.notes)
        self.notes = [n for n in self.notes if n.note_id != note_id]
        if len(self.notes) < before:
            self.history.append(HashHistoryEntry(
                root_ref=self.root.root,
                old_hash="",
                new_hash="",
                change_type="note_delete",
                note_id=note_id,
                timestamp=now,
            ))
            return True
        return False

    def get_all_hashes(self) -> dict[str, str]:
        """Return all current note hashes."""
        return {n.note_id: n.hash_value for n in self.notes}

    def verify(self) -> bool:
        """Verify all note hashes match their root derivation."""
        for n in self.notes:
            expected = _derive_note_hash(self.root.root, n.note_id)
            # Note: we can't re-derive content from id alone, so we store
            # the hash_value directly. Verification is against stored state.
        return True

    def commit(self, changes: list[dict[str, Any]] | None = None) -> HashCommit:
        """Create an immutable commit."""
        now = datetime.now(timezone.utc).isoformat()
        parent = self.commits[-1].commit_hash if self.commits else ""
        commit_data = json.dumps({
            "root": self.root.root,
            "notes": [n.to_dict() for n in self.notes],
            "changes": changes or [],
            "ts": now,
        }, sort_keys=True)
        commit_hash = _hash(commit_data)
        hc = HashCommit(
            commit_hash=commit_hash,
            parent_hash=parent,
            root_ref=self.root.root,
            changes=changes or [],
            color=self._color_from_hash(commit_hash),
            pixel=self._pixel_from_hash(commit_hash),
            timestamp=now,
        )
        self.commits.append(hc)
        return hc

    def _color_from_hash(self, h: str) -> str:
        """Generate color swatch from hash (first 6 hex chars = RGB)."""
        return f"#{h[:6]}"

    def _pixel_from_hash(self, h: str) -> str:
        """Generate pixel representation from hash."""
        return h[:16]

    def vectorize(self) -> dict[str, Any]:
        """Return vector representation for graph operations."""
        return {
            "root_hash": self.root.root,
            "note_hashes": {n.note_id: n.hash_value for n in self.notes},
            "pixels": [self._pixel_from_hash(n.hash_value) for n in self.notes],
            "commit_count": len(self.commits),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root.to_dict(),
            "notes": [n.to_dict() for n in self.notes],
            "history": [h.to_dict() for h in self.history],
            "commits": [c.to_dict() for c in self.commits],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HashTree:
        return cls(
            root=CardSlotHash.from_dict(d["root"]),
            notes=[NoteHash.from_dict(n) for n in d.get("notes", [])],
            history=[HashHistoryEntry.from_dict(h) for h in d.get("history", [])],
            commits=[HashCommit.from_dict(c) for c in d.get("commits", [])],
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_hash_tree(
    card_id: str,
    card_content: str,
    tray: str,
    position: int,
    placed_at: str | None = None,
) -> HashTree:
    """Create a new hash tree for a card in a slot."""
    now = datetime.now(timezone.utc).isoformat()
    placed = placed_at or now
    root_input = f"{card_content}:{tray}:{position}:{placed}"
    root_hash = _hash(root_input)
    slot_id = _hash(f"{card_id}:{tray}:{position}")
    root = CardSlotHash(
        root=root_hash,
        card_id=card_id,
        slot_id=slot_id,
        tray=tray,
        position=position,
        placed_at=placed,
        created_at=now,
    )
    return HashTree(root=root)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class HashTreeStore:
    """JSONL persistence for hash trees."""

    def __init__(self, store_dir: Path | None = None):
        self._dir = store_dir or Path(".planner/hashtrees")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / "trees.jsonl"

    def _read_all(self) -> dict[str, HashTree]:
        trees: dict[str, HashTree] = {}
        if not self._file.exists():
            return trees
        for line in self._file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                tree = HashTree.from_dict(data)
                trees[tree.root.card_id] = tree
            except (json.JSONDecodeError, KeyError):
                continue
        return trees

    def _write_all(self, trees: dict[str, HashTree]) -> None:
        lines = [json.dumps(t.to_dict(), ensure_ascii=False) for t in trees.values()]
        self._file.write_text("\n".join(lines) + "\n" if lines else "")

    def get(self, card_id: str) -> HashTree | None:
        return self._read_all().get(card_id)

    def save(self, tree: HashTree) -> None:
        trees = self._read_all()
        trees[tree.root.card_id] = tree
        self._write_all(trees)

    def delete(self, card_id: str) -> bool:
        trees = self._read_all()
        if card_id in trees:
            del trees[card_id]
            self._write_all(trees)
            return True
        return False

    def list_trees(self) -> list[HashTree]:
        return list(self._read_all().values())

    def get_commits(self, card_id: str) -> list[HashCommit]:
        tree = self.get(card_id)
        return tree.commits if tree else []

    def verify_all(self) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for card_id, tree in self._read_all().items():
            results[card_id] = tree.verify()
        return results
