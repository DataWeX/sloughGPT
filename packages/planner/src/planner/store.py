"""
planner.store — JSONL board operations.

Manages the kanban board stored as JSONL in `.kanban/board.jsonl`.
Each line is a JSON object representing a card with fields:
  id, title, description, column, priority, tags, assignee, dueDate, createdAt, updatedAt
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config


@dataclass
class Card:
    """A single kanban card."""
    id: str = ""
    title: str = ""
    description: str = ""
    column: str = "todo"
    priority: str = "medium"
    tags: list[str] = field(default_factory=list)
    assignee: str = ""
    dueDate: str = ""
    createdAt: str = ""
    updatedAt: str = ""
    root_hash: str = ""
    notes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Card:
        return cls(
            id=d.get("id", ""),
            title=d.get("title", ""),
            description=d.get("description", ""),
            column=d.get("column", "todo"),
            priority=d.get("priority", "medium"),
            tags=d.get("tags", []),
            assignee=d.get("assignee", ""),
            dueDate=d.get("dueDate", ""),
            createdAt=d.get("createdAt", ""),
            updatedAt=d.get("updatedAt", ""),
            root_hash=d.get("root_hash", ""),
            notes=d.get("notes", []),
        )


@dataclass
class Board:
    """The kanban board state."""
    cards: list[Card] = field(default_factory=list)
    columns: list[str] = field(default_factory=lambda: ["todo", "in_progress", "review", "done"])


class Store:
    """JSONL-backed board store."""

    def __init__(self, board_dir: Path | None = None):
        self._dir = Path(board_dir) if board_dir else config.default_board_dir()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._board_file = self._dir / "board.jsonl"

    def _read_cards(self) -> list[Card]:
        if not self._board_file.exists():
            return []
        cards = []
        for line in self._board_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                cards.append(Card.from_dict(data))
            except json.JSONDecodeError:
                continue
        return cards

    def _write_cards(self, cards: list[Card]) -> None:
        lines = [json.dumps(card.to_dict(), ensure_ascii=False) for card in cards]
        self._board_file.write_text("\n".join(lines) + "\n" if lines else "")

    def load_board(self) -> Board:
        return Board(cards=self._read_cards())

    def save_board(self, board: Board) -> None:
        self._write_cards(board.cards)

    def get_card(self, card_id: str) -> Card | None:
        for card in self._read_cards():
            if card.id == card_id:
                return card
        return None

    def create_card(self, title: str, column: str = "todo", priority: str = "medium",
                    description: str = "", tags: list[str] | None = None,
                    assignee: str = "", dueDate: str = "") -> Card:
        now = datetime.now(timezone.utc).isoformat()
        card = Card(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            column=column,
            priority=priority,
            tags=tags or [],
            assignee=assignee,
            dueDate=dueDate,
            createdAt=now,
            updatedAt=now,
        )
        cards = self._read_cards()
        cards.append(card)
        self._write_cards(cards)
        return card

    def update_card(self, card_id: str, **kwargs: Any) -> Card | None:
        cards = self._read_cards()
        for i, card in enumerate(cards):
            if card.id == card_id:
                for key, value in kwargs.items():
                    if hasattr(card, key):
                        setattr(card, key, value)
                card.updatedAt = datetime.now(timezone.utc).isoformat()
                cards[i] = card
                self._write_cards(cards)
                return card
        return None

    def delete_card(self, card_id: str) -> bool:
        cards = self._read_cards()
        new_cards = [c for c in cards if c.id != card_id]
        if len(new_cards) < len(cards):
            self._write_cards(new_cards)
            return True
        return False

    def move_card(self, card_id: str, to_column: str, to_index: int | None = None) -> bool:
        cards = self._read_cards()
        card = None
        for c in cards:
            if c.id == card_id:
                card = c
                break
        if card is None:
            return False
        cards = [c for c in cards if c.id != card_id]
        card.column = to_column
        card.updatedAt = datetime.now(timezone.utc).isoformat()
        if to_index is not None and 0 <= to_index <= len(cards):
            cards.insert(to_index, card)
        else:
            cards.append(card)
        self._write_cards(cards)
        return True

    def get_cards_by_column(self, column: str) -> list[Card]:
        return [c for c in self._read_cards() if c.column == column]

    def get_all_tags(self) -> dict[str, int]:
        tag_counts: dict[str, int] = {}
        for card in self._read_cards():
            for tag in card.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return tag_counts

    def get_stats(self) -> dict[str, Any]:
        cards = self._read_cards()
        by_column: dict[str, int] = {}
        by_priority: dict[str, int] = {}
        for card in cards:
            by_column[card.column] = by_column.get(card.column, 0) + 1
            by_priority[card.priority] = by_priority.get(card.priority, 0) + 1
        return {
            "total": len(cards),
            "byColumn": by_column,
            "byPriority": by_priority,
        }


_store: Store | None = None


def get_store(board_dir: Path | None = None) -> Store:
    global _store
    if _store is None:
        _store = Store(board_dir=board_dir)
    return _store


def reset_store() -> None:
    global _store
    _store = None
