"""Loadable memory cards — save / list / load / delete long-term memory.

A card is a portable, checksummed JSON snapshot of stored memory facts: the
user's "memory card like a real memory card that can be loaded so the model
remembers". Cards live in ``data/memory_cards/<name>.card.json`` (override
with ``SLO_MEMORY_CARDS_DIR``) and can be moved between installs.

Format ``slo-memory-card/v1``::

    {
      "format": "slo-memory-card/v1",
      "name": "my-card",
      "created_at": 1790700000.0,
      "facts_count": 42,
      "checksum": "sha256:...",   # over the canonicalized facts payload
      "facts": [{"content", "topic", "source", "url",
                 "timestamp", "importance", "workspace_id"}]
    }

Invariants:
- only whitelisted, portable fields are stored (no vector-store ids/scores)
- the checksum is verified BEFORE callers may touch the store — a corrupt or
  truncated card can never destroy memory
- card names are restricted to ``[A-Za-z0-9._-]`` (no separators, no ``..``)
- writes are atomic (tmp file + ``os.replace``)

Every failure raises a :class:`MemoryCardError` subclass with a user-facing
message; callers map subclasses to ``error_code``s via ``MemoryCardError``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CARD_FORMAT = "slo-memory-card/v1"
DEFAULT_CARDS_DIR = "data/memory_cards"
CARDS_DIR_ENV = "SLO_MEMORY_CARDS_DIR"

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SAFE_FIELDS = (
    "content",
    "topic",
    "source",
    "url",
    "timestamp",
    "importance",
    "workspace_id",
)


class MemoryCardError(Exception):
    """Base for memory-card failures; ``error_code`` drives HTTP mapping."""

    error_code = "card_error"


class CardNotFound(MemoryCardError):
    """Named card does not exist in the cards directory."""

    error_code = "not_found"


class CardCorrupt(MemoryCardError):
    """Card file is unreadable, wrong format, or fails its checksum."""

    error_code = "corrupt"


class CardExists(MemoryCardError):
    """Card name already present and overwrite was not requested."""

    error_code = "exists"


class CardInvalidName(MemoryCardError):
    """Card name contains separators/``..`` or is otherwise unusable."""

    error_code = "invalid_name"


@dataclass(frozen=True)
class CardInfo:
    """Card metadata as exposed by :func:`list_cards`."""

    name: str
    created_at: float
    facts_count: int
    size_bytes: int
    valid: bool  # checksum matches (False = present but corrupt)


def resolve_cards_dir(directory: str | Path | None = None) -> Path:
    """Resolve the cards directory: arg > ``SLO_MEMORY_CARDS_DIR`` > default."""
    return Path(directory or os.environ.get(CARDS_DIR_ENV) or DEFAULT_CARDS_DIR)


def validate_name(name: str) -> str:
    """Return ``name`` if safe as a filename; raise :class:`CardInvalidName`."""
    if not isinstance(name, str) or not _NAME_RE.match(name) or ".." in name:
        raise CardInvalidName(f"invalid card name: {name!r}")
    return name


def _card_path(name: str, directory: Path) -> Path:
    return directory / f"{validate_name(name)}.card.json"


def _normalize_fact(fact: dict[str, Any]) -> dict[str, Any]:
    """Whitelist portable fields and coerce types (raises on missing content)."""
    content = str(fact.get("content") or "").strip()
    if not content:
        raise CardCorrupt("card contains a fact without content")
    out: dict[str, Any] = {"content": content}
    out["topic"] = str(fact.get("topic") or "general")
    out["source"] = str(fact.get("source") or "")
    out["url"] = str(fact.get("url") or "")
    try:
        ts = fact.get("timestamp")
        imp = fact.get("importance")
        out["timestamp"] = float(ts) if ts is not None else 0.0
        out["importance"] = float(imp) if imp is not None else 0.5
    except (TypeError, ValueError) as e:
        raise CardCorrupt(f"fact has non-numeric fields: {e}") from e
    out["workspace_id"] = str(fact.get("workspace_id") or "")
    return out


def _checksum(facts: list[dict[str, Any]]) -> str:
    payload = json.dumps(facts, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_card(
    facts: list[dict[str, Any]],
    name: str | None = None,
    *,
    cards_dir: str | Path | None = None,
    overwrite: bool = False,
) -> CardInfo:
    """Write ``facts`` to a named (or auto-named) card file.

    Args:
        facts: portable fact dicts (``list_all``-shaped); empty-content facts
            are dropped, fields outside the whitelist are ignored.
        name: card name; ``None`` auto-generates ``card-<timestamp>``.
        cards_dir: override directory (tests/temporary saves).
        overwrite: allow replacing an existing card.

    Returns:
        CardInfo for the written card.

    Raises:
        CardInvalidName: name is unsafe.
        CardExists: name present and ``overwrite`` is False.
        MemoryCardError: write failed (no partial file is left behind).
    """
    directory = resolve_cards_dir(cards_dir)
    clean = [_normalize_fact(f) for f in facts]
    clean = [f for f in clean if f["content"]]

    if name is None:
        name = (
            time.strftime("card-%Y%m%d-%H%M%S-", time.localtime())
            + f"{time.time_ns() % 1_000_000:06d}"
        )
    path = _card_path(name, directory)

    if path.exists() and not overwrite:
        raise CardExists(f"card already exists: {name}")

    now = time.time()
    card = {
        "format": CARD_FORMAT,
        "name": name,
        "created_at": now,
        "facts_count": len(clean),
        "checksum": _checksum(clean),
        "facts": clean,
    }
    try:
        directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".card.json.tmp")
        tmp.write_text(json.dumps(card, ensure_ascii=False, indent=None))
        os.replace(tmp, path)
    except MemoryCardError:
        raise
    except Exception as e:
        raise MemoryCardError(f"failed to write card {name}: {e}") from e

    size = path.stat().st_size
    logger.info("memory card saved: %s (%d facts, %d bytes)", name, len(clean), size)
    return CardInfo(name=name, created_at=now, facts_count=len(clean), size_bytes=size, valid=True)


def load_card(name: str, *, cards_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Read and verify a card, returning its normalized facts.

    The checksum and format are validated here — callers MUST NOT mutate the
    store until this returns successfully.

    Raises:
        CardNotFound: no such card.
        CardCorrupt: unreadable/foreign/tampered card.
        CardInvalidName: unsafe name.
    """
    directory = resolve_cards_dir(cards_dir)
    path = _card_path(name, directory)
    if not path.exists():
        raise CardNotFound(f"card not found: {name}")
    try:
        card = json.loads(path.read_text())
    except Exception as e:
        raise CardCorrupt(f"card {name} is not valid JSON: {e}") from e
    if not isinstance(card, dict) or card.get("format") != CARD_FORMAT:
        raise CardCorrupt(
            f"card {name} has unknown format: {card.get('format') if isinstance(card, dict) else type(card).__name__}"
        )
    facts = card.get("facts")
    if not isinstance(facts, list):
        raise CardCorrupt(f"card {name} has no facts list")
    normalized = [_normalize_fact(f) for f in facts]
    if card.get("checksum") != _checksum(normalized):
        raise CardCorrupt(f"card {name} failed checksum verification")
    return normalized


def list_cards(*, cards_dir: str | Path | None = None) -> list[CardInfo]:
    """List cards, newest first. Corrupt cards are reported, never raised."""
    directory = resolve_cards_dir(cards_dir)
    infos: list[CardInfo] = []
    if not directory.exists():
        return infos
    for path in directory.glob("*.card.json"):
        name = path.name[: -len(".card.json")]
        try:
            card = json.loads(path.read_text())
            facts = card.get("facts")
            normalized = [_normalize_fact(f) for f in facts] if isinstance(facts, list) else []
            valid = (
                isinstance(card, dict)
                and card.get("format") == CARD_FORMAT
                and card.get("checksum") == _checksum(normalized)
            )
            infos.append(
                CardInfo(
                    name=str(card.get("name") or name),
                    created_at=float(card.get("created_at") or 0.0),
                    facts_count=len(normalized),
                    size_bytes=path.stat().st_size,
                    valid=bool(valid),
                )
            )
        except Exception:
            infos.append(
                CardInfo(
                    name=name,
                    created_at=0.0,
                    facts_count=0,
                    size_bytes=path.stat().st_size,
                    valid=False,
                )
            )
    infos.sort(key=lambda c: (-c.created_at, c.name))
    return infos


def delete_card(name: str, *, cards_dir: str | Path | None = None) -> bool:
    """Delete a card file; ``False`` when it did not exist."""
    directory = resolve_cards_dir(cards_dir)
    path = _card_path(name, directory)
    if not path.exists():
        return False
    path.unlink()
    logger.info("memory card deleted: %s", name)
    return True


def card_exists(name: str, *, cards_dir: str | Path | None = None) -> bool:
    """Whether a card file exists for ``name`` (name validity enforced)."""
    return _card_path(name, resolve_cards_dir(cards_dir)).exists()


def _info_dict(info: CardInfo) -> dict[str, Any]:
    return asdict(info)


__all__ = [
    "CARD_FORMAT",
    "CARDS_DIR_ENV",
    "DEFAULT_CARDS_DIR",
    "CardCorrupt",
    "CardExists",
    "CardInfo",
    "CardInvalidName",
    "CardNotFound",
    "MemoryCardError",
    "card_exists",
    "delete_card",
    "list_cards",
    "load_card",
    "save_card",
    "validate_name",
]
