"""Corpus loading: decode files and structured records into plain-text documents.

Single serialization seam for the training/tokenizer pipeline: every input
format (text files, JSONL, JSON, CSV) is decoded here into ``list[str]``
documents, so tokenizer training (``train_from_directory``) and
``prepare_data`` consume the same format-agnostic text. Tokenizers themselves
stay str-only — file/JSON decoding never happens inside ``encode``/``train``.

Message records render with the chat-trainer convention
(``chat_trainer._format_pairs_text``): ``User: ...\\nAssistant: ...\\n\\n``.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("slo.training.corpus_loader")

TEXT_PATTERNS = ("*.txt", "*.md", "*.py")
DEFAULT_PATTERNS = TEXT_PATTERNS + ("*.json", "*.jsonl", "*.csv")
STRUCTURED_SUFFIXES = frozenset({".json", ".jsonl", ".csv"})
_VALID_ROLES = frozenset({"user", "assistant", "system"})
_TEXT_KEYS = (
    "text",
    "content",
    "input",
    "output",
    "prompt",
    "completion",
    "instruction",
    "response",
    "question",
    "answer",
    "document",
    "body",
    "message",
)


def is_structured(path: str | Path) -> bool:
    """True when the path's suffix has a structured (non raw-text) decoder."""
    return Path(path).suffix.lower() in STRUCTURED_SUFFIXES


def render_messages(messages: list[Any]) -> str:
    """Render a ``[{"role", "content"}, ...]`` record to conversation text."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError(f"messages entry must be an object, got {type(msg).__name__}")
        role = msg.get("role")
        content = msg.get("content")
        if role not in _VALID_ROLES:
            raise ValueError(f"invalid role {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message content must be a non-empty string")
        parts.append(f"{role.capitalize()}: {content}\n")
    return "".join(parts) + "\n"


def _doc_from_record(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        if "messages" in obj:
            return render_messages(obj["messages"])
        texts = [
            str(obj[k]) for k in _TEXT_KEYS if k in obj and isinstance(obj[k], (str, int, float))
        ]
        if texts:
            return "\n".join(texts)
        return json.dumps(obj, ensure_ascii=False)
    return json.dumps(obj, ensure_ascii=False)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_jsonl(path: Path) -> list[str]:
    docs: list[str] = []
    text = _read_text(path)
    for line_num, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("%s:%d invalid JSON, skipping", path, line_num)
            continue
        try:
            docs.append(_doc_from_record(obj))
        except ValueError as exc:
            logger.warning("%s:%d %s, skipping", path, line_num, exc)
    return docs


def _load_json(path: Path) -> list[str]:
    try:
        obj = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        logger.warning("%s invalid JSON: %s", path, exc)
        return []
    if isinstance(obj, list):
        docs: list[str] = []
        for item in obj:
            try:
                docs.append(_doc_from_record(item))
            except ValueError as exc:
                logger.warning("%s record skipped: %s", path, exc)
        return docs
    return [_doc_from_record(obj)]


def _load_csv(path: Path) -> list[str]:
    docs: list[str] = []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return []
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            cells = list(row)[: len(header)]
            cells += [""] * (len(header) - len(cells))
            docs.append("\n".join(f"{h}: {c}" for h, c in zip(header, cells)))
    return docs


def load_corpus_file(path: str | Path) -> list[str]:
    """Decode one file into documents (empty files yield ``[]``)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Corpus file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        docs = _load_jsonl(p)
    elif suffix == ".json":
        docs = _load_json(p)
    elif suffix == ".csv":
        docs = _load_csv(p)
    else:
        text = _read_text(p)
        docs = [text] if text.strip() else []
    return docs


def load_corpus_dir(
    dir_path: str | Path,
    patterns: tuple[str, ...] | list[str] | None = None,
    recursive: bool = True,
) -> list[str]:
    """Decode every matching file under ``dir_path`` (deterministic order)."""
    root = Path(dir_path)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    pats = tuple(patterns) if patterns else DEFAULT_PATTERNS
    files: set[Path] = set()
    for pat in pats:
        files.update(root.rglob(pat) if recursive else root.glob(pat))
    docs: list[str] = []
    for p in sorted(files):
        if p.is_file():
            docs.extend(load_corpus_file(p))
    return docs


def load_corpus(spec: str | Path) -> list[str]:
    """Decode a file path or directory path into documents."""
    p = Path(spec)
    if p.is_dir():
        return load_corpus_dir(p)
    return load_corpus_file(p)
