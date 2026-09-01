"""
Pair Extractor

Extracts (user_msg, assistant_msg) training pairs from server-side
inference logs (session JSON files and response log JSONL files).

Used by the /mobile/train/from-sessions endpoint and AutoTrainer
to avoid the mobile→server round-trip for training data.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from domains.shared import find_repo_root

logger = logging.getLogger("slo.training.pair_extractor")

_REPO_ROOT = find_repo_root(Path(__file__).resolve())
_SESSIONS_DIR = _REPO_ROOT / "data" / "chat_sessions"
_RESPONSE_LOGS_DIR = _REPO_ROOT / "data" / "response_logs"
_CAPTURED_DIR = _REPO_ROOT / "datasets" / "api_conversations"

# Ensure directories exist (auto-created on first access)
_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
_RESPONSE_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def extract_pairs_from_sessions(
    limit: int = 50,
    min_length: int = 5,
    session_ids: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """
    Extract (user_msg, assistant_msg) pairs from session JSON files.

    Sessions are stored as JSON with a messages[] array where each entry
    has {"role": "user"|"assistant", "content": "..."}.

    Args:
        limit: Maximum number of pairs to return.
        min_length: Minimum character length for both user and assistant messages.
        session_ids: If provided, only extract from these sessions. Otherwise all.

    Returns:
        List of dicts with keys: user_msg, assistant_msg, session_id.
        Deduplicated by content hash. Newest first.
    """
    if not _SESSIONS_DIR.exists():
        logger.info("Sessions directory not found: %s", _SESSIONS_DIR,
            extra={"tag": "TRAIN"},)
        return []

    seen_hashes: set = set()
    pairs: List[Dict[str, str]] = []

    # Get session files, newest first
    session_files = sorted(
        _SESSIONS_DIR.glob("*.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    for sf in session_files:
        sid = sf.stem
        if session_ids and sid not in session_ids:
            continue

        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read session %s: %s", sid, e,
                extra={"tag": "TRAIN"},)
            continue

        messages = data.get("messages", [])
        if len(messages) < 2:
            continue

        # Walk consecutive user→assistant pairs
        for i in range(len(messages) - 1):
            if len(pairs) >= limit:
                break

            msg_a = messages[i]
            msg_b = messages[i + 1]

            if msg_a.get("role") != "user" or msg_b.get("role") != "assistant":
                continue

            user_msg = (msg_a.get("content") or "").strip()
            assistant_msg = (msg_b.get("content") or "").strip()

            if len(user_msg) < min_length or len(assistant_msg) < min_length:
                continue

            # Deduplicate by content hash
            h = hashlib.md5(f"{user_msg}|||{assistant_msg}".encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            pairs.append({
                "user_msg": user_msg,
                "assistant_msg": assistant_msg,
                "session_id": sid,
            })

        if len(pairs) >= limit:
            break

    logger.info("Extracted %d pairs from %d session files", len(pairs), len(session_files),
        extra={"tag": "TRAIN"},)
    return pairs


def extract_pairs_from_logs(
    limit: int = 100,
    min_length: int = 5,
    model: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Extract (user_msg, assistant_msg) pairs from response log JSONL files.

    Each line is a JSON object with user_message, assistant_response, model, etc.

    Args:
        limit: Maximum number of pairs to return.
        min_length: Minimum character length for both messages.
        model: If provided, only extract pairs from this model.

    Returns:
        List of dicts with keys: user_msg, assistant_msg, session_id, model.
        Newest first (file order).
    """
    if not _RESPONSE_LOGS_DIR.exists():
        logger.info("Response logs directory not found: %s", _RESPONSE_LOGS_DIR,
            extra={"tag": "TRAIN"},)
        return []

    seen_hashes: set = set()
    pairs: List[Dict[str, str]] = []

    # Get log files, newest first
    log_files = sorted(
        _RESPONSE_LOGS_DIR.glob("*.jsonl"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    for lf in log_files:
        try:
            with open(lf, encoding="utf-8") as f:
                for line in f:
                    if len(pairs) >= limit:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if model and entry.get("model") != model:
                        continue

                    user_msg = (entry.get("user_message") or "").strip()
                    assistant_msg = (entry.get("assistant_response") or "").strip()

                    if len(user_msg) < min_length or len(assistant_msg) < min_length:
                        continue

                    h = hashlib.md5(f"{user_msg}|||{assistant_msg}".encode()).hexdigest()
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)

                    pairs.append({
                        "user_msg": user_msg,
                        "assistant_msg": assistant_msg,
                        "session_id": entry.get("session_id", ""),
                        "model": entry.get("model", ""),
                    })
        except OSError as e:
            logger.warning("Failed to read log %s: %s", lf, e,
                extra={"tag": "TRAIN"},)
            continue

        if len(pairs) >= limit:
            break

    logger.info("Extracted %d pairs from %d log files", len(pairs), len(log_files),
        extra={"tag": "TRAIN"},)
    return pairs


def extract_pairs_from_corpus(
    limit: int = 100,
    min_length: int = 5,
    model: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Extract (user_msg, assistant_msg) pairs from captured API conversations.

    Reads ``datasets/api_conversations/corpus.jsonl``, where each line is a
    messages-format exchange: ``{"messages": [{"role": "user", ...}, {"role":
    "assistant", ...}], "meta": {...}}``.

    Args:
        limit: Maximum number of pairs to return.
        min_length: Minimum character length for both messages.
        model: If provided, only extract pairs from this model.

    Returns:
        List of dicts with keys: user_msg, assistant_msg, session_id, model.
        Deduplicated by content hash. File order.
    """
    corpus_file = _CAPTURED_DIR / "corpus.jsonl"
    if not corpus_file.exists():
        logger.info("Captured corpus not found: %s", corpus_file,
            extra={"tag": "TRAIN"},)
        return []

    seen_hashes: set = set()
    pairs: List[Dict[str, str]] = []

    try:
        with open(corpus_file, encoding="utf-8") as f:
            for line in f:
                if len(pairs) >= limit:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                meta = entry.get("meta") or {}
                if model and meta.get("model") != model:
                    continue

                user_msg = ""
                assistant_msg = ""
                for m in entry.get("messages") or []:
                    if m.get("role") == "user":
                        user_msg = (m.get("content") or "").strip()
                    elif m.get("role") == "assistant":
                        assistant_msg = (m.get("content") or "").strip()

                if len(user_msg) < min_length or len(assistant_msg) < min_length:
                    continue

                h = hashlib.md5(f"{user_msg}|||{assistant_msg}".encode()).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                pairs.append({
                    "user_msg": user_msg,
                    "assistant_msg": assistant_msg,
                    "session_id": meta.get("session_id", ""),
                    "model": meta.get("model", ""),
                })
    except OSError as e:
        logger.warning("Failed to read captured corpus %s: %s", corpus_file, e,
            extra={"tag": "TRAIN"},)

    logger.info("Extracted %d pairs from captured corpus", len(pairs),
        extra={"tag": "TRAIN"},)
    return pairs


def write_training_text(
    pairs: List[Dict[str, str]],
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Write training pairs to a text file in 'User: ...\nAssistant: ...\n\n' format.

    Args:
        pairs: List of dicts with 'user_msg' and 'assistant_msg' keys.
        output_dir: Directory to write to. Defaults to data/mobile_training/.

    Returns:
        Path to the written text file.
    """
    if output_dir is None:
        output_dir = _REPO_ROOT / "data" / "mobile_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    import time
    ts = int(time.time())
    text_file = output_dir / f"sessions_{ts}.txt"

    with open(text_file, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(f"User: {pair['user_msg']}\nAssistant: {pair['assistant_msg']}\n\n")

    logger.info("Wrote %d pairs to %s", len(pairs), text_file,
        extra={"tag": "TRAIN"},)
    return text_file


def count_pairs_in_sessions() -> int:
    """Count total extractable pairs across all session files (no read of content)."""
    if not _SESSIONS_DIR.exists():
        return 0

    total = 0
    for sf in _SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
            messages = data.get("messages", [])
            for i in range(len(messages) - 1):
                if messages[i].get("role") == "user" and messages[i + 1].get("role") == "assistant":
                    total += 1
        except (json.JSONDecodeError, OSError):
            continue

    return total


def count_pairs_in_logs() -> int:
    """Count total entries in response log files."""
    if not _RESPONSE_LOGS_DIR.exists():
        return 0

    total = 0
    for lf in _RESPONSE_LOGS_DIR.glob("*.jsonl"):
        try:
            with open(lf) as f:
                total += sum(1 for line in f if line.strip())
        except OSError:
            continue

    return total


def count_pairs_in_corpus() -> int:
    """Count total exchanges in the captured API conversations corpus."""
    corpus_file = _CAPTURED_DIR / "corpus.jsonl"
    if not corpus_file.exists():
        return 0
    try:
        with open(corpus_file, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


__all__ = [
    "extract_pairs_from_sessions",
    "extract_pairs_from_logs",
    "extract_pairs_from_corpus",
    "write_training_text",
    "count_pairs_in_sessions",
    "count_pairs_in_logs",
    "count_pairs_in_corpus",
]
