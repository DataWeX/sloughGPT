"""Training data feed — subscribe to a live corpus instead of a static dataset.

The model's own API conversations are appended continuously to a JSONL corpus
(``ConversationLogger`` → ``data/api_conversations/corpus.jsonl``). This module
turns that stream into consumable training data in two ways:

1. **Feed endpoints over HTTP** — ``TrainingFeedClient`` fetches records past a
   persisted offset cursor from any ``/training/feed``-style URL (local corpus
   or remote network feed).
2. **Batch consumption** — ``FeedBatchSampler`` implements the ``BatchSampler``
   protocol consumed by ``TrainingLoop``: it drains new records on a refresh
   interval and builds structured training text (train ≡ serve via PromptEngine).

The batch trainer path accepts ``feed:<source>`` and ``http(s)://...`` as a
``data_path`` (see ``load_feed_text``): point a URL at the training flow, and it
reads the feed and trains on whatever it returns.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from domain.shared import find_repo_root

from .training_handler import BatchSampler

logger = logging.getLogger("slo.training.feed")

_REPO_ROOT = find_repo_root(Path(__file__).resolve())

_PAGE_SIZE = 1000
_MAX_PAGES = 10000
_MIN_CONTENT_LEN = 2


def is_feed_source(data_path: Any) -> bool:
    """True when a data_path is a feed specification, not a file/dataset name.

    Accepts ``feed:<source>`` (local corpus) and ``http(s)://...`` (network feed URL).

    Args:
        data_path: the value passed as a training ``data_path``.

    Returns:
        True when the value resolves to a training feed.
    """
    return isinstance(data_path, str) and (
        data_path.startswith("feed:")
        or data_path.startswith("http://")
        or data_path.startswith("https://")
    )


def extract_pair(record: Any) -> dict[str, str] | None:
    """Extract one (user, assistant) pair from a corpus record.

    Handles the ``{"messages": [{role, content}, ...]}`` format written by
    ``ConversationLogger`` and flat formats (``user_msg``/``assistant_msg``,
    ``prompt``/``completion``, ``user``/``assistant``).

    Args:
        record: a parsed JSONL record (dict).

    Returns:
        ``{"user_msg", "assistant_msg"}`` or None when no valid pair exists.
    """
    if not isinstance(record, dict):
        return None

    user: Any = None
    assistant: Any = None
    messages = record.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role == "user" and user is None:
                user = content
            elif role == "assistant" and assistant is None and user is not None:
                assistant = content
                break
    else:
        user = record.get("user_msg") or record.get("prompt") or record.get("user")
        assistant = (
            record.get("assistant_msg") or record.get("completion") or record.get("assistant")
        )

    if user is None or assistant is None:
        return None
    user = str(user).strip()
    assistant = str(assistant).strip()
    if len(user) < _MIN_CONTENT_LEN or len(assistant) < _MIN_CONTENT_LEN:
        return None
    return {"user_msg": user, "assistant_msg": assistant}


def pairs_to_text(
    pairs: list[dict[str, str]],
    use_prompt_engine: bool = True,
    block_size: int | None = None,
) -> str:
    """Build structured training text from feed pairs.

    Delegates to the same template used by ``ExperienceSampler`` (PromptEngine
    chat template when available, explicit ``<|im_start|>`` delimiters otherwise)
    so train ≡ serve.

    Args:
        pairs: list of ``{"user_msg", "assistant_msg"}`` dicts.
        use_prompt_engine: try PromptEngine for full train/serve parity.
        block_size: kept for API compatibility (unused in text building).

    Returns:
        Concatenated training text.
    """
    from .experience_adapter import _build_experience_text

    return _build_experience_text(pairs, block_size=block_size, use_prompt_engine=use_prompt_engine)


@dataclass(slots=True)
class FeedPage:
    """One page of a training feed read."""

    name: str
    source: str
    offset: int
    next_offset: int
    total: int
    records: list[dict[str, Any]] = field(default_factory=list)


class TrainingFeedClient:
    """Fetch training records from a feed past a persisted offset cursor.

    Args:
        feed_url: network feed URL (``http(s)://...``). When None, reads the
            local corpus for ``source`` (live API logs on disk).
        source: corpus/source name (``api-conversations``, ``feedback``, ...).
        timeout: HTTP timeout in seconds.
        offset_store_path: JSON file persisting ``{feed_key: offset}`` so a
            consumer resumes where it left off. None disables persistence.
        local_base: explicit base dir for local corpus lookups (testing). When
            None, candidates are resolved under the repo root.
    """

    def __init__(
        self,
        feed_url: str | None = None,
        source: str = "api-conversations",
        timeout: float = 10.0,
        offset_store_path: str | Path | None = None,
        local_base: str | Path | None = None,
    ) -> None:
        self._source = source
        self._url = feed_url
        self._timeout = timeout
        self._local_base = Path(local_base) if local_base else None
        self._store_path = Path(offset_store_path) if offset_store_path else None
        self._offset: int | None = None
        self._feed_key = self._url or f"local:{source}"

    @property
    def url(self) -> str:
        """Canonical feed identifier (URL for network, ``local:<source>`` otherwise)."""
        return self._feed_key

    @property
    def source(self) -> str:
        """Feed source name."""
        return self._source

    @property
    def offset(self) -> int:
        """Persisted/known read offset (0 when no cursor exists)."""
        if self._offset is None:
            self._offset = self._load_offset()
        return self._offset

    # ── cursor persistence ────────────────────────────────────────────────

    def _load_offset(self) -> int:
        if self._store_path is None:
            return 0
        try:
            data = json.loads(self._store_path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        return int(data.get(self._feed_key, 0))

    def save_offset(self, new_offset: int) -> None:
        """Advance the read cursor (in-memory and in the offset store if set)."""
        self._offset = int(new_offset)
        if self._store_path is None:
            return
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            data: dict[str, Any] = {}
            if self._store_path.exists():
                try:
                    data = json.loads(self._store_path.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            data[self._feed_key] = self._offset
            self._store_path.write_text(json.dumps(data), encoding="utf-8")
        except Exception as exc:
            logger.debug("Failed to persist feed cursor %s: %s", self._feed_key, exc)

    def clear_offset(self) -> None:
        """Reset the cursor to 0 in-memory and in the store."""
        self.save_offset(0)

    # ── local corpus resolution ───────────────────────────────────────────

    def _resolve_local_path(self) -> Path:
        base = self._local_base
        # Source names use hyphens (api-conversations); corpus dirs in this repo
        # use underscores (data/api_conversations/). Try both spellings.
        underscored = self._source.replace("-", "_")
        names = {self._source, underscored}
        if base is not None:
            candidates: list[Path] = []
            for name in names:
                candidates.extend([base / name / "corpus.jsonl", base / f"{name}.jsonl"])
        else:
            candidates = []
            for name in names:
                candidates.extend(
                    [
                        _REPO_ROOT / "data" / name / "corpus.jsonl",
                        _REPO_ROOT / "data" / f"{name}.jsonl",
                        _REPO_ROOT / "datasets" / name / "corpus.jsonl",
                    ]
                )
        for p in candidates:
            if p.exists():
                return p
        raise ValueError(f"Feed source not found: {self._source}")

    def _read_local(self, limit: int, start: int) -> FeedPage:
        path = self._resolve_local_path()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(f"Unreadable feed source {self._source}: {exc}") from exc
        total = len(lines)
        if limit and limit > 0:
            taken = lines[start : start + limit]
        else:
            taken = lines[start:]

        records: list[dict[str, Any]] = []
        for line in taken:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        next_offset = min(total, start + len(taken))
        return FeedPage(
            name=self._source,
            source=self._source,
            offset=start,
            next_offset=next_offset,
            total=total,
            records=records,
        )

    # ── network fetch ─────────────────────────────────────────────────────

    def _http_get(self, url: str) -> dict[str, Any]:
        with urllib.request.urlopen(url, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _build_url(self, offset: int, limit: int) -> str:
        sep = "&" if "?" in self._url else "?"
        url = f"{self._url}{sep}offset={offset}"
        if limit and limit > 0:
            url += f"&limit={limit}"
        return url

    def _fetch_page(self, offset: int, limit: int) -> dict[str, Any]:
        payload = self._http_get(self._build_url(offset, limit))
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            payload = payload["data"]
        return payload

    def _read_network(self, limit: int, start: int) -> FeedPage:
        if limit and limit > 0:
            body = self._fetch_page(start, limit)
            return self._page_from_body(body, start)
        # Read everything: page through until the feed reports no more records.
        records: list[dict[str, Any]] = []
        pos = start
        next_offset = start
        total = start
        for _ in range(_MAX_PAGES):
            body = self._fetch_page(pos, _PAGE_SIZE)
            page_records = body.get("records") or []
            records.extend(page_records)
            next_offset = int(body.get("next_offset") or (pos + len(page_records)))
            total = int(body.get("total") or next_offset)
            if next_offset >= total or not page_records:
                break
            pos = next_offset
        return FeedPage(
            name=str(body.get("name") or self._source),
            source=str(body.get("source") or self._source),
            offset=start,
            next_offset=next_offset,
            total=total,
            records=records,
        )

    @staticmethod
    def _page_from_body(body: dict[str, Any], start: int) -> FeedPage:
        records = list(body.get("records") or [])
        next_offset = int(body.get("next_offset") or (start + len(records)))
        total = int(body.get("total") or next_offset)
        return FeedPage(
            name=str(body.get("name") or body.get("source") or "feed"),
            source=str(body.get("source") or body.get("name") or "feed"),
            offset=start,
            next_offset=next_offset,
            total=total,
            records=records,
        )

    # ── public read API ───────────────────────────────────────────────────

    def read(self, limit: int = 0, offset: int | None = None) -> FeedPage:
        """Read one feed page starting at the given offset (default: cursor).

        Args:
            limit: max records to return (0 = all remaining).
            offset: explicit start position; None uses the persisted cursor.

        Returns:
            FeedPage with raw records (not yet converted to pairs).
        """
        start = self.offset if offset is None else int(offset)
        if self._url is not None:
            return self._read_network(limit, start)
        return self._read_local(limit, start)

    def read_pairs(self, limit: int = 0) -> list[dict[str, str]]:
        """Read a page and convert records to validated (user, assistant) pairs."""
        page = self.read(limit=limit)
        return [p for p in (extract_pair(r) for r in page.records) if p is not None]

    def stats(self) -> dict[str, Any]:
        """Small summary for logs/UIs (never a raw dump)."""
        return {
            "url": self.url,
            "source": self._source,
            "offset": self.offset,
        }


class FeedBatchSampler(BatchSampler):
    """BatchSampler that drains a training feed for ``TrainingLoop``.

    Reads records past the persisted cursor on construction, then refreshes on a
    timer so training consumes newly captured experience as it arrives. Falls
    back to synthetic seed pairs when the feed has no usable data yet.

    Args:
        stoi: token -> id map (char-level or BPE) for tokenization.
        block_size: context length.
        feed_url: network feed URL; None reads the local corpus for ``source``.
        source: feed source name.
        max_pairs: rolling window cap on retained pairs (bounded memory).
        refresh_interval: seconds between automatic feed refreshes; 0 disables.
        offset_store_path: JSON file persisting the consumed offset.
        local_base: local corpus base dir (testing).
        seed: RNG seed.
        use_prompt_engine: template train≡serve via PromptEngine when possible.
    """

    def __init__(
        self,
        stoi: dict[str, int],
        block_size: int,
        feed_url: str | None = None,
        source: str = "api-conversations",
        max_pairs: int = 2048,
        refresh_interval: float = 30.0,
        offset_store_path: str | Path | None = None,
        local_base: str | Path | None = None,
        seed: int = 42,
        use_prompt_engine: bool = True,
    ) -> None:
        self.stoi = stoi
        self.block_size = block_size
        self._max_pairs = int(max_pairs)
        self._refresh_interval = float(refresh_interval)
        self._use_prompt_engine = use_prompt_engine
        self._rng = np.random.default_rng(seed)
        self._client = TrainingFeedClient(
            feed_url=feed_url,
            source=source,
            offset_store_path=offset_store_path,
            local_base=local_base,
        )

        page = self._client.read()
        pairs = [p for p in (extract_pair(r) for r in page.records) if p is not None]
        self._client.save_offset(page.next_offset)
        self._pairs = pairs
        if not self._pairs:
            logger.info(
                "Feed %r returned no usable pairs — using synthetic seed",
                self._client.url,
                extra={"tag": "TRAIN"},
            )
            self._pairs = [
                {"user_msg": "hello", "assistant_msg": "hi there"},
                {"user_msg": "how are you", "assistant_msg": "doing well, how can I help?"},
            ]
        self._rebuild()
        self._last_refresh = time.monotonic()

    def _rebuild(self) -> None:
        """Re-tokenize the current pair window into a flat id array."""
        text = pairs_to_text(
            self._pairs, block_size=self.block_size, use_prompt_engine=self._use_prompt_engine
        )
        self._text = text
        self.ids = np.array([self.stoi.get(c, 0) for c in text], dtype=np.int32)
        self.n_samples = max(1, len(self.ids) - self.block_size - 1)
        logger.info(
            "FeedBatchSampler: %d pairs -> %d tokens -> %d samples (block=%d)",
            len(self._pairs),
            len(self.ids),
            self.n_samples,
            self.block_size,
            extra={"tag": "TRAIN"},
        )

    def refresh(self) -> int:
        """Fetch new records past the cursor and rebuild the token window.

        Returns:
            Number of newly ingested pairs (0 when the feed has no new data).
        """
        page = self._client.read(limit=_PAGE_SIZE)
        new_pairs = [p for p in (extract_pair(r) for r in page.records) if p is not None]
        if new_pairs:
            self._pairs.extend(new_pairs)
            if self._max_pairs > 0 and len(self._pairs) > self._max_pairs:
                self._pairs = self._pairs[-self._max_pairs :]
            self._client.save_offset(page.next_offset)
            self._rebuild()
        self._last_refresh = time.monotonic()
        return len(new_pairs)

    def __len__(self) -> int:
        return self.n_samples

    def get_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample a random block batch, refreshing the feed first when due.

        Args:
            batch_size: number of samples in the batch.

        Returns:
            (x, y) int32 arrays of shape (batch_size, block_size).
        """
        if (
            self._refresh_interval > 0
            and time.monotonic() - self._last_refresh >= self._refresh_interval
        ):
            self.refresh()
            self._last_refresh = time.monotonic()
        indices = self._rng.integers(0, self.n_samples, size=batch_size)
        offsets = np.arange(self.block_size)
        pos = indices[:, None] + offsets
        x = self.ids[pos]
        y = self.ids[pos + 1]
        return x.astype(np.int32), y.astype(np.int32)

    def stats(self) -> dict[str, Any]:
        """Concise summary for logs/UIs."""
        return {
            "pairs": len(self._pairs),
            "tokens": int(len(self.ids)),
            "samples": int(self.n_samples),
            "block_size": int(self.block_size),
            "offset": self._client.offset,
        }


def load_feed_text(
    data_path: str,
    use_prompt_engine: bool = True,
    local_base: str | Path | None = None,
    offset_store_path: str | Path | None = None,
) -> str:
    """Load a feed specification as structured training text.

    Used by the batch trainer path (``prepare_data``) so a ``feed:<source>`` or
    ``http(s)://...`` data_path trains directly on the feed contents.

    Args:
        data_path: feed spec (``feed:<source>`` or a feed URL).
        use_prompt_engine: template train≡serve via PromptEngine when possible.
        local_base: local corpus base dir (testing).
        offset_store_path: cursor store for network feeds.

    Returns:
        Training text built from every usable record in the feed.
    """
    if data_path.startswith("feed:"):
        client = TrainingFeedClient(source=data_path.split(":", 1)[1], local_base=local_base)
    else:
        client = TrainingFeedClient(
            feed_url=data_path,
            offset_store_path=offset_store_path,
            local_base=local_base,
        )
    pairs = client.read_pairs()
    logger.info(
        "Feed %s: loaded %d pairs",
        client.url,
        len(pairs),
        extra={"tag": "TRAIN"},
    )
    return pairs_to_text(pairs, use_prompt_engine=use_prompt_engine)
