"""Crawl page store — bawl-format pages to an immutable raw ingest + corpus rows.

The server half of the internet-acquisition lane: clients (``bawl`` CLI,
cron, other machines) POST crawled pages; this module owns validation and
durable storage. One ingest writes exactly one immutable
``raw/crawl_<utc>[_<n>].jsonl`` (never overwritten — a bad batch rolls back by
pointing at the previous file, same immutability rule as the cruncher
snapshots) and appends derived ``{"content", "url", "title"}`` rows to the
dataset's ``corpus.jsonl`` — the shape the existing importers and the trainer
pipeline already consume.

Page schema is ``packages/bawl``'s JSONL line: ``{url, domain?, title?, text,
headings?, ...}``; only ``url`` + ``text`` are required, unknown keys are kept
out of the normalized record on purpose (the raw store is ours, not a mirror
of every crawler quirk).

No FastAPI / server imports here — pure stdlib, unit-testable with tmp_path.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

#: Same identity rule the datasets router enforces on dataset ids.
DATASET_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")

__all__ = [
    "CrawlIngestResult",
    "CrawlStoreError",
    "DATASET_ID_RE",
    "ingest_crawl",
    "normalize_page",
    "normalize_pages",
    "parse_jsonl_pages",
]


class CrawlStoreError(ValueError):
    """Invalid crawl ingest request (bad name, no usable pages, bad JSONL)."""


@dataclass(frozen=True)
class CrawlIngestResult:
    """Audit summary of one successful ingest."""

    pages_received: int
    pages_stored: int
    pages_dropped: int
    raw_path: str
    corpus_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "pages_received": self.pages_received,
            "pages_stored": self.pages_stored,
            "pages_dropped": self.pages_dropped,
            "raw_path": self.raw_path,
            "corpus_path": self.corpus_path,
        }


def normalize_page(page: Any) -> dict[str, Any] | None:
    """Return a normalized page, or ``None`` when the record is unusable.

    A usable page has non-empty string ``url`` and ``text``; ``title`` and
    ``domain`` are kept when present (empty string otherwise), ``word_count``
    is derived.
    """
    if not isinstance(page, dict):
        return None
    url = page.get("url")
    text = page.get("text")
    if not isinstance(url, str) or not url.strip():
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    title = page.get("title")
    domain = page.get("domain")
    return {
        "url": url.strip(),
        "text": text,
        "title": title.strip() if isinstance(title, str) else "",
        "domain": domain.strip() if isinstance(domain, str) else "",
        "word_count": len(text.split()),
    }


def normalize_pages(pages: Iterable[Any]) -> tuple[list[dict[str, Any]], int]:
    """Normalize a materialized page list; returns ``(kept, dropped_count)``."""
    kept: list[dict[str, Any]] = []
    dropped = 0
    for page in pages:
        normalized = normalize_page(page)
        if normalized is None:
            dropped += 1
        else:
            kept.append(normalized)
    return kept, dropped


def parse_jsonl_pages(raw: str) -> list[Any]:
    """Parse a raw JSONL body into page objects (one JSON value per line).

    Blank lines are skipped; a non-JSON line raises ``CrawlStoreError`` naming
    the line number — silent partial parses would corrupt provenance.
    """
    pages: list[Any] = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            pages.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise CrawlStoreError(f"jsonl line {lineno}: invalid JSON: {exc}") from exc
    return pages


def _unique_raw_path(raw_dir: Path, ts: str) -> Path:
    """``crawl_<utc>.jsonl``, disambiguated with ``_2``, ``_3``… in the same second."""
    candidate = raw_dir / f"crawl_{ts}.jsonl"
    n = 2
    while candidate.exists():
        candidate = raw_dir / f"crawl_{ts}_{n}.jsonl"
        n += 1
    return candidate


def ingest_crawl(
    name: str,
    pages: Iterable[Any],
    *,
    datasets_dir: Path | str,
    source: str | None = None,
) -> CrawlIngestResult:
    """Validate and durably store one crawl batch under ``datasets_dir/name``.

    Layout::

        datasets_dir/<name>/raw/crawl_<utc>.jsonl   # immutable, one per ingest
        datasets_dir/<name>/corpus.jsonl             # appended derived rows

    Raises:
        CrawlStoreError: invalid ``name``, empty batch, or zero usable pages.
        OSError: storage write failure (propagates — the caller classifies).
    """
    if not name or not DATASET_ID_RE.match(name):
        raise CrawlStoreError(
            f"invalid dataset name {name!r}: expected letters, digits, '_' or '-'"
        )
    pages_list = list(pages)
    if not pages_list:
        raise CrawlStoreError("pages batch is empty: nothing to ingest")
    kept, dropped = normalize_pages(pages_list)
    if not kept:
        raise CrawlStoreError(
            f"no usable pages: all {len(pages_list)} dropped (each page needs "
            "non-empty string 'url' and 'text')"
        )

    dataset_dir = Path(datasets_dir) / name
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    raw_path = _unique_raw_path(raw_dir, ts)

    stored_at = time.time()
    with open(raw_path, "w", encoding="utf-8") as fh:
        for page in kept:
            record = dict(page)
            if source:
                record["source"] = source
            record["_stored_at"] = stored_at
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    corpus_path = dataset_dir / "corpus.jsonl"
    with open(corpus_path, "a", encoding="utf-8") as fh:
        for page in kept:
            row: dict[str, Any] = {
                "content": page["text"],
                "url": page["url"],
                "title": page["title"],
            }
            if source:
                row["source"] = source
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return CrawlIngestResult(
        pages_received=len(pages_list),
        pages_stored=len(kept),
        pages_dropped=dropped,
        raw_path=str(raw_path),
        corpus_path=str(corpus_path),
    )
