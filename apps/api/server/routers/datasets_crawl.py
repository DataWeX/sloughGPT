"""Crawl ingest — descriptor-projected ``POST /datasets/import/crawl``.

Server half of the internet-acquisition lane (user ruling: server well first).
Clients running ``bawl`` (CLI cron, other machines) POST crawled pages here;
the server owns validation and the durable raw store (``crawl_store``), so
crawling can stay a thin client and storage stays one contract.

Tier: external utility tooling (site data acquisition of the app itself) — it
never enters the chat tool registry, ``domain.tools``, or anything the model
reasons over. Family: the existing ``/datasets/import/*`` journey (same
purpose, one more source); this module is a descriptor projection
(``create_router``), not a hand-written route, per AGENTS.md Endpoint &
Transport Rule — the grandfathered ``datasets.py`` is untouched.

Body (contract-derived): ``name`` (dataset id), plus at least one of
``pages`` (list of bawl page objects) or ``jsonl`` (raw JSONL text — the
``bawl`` output file's bytes, for curl-friendly pushes). ``source`` is an
optional provenance label recorded on every stored record.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from infrastructure.contract import RouteSpec, create_router
from schemas.common import raise_error, safe_audit_log

from domain.agents import ToolParam, ToolSpec

#: Datasets root — same resolution the grandfathered datasets router uses
#: (``routers/datasets.py``): repo-root ``data/``. Tests monkeypatch this.
DATASETS_DIR = Path(__file__).resolve().parents[4] / "data"

_import_locks: dict[str, asyncio.Lock] = {}
_import_locks_guard = asyncio.Lock()


async def _get_import_lock(name: str) -> asyncio.Lock:
    """One lock per dataset id — mirrors the datasets router's import locks."""
    async with _import_locks_guard:
        lock = _import_locks.get(name)
        if lock is None:
            lock = asyncio.Lock()
            _import_locks[name] = lock
        return lock


async def import_crawl_pages(
    name: str,
    pages: list | None = None,
    jsonl: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Ingest bawl-format crawl pages into the dataset's raw store + corpus."""
    from domain.training._internal.crawl_store import (
        CrawlStoreError,
        ingest_crawl,
        parse_jsonl_pages,
    )

    merged: list[Any] = list(pages or [])
    if jsonl:
        try:
            merged.extend(parse_jsonl_pages(jsonl))
        except CrawlStoreError as exc:
            raise_error(str(exc), "E_BAD_REQUEST")
    if not merged:
        raise_error(
            "provide 'pages' (list of page objects) and/or 'jsonl' (JSONL text)",
            "E_BAD_REQUEST",
        )

    lock = await _get_import_lock(name)
    if lock.locked():
        raise_error(f"Import already in progress for '{name}'", "E_INFRA_BUSY")
    async with lock:
        try:
            _t0 = time.monotonic()
            result = await asyncio.to_thread(
                ingest_crawl,
                name,
                merged,
                datasets_dir=DATASETS_DIR,
                source=source,
            )
            _elapsed_ms = (time.monotonic() - _t0) * 1000
            safe_audit_log(
                "dataset.import_crawl",
                resource=name,
                detail=f"crawl elapsed={_elapsed_ms:.0f}ms",
                pages_stored=result.pages_stored,
                pages_dropped=result.pages_dropped,
            )
            return {
                "dataset_id": name,
                "pages_received": result.pages_received,
                "pages_stored": result.pages_stored,
                "pages_dropped": result.pages_dropped,
                "raw_path": result.raw_path,
                "corpus_path": result.corpus_path,
            }
        except CrawlStoreError as exc:
            raise_error(str(exc), "E_BAD_REQUEST")
        except OSError as exc:
            raise_error(f"failed to write crawl store: {exc}", "E_INTERNAL")


SPEC = ToolSpec(
    name="datasets.import_crawl",
    description=(
        "Ingest bawl-format crawled pages into a dataset: validates each page "
        "(non-empty url + text), writes an immutable raw/crawl_<utc>.jsonl "
        "batch and appends derived content rows to the dataset corpus.jsonl."
    ),
    parameters=[
        ToolParam(
            name="name",
            type="string",
            description="Dataset id to ingest under ([a-zA-Z0-9_-]+).",
            required=True,
        ),
        ToolParam(
            name="pages",
            type="array",
            description=(
                "Crawl page objects: {url, text, title?, domain?, ...} (bawl page schema)."
            ),
        ),
        ToolParam(
            name="jsonl",
            type="string",
            description="Raw JSONL text: one page object per line (bawl -o output).",
        ),
        ToolParam(
            name="source",
            type="string",
            description="Provenance label recorded on every stored record.",
        ),
    ],
    execute=import_crawl_pages,
    auth_scope="authenticated",
    idempotent=False,
    result={
        "type": "object",
        "properties": {
            "dataset_id": {"type": "string"},
            "pages_received": {"type": "integer"},
            "pages_stored": {"type": "integer"},
            "pages_dropped": {"type": "integer"},
            "raw_path": {"type": "string"},
            "corpus_path": {"type": "string"},
        },
    },
    version="1",
)

router = create_router(
    SPEC,
    RouteSpec(
        path="/import/crawl",
        method="POST",
        summary="Ingest crawled pages into a dataset raw store",
    ),
    import_crawl_pages,
    prefix="/datasets",
    tags=("datasets",),
)
