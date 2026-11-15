"""Uploaded-files adapter — the MogDB ``files`` collection.

Same data source as the legacy ``/files/search`` endpoint (filename
substring), queried directly so no legacy contract moves. The upload
blob must still exist on disk — a hit the caller cannot open is not
a hit.
"""

from __future__ import annotations

from pathlib import Path

from domain.search.matching import substring_score
from domain.search.types import SearchContext, SearchHit

_ROUTE = "route:/files"
# domain/search/adapters/files.py -> repo root is four levels up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_UPLOADS_DIR = _REPO_ROOT / "data" / "uploads"


class FilesAdapter:
    store = "files"
    capability = "live"
    workspace_scoped = False

    def __init__(self, docs_factory=None, uploads_dir: Path = _UPLOADS_DIR):
        # docs_factory: () -> iterable[dict] — injectable for tests.
        self._docs_factory = docs_factory
        self._uploads_dir = uploads_dir

    def _docs(self):
        if self._docs_factory is not None:
            return self._docs_factory()
        # Same database name the legacy /files endpoints use.
        from infrastructure.db_pool import get_db

        return get_db("uploads_mogdb").collection("files").find()

    async def search(self, q: str, limit: int, ctx: SearchContext) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for doc in self._docs():
            name = doc.get("original_name") or doc.get("filename", "")
            tags = ", ".join(doc.get("tags", []))
            score = substring_score(q, name, tags)
            if score is None:
                continue
            # Mirror the legacy endpoint: skip entries whose upload blob
            # is gone (dead files are not results).
            blob = doc.get("filename", "")
            if blob and not (self._uploads_dir / blob).exists():
                continue
            hits.append(
                SearchHit(
                    id=str(doc.get("file_id", "")),
                    store=self.store,
                    title=name,
                    detail=tags or doc.get("extension", ""),
                    score=score,
                    locator=_ROUTE,
                )
            )
            if len(hits) >= limit:
                break
        return hits
