"""DocStore Router — server-side document store for the browser chat DB.

The frontend chat store (formerly Dexie/IndexedDB in ``apps/web/lib/db.ts``)
persists conversations, pending messages, knowledge, bookmarks, prompts,
drafts, key-value settings and error logs. This router exposes CRUD over a
fixed set of MogDB collections so the browser can persist to the server
instead of local IndexedDB.

This module is a thin HTTP adapter only: all business logic lives in the
frontend client (``apps/web/lib/db.ts``). Documents are keyed by an
explicit string ``_id`` taken from the URL path (``doc_id``); the client's
logical ``id`` field is preserved inside the stored document. MogDB-internal
fields (``_id``, ``_created``, ``_updated``) are stripped from responses.

Collections:
    sessions, pendingMessages, knowledge, bookmarks, prompts, drafts, kv,
    errors
"""

import os
from pathlib import Path as PathLib
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Path, Query

from schemas.common import success_response, raise_error, safe_audit_log
from mogdb import MogDB

COLLECTIONS = frozenset({
    "sessions",
    "pendingMessages",
    "knowledge",
    "bookmarks",
    "prompts",
    "drafts",
    "kv",
    "errors",
})

_DEFAULT_PATH = str(PathLib(__file__).resolve().parents[4] / "data" / "docstore")

_db: Optional[MogDB] = None


def _get_db() -> MogDB:
    """Return the shared MogDB instance, creating it from config on first use.

    The storage path comes from ``MOGDB_DOCSTORE_PATH`` (set by tests) and
    defaults to ``<repo>/data/docstore``.
    """
    global _db
    if _db is None:
        _db = MogDB(os.environ.get("MOGDB_DOCSTORE_PATH", _DEFAULT_PATH))
    return _db


def _collection(name: str) -> Any:
    """Return the named MogDB collection from the shared database."""
    return _get_db().collection(name)


def _strip_meta(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Return *doc* without MogDB-internal fields (``_id``/``_created``/``_updated``)."""
    return {k: v for k, v in doc.items() if not k.startswith("_")}


class DocStoreRouter:
    """CRUD router over the fixed set of MogDB-backed collections."""

    def __init__(self) -> None:
        self.router = APIRouter(prefix="/docstore", tags=["docstore"])
        self._register_routes()

    def _register_routes(self) -> None:
        """Register all routes on this router."""
        self.router.add_api_route("/{collection}/bulk", self.bulk_put, methods=["POST"])
        self.router.add_api_route("/{collection}", self.list_docs, methods=["GET"])
        self.router.add_api_route("/{collection}", self.clear_collection, methods=["DELETE"])
        self.router.add_api_route("/{collection}/{doc_id}", self.get_doc, methods=["GET"])
        self.router.add_api_route("/{collection}/{doc_id}", self.put_doc, methods=["PUT"])
        self.router.add_api_route("/{collection}/{doc_id}", self.patch_doc, methods=["PATCH"])
        self.router.add_api_route("/{collection}/{doc_id}", self.delete_doc, methods=["DELETE"])

    def _validate(self, collection: str) -> Optional[dict]:
        """Return an error response dict for an unknown collection, else ``None``."""
        if collection not in COLLECTIONS:
            raise_error(
                f"Unknown collection: {collection}",
                code="E_UNKNOWN_COLLECTION",
                details={"allowed": sorted(COLLECTIONS)},
            )
        return None

    def list_docs(
        self,
        collection: str = Path(...),
        sort: Optional[str] = Query(default=None, description="Field to sort by"),
        direction: int = Query(
            default=-1, ge=-1, le=1, alias="dir", description="Sort direction: 1 or -1"
        ),
        limit: Optional[int] = Query(default=None, gt=0, description="Max results"),
    ) -> dict:
        """List all documents in a collection, optionally sorted/limited.

        Args:
            collection: Whitelisted collection name.
            sort: Field name to sort on (any document field).
            direction: 1 = ascending, -1 = descending.
            limit: Maximum number of documents to return.

        Returns:
            success_response with a list of docs (MogDB meta stripped).
        """
        err = self._validate(collection)
        if err:
            return err
        sort_by = [(sort, direction)] if sort else None
        docs = _collection(collection).find(sort=sort_by, limit=limit)
        return success_response(data=[_strip_meta(d) for d in docs])

    def get_doc(
        self,
        collection: str = Path(...),
        doc_id: str = Path(...),
    ) -> dict:
        """Get a single document by ``doc_id``.

        Returns:
            success_response with the doc (meta stripped) or ``None``.
        """
        err = self._validate(collection)
        if err:
            return err
        doc = _collection(collection).find_one({"_id": doc_id})
        return success_response(data=_strip_meta(doc) if doc else None)

    def put_doc(
        self,
        collection: str = Path(...),
        doc_id: str = Path(...),
        body: Dict[str, Any] = Body(...),
    ) -> dict:
        """Upsert a document: replace it if it exists, otherwise insert.

        Args:
            collection: Whitelisted collection name.
            doc_id: Document key (stored as ``_id``).
            body: Full document body; the client's ``id`` field is preserved.

        Returns:
            success_response with ``{"id": doc_id, "created": bool}``.
        """
        err = self._validate(collection)
        if err:
            return err
        coll = _collection(collection)
        doc = dict(body)
        doc["_id"] = doc_id
        created = coll.find_one({"_id": doc_id}) is None
        if not created:
            coll.delete_one({"_id": doc_id})
        coll.insert_one(doc)
        return success_response(data={"id": doc_id, "created": created})

    def patch_doc(
        self,
        collection: str = Path(...),
        doc_id: str = Path(...),
        body: Dict[str, Any] = Body(...),
    ) -> dict:
        """Merge fields into an existing document (no-op if it does not exist).

        Mirrors Dexie ``Table.update``: a missing document is not created.

        Returns:
            success_response with ``{"modified": 0|1}``.
        """
        err = self._validate(collection)
        if err:
            return err
        update = dict(body)
        update.pop("_id", None)
        if not update:
            return success_response(data={"modified": 0})
        modified = _collection(collection).update_one({"_id": doc_id}, {"$set": update})
        return success_response(data={"modified": modified})

    def delete_doc(
        self,
        collection: str = Path(...),
        doc_id: str = Path(...),
    ) -> dict:
        """Delete a single document by ``doc_id``.

        Returns:
            success_response with ``{"deleted": bool}``.
        """
        err = self._validate(collection)
        if err:
            return err
        deleted = _collection(collection).delete_one({"_id": doc_id})
        safe_audit_log("docstore.delete", resource=f"{collection}/{doc_id}")
        return success_response(data={"deleted": bool(deleted)})

    def clear_collection(self, collection: str = Path(...)) -> dict:
        """Delete every document in a collection (drops its journal files).

        Returns:
            success_response with ``{"cleared": True}``.
        """
        err = self._validate(collection)
        if err:
            return err
        _collection(collection).drop()
        safe_audit_log("docstore.clear", resource=collection)
        return success_response(data={"cleared": True})

    def bulk_put(
        self,
        collection: str = Path(...),
        body: Dict[str, Any] = Body(...),
    ) -> dict:
        """Upsert many documents in one request.

        Args:
            collection: Whitelisted collection name.
            body: ``{"docs": [...]}``; each doc must carry an ``id`` field.

        Returns:
            success_response with ``{"imported": n}``.
        """
        err = self._validate(collection)
        if err:
            return err
        docs = body.get("docs")
        if not isinstance(docs, list):
            raise_error("body.docs must be an array", code="E_BAD_REQUEST")
        coll = _collection(collection)
        count = 0
        for raw in docs:
            if not isinstance(raw, dict):
                continue
            doc_id = raw.get("id")
            if not doc_id:
                continue
            doc = dict(raw)
            doc["_id"] = doc_id
            if coll.find_one({"_id": doc_id}):
                coll.delete_one({"_id": doc_id})
            coll.insert_one(doc)
            count += 1
        safe_audit_log("docstore.bulk_put", resource=collection, detail=f"imported={count}")
        return success_response(data={"imported": count})


router = DocStoreRouter().router
