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

import logging
import os
from pathlib import Path as PathLib
from typing import Any

from fastapi import APIRouter, Body, Depends, Path, Query
from infrastructure.auth import require_auth_if_enabled
from mogdb import MogDB
from schemas.common import classify_and_raise, raise_error, safe_audit_log, success_response

logger = logging.getLogger("slo.docstore")

COLLECTIONS = frozenset(
    {
        "sessions",
        "pendingMessages",
        "knowledge",
        "bookmarks",
        "prompts",
        "drafts",
        "kv",
        "errors",
    }
)

_DEFAULT_PATH = str(PathLib(__file__).resolve().parents[4] / "data" / "docstore")

_db: MogDB | None = None


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


def _strip_meta(doc: dict[str, Any]) -> dict[str, Any]:
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

    def _validate(self, collection: str) -> dict | None:
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
        sort: str | None = Query(default=None, description="Field to sort by"),
        direction: int = Query(
            default=-1, ge=-1, le=1, alias="dir", description="Sort direction: 1 or -1"
        ),
        limit: int | None = Query(default=None, gt=0, description="Max results"),
    ) -> dict:
        """List all documents in a collection, optionally sorted/limited."""
        try:
            err = self._validate(collection)
            if err:
                return err
            sort_by = [(sort, direction)] if sort else None
            docs = _collection(collection).find(sort=sort_by, limit=limit)
            return success_response(data=[_strip_meta(d) for d in docs])
        except Exception as e:
            classify_and_raise(e, source="docstore.list")

    def get_doc(
        self,
        collection: str = Path(...),
        doc_id: str = Path(...),
    ) -> dict:
        """Get a single document by ``doc_id``."""
        try:
            err = self._validate(collection)
            if err:
                return err
            doc = _collection(collection).find_one({"_id": doc_id})
            return success_response(data=_strip_meta(doc) if doc else None)
        except Exception as e:
            classify_and_raise(e, source="docstore.get")

    def put_doc(
        self,
        collection: str = Path(...),
        doc_id: str = Path(...),
        body: dict[str, Any] = Body(...),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Upsert a document: replace it if it exists, otherwise insert."""
        try:
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
        except Exception as e:
            logger.warning(
                "docstore.put failed: collection=%s doc_id=%s error=%s",
                collection,
                doc_id,
                str(e)[:200],
                extra={
                    "tag": "DOCSTORE",
                    "collection": collection,
                    "doc_id": doc_id,
                    "error": str(e)[:200],
                },
            )
            classify_and_raise(e, source="docstore.put")

    def patch_doc(
        self,
        collection: str = Path(...),
        doc_id: str = Path(...),
        body: dict[str, Any] = Body(...),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Merge fields into an existing document (no-op if it does not exist)."""
        try:
            err = self._validate(collection)
            if err:
                return err
            update = dict(body)
            update.pop("_id", None)
            if not update:
                return success_response(data={"modified": 0})
            modified = _collection(collection).update_one({"_id": doc_id}, {"$set": update})
            return success_response(data={"modified": modified})
        except Exception as e:
            classify_and_raise(e, source="docstore.patch")

    def delete_doc(
        self,
        collection: str = Path(...),
        doc_id: str = Path(...),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Delete a single document by ``doc_id``."""
        try:
            err = self._validate(collection)
            if err:
                return err
            deleted = _collection(collection).delete_one({"_id": doc_id})
            safe_audit_log("docstore.delete", resource=f"{collection}/{doc_id}")
            return success_response(data={"deleted": bool(deleted)})
        except Exception as e:
            classify_and_raise(e, source="docstore.delete")

    def clear_collection(
        self, collection: str = Path(...), auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Delete every document in a collection."""
        try:
            err = self._validate(collection)
            if err:
                return err
            _collection(collection).drop()
            safe_audit_log("docstore.clear", resource=collection)
            return success_response(data={"cleared": True})
        except Exception as e:
            classify_and_raise(e, source="docstore.clear")

    def bulk_put(
        self,
        collection: str = Path(...),
        body: dict[str, Any] = Body(...),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Upsert many documents in one request."""
        try:
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
        except Exception as e:
            classify_and_raise(e, source="docstore.bulk_put")


router = DocStoreRouter().router
