"""
Model Catalog — persistent model registry backed by MogDB.

Tracks all known models: loaded, on-disk, and from HuggingFace.
Stores metadata (format, path, parameters, quantization, source)
so the server knows what's available without re-scanning on every request.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("slo.infrastructure.model_catalog")

_DEFAULT_DB_PATH = "data/model_catalog"


class ModelCatalog:
    """Persistent model catalog backed by MogDB.

    Each model is a document with:
      - model_id: unique identifier (e.g. "gpt2", "qwen-0.5b")
      - source: "huggingface" | "local" | "soul"
      - format: "slnc" | "safetensors" | "soul"
      - path: filesystem path to weights (if local)
      - downloaded: whether weights are present on disk (loadable now)
      - status: "loaded" | "available" | "error"
      - parameters: parameter count
      - vocab_size: vocabulary size
      - n_layer: number of layers
      - n_embed: embedding dimension
      - n_head: number of attention heads
      - max_seq_len: maximum sequence length
      - quantized: whether quantized
      - quant_bits: quantization bits (8, 4, etc.)
      - device: current device ("cpu", "cuda", "mps")
      - loaded_at: timestamp when loaded (epoch seconds)
      - last_used: timestamp of last inference
      - inference_count: total inferences served
      - error: last error message (if status == "error")
      - tags: list of tags (e.g. ["chat", "code", "small"])
      - description: human-readable description
      - created_at: when cataloged
      - updated_at: last metadata update
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB_PATH):
        from mogdb import MogDB

        # Add JSON sync directory for human-readable backup
        sync_dir = str(Path(db_path).parent / "model_catalog_json")
        self._db = MogDB(str(db_path), sync_dir=sync_dir)
        self._models = self._db.collection("models")
        # Unique index on model_id
        self._models.create_index("model_id", unique=True)
        logger.info("ModelCatalog: opened at %s (sync: %s)", db_path, sync_dir)

    def add(
        self,
        model_id: str,
        source: str = "local",
        format: str = "slnc",
        path: str | None = None,
        parameters: int = 0,
        vocab_size: int = 0,
        n_layer: int = 0,
        n_embed: int = 0,
        n_head: int = 0,
        max_seq_len: int = 0,
        downloaded: bool = False,
        tags: list[str] | None = None,
        description: str = "",
        **extra: Any,
    ) -> str:
        """Add or update a model in the catalog.

        If model_id already exists, updates the fields provided.
        Returns the document ID.

        ``downloaded`` is the honesty flag: it says weights for this model
        were found on disk, not merely that the id is a known Hub repo. A
        model can be catalogued (known, tagged, sized) without being
        downloaded; only ``downloaded`` entries are loadable right now.
        """
        now = time.time()
        existing = self._models.find_one({"model_id": model_id})

        doc: dict[str, Any] = {
            "model_id": model_id,
            "source": source,
            "format": format,
            "path": path,
            "downloaded": bool(downloaded),
            "status": "available",
            "parameters": parameters,
            "vocab_size": vocab_size,
            "n_layer": n_layer,
            "n_embed": n_embed,
            "n_head": n_head,
            "max_seq_len": max_seq_len,
            "quantized": extra.get("quantized", False),
            "quant_bits": extra.get("quant_bits", None),
            "device": extra.get("device", "cpu"),
            "loaded_at": None,
            "last_used": None,
            "inference_count": 0,
            "error": None,
            "tags": tags or [],
            "description": description,
            "created_at": now,
            "updated_at": now,
        }

        if existing:
            # Merge: keep runtime fields, update metadata
            doc["status"] = existing.get("status", "available")
            doc["loaded_at"] = existing.get("loaded_at")
            doc["last_used"] = existing.get("last_used")
            doc["inference_count"] = existing.get("inference_count", 0)
            doc["error"] = existing.get("error")
            doc["created_at"] = existing.get("created_at", now)
            self._models.update_one(
                {"model_id": model_id},
                {"$set": doc},
            )
            logger.info("ModelCatalog: updated '%s'", model_id)
        else:
            self._models.insert_one(doc)
            logger.info("ModelCatalog: added '%s' (source=%s, format=%s)", model_id, source, format)

        return model_id

    def mark_loaded(self, model_id: str, device: str = "cpu") -> None:
        """Mark a model as loaded in memory."""
        self._models.update_one(
            {"model_id": model_id},
            {
                "$set": {
                    "status": "loaded",
                    "device": device,
                    "loaded_at": time.time(),
                    "error": None,
                    "updated_at": time.time(),
                }
            },
        )

    def mark_unloaded(self, model_id: str) -> None:
        """Mark a model as unloaded."""
        self._models.update_one(
            {"model_id": model_id},
            {
                "$set": {
                    "status": "available",
                    "loaded_at": None,
                    "updated_at": time.time(),
                }
            },
        )

    def mark_error(self, model_id: str, error: str) -> None:
        """Mark a model as having an error."""
        self._models.update_one(
            {"model_id": model_id},
            {
                "$set": {
                    "status": "error",
                    "error": error,
                    "updated_at": time.time(),
                }
            },
        )

    def record_inference(self, model_id: str) -> None:
        """Record that an inference was served.

        Uses find_one_and_update for atomic read-modify-write.
        """
        now = time.time()
        doc = self._models.find_one_and_update(
            {"model_id": model_id},
            {
                "$set": {
                    "last_used": now,
                    "updated_at": now,
                },
                "$inc": {
                    "inference_count": 1,
                },
            },
            return_document="after",
        )
        # Fallback if $inc isn't supported: read-modify-write
        if doc is None:
            doc = self._models.find_one({"model_id": model_id})
            if doc:
                self._models.update_one(
                    {"model_id": model_id},
                    {
                        "$set": {
                            "inference_count": doc.get("inference_count", 0) + 1,
                            "last_used": now,
                            "updated_at": now,
                        }
                    },
                )

    def get(self, model_id: str) -> dict | None:
        """Get a model by ID."""
        return self._models.find_one({"model_id": model_id})

    def list_all(self) -> list[dict]:
        """List all models in the catalog."""
        return self._models.find()

    def list_loaded(self) -> list[dict]:
        """List all currently loaded models."""
        return self._models.find({"status": "loaded"})

    def list_available(self) -> list[dict]:
        """List all available (not loaded) models."""
        return self._models.find({"status": "available"})

    def list_by_source(self, source: str) -> list[dict]:
        """List models from a specific source."""
        return self._models.find({"source": source})

    def list_by_tag(self, tag: str) -> list[dict]:
        """List models with a specific tag."""
        return [m for m in self._models.find() if tag in (m.get("tags") or [])]

    def remove(self, model_id: str) -> bool:
        """Remove a model from the catalog."""
        count = self._models.delete_one({"model_id": model_id})
        if count > 0:
            logger.info("ModelCatalog: removed '%s'", model_id)
            return True
        return False

    def count(self) -> int:
        """Total models in catalog."""
        return self._models.count()

    def stats(self) -> dict:
        """Catalog statistics using server-side aggregation."""
        try:
            # Use MogDB aggregation pipeline
            pipeline = [
                {
                    "$group": {
                        "_id": "$status",
                        "count": {"$sum": 1},
                        "total_params": {"$sum": "$parameters"},
                        "total_inferences": {"$sum": "$inference_count"},
                        "sources": {"$addToSet": "$source"},
                    }
                },
            ]
            results = self._models.aggregate(pipeline)

            stats = {
                "total": 0,
                "loaded": 0,
                "available": 0,
                "errors": 0,
                "total_parameters": 0,
                "total_inferences": 0,
                "sources": set(),
            }

            for group in results:
                status = group["_id"]
                count = group["count"]
                stats["total"] += count
                stats["total_parameters"] += group.get("total_params", 0)
                stats["total_inferences"] += group.get("total_inferences", 0)
                stats["sources"].update(group.get("sources", []))

                if status == "loaded":
                    stats["loaded"] = count
                elif status == "available":
                    stats["available"] = count
                elif status == "error":
                    stats["errors"] = count

            stats["sources"] = list(stats["sources"])
            return stats

        except Exception as e:
            # Fallback to client-side aggregation if pipeline fails
            logger.debug("Aggregation failed, using fallback: %s", e)
            all_models = self.list_all()
            loaded = [m for m in all_models if m.get("status") == "loaded"]
            available = [m for m in all_models if m.get("status") == "available"]
            errors = [m for m in all_models if m.get("status") == "error"]
            total_params = sum(m.get("parameters", 0) for m in loaded)
            total_inferences = sum(m.get("inference_count", 0) for m in all_models)

            return {
                "total": len(all_models),
                "loaded": len(loaded),
                "available": len(available),
                "errors": len(errors),
                "total_parameters": total_params,
                "total_inferences": total_inferences,
                "sources": list({m.get("source", "unknown") for m in all_models}),
            }

    @staticmethod
    def _largest_weight_file(model_dir: Path) -> Path | None:
        """Largest weight file (>1 KB) under a cache dir, or ``None``.

        ``safetensors`` outranks ``bin`` regardless of size: dual-format
        repos ship the same weights twice and the downloader fetches the
        safetensors copy when it exists (same rule as
        ``model_size._get_hub_file_size_gb``).
        """
        best: Path | None = None
        best_rank: tuple[int, int] = (2, 0)
        for suffix in (".safetensors", ".bin"):
            for candidate in model_dir.rglob(f"*{suffix}"):
                try:
                    size = candidate.stat().st_size
                except OSError:
                    continue  # broken symlink: blob gone, not a model
                if size <= 1_000:
                    continue
                rank = (0 if suffix == ".safetensors" else 1, -size)
                if rank < best_rank:
                    best, best_rank = candidate, rank
        return best

    def sync_from_disk(self, cache_dirs: list[str | Path] | None = None) -> int:
        """Scan cache directories and catalog models whose weights are on disk.

        Returns the number of new models added.

        Two on-disk shapes are recognized in the HuggingFace cache:

        * ``model.slnc`` — this project's own serialized format, in any
          snapshot (``snapshots/<commit>/``) *or* flat at the dir root (the
          project-local mirror's shape), catalogued as ``format="slnc"``.
        * standard HF weights (``*.safetensors`` / ``*.bin``) — catalogued
          only when :func:`hf_hub.is_cache_dir_complete` says the download
          finished, as ``format="safetensors"`` / ``"bin"``.

        The completeness gate is the whole point: a refs-only stub (refs
        kept, ``snapshots/`` deleted) or an interrupted download has a
        directory but no loadable model, and listing it would recreate the
        "available but not actually here" lie this registry exists to
        prevent. Nothing is ever deleted here — the scan only reads.

        Custom ``cache_dirs`` are scanned for ``*.slnc`` only; they are
        our own model directories, not HF repo layouts.
        """

        added = 0
        # Scan HuggingFace cache
        hf_home = os.environ.get("HF_HOME")
        hf_cache = (Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface") / "hub"
        if hf_cache.exists():
            from .hf_hub import is_cache_dir_complete

            for model_dir in hf_cache.iterdir():
                if not model_dir.is_dir() or not model_dir.name.startswith("models--"):
                    continue
                # HF cache dirs are like models--gpt2
                model_id = model_dir.name.replace("models--", "").replace("--", "/")
                slnc_path = model_dir / "snapshots" / "main" / "model.slnc"
                if not slnc_path.exists():
                    # Check parent dirs
                    for snap in (model_dir / "snapshots").glob("*"):
                        candidate = snap / "model.slnc"
                        if candidate.exists():
                            slnc_path = candidate
                            break
                if not slnc_path.exists():
                    # Flat layout: the project-local mirror keeps model.slnc at
                    # the dir root (``models/hf-cache/hub/models--<id>/model.slnc``),
                    # which is exactly what the loader reads. Without this the
                    # 3.8 GB Qwen model on this machine catalogued as nothing.
                    flat_slnc = model_dir / "model.slnc"
                    if flat_slnc.exists():
                        slnc_path = flat_slnc

                if slnc_path.exists():
                    if not self.get(model_id):
                        self.add(
                            model_id=model_id,
                            source="huggingface",
                            format="slnc",
                            path=str(slnc_path),
                            downloaded=True,
                        )
                        added += 1
                    continue

                # Standard HF weights — only a complete download is a model.
                if not is_cache_dir_complete(model_dir, model_id):
                    continue
                weight = self._largest_weight_file(model_dir)
                if weight is None or self.get(model_id):
                    continue
                self.add(
                    model_id=model_id,
                    source="huggingface",
                    format=weight.suffix.lstrip("."),
                    path=str(weight),
                    downloaded=True,
                )
                added += 1

        # Scan custom cache dirs
        if cache_dirs:
            for cache_dir in cache_dirs:
                cache_path = Path(cache_dir)
                if not cache_path.exists():
                    continue
                for slnc_file in cache_path.glob("**/*.slnc"):
                    model_id = slnc_file.stem
                    existing = self.get(model_id)
                    if not existing:
                        self.add(
                            model_id=model_id,
                            source="local",
                            format="slnc",
                            path=str(slnc_file),
                        )
                        added += 1

        if added:
            logger.info("ModelCatalog: synced %d new models from disk", added)
        return added


# Module-level singleton
_catalog: ModelCatalog | None = None
_catalog_lock = threading.Lock()


def get_model_catalog(db_path: str | Path | None = None) -> ModelCatalog:
    """Get or create the global model catalog."""
    global _catalog
    if _catalog is None:
        with _catalog_lock:
            if _catalog is None:
                _catalog = ModelCatalog(db_path or _DEFAULT_DB_PATH)
    return _catalog


def reset_model_catalog() -> None:
    """Reset the singleton (for testing)."""
    global _catalog
    with _catalog_lock:
        _catalog = None
