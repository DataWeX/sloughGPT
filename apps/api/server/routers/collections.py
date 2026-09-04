"""
Collections Router - Data collection pipeline management.

Provides endpoints to:
- List/create/manage collection pipelines
- Run collections (one-shot or continuous)
- View collection stats and records
- Manage sources, stores, and filters
"""
import logging
from fastapi import APIRouter, Depends, Query
from typing import List
from pydantic import BaseModel, Field

from infrastructure.auth import require_auth_if_enabled
from schemas.common import success_response, raise_error, classify_and_raise, safe_audit_log
from domains.infrastructure.errors import AppError

logger = logging.getLogger("slo.api.collections")


class PipelineConfigRequest(BaseModel):
    """Configuration for creating a collection pipeline."""
    name: str = Field(..., max_length=200)
    source_type: str = Field(..., description="Source type: file, url, rss, api, sse, watch, generator")
    source_config: dict = Field(default_factory=dict)
    store_type: str = Field(default="memory", description="Store type: file, memory, callback, chained, stats")
    store_config: dict = Field(default_factory=dict)
    filter_chain: List[dict] = Field(default_factory=list, description="Ordered list of filter configs")


class CollectRequest(BaseModel):
    """Direct collection request (no pre-registered pipeline)."""
    source_type: str = Field(..., description="Source type: file, url, rss, api")
    source_config: dict = Field(default_factory=dict)
    min_length: int = Field(default=10, ge=0)
    dedup: bool = Field(default=True)
    max_records: int | None = Field(default=None, ge=1)


class CollectionsRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/collections", tags=["collections"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("", self.list_pipelines, methods=["GET"])
        self.router.add_api_route("/create", self.create_pipeline, methods=["POST"])
        self.router.add_api_route("/run", self.run_pipeline, methods=["POST"])
        self.router.add_api_route("/collect", self.collect_direct, methods=["POST"])
        self.router.add_api_route("/stats", self.get_stats, methods=["GET"])
        self.router.add_api_route("/{pipeline_id}", self.get_pipeline, methods=["GET"])
        self.router.add_api_route("/{pipeline_id}", self.delete_pipeline, methods=["DELETE"])
        self.router.add_api_route("/{pipeline_id}/collect", self.collect, methods=["POST"])
        self.router.add_api_route("/{pipeline_id}/records", self.get_records, methods=["GET"])

    async def list_pipelines(self) -> dict:
        """List all registered collection pipelines."""
        try:
            from domains.collections import get_registry
            registry = get_registry()
            pipelines = registry.list_pipelines()
            sources = registry.list_sources()
            stores = registry.list_stores()
            filters = registry.list_filters()
            return success_response(data={
                "pipelines": pipelines,
                "sources": sources,
                "stores": stores,
                "filters": filters,
                "counts": {
                    "pipelines": len(pipelines),
                    "sources": len(sources),
                    "stores": len(stores),
                    "filters": len(filters),
                },
            })
        except Exception as e:
            classify_and_raise(e, source="collections.list_pipelines")

    async def create_pipeline(self, req: PipelineConfigRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Create and register a new collection pipeline."""
        try:
            from domains.collections.registry import get_registry

            registry = get_registry()
            pipeline = registry.create_pipeline(
                name=req.name,
                source_name=req.source_type,
                store_name=req.store_type,
                filter_names=[fc.get("type", "") for fc in req.filter_chain if fc.get("type")],
            )
            if pipeline is None:
                raise_error(f"Failed to create pipeline: source '{req.source_type}' or store '{req.store_type}' not registered", code="E_CREATE_FAILED", status_code=400)

            safe_audit_log("collection.create", resource=req.name, detail=f"source={req.source_type} store={req.store_type}")
            return success_response(data={
                "id": req.name,
                "name": req.name,
                "source_type": req.source_type,
                "store_type": req.store_type,
                "filters": len(req.filter_chain),
            })
        except AppError as e:
            classify_and_raise(e, source="collections.create_pipeline")
        except Exception as e:
            logger.warning("Create pipeline failed: %s", e)
            classify_and_raise(e, source="create_pipeline")

    async def run_pipeline(self, name: str = Query(..., description="Pipeline name"), auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Run a collection pipeline once."""
        import time as _time
        try:
            _t0 = _time.monotonic()
            from domains.collections.registry import get_registry
            registry = get_registry()
            count = registry.collect(name)
            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            pipeline = registry.get_pipeline(name)
            if not pipeline:
                raise_error(f"Pipeline '{name}' not found", code="E_NOT_FOUND", status_code=404)
            safe_audit_log("collection.run", resource=name, detail=f"collected={count} elapsed={_elapsed_ms:.0f}ms")
            return success_response(data={
                "pipeline": name,
                "collected": count,
                "stats": pipeline.stats,
                "elapsed_ms": round(_elapsed_ms, 1),
            })
        except AppError as e:
            classify_and_raise(e, source="collections.run_pipeline")
        except Exception as e:
            logger.warning("Run pipeline failed: %s", e)
            classify_and_raise(e, source="run_pipeline")

    async def collect_direct(self, req: CollectRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Collect data directly without pre-creating a pipeline."""
        import time as _time
        try:
            _t0 = _time.monotonic()
            from domains.collections import (
                Collector, MemoryStore, LengthFilter, DedupFilter,
            )
            source = _build_source(req.source_type, req.source_config)
            store = MemoryStore()
            filters = []
            if req.min_length > 0:
                filters.append(LengthFilter(min_length=req.min_length))
            if req.dedup:
                filters.append(DedupFilter())

            collector = Collector(source, store, filters=filters)
            count = collector.collect()
            _elapsed_ms = (_time.monotonic() - _t0) * 1000

            records = []
            for record in store.read_all():
                records.append({"content": record.content[:200], "metadata": record.metadata})

            safe_audit_log("collection.direct", resource=req.source_type, detail=f"collected={count} elapsed={_elapsed_ms:.0f}ms")
            return success_response(data={
                "collected": count,
                "stats": collector.stats,
                "records": records[:50],
                "elapsed_ms": round(_elapsed_ms, 1),
            })
        except AppError as e:
            classify_and_raise(e, source="collections.collect_direct")
        except Exception as e:
            logger.warning("Direct collect failed: %s", e)
            classify_and_raise(e, source="collect_direct")

    async def get_stats(self) -> dict:
        """Get overall collection stats."""
        try:
            from domains.collections.registry import get_registry
            registry = get_registry()
            stats = registry.stats()
            return success_response(data=stats)
        except Exception as e:
            classify_and_raise(e, source="collections.get_stats")

    async def get_pipeline(self, pipeline_id: str) -> dict:
        """Get details of a specific pipeline."""
        try:
            from domains.collections.registry import get_registry
            registry = get_registry()
            pipeline = registry.get_pipeline(pipeline_id)
            if not pipeline:
                raise_error(f"Pipeline '{pipeline_id}' not found", code="E_NOT_FOUND", status_code=404)
            return success_response(data={
                "id": pipeline_id,
                "name": getattr(pipeline, 'name', pipeline_id),
                "stats": pipeline.stats,
            })
        except AppError as e:
            classify_and_raise(e, source="collections.get_pipeline")
        except Exception as e:
            classify_and_raise(e, source="get_pipeline")

    async def delete_pipeline(self, pipeline_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Delete a pipeline from the registry."""
        try:
            from domains.collections.registry import get_registry
            registry = get_registry()
            removed = registry.remove_pipeline(pipeline_id)
            if not removed:
                raise_error(f"Pipeline '{pipeline_id}' not found", code="E_NOT_FOUND", status_code=404)
            safe_audit_log("collection.delete", resource=pipeline_id)
            return success_response(data={"deleted": pipeline_id})
        except Exception as e:
            classify_and_raise(e, source="collections.delete_pipeline")

    async def collect(self, pipeline_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Run collection for a specific pipeline."""
        try:
            from domains.collections.registry import get_registry
            registry = get_registry()
            pipeline = registry.get_pipeline(pipeline_id)
            if not pipeline:
                raise_error(f"Pipeline '{pipeline_id}' not found", code="E_NOT_FOUND", status_code=404)
            count = pipeline.collect()
            return success_response(data={
                "pipeline": pipeline_id,
                "collected": count,
                "stats": pipeline.stats,
            })
        except AppError as e:
            classify_and_raise(e, source="collections.collect")
        except Exception as e:
            classify_and_raise(e, source="collect")

    async def get_records(
        self,
        pipeline_id: str,
        limit: int = Query(default=50, ge=1, le=500),
    ) -> dict:
        """Get records from a pipeline's store."""
        try:
            from domains.collections.registry import get_registry
            registry = get_registry()
            pipeline = registry.get_pipeline(pipeline_id)
            if not pipeline:
                raise_error(f"Pipeline '{pipeline_id}' not found", code="E_NOT_FOUND", status_code=404)
            records = list(pipeline.read())
            limited = records[:limit]
            return success_response(data={
                "pipeline": pipeline_id,
                "records": [{"content": r.content, "metadata": r.metadata} for r in limited],
                "total": len(records),
                "returned": len(limited),
            })
        except AppError as e:
            classify_and_raise(e, source="collections.get_records")
        except Exception as e:
            classify_and_raise(e, source="get_records")


def _build_source(source_type: str, config: dict):
    """Build a Source from type and config."""
    from domains.collections.sources import (
        FileSource, UrlSource, RssSource, ApiSource,
        SseSource, WatchSource, GeneratorSource,
    )

    constructors = {
        "file": FileSource,
        "url": UrlSource,
        "rss": RssSource,
        "api": ApiSource,
        "sse": SseSource,
        "watch": WatchSource,
        "generator": GeneratorSource,
    }

    cls = constructors.get(source_type)
    if not cls:
        raise ValueError(f"Unknown source type: {source_type}")
    return cls(**config)


def _build_store(store_type: str, config: dict):
    """Build a Store from type and config."""
    from domains.collections.stores import (
        FileStore, MemoryStore, CallbackStore,
        ChainedStore, StatsStore,
    )

    constructors = {
        "file": FileStore,
        "memory": MemoryStore,
        "callback": CallbackStore,
        "chained": ChainedStore,
        "stats": StatsStore,
    }

    cls = constructors.get(store_type)
    if not cls:
        raise ValueError(f"Unknown store type: {store_type}")
    return cls(**config)


def _build_filter(config: dict):
    """Build a Filter from config dict."""
    from domains.collections.filters import (
        LengthFilter, DedupFilter, KeywordFilter, RegexFilter,
        LanguageFilter, SamplerFilter, TransformFilter,
        TruncateFilter, PrefixFilter, MetadataFilter,
    )

    filter_type = config.get("type", "")
    constructors = {
        "length": LengthFilter,
        "dedup": DedupFilter,
        "keyword": KeywordFilter,
        "regex": RegexFilter,
        "language": LanguageFilter,
        "sampler": SamplerFilter,
        "transform": TransformFilter,
        "truncate": TruncateFilter,
        "prefix": PrefixFilter,
        "metadata": MetadataFilter,
    }

    cls = constructors.get(filter_type)
    if not cls:
        raise ValueError(f"Unknown filter type: {filter_type}")
    return cls(**{k: v for k, v in config.items() if k != "type"})


router = CollectionsRouter().router
