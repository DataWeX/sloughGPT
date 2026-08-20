"""
Collections Router - Data collection pipeline management.

Provides endpoints to:
- List/create/manage collection pipelines
- Run collections (one-shot or continuous)
- View collection stats and records
- Manage sources, stores, and filters
"""
from fastapi import APIRouter, Query
from typing import Optional, List
from pydantic import BaseModel, Field

from schemas.common import success_response, raise_error


class PipelineConfigRequest(BaseModel):
    """Configuration for creating a collection pipeline."""
    name: str = Field(..., max_length=200)
    source_type: str = Field(..., description="Source type: file, url, rss, api, sse, watch, generator")
    source_config: dict = Field(default_factory=dict)
    store_type: str = Field(default_factory="memory", description="Store type: file, memory, callback, chained, stats")
    store_config: dict = Field(default_factory=dict)
    filter_chain: List[dict] = Field(default_factory=list, description="Ordered list of filter configs")


class LogMetricRequest(BaseModel):
    """Log a metric to an experiment."""
    metric_name: str = Field(..., max_length=200)
    value: float


class LogParamRequest(BaseModel):
    """Log a parameter to an experiment."""
    param_name: str = Field(..., max_length=200)
    value: str = Field(..., max_length=1000)


class CollectionsRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/collections", tags=["collections"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("", self.list_pipelines, methods=["GET"])
        self.router.add_api_route("", self.create_pipeline, methods=["POST"])
        self.router.add_api_route("/run", self.run_pipeline, methods=["POST"])
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
            pipelines = registry.list()
            return success_response(data={"pipelines": pipelines, "count": len(pipelines)})
        except Exception as e:
            return success_response(data={"pipelines": [], "count": 0, "error": str(e)})

    async def create_pipeline(self, req: PipelineConfigRequest) -> dict:
        """Create and register a new collection pipeline."""
        try:
            from domains.collections import (
                CullerPipeline, SourceConfig, StoreConfig, FilterConfig
            )
            from domains.collections.registry import get_registry

            # Build source from config
            source = _build_source(req.source_type, req.source_config)
            store = _build_store(req.store_type, req.store_config)
            filters = [_build_filter(fc) for fc in req.filter_chain if fc.get("type")]

            pipeline = CullerPipeline(
                source=source,
                store=store,
                filters=filters,
                name=req.name,
            )

            registry = get_registry()
            registry.register(pipeline)

            return success_response(data={
                "id": req.name,
                "name": req.name,
                "source_type": req.source_type,
                "store_type": req.store_type,
                "filters": len(filters),
            })
        except Exception as e:
            raise_error(500, f"Failed to create pipeline: {e}")

    async def run_pipeline(self, name: str = Query(..., description="Pipeline name")) -> dict:
        """Run a collection pipeline once."""
        try:
            from domains.collections.registry import get_registry
            registry = get_registry()
            pipeline = registry.get(name)
            if not pipeline:
                raise_error(404, f"Pipeline '{name}' not found")
            count = pipeline.collect()
            return success_response(data={
                "pipeline": name,
                "collected": count,
                "stats": pipeline.stats,
            })
        except Exception as e:
            raise_error(500, f"Failed to run pipeline: {e}")

    async def get_stats(self) -> dict:
        """Get overall collection stats."""
        try:
            from domains.collections.registry import get_registry
            registry = get_registry()
            pipelines = registry.list()
            total = len(pipelines)
            return success_response(data={
                "total_pipelines": total,
                "pipelines": pipelines,
            })
        except Exception as e:
            return success_response(data={"total_pipelines": 0, "error": str(e)})

    async def get_pipeline(self, pipeline_id: str) -> dict:
        """Get details of a specific pipeline."""
        try:
            from domains.collections.registry import get_registry
            registry = get_registry()
            pipeline = registry.get(pipeline_id)
            if not pipeline:
                raise_error(404, f"Pipeline '{pipeline_id}' not found")
            return success_response(data={
                "id": pipeline_id,
                "name": getattr(pipeline, 'name', pipeline_id),
                "stats": pipeline.stats,
            })
        except Exception as e:
            raise_error(500, f"Failed to get pipeline: {e}")

    async def delete_pipeline(self, pipeline_id: str) -> dict:
        """Delete a pipeline from the registry."""
        try:
            from domains.collections.registry import get_registry
            registry = get_registry()
            registry.unregister(pipeline_id)
            return success_response(data={"deleted": pipeline_id})
        except Exception as e:
            raise_error(500, f"Failed to delete pipeline: {e}")

    async def collect(self, pipeline_id: str) -> dict:
        """Run collection for a specific pipeline."""
        try:
            from domains.collections.registry import get_registry
            registry = get_registry()
            pipeline = registry.get(pipeline_id)
            if not pipeline:
                raise_error(404, f"Pipeline '{pipeline_id}' not found")
            count = pipeline.collect()
            return success_response(data={
                "pipeline": pipeline_id,
                "collected": count,
                "stats": pipeline.stats,
            })
        except Exception as e:
            raise_error(500, f"Failed to collect: {e}")

    async def get_records(
        self,
        pipeline_id: str,
        limit: int = Query(default=50, ge=1, le=500),
    ) -> dict:
        """Get records from a pipeline's store."""
        try:
            from domains.collections.registry import get_registry
            registry = get_registry()
            pipeline = registry.get(pipeline_id)
            if not pipeline:
                raise_error(404, f"Pipeline '{pipeline_id}' not found")
            records = list(pipeline.read())
            # Apply limit
            limited = records[:limit]
            return success_response(data={
                "pipeline": pipeline_id,
                "records": [{"content": r.content, "source": r.source} for r in limited],
                "total": len(records),
                "returned": len(limited),
            })
        except Exception as e:
            raise_error(500, f"Failed to get records: {e}")


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

    filter_type = config.pop("type", "")
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
