"""collections — Data collection pipeline: sources, stores, filters, collectors.

Public API:
    Record, Source, FileSource, UrlSource, RssSource, ApiSource, SseSource, WatchSource, GeneratorSource
    Store, FileStore, MemoryStore, CallbackStore, ChainedStore, StatsStore
    Filter, LengthFilter, DedupFilter, KeywordFilter, RegexFilter, LanguageFilter, FilterChain
    Collector, ParallelCollector, BatchCollector
    Schema, DataValidator, DataEnricher, EnrichmentRule, RateLimiter
    JobConfig, JobScheduler, CollectorMonitor, CollectorExporter
    CollectorBuilder, DataSource, DataSink, DataTransformer
    CollectionPipeline, CollectionRegistry, get_registry
    collect_file, collect_url, collect_rss, collect_api, collect_records
"""

from domain.collections._internal.sources import (
    Record, Source, FileSource, UrlSource, RssSource, ApiSource,
    SseSource, WatchSource, GeneratorSource,
)
from domain.collections._internal.stores import (
    Store, FileStore, MemoryStore, CallbackStore,
    ChainedStore, StatsStore,
)
from domain.collections._internal.filters import (
    Filter, LengthFilter, DedupFilter, KeywordFilter, RegexFilter,
    LanguageFilter, FilterChain, SamplerFilter, TransformFilter,
    TruncateFilter, PrefixFilter, MetadataFilter,
)
from domain.collections._internal.collector import Collector, ParallelCollector, BatchCollector
from domain.collections._internal.validators import (
    Schema, DataValidator, DataEnricher, EnrichmentRule,
    RateLimiter, CallableSource, CallableStore, CollectorRunner,
)
from domain.collections._internal.scheduler import (
    JobConfig, JobScheduler, CollectorMonitor, CollectorExporter,
)
from domain.collections._internal.builders import (
    CollectorBuilder, DataSource, DataSink, DataTransformer,
)
from domain.collections._internal.world_bridge import (
    WorldFeedConfig, RecordToWorldMapper, WorldGridBridge,
    WorldGridSource, WorldStoreAdapter, CollectionWorldPipeline,
)
from domain.collections._internal.training_bridge import (
    TrainingDataConfig, TrainingDataAdapter, RecordToTrainingSource,
    TrainingDatasetBuilder, CollectorTrainingBridge,
)
from domain.collections._internal.pipeline import CollectionPipeline
from domain.collections._internal.registry import CollectionRegistry, get_registry
from domain.collections._internal.config import SourceConfig, StoreConfig, FilterConfig, PipelineConfig


def collect_file(path: str, output_path: str | None = None, **kwargs) -> int:
    source = FileSource(path)
    store = FileStore(output_path) if output_path else MemoryStore()
    collector = Collector(source, store, **kwargs)
    return collector.collect()


def collect_url(url: str, output_path: str | None = None, **kwargs) -> int:
    source = UrlSource(url)
    store = FileStore(output_path) if output_path else MemoryStore()
    collector = Collector(source, store, **kwargs)
    return collector.collect()


def collect_rss(url: str, output_path: str | None = None, **kwargs) -> int:
    source = RssSource(url)
    store = FileStore(output_path) if output_path else MemoryStore()
    collector = Collector(source, store, **kwargs)
    return collector.collect()


def collect_api(url: str, output_path: str | None = None, **kwargs) -> int:
    source = ApiSource(url)
    store = FileStore(output_path) if output_path else MemoryStore()
    collector = Collector(source, store, **kwargs)
    return collector.collect()


def collect_records(records: list[Record], output_path: str | None = None, **kwargs) -> int:
    source = GeneratorSource(lambda: iter(records))
    store = FileStore(output_path) if output_path else MemoryStore()
    collector = Collector(source, store, **kwargs)
    return collector.collect()


__all__ = [
    "Record", "Source", "FileSource", "UrlSource", "RssSource", "ApiSource",
    "SseSource", "WatchSource", "GeneratorSource",
    "Store", "FileStore", "MemoryStore", "CallbackStore",
    "ChainedStore", "StatsStore",
    "Filter", "LengthFilter", "DedupFilter", "KeywordFilter", "RegexFilter",
    "LanguageFilter", "FilterChain", "SamplerFilter", "TransformFilter",
    "TruncateFilter", "PrefixFilter", "MetadataFilter",
    "Collector", "ParallelCollector", "BatchCollector",
    "Schema", "DataValidator", "DataEnricher", "EnrichmentRule",
    "RateLimiter", "CallableSource", "CallableStore", "CollectorRunner",
    "JobConfig", "JobScheduler", "CollectorMonitor", "CollectorExporter",
    "CollectorBuilder", "DataSource", "DataSink", "DataTransformer",
    "WorldFeedConfig", "RecordToWorldMapper", "WorldGridBridge",
    "WorldGridSource", "WorldStoreAdapter", "CollectionWorldPipeline",
    "TrainingDataConfig", "TrainingDataAdapter", "RecordToTrainingSource",
    "TrainingDatasetBuilder", "CollectorTrainingBridge",
    "CollectionPipeline", "CollectionRegistry", "get_registry",
    "SourceConfig", "StoreConfig", "FilterConfig", "PipelineConfig",
    "collect_file", "collect_url", "collect_rss", "collect_api", "collect_records",
]
