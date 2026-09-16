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

from domain.collections._internal.builders import (
    CollectorBuilder,
    DataSink,
    DataSource,
    DataTransformer,
)
from domain.collections._internal.collector import BatchCollector, Collector, ParallelCollector
from domain.collections._internal.config import (
    FilterConfig,
    PipelineConfig,
    SourceConfig,
    StoreConfig,
)
from domain.collections._internal.filters import (
    DedupFilter,
    Filter,
    FilterChain,
    KeywordFilter,
    LanguageFilter,
    LengthFilter,
    MetadataFilter,
    PrefixFilter,
    RegexFilter,
    SamplerFilter,
    TransformFilter,
    TruncateFilter,
)
from domain.collections._internal.pipeline import CollectionPipeline
from domain.collections._internal.registry import CollectionRegistry, get_registry
from domain.collections._internal.scheduler import (
    CollectorExporter,
    CollectorMonitor,
    JobConfig,
    JobScheduler,
)
from domain.collections._internal.sources import (
    ApiSource,
    FileSource,
    GeneratorSource,
    Record,
    RssSource,
    Source,
    SseSource,
    UrlSource,
    WatchSource,
)
from domain.collections._internal.stores import (
    CallbackStore,
    ChainedStore,
    FileStore,
    MemoryStore,
    StatsStore,
    Store,
)
from domain.collections._internal.training_bridge import (
    CollectorTrainingBridge,
    RecordToTrainingSource,
    TrainingDataAdapter,
    TrainingDataConfig,
    TrainingDatasetBuilder,
)
from domain.collections._internal.validators import (
    CallableSource,
    CallableStore,
    CollectorRunner,
    DataEnricher,
    DataValidator,
    EnrichmentRule,
    RateLimiter,
    Schema,
)
from domain.collections._internal.world_bridge import (
    CollectionWorldPipeline,
    RecordToWorldMapper,
    WorldFeedConfig,
    WorldGridBridge,
    WorldGridSource,
    WorldStoreAdapter,
)


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
    "Record",
    "Source",
    "FileSource",
    "UrlSource",
    "RssSource",
    "ApiSource",
    "SseSource",
    "WatchSource",
    "GeneratorSource",
    "Store",
    "FileStore",
    "MemoryStore",
    "CallbackStore",
    "ChainedStore",
    "StatsStore",
    "Filter",
    "LengthFilter",
    "DedupFilter",
    "KeywordFilter",
    "RegexFilter",
    "LanguageFilter",
    "FilterChain",
    "SamplerFilter",
    "TransformFilter",
    "TruncateFilter",
    "PrefixFilter",
    "MetadataFilter",
    "Collector",
    "ParallelCollector",
    "BatchCollector",
    "Schema",
    "DataValidator",
    "DataEnricher",
    "EnrichmentRule",
    "RateLimiter",
    "CallableSource",
    "CallableStore",
    "CollectorRunner",
    "JobConfig",
    "JobScheduler",
    "CollectorMonitor",
    "CollectorExporter",
    "CollectorBuilder",
    "DataSource",
    "DataSink",
    "DataTransformer",
    "WorldFeedConfig",
    "RecordToWorldMapper",
    "WorldGridBridge",
    "WorldGridSource",
    "WorldStoreAdapter",
    "CollectionWorldPipeline",
    "TrainingDataConfig",
    "TrainingDataAdapter",
    "RecordToTrainingSource",
    "TrainingDatasetBuilder",
    "CollectorTrainingBridge",
    "CollectionPipeline",
    "CollectionRegistry",
    "get_registry",
    "SourceConfig",
    "StoreConfig",
    "FilterConfig",
    "PipelineConfig",
    "collect_file",
    "collect_url",
    "collect_rss",
    "collect_api",
    "collect_records",
]
