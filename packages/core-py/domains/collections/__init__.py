from .sources import (
    Record, Source, FileSource, UrlSource, RssSource, ApiSource,
    SseSource, WatchSource, GeneratorSource,
)
from .stores import (
    Store, FileStore, MemoryStore, CallbackStore,
    ChainedStore, StatsStore,
)
from .filters import (
    Filter, LengthFilter, DedupFilter, KeywordFilter, RegexFilter,
    LanguageFilter, FilterChain, SamplerFilter, TransformFilter,
    TruncateFilter, PrefixFilter, MetadataFilter,
)
from .collector import Collector, ParallelCollector, BatchCollector
from .validators import (
    Schema, DataValidator, DataEnricher, EnrichmentRule,
    RateLimiter, CallableSource, CallableStore, CollectorRunner,
)
from .scheduler import (
    JobConfig, JobScheduler, CollectorMonitor, CollectorExporter,
)
from .pipeline import CollectionPipeline
from .registry import CollectionRegistry, get_registry
from .config import SourceConfig, StoreConfig, FilterConfig, PipelineConfig


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
    "CollectionPipeline", "CollectionRegistry", "get_registry",
    "SourceConfig", "StoreConfig", "FilterConfig", "PipelineConfig",
    "collect_file", "collect_url", "collect_rss", "collect_api", "collect_records",
]
