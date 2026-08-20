from __future__ import annotations

from typing import Any, Callable, Iterator
from pathlib import Path

from .sources import Record, Source, FileSource, UrlSource, RssSource, ApiSource, SseSource, WatchSource, GeneratorSource
from .stores import Store, FileStore, MemoryStore, CallbackStore, ChainedStore, StatsStore
from .filters import (
    Filter, LengthFilter, DedupFilter, KeywordFilter, RegexFilter,
    LanguageFilter, FilterChain, SamplerFilter, TransformFilter,
    TruncateFilter, PrefixFilter, MetadataFilter,
)
from .collector import Collector, ParallelCollector, BatchCollector


class CollectorBuilder:
    def __init__(self):
        self._source: Source | None = None
        self._store: Store | None = None
        self._filters: list[Filter] = []
        self._batch_size: int | None = None
        self._max_retries: int = 3
        self._name: str = ""

    def name(self, name: str) -> CollectorBuilder:
        self._name = name
        return self

    def source(self, source: Source) -> CollectorBuilder:
        self._source = source
        return self

    def file_source(self, path: str) -> CollectorBuilder:
        self._source = FileSource(path)
        return self

    def url_source(self, url: str) -> CollectorBuilder:
        self._source = UrlSource(url)
        return self

    def rss_source(self, url: str) -> CollectorBuilder:
        self._source = RssSource(url)
        return self

    def api_source(self, url: str) -> CollectorBuilder:
        self._source = ApiSource(url)
        return self

    def sse_source(self, url: str, event: str = "message") -> CollectorBuilder:
        self._source = SseSource(url, event=event)
        return self

    def watch_source(self, path: str, patterns: list[str] | None = None) -> CollectorBuilder:
        self._source = WatchSource(path, patterns=patterns)
        return self

    def generator_source(self, fn: Callable[[], Iterator]) -> CollectorBuilder:
        self._source = GeneratorSource(fn)
        return self

    def store(self, store: Store) -> CollectorBuilder:
        self._store = store
        return self

    def file_store(self, path: str, append: bool = True) -> CollectorBuilder:
        self._store = FileStore(path, append=append)
        return self

    def memory_store(self, max_size: int = 10000) -> CollectorBuilder:
        self._store = MemoryStore(max_size=max_size)
        return self

    def callback_store(self, callback: Callable[[Record], None]) -> CollectorBuilder:
        self._store = CallbackStore(callback)
        return self

    def stats_store(self) -> CollectorBuilder:
        if self._store:
            self._store = StatsStore(self._store)
        return self

    def filter(self, f: Filter) -> CollectorBuilder:
        self._filters.append(f)
        return self

    def length_filter(self, min_length: int = 10, max_length: int = 100000) -> CollectorBuilder:
        self._filters.append(LengthFilter(min_length=min_length, max_length=max_length))
        return self

    def dedup_filter(self) -> CollectorBuilder:
        self._filters.append(DedupFilter())
        return self

    def keyword_filter(self, keywords: list[str], mode: str = "include") -> CollectorBuilder:
        self._filters.append(KeywordFilter(keywords=keywords, mode=mode))
        return self

    def regex_filter(self, pattern: str, mode: str = "include") -> CollectorBuilder:
        self._filters.append(RegexFilter(pattern=pattern, mode=mode))
        return self

    def language_filter(self, allowed_chars_ratio: float = 0.8) -> CollectorBuilder:
        self._filters.append(LanguageFilter(allowed_chars_ratio=allowed_chars_ratio))
        return self

    def sampler_filter(self, rate: float = 0.1) -> CollectorBuilder:
        self._filters.append(SamplerFilter(rate=rate))
        return self

    def transform_filter(self, transform_fn: Callable[[Record], Record]) -> CollectorBuilder:
        self._filters.append(TransformFilter(transform_fn=transform_fn))
        return self

    def truncate_filter(self, max_length: int = 1000) -> CollectorBuilder:
        self._filters.append(TruncateFilter(max_length=max_length))
        return self

    def prefix_filter(self, prefix: str) -> CollectorBuilder:
        self._filters.append(PrefixFilter(prefix=prefix))
        return self

    def metadata_filter(self, key: str, values: list[str], mode: str = "include") -> CollectorBuilder:
        self._filters.append(MetadataFilter(key=key, values=values, mode=mode))
        return self

    def batch(self, batch_size: int, max_retries: int = 3) -> CollectorBuilder:
        self._batch_size = batch_size
        self._max_retries = max_retries
        return self

    def build(self) -> Collector | BatchCollector:
        if not self._source:
            raise ValueError("Source is required")
        if not self._store:
            self._store = MemoryStore()

        if self._batch_size:
            return BatchCollector(
                source=self._source,
                store=self._store,
                filters=self._filters if self._filters else None,
                batch_size=self._batch_size,
                max_retries=self._max_retries,
            )
        return Collector(
            source=self._source,
            store=self._store,
            filters=self._filters if self._filters else None,
        )

    def build_parallel(self, builders: list[CollectorBuilder]) -> ParallelCollector:
        collectors = [b.build() for b in builders]
        return ParallelCollector(collectors)


class DataSource:
    def __init__(self, sources: list[Source] | None = None):
        self._sources = sources or []

    def add(self, source: Source) -> DataSource:
        self._sources.append(source)
        return self

    def add_file(self, path: str) -> DataSource:
        self._sources.append(FileSource(path))
        return self

    def add_url(self, url: str) -> DataSource:
        self._sources.append(UrlSource(url))
        return self

    def add_rss(self, url: str) -> DataSource:
        self._sources.append(RssSource(url))
        return self

    def add_api(self, url: str) -> DataSource:
        self._sources.append(ApiSource(url))
        return self

    def read_all(self) -> Iterator[Record]:
        for source in self._sources:
            yield from source.read()

    def read(self, source_index: int = 0) -> Iterator[Record]:
        if 0 <= source_index < len(self._sources):
            yield from self._sources[source_index].read()

    def list_sources(self) -> list[str]:
        return [s.name for s in self._sources]

    def count(self) -> int:
        return len(self._sources)


class DataSink:
    def __init__(self, stores: list[Store] | None = None):
        self._stores = stores or []

    def add(self, store: Store) -> DataSink:
        self._stores.append(store)
        return self

    def add_file(self, path: str, append: bool = True) -> DataSink:
        self._stores.append(FileStore(path, append=append))
        return self

    def add_memory(self, max_size: int = 10000) -> DataSink:
        self._stores.append(MemoryStore(max_size=max_size))
        return self

    def add_callback(self, callback: Callable[[Record], None]) -> DataSink:
        self._stores.append(CallbackStore(callback))
        return self

    def write(self, record: Record) -> None:
        for store in self._stores:
            store.write(record)

    def write_all(self, records: list[Record]) -> int:
        count = 0
        for record in records:
            self.write(record)
            count += 1
        return count

    def flush(self) -> None:
        pass

    def list_stores(self) -> list[str]:
        return [s.name for s in self._stores]

    def count(self) -> int:
        return sum(s.count() for s in self._stores)


class DataTransformer:
    def __init__(self, transforms: list[Callable[[Record], Record]] | None = None):
        self._transforms = transforms or []
        self.stats = {"transformed": 0, "errors": 0}

    def add(self, transform_fn: Callable[[Record], Record]) -> DataTransformer:
        self._transforms.append(transform_fn)
        return self

    def add_field(self, key: str, value: Any) -> DataTransformer:
        def transform(r: Record) -> Record:
            r.metadata[key] = value
            return r
        self._transforms.append(transform)
        return self

    def add_field_fn(self, key: str, fn: Callable[[Record], Any]) -> DataTransformer:
        def transform(r: Record) -> Record:
            r.metadata[key] = fn(r)
            return r
        self._transforms.append(transform)
        return self

    def add_content_transform(self, fn: Callable[[str], str]) -> DataTransformer:
        def transform(r: Record) -> Record:
            r.content = fn(r.content)
            return r
        self._transforms.append(transform)
        return self

    def transform(self, record: Record) -> Record:
        for t in self._transforms:
            try:
                record = t(record)
                self.stats["transformed"] += 1
            except Exception:
                self.stats["errors"] += 1
        return record

    def transform_all(self, records: list[Record]) -> list[Record]:
        return [self.transform(r) for r in records]

    def reset_stats(self):
        self.stats = {"transformed": 0, "errors": 0}
