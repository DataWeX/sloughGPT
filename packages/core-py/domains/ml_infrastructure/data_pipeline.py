"""
Data Pipeline - ETL and Data Processing for ML

Production-grade data pipeline with:
- Data sources (files, databases, APIs)
- Transformations (transform, filter, split)
- Data validation
- Caching and persistence
- Streaming support
"""

import json
import time
import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Iterator
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger("sloughgpt.pipeline")


class DataType(Enum):
    """Data types supported."""

    TEXT = "text"
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    EMBEDDING = "embedding"
    MIXED = "mixed"


class PipelineStatus(Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DataSchema:
    """Schema definition for data."""

    columns: Dict[str, DataType]
    required: List[str] = field(default_factory=list)
    nullable: List[str] = field(default_factory=list)
    default_values: Dict[str, Any] = field(default_factory=dict)

    def validate(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate data against schema."""
        errors = []

        for col in self.required:
            if col not in data:
                errors.append(f"Missing required column: {col}")

        for col, value in data.items():
            if col in self.columns:
                expected_type = self.columns[col]
                if not self._check_type(value, expected_type):
                    errors.append(f"Invalid type for {col}: expected {expected_type.value}")

        return len(errors) == 0, errors

    def _check_type(self, value: Any, expected_type: DataType) -> bool:
        if value is None:
            return True
        if expected_type == DataType.NUMERICAL:
            return isinstance(value, (int, float, np.number))
        elif expected_type == DataType.CATEGORICAL:
            return isinstance(value, (str, int))
        elif expected_type == DataType.TEXT:
            return isinstance(value, str)
        elif expected_type == DataType.EMBEDDING:
            return isinstance(value, (list, np.ndarray))
        return True


@dataclass
class DataStats:
    """Statistics for a dataset."""

    row_count: int
    column_stats: Dict[str, Dict[str, Any]]
    null_counts: Dict[str, int]
    unique_counts: Dict[str, int]
    generated_at: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DataSource(ABC):
    """Abstract base for data sources."""

    @abstractmethod
    def read(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Read data from source."""
        pass

    @abstractmethod
    def write(self, data: Iterator[Dict[str, Any]]) -> int:
        """Write data to source. Returns count."""
        pass

    @abstractmethod
    def exists(self) -> bool:
        """Check if source exists."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Count records in source."""
        pass


class FileSource(DataSource):
    """File-based data source (JSON, CSV, Parquet)."""

    def __init__(self, path: str, format: str = "json"):
        self.path = Path(path)
        self.format = format.lower()

    def read(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        if self.format == "json":
            return self._read_json(limit)
        elif self.format == "csv":
            return self._read_csv(limit)
        return iter([])

    def _read_json(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        import json

        with open(self.path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                yield json.loads(line.strip())

    def _read_csv(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        import csv

        with open(self.path) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break
                yield {
                    k: (float(v) if v.replace(".", "", 1).isdigit() else v) for k, v in row.items()
                }

    def write(self, data: Iterator[Dict[str, Any]]) -> int:
        count = 0
        if self.format == "json":
            with open(self.path, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
                    count += 1
        return count

    def exists(self) -> bool:
        return self.path.exists()

    def count(self) -> int:
        if not self.exists():
            return 0
        return sum(1 for _ in self.read())


class InMemorySource(DataSource):
    """In-memory data source."""

    def __init__(self, data: Optional[List[Dict[str, Any]]] = None):
        self.data = data or []

    def read(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        items = self.data[:limit] if limit else self.data
        return iter(items)

    def write(self, data: Iterator[Dict[str, Any]]) -> int:
        count = 0
        self.data = []
        for item in data:
            self.data.append(item)
            count += 1
        return count

    def exists(self) -> bool:
        return True

    def count(self) -> int:
        return len(self.data)


class Transform(ABC):
    """Abstract base for data transformations."""

    @abstractmethod
    def transform(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single item."""
        pass

    def transform_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform a batch of items. Override for optimization."""
        return [self.transform(item) for item in items]


class FilterTransform(Transform):
    """Filter items based on condition."""

    def __init__(self, condition: Callable[[Dict[str, Any]], bool]):
        self.condition = condition

    def transform(self, item: Dict[str, Any]) -> Dict[str, Any]:
        if self.condition(item):
            return item
        raise FilteredOut()


class MapTransform(Transform):
    """Map/transform values."""

    def __init__(self, mapping: Dict[str, Callable[[Any], Any]]):
        self.mapping = mapping

    def transform(self, item: Dict[str, Any]) -> Dict[str, Any]:
        result = item.copy()
        for key, fn in self.mapping.items():
            if key in result:
                result[key] = fn(result[key])
        return result


class RenameTransform(Transform):
    """Rename columns."""

    def __init__(self, renames: Dict[str, str]):
        self.renames = renames

    def transform(self, item: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, value in item.items():
            new_key = self.renames.get(key, key)
            result[new_key] = value
        return result


class DropTransform(Transform):
    """Drop columns."""

    def __init__(self, columns: List[str]):
        self.columns = set(columns)

    def transform(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in item.items() if k not in self.columns}


class FillNaTransform(Transform):
    """Fill missing values."""

    def __init__(self, fill_values: Dict[str, Any]):
        self.fill_values = fill_values

    def transform(self, item: Dict[str, Any]) -> Dict[str, Any]:
        result = item.copy()
        for key, value in self.fill_values.items():
            if key not in result or result[key] is None:
                result[key] = value
        return result


class CompositeTransform(Transform):
    """Compose multiple transforms."""

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def transform(self, item: Dict[str, Any]) -> Dict[str, Any]:
        result = item
        for transform in self.transforms:
            result = transform.transform(result)
        return result


class FilteredOut(Exception):
    """Exception raised when item is filtered out."""

    pass


class DataPipeline:
    """
    Data pipeline for ETL operations.

    Features:
    - Multiple data sources
    - Chained transformations
    - Data validation
    - Caching
    - Streaming
    """

    def __init__(
        self,
        source: Optional[DataSource] = None,
        transforms: Optional[List[Transform]] = None,
        schema: Optional[DataSchema] = None,
        cache_dir: Optional[str] = None,
    ):
        self.source = source
        self.transforms = transforms or []
        self.schema = schema
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._stats: Optional[DataStats] = None

    def add_transform(self, transform: Transform) -> "DataPipeline":
        """Add a transformation to the pipeline."""
        self.transforms.append(transform)
        return self

    def filter(self, condition: Callable[[Dict[str, Any]], bool]) -> "DataPipeline":
        """Add a filter transform."""
        return self.add_transform(FilterTransform(condition))

    def map(self, mapping: Dict[str, Callable[[Any], Any]]) -> "DataPipeline":
        """Add a map transform."""
        return self.add_transform(MapTransform(mapping))

    def rename(self, renames: Dict[str, str]) -> "DataPipeline":
        """Add a rename transform."""
        return self.add_transform(RenameTransform(renames))

    def drop(self, columns: List[str]) -> "DataPipeline":
        """Add a drop transform."""
        return self.add_transform(DropTransform(columns))

    def fill_na(self, fill_values: Dict[str, Any]) -> "DataPipeline":
        """Add fill NA transform."""
        return self.add_transform(FillNaTransform(fill_values))

    def _get_cache_key(self) -> str:
        """Generate cache key based on pipeline config."""
        config = f"{self.source.path if self.source else 'none'}_{len(self.transforms)}"
        return hashlib.md5(config.encode()).hexdigest()

    def _load_from_cache(self) -> Optional[InMemorySource]:
        """Load cached data if available."""
        if not self.cache_dir:
            return None

        cache_file = self.cache_dir / f"{self._get_cache_key()}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    def _save_to_cache(self, data: InMemorySource):
        """Save data to cache."""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / f"{self._get_cache_key()}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")

    def run(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Run the pipeline and return data."""
        if use_cache:
            cached = self._load_from_cache()
            if cached:
                logger.info("Loaded data from cache")
                return list(cached.read())

        if not self.source:
            return []

        results = []

        for item in self.source.read():
            try:
                for transform in self.transforms:
                    item = transform.transform(item)

                if self.schema:
                    valid, errors = self.schema.validate(item)
                    if not valid:
                        logger.warning(f"Validation errors: {errors}")
                        continue

                results.append(item)
            except FilteredOut:
                continue
            except Exception as e:
                logger.warning(f"Transform error: {e}")
                continue

        if use_cache:
            self._save_to_cache(InMemorySource(results))

        return results

    def stream(self) -> Iterator[Dict[str, Any]]:
        """Stream data through pipeline."""
        if not self.source:
            return

        for item in self.source.read():
            try:
                for transform in self.transforms:
                    item = transform.transform(item)

                if self.schema:
                    valid, _ = self.schema.validate(item)
                    if not valid:
                        continue

                yield item
            except FilteredOut:
                continue

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        seed: int = 42,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split data into train/val/test sets."""
        data = self.run()

        if shuffle:
            np.random.seed(seed)
            indices = np.random.permutation(len(data))
            data = [data[i] for i in indices]

        n = len(data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train = data[:train_end]
        val = data[train_end:val_end]
        test = data[val_end:]

        return train, val, test

    def compute_stats(self) -> DataStats:
        """Compute statistics for the dataset."""
        data = self.run(use_cache=False)

        if not data:
            return DataStats(0, {}, {}, {}, time.time())

        columns = set()
        for item in data:
            columns.update(item.keys())

        column_stats = {}
        null_counts = {col: 0 for col in columns}
        unique_counts = {col: 0 for col in columns}

        for col in columns:
            values = []
            for item in data:
                if col in item:
                    values.append(item[col])

            if not values:
                continue

            null_counts[col] = sum(1 for v in values if v is None)
            unique_counts[col] = len(set(str(v) for v in values if v is not None))

            numeric_values = [v for v in values if isinstance(v, (int, float, np.number))]
            if numeric_values:
                column_stats[col] = {
                    "type": "numerical",
                    "min": float(min(numeric_values)),
                    "max": float(max(numeric_values)),
                    "mean": float(np.mean(numeric_values)),
                    "std": float(np.std(numeric_values)),
                }
            else:
                column_stats[col] = {"type": "categorical", "unique_count": unique_counts[col]}

        self._stats = DataStats(
            row_count=len(data),
            column_stats=column_stats,
            null_counts=null_counts,
            unique_counts=unique_counts,
            generated_at=time.time(),
        )

        return self._stats

    def save(self, destination: DataSource) -> int:
        """Save pipeline output to destination."""
        data = self.run()
        return destination.write(iter(data))


pipeline = DataPipeline


def from_file(path: str, format: str = "json") -> DataPipeline:
    """Create pipeline from file."""
    return DataPipeline(source=FileSource(path, format))


def from_list(data: List[Dict[str, Any]]) -> DataPipeline:
    """Create pipeline from list."""
    return DataPipeline(source=InMemorySource(data))


__all__ = [
    "DataPipeline",
    "DataSource",
    "FileSource",
    "InMemorySource",
    "DataSchema",
    "DataStats",
    "DataType",
    "PipelineStatus",
    "Transform",
    "FilterTransform",
    "MapTransform",
    "RenameTransform",
    "DropTransform",
    "FillNaTransform",
    "CompositeTransform",
    "FilteredOut",
    "pipeline",
    "from_file",
    "from_list",
]
