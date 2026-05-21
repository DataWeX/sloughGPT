"""
Feature Store - ML Feature Management

Production-grade feature store with:
- Feature definitions and versioning
- Feature groups
- Online/offline feature serving
- Feature validation
- Feature monitoring
"""

import json
import time
import hashlib
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import numpy as np

logger = logging.getLogger("sloughgpt.features")


class FeatureType(Enum):
    """Feature data types."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"
    EMBEDDING = "embedding"
    TIMESTAMP = "timestamp"


class FeatureStatus(Enum):
    """Feature registration status."""

    EXPERIMENTAL = "experimental"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


@dataclass
class FeatureStats:
    """Statistical summary of a feature."""

    count: int
    null_count: int
    unique_count: int
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    quantiles: Dict[str, float] = field(default_factory=dict)
    histogram: List[int] = field(default_factory=list)
    top_values: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureStats":
        return cls(**data)


@dataclass
class Feature:
    """Feature definition."""

    feature_id: str
    name: str
    feature_group: str
    feature_type: FeatureType
    description: str
    default_value: Any
    transformation: str
    status: FeatureStatus
    version: str
    created_at: float
    updated_at: float
    tags: Dict[str, str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["feature_type"] = self.feature_type.value
        data["status"] = self.status.value
        return data


@dataclass
class FeatureGroup:
    """Group of related features."""

    group_id: str
    name: str
    description: str
    features: List[str]
    version: str
    created_at: float
    updated_at: float
    tags: Dict[str, str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureVector:
    """A feature vector for inference."""

    features: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeatureValidator:
    """Validates feature values against definitions."""

    @staticmethod
    def validate_numerical(
        value: Any, min_val: Optional[float] = None, max_val: Optional[float] = None
    ) -> bool:
        try:
            num = float(value)
            if min_val is not None and num < min_val:
                return False
            if max_val is not None and num > max_val:
                return False
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_categorical(value: Any, allowed_values: Set[str]) -> bool:
        return str(value) in allowed_values

    @staticmethod
    def validate_boolean(value: Any) -> bool:
        return isinstance(value, bool) or str(value).lower() in ("true", "false", "1", "0")

    @staticmethod
    def validate_embedding(value: Any, expected_dim: int) -> bool:
        try:
            arr = np.array(value)
            return arr.shape == (expected_dim,)
        except Exception:
            return False


class FeatureStore:
    """
    Feature store for ML feature management.

    Features:
    - Feature definitions with versioning
    - Feature groups for related features
    - Online feature serving
    - Feature statistics
    - Feature validation
    - Feature monitoring
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, storage_path: str = "data/features"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, storage_path: str = "data/features"):
        if self._initialized:
            return

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.features: Dict[str, Feature] = {}
        self.feature_groups: Dict[str, FeatureGroup] = {}
        self.feature_values: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.feature_stats: Dict[str, FeatureStats] = {}

        self._lock = threading.Lock()
        self._validators = {
            FeatureType.NUMERICAL: FeatureValidator.validate_numerical,
            FeatureType.CATEGORICAL: FeatureValidator.validate_categorical,
            FeatureType.BOOLEAN: FeatureValidator.validate_boolean,
            FeatureType.EMBEDDING: FeatureValidator.validate_embedding,
        }

        self._load_features()
        self._initialized = True
        logger.info(f"FeatureStore initialized at {self.storage_path}")

    def _load_features(self):
        """Load features from storage."""
        features_file = self.storage_path / "features.json"
        groups_file = self.storage_path / "feature_groups.json"

        if features_file.exists():
            try:
                with open(features_file) as f:
                    data = json.load(f)
                    for feat_data in data.values():
                        feat_data["feature_type"] = FeatureType(feat_data["feature_type"])
                        feat_data["status"] = FeatureStatus(feat_data["status"])
                        self.features[feat_data["feature_id"]] = Feature(**feat_data)
            except Exception as e:
                logger.warning(f"Failed to load features: {e}")

        if groups_file.exists():
            try:
                with open(groups_file) as f:
                    data = json.load(f)
                    for group_data in data.values():
                        self.feature_groups[group_data["group_id"]] = FeatureGroup(**group_data)
            except Exception as e:
                logger.warning(f"Failed to load feature groups: {e}")

    def _save_features(self):
        """Persist features to storage."""
        features_file = self.storage_path / "features.json"
        try:
            with open(features_file, "w") as f:
                data = {k: v.to_dict() for k, v in self.features.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save features: {e}")

    def _save_feature_groups(self):
        """Persist feature groups to storage."""
        groups_file = self.storage_path / "feature_groups.json"
        try:
            with open(groups_file, "w") as f:
                data = {k: v.to_dict() for k, v in self.feature_groups.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save feature groups: {e}")

    def register_feature(
        self,
        name: str,
        feature_group: str,
        feature_type: FeatureType,
        description: str = "",
        default_value: Any = None,
        transformation: str = "identity",
        status: FeatureStatus = FeatureStatus.EXPERIMENTAL,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register a new feature."""
        feature_id = hashlib.md5(f"{feature_group}:{name}".encode()).hexdigest()[:12]

        if feature_id in self.features:
            raise ValueError(f"Feature '{feature_group}.{name}' already exists")

        existing_in_group = [f for f in self.features.values() if f.feature_group == feature_group]
        version = f"v1.0.{len(existing_in_group)}"

        feature = Feature(
            feature_id=feature_id,
            name=name,
            feature_group=feature_group,
            feature_type=feature_type,
            description=description,
            default_value=default_value,
            transformation=transformation,
            status=status,
            version=version,
            created_at=time.time(),
            updated_at=time.time(),
            tags=tags or {},
            metadata=metadata or {},
        )

        with self._lock:
            self.features[feature_id] = feature
            self._save_features()

        logger.info(f"Registered feature: {feature_group}.{name}")
        return feature_id

    def register_feature_group(
        self,
        name: str,
        features: List[str],
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register a feature group."""
        group_id = hashlib.md5(name.encode()).hexdigest()[:12]

        if group_id in self.feature_groups:
            raise ValueError(f"Feature group '{name}' already exists")

        version = "v1.0"

        group = FeatureGroup(
            group_id=group_id,
            name=name,
            description=description,
            features=features,
            version=version,
            created_at=time.time(),
            updated_at=time.time(),
            tags=tags or {},
            metadata=metadata or {},
        )

        with self._lock:
            self.feature_groups[group_id] = group
            self._save_feature_groups()

        logger.info(f"Registered feature group: {name}")
        return group_id

    def get_feature(self, feature_id: str) -> Optional[Feature]:
        return self.features.get(feature_id)

    def get_feature_by_name(self, feature_group: str, name: str) -> Optional[Feature]:
        for feat in self.features.values():
            if feat.feature_group == feature_group and feat.name == name:
                return feat
        return None

    def get_feature_group(self, group_id: str) -> Optional[FeatureGroup]:
        return self.feature_groups.get(group_id)

    def get_features_by_group(self, feature_group: str) -> List[Feature]:
        return [f for f in self.features.values() if f.feature_group == feature_group]

    def get_feature_vector(
        self, feature_group: str, entity_id: str, features: Optional[List[str]] = None
    ) -> Optional[FeatureVector]:
        """Get feature vector for an entity."""
        key = f"{feature_group}:{entity_id}"

        if key not in self.feature_values:
            return None

        all_features = self.feature_values[key]

        if features:
            all_features = {k: v for k, v in all_features.items() if k in features}

        return FeatureVector(
            features=all_features,
            metadata={"entity_id": entity_id, "feature_group": feature_group},
            timestamp=time.time(),
        )

    def set_feature_value(
        self,
        feature_group: str,
        entity_id: str,
        feature_name: str,
        value: Any,
        validate: bool = True,
    ) -> bool:
        """Set feature value for an entity."""
        feature = self.get_feature_by_name(feature_group, feature_name)
        if not feature:
            logger.warning(f"Feature not found: {feature_group}.{feature_name}")
            return False

        if validate and feature.feature_type in self._validators:
            validator = self._validators[feature.feature_type]

            if feature.feature_type == FeatureType.CATEGORICAL:
                allowed = feature.metadata.get("allowed_values", set())
                if allowed and not validator(value, allowed):
                    raise ValueError(f"Invalid value for {feature_name}: {value}")
            elif feature.feature_type == FeatureType.EMBEDDING:
                dim = feature.metadata.get("dimension", 128)
                if not validator(value, dim):
                    raise ValueError(f"Invalid embedding dimension for {feature_name}")
            else:
                min_val = feature.metadata.get("min_value")
                max_val = feature.metadata.get("max_value")
                if not validator(value, min_val, max_val):
                    raise ValueError(f"Invalid value for {feature_name}: {value}")

        key = f"{feature_group}:{entity_id}"

        with self._lock:
            self.feature_values[key][feature_name] = value

        return True

    def compute_stats(self, feature_group: str, feature_name: str) -> FeatureStats:
        """Compute statistics for a feature."""
        feature = self.get_feature_by_name(feature_group, feature_name)
        if not feature:
            raise ValueError(f"Feature not found: {feature_group}.{feature_name}")

        values = []
        for entity_features in self.feature_values.values():
            if feature_name in entity_features:
                values.append(entity_features[feature_name])

        if not values:
            return FeatureStats(count=0, null_count=0, unique_count=0)

        null_count = sum(1 for v in values if v is None)
        non_null_values = [v for v in values if v is not None]

        stats = FeatureStats(
            count=len(values),
            null_count=null_count,
            unique_count=len(set(str(v) for v in non_null_values)),
        )

        if feature.feature_type == FeatureType.NUMERICAL and non_null_values:
            try:
                numeric_values = [float(v) for v in non_null_values]
                stats.min_value = min(numeric_values)
                stats.max_value = max(numeric_values)
                stats.mean = sum(numeric_values) / len(numeric_values)
                stats.std = float(np.std(numeric_values))

                quantiles = [0.25, 0.5, 0.75, 0.95, 0.99]
                sorted_vals = sorted(numeric_values)
                for q in quantiles:
                    idx = int(len(sorted_vals) * q)
                    stats.quantiles[f"q{int(q * 100)}"] = sorted_vals[
                        min(idx, len(sorted_vals) - 1)
                    ]

                hist, _ = np.histogram(numeric_values, bins=10)
                stats.histogram = hist.tolist()
            except (ValueError, TypeError):
                pass

        if feature.feature_type == FeatureType.CATEGORICAL and non_null_values:
            from collections import Counter

            counter = Counter(str(v) for v in non_null_values)
            stats.top_values = [{"value": v, "count": c} for v, c in counter.most_common(10)]

        with self._lock:
            self.feature_stats[f"{feature_group}.{feature_name}"] = stats

        return stats

    def list_features(
        self,
        feature_group: Optional[str] = None,
        feature_type: Optional[FeatureType] = None,
        status: Optional[FeatureStatus] = None,
    ) -> List[Feature]:
        """List features with optional filtering."""
        results = list(self.features.values())

        if feature_group:
            results = [f for f in results if f.feature_group == feature_group]

        if feature_type:
            results = [f for f in results if f.feature_type == feature_type]

        if status:
            results = [f for f in results if f.status == status]

        return sorted(results, key=lambda f: f.name)

    def list_feature_groups(self) -> List[FeatureGroup]:
        return sorted(self.feature_groups.values(), key=lambda g: g.name)


feature_store = FeatureStore()


def register_feature(
    name: str,
    feature_group: str,
    feature_type: FeatureType,
    description: str = "",
    default_value: Any = None,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    return feature_store.register_feature(
        name, feature_group, feature_type, description, default_value, tags=tags
    )


def register_feature_group(
    name: str, features: List[str], description: str = "", tags: Optional[Dict[str, str]] = None
) -> str:
    return feature_store.register_feature_group(name, features, description, tags=tags)


def get_feature(feature_group: str, name: str) -> Optional[Feature]:
    return feature_store.get_feature_by_name(feature_group, name)


def get_feature_vector(feature_group: str, entity_id: str) -> Optional[FeatureVector]:
    return feature_store.get_feature_vector(feature_group, entity_id)


__all__ = [
    "FeatureStore",
    "Feature",
    "FeatureGroup",
    "FeatureVector",
    "FeatureStats",
    "FeatureType",
    "FeatureStatus",
    "FeatureValidator",
    "feature_store",
    "register_feature",
    "register_feature_group",
    "get_feature",
    "get_feature_vector",
]
