"""
Model Versioning - ML Model Lifecycle Management

Production-grade model versioning system with:
- Semantic versioning
- Model registry with stages (staging, production, archived)
- Model lineage tracking
- A/B testing support
- Rollback capabilities
"""

import json
import time
import uuid
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading

logger = logging.getLogger("sloughgpt.model_registry")


class ModelStage(Enum):
    """Model deployment stage."""

    NONE = "none"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DELETED = "deleted"


class ModelStatus(Enum):
    """Model registration status."""

    PENDING = "pending"
    VALIDATING = "validating"
    VALIDATED = "validated"
    FAILED = "failed"


@dataclass
class ModelMetrics:
    """Model performance metrics."""

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    loss: Optional[float] = None
    perplexity: Optional[float] = None
    latency_ms: Optional[float] = None
    throughput: Optional[float] = None
    custom: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetrics":
        return cls(
            **{k: v for k, v in data.items() if k != "custom"}, custom=data.get("custom", {})
        )


@dataclass
class ModelVersion:
    """A single model version."""

    version_id: str
    model_name: str
    version: str
    description: str
    stage: ModelStage
    status: ModelStatus
    metrics: ModelMetrics
    parameters: Dict[str, Any]
    artifacts: Dict[str, str]
    metadata: Dict[str, Any]
    parent_version: Optional[str]
    created_at: float
    updated_at: float
    created_by: str
    tags: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["stage"] = self.stage.value
        data["status"] = self.status.value
        data["metrics"] = self.metrics.to_dict()
        return data


@dataclass
class Model:
    """Model entity containing all versions."""

    model_id: str
    name: str
    description: str
    model_type: str
    versions: List[ModelVersion]
    current_stage: ModelStage
    tags: Dict[str, str]
    created_at: float
    updated_at: float

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["current_stage"] = self.current_stage.value
        data["versions"] = [v.to_dict() for v in self.versions]
        return data

    def get_version(self, version: str) -> Optional[ModelVersion]:
        for v in self.versions:
            if v.version == version:
                return v
        return None

    def get_latest_version(self) -> Optional[ModelVersion]:
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.created_at)

    def get_by_stage(self, stage: ModelStage) -> List[ModelVersion]:
        return [v for v in self.versions if v.stage == stage]


class ModelVersioning:
    """
    Model versioning and registry system.

    Features:
    - Semantic versioning
    - Stage management (staging, production, archived)
    - Metrics tracking
    - Lineage tracking
    - A/B testing support
    - Rollback capabilities
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, storage_path: str = "./models/registry"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, storage_path: str = "./models/registry"):
        if self._initialized:
            return

        self.storage_path = Path(storage_path)
        self.models_path = self.storage_path / "models"
        self.artifacts_path = self.storage_path / "artifacts"

        self.models_path.mkdir(parents=True, exist_ok=True)
        self.artifacts_path.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, Model] = {}
        self._lock = threading.Lock()

        self._load_models()
        self._initialized = True
        logger.info(f"ModelVersioning initialized at {self.storage_path}")

    def _load_models(self):
        """Load models from storage."""
        models_file = self.storage_path / "models.json"
        if models_file.exists():
            try:
                with open(models_file) as f:
                    data = json.load(f)
                    for model_data in data.values():
                        model_data["versions"] = [
                            ModelVersion(
                                **{
                                    **v,
                                    "metrics": ModelMetrics.from_dict(v["metrics"]),
                                    "stage": ModelStage(v["stage"]),
                                    "status": ModelStatus(v["status"]),
                                }
                            )
                            for v in model_data.get("versions", [])
                        ]
                        model_data["current_stage"] = ModelStage(model_data["current_stage"])
                        self.models[model_data["model_id"]] = Model(**model_data)
            except Exception as e:
                logger.warning(f"Failed to load models: {e}")

    def _save_models(self):
        """Persist models to storage."""
        models_file = self.storage_path / "models.json"
        try:
            with open(models_file, "w") as f:
                data = {k: v.to_dict() for k, v in self.models.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    def _generate_version_id(self) -> str:
        return str(uuid.uuid4())[:12]

    def _generate_semver(self, model_name: str) -> str:
        """Generate semantic version (major.minor.patch)."""
        versions = [
            v.version
            for v in self.models.get(
                model_name, Model("", "", "", "", [], ModelStage.NONE, {}, 0, 0)
            ).versions
            if v.version.startswith("v")
        ]

        if not versions:
            return "v0.1.0"

        latest = max(versions, key=lambda v: [int(x) for x in v[1:].split(".")])
        parts = [int(x) for x in latest[1:].split(".")]

        return f"v{parts[0]}.{parts[1]}.{parts[2] + 1}"

    def register_model(
        self,
        name: str,
        model_type: str,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        created_by: str = "system",
    ) -> str:
        """Register a new model."""
        model_id = hashlib.md5(name.encode()).hexdigest()[:12]

        if model_id in self.models:
            raise ValueError(f"Model '{name}' already exists")

        model = Model(
            model_id=model_id,
            name=name,
            description=description,
            model_type=model_type,
            versions=[],
            current_stage=ModelStage.NONE,
            tags=tags or {},
            created_at=time.time(),
            updated_at=time.time(),
        )

        with self._lock:
            self.models[model_id] = model
            self._save_models()

        logger.info(f"Registered model: {name} ({model_id})")
        return model_id

    def create_version(
        self,
        model_name: str,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        created_by: str = "system",
    ) -> Optional[ModelVersion]:
        """Create a new version of an existing model."""
        model = self.get_model_by_name(model_name)
        if not model:
            return None

        version_id = self._generate_version_id()
        version_str = self._generate_semver(model_name)

        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            version=version_str,
            description=description,
            stage=ModelStage.NONE,
            status=ModelStatus.PENDING,
            metrics=ModelMetrics(),
            parameters=parameters or {},
            artifacts={},
            metadata=metadata or {},
            parent_version=model.get_latest_version().version_id if model.versions else None,
            created_at=time.time(),
            updated_at=time.time(),
            created_by=created_by,
            tags=tags or {},
        )

        with self._lock:
            model.versions.append(version)
            model.updated_at = time.time()
            self._save_models()

        logger.info(f"Created version {version_str} for model {model_name}")
        return version

    def update_version_metrics(self, model_name: str, version: str, metrics: ModelMetrics) -> bool:
        """Update metrics for a model version."""
        model = self.get_model_by_name(model_name)
        if not model:
            return False

        model_version = model.get_version(version)
        if not model_version:
            return False

        with self._lock:
            model_version.metrics = metrics
            model_version.updated_at = time.time()
            model.updated_at = time.time()
            self._save_models()

        return True

    def add_artifact(
        self, model_name: str, version: str, artifact_name: str, artifact_path: str
    ) -> bool:
        """Add an artifact to a model version."""
        model = self.get_model_by_name(model_name)
        if not model:
            return False

        model_version = model.get_version(version)
        if not model_version:
            return False

        artifact_hash = hashlib.md5(artifact_path.encode()).hexdigest()[:12]
        dest_path = self.artifacts_path / artifact_hash

        try:
            shutil.copy2(artifact_path, dest_path)
        except FileNotFoundError:
            pass

        with self._lock:
            model_version.artifacts[artifact_name] = str(dest_path)
            model_version.updated_at = time.time()
            self._save_models()

        return True

    def transition_stage(
        self, model_name: str, version: str, stage: ModelStage, target_version: Optional[str] = None
    ) -> bool:
        """Transition model version to a different stage."""
        model = self.get_model_by_name(model_name)
        if not model:
            return False

        model_version = model.get_version(version)
        if not model_version:
            return False

        with self._lock:
            if stage == ModelStage.PRODUCTION:
                for v in model.versions:
                    if v.stage == ModelStage.PRODUCTION and v.version != version:
                        v.stage = ModelStage.STAGING

            model_version.stage = stage
            model_version.updated_at = time.time()
            model.current_stage = stage
            model.updated_at = time.time()
            self._save_models()

        logger.info(f"Transitioned {model_name}:{version} to {stage.value}")
        return True

    def get_model(self, model_id: str) -> Optional[Model]:
        return self.models.get(model_id)

    def get_model_by_name(self, name: str) -> Optional[Model]:
        for model in self.models.values():
            if model.name == name:
                return model
        return None

    def list_models(
        self,
        model_type: Optional[str] = None,
        stage: Optional[ModelStage] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[Model]:
        """List models with optional filtering."""
        results = list(self.models.values())

        if model_type:
            results = [m for m in results if m.model_type == model_type]

        if stage:
            results = [m for m in results if m.current_stage == stage]

        if tags:
            results = [m for m in results if all(m.tags.get(k) == v for k, v in tags.items())]

        return sorted(results, key=lambda m: m.updated_at, reverse=True)

    def get_versions(
        self,
        model_name: str,
        stage: Optional[ModelStage] = None,
        status: Optional[ModelStatus] = None,
    ) -> List[ModelVersion]:
        """Get versions of a model with optional filtering."""
        model = self.get_model_by_name(model_name)
        if not model:
            return []

        results = model.versions

        if stage:
            results = [v for v in results if v.stage == stage]

        if status:
            results = [v for v in results if v.status == status]

        return sorted(results, key=lambda v: v.created_at, reverse=True)

    def rollback(self, model_name: str, target_version: str) -> bool:
        """Rollback to a previous version."""
        model = self.get_model_by_name(model_name)
        if not model:
            return False

        target = model.get_version(target_version)
        if not target:
            return False

        return self.transition_stage(model_name, target_version, ModelStage.PRODUCTION)

    def archive_model(self, model_name: str, version: str) -> bool:
        """Archive a model version."""
        return self.transition_stage(model_name, version, ModelStage.ARCHIVED)

    def delete_model(self, model_name: str) -> bool:
        """Delete a model (soft delete - marks as deleted)."""
        model = self.get_model_by_name(model_name)
        if not model:
            return False

        with self._lock:
            for version in model.versions:
                version.stage = ModelStage.DELETED
            model.current_stage = ModelStage.DELETED
            self._save_models()

        return True


model_registry = ModelVersioning()


def register_model(
    name: str,
    model_type: str,
    description: str = "",
    parameters: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    return model_registry.register_model(name, model_type, description, parameters, tags=tags)


def create_model_version(
    model_name: str,
    description: str = "",
    parameters: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Optional[ModelVersion]:
    return model_registry.create_version(model_name, description, parameters, tags=tags)


def get_model(name: str) -> Optional[Model]:
    return model_registry.get_model_by_name(name)


def list_models(
    model_type: Optional[str] = None, stage: Optional[ModelStage] = None
) -> List[Model]:
    return model_registry.list_models(model_type, stage)


__all__ = [
    "ModelVersioning",
    "Model",
    "ModelVersion",
    "ModelMetrics",
    "ModelStage",
    "ModelStatus",
    "model_registry",
    "register_model",
    "create_model_version",
    "get_model",
    "list_models",
]
