"""Pydantic models for training HTTP API (dataset manifests, job payloads).

Optional-field defaults on ``TrainRequest`` / ``TrainingRequest`` should match the web UI
constants in ``apps/web/lib/training-defaults.ts`` (``TRAINING_API_DEFAULTS``).

Trainer ``step_*.pt`` files on disk include ``stoi`` / ``itos`` / ``chars`` so char-LM
eval can decode without vocab warnings; see ``docs/policies/CONTRIBUTING.md``
(*Checkpoint vocabulary*).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class TrainDatasetRef(BaseModel):
    source: Optional[str] = None
    path: Optional[str] = None
    manifest_uri: Optional[str] = None
    dataset_id: Optional[str] = None
    version: Optional[str] = None


class TrainDataSourceBody(BaseModel):
    dataset: Optional[str] = None
    manifest_uri: Optional[str] = None
    dataset_ref: Optional[TrainDatasetRef] = None

    @model_validator(mode="after")
    def _exactly_one_dataset_source(self) -> TrainDataSourceBody:
        has_d = self.dataset is not None and str(self.dataset).strip() != ""
        has_m = self.manifest_uri is not None and str(self.manifest_uri).strip() != ""
        has_r = self.dataset_ref is not None
        if sum(bool(x) for x in (has_d, has_m, has_r)) != 1:
            raise ValueError(
                "Specify exactly one of: `dataset` (folder under datasets/), "
                "`manifest_uri`, or `dataset_ref`."
            )
        return self


class _TrainHyperparameters(BaseModel):
    """SloughGPTTrainer keyword arguments shared by ``TrainRequest`` and ``TrainingRequest``.

    On-disk ``checkpoint_step_*.pt`` bundles carry charset maps for fair ``cli.py eval``;
    vocabulary formats are documented under *Checkpoint vocabulary* in CONTRIBUTING.
    """

    epochs: Optional[int] = 3
    batch_size: Optional[int] = 32
    learning_rate: Optional[float] = 1e-3
    n_embed: Optional[int] = 128
    n_layer: Optional[int] = 4
    n_head: Optional[int] = 4
    block_size: Optional[int] = 128
    max_steps: Optional[int] = None
    log_interval: int = Field(default=10, ge=1, le=50_000)
    eval_interval: int = Field(default=100, ge=1, le=1_000_000)
    dropout: float = Field(default=0.1, ge=0.0, le=0.9)
    weight_decay: float = Field(default=0.01, ge=0.0)
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=10_000)
    max_grad_norm: float = Field(default=1.0, ge=0.0)
    use_mixed_precision: bool = True
    mixed_precision_dtype: Literal["bf16", "fp16"] = "bf16"
    warmup_steps: int = Field(default=100, ge=0, le=1_000_000)
    min_lr: float = Field(default=1e-5, ge=0.0)
    scheduler: str = "cosine"
    use_lora: bool = False
    lora_rank: int = Field(default=8, ge=1, le=256)
    lora_alpha: int = Field(default=16, ge=1, le=1024)
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = Field(default=500, ge=1, le=1_000_000)
    save_best_only: bool = False
    max_checkpoints: int = Field(default=5, ge=1, le=100)
    device: Optional[str] = None
    use_compile: bool = False

    @field_validator("mixed_precision_dtype", mode="before")
    @classmethod
    def _normalize_mixed_precision_dtype(cls, value: object) -> str:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return "bf16"
        s = str(value).strip().lower()
        if s in ("bf16", "fp16"):
            return s
        raise ValueError("mixed_precision_dtype must be 'bf16' or 'fp16'")

    @field_validator("device", mode="before")
    @classmethod
    def _normalize_device(cls, value: object) -> Optional[str]:
        if value is None:
            return None
        s = str(value).strip()
        return s if s else None


class TrainRequest(TrainDataSourceBody, _TrainHyperparameters):
    """Train char-level model from ``datasets/<name>/input.txt`` or from a v1 manifest.

    Checkpoints under ``checkpoint_dir`` include charset maps on native ``step_*.pt``;
    see *Checkpoint vocabulary* in CONTRIBUTING.
    """

    pass


class TrainResolveRequest(TrainDataSourceBody):
    """Preview resolved training file path without starting a job.

    Training endpoints (not this one) write ``step_*.pt`` with ``stoi`` / ``itos`` /
    ``chars``; see *Checkpoint vocabulary* in CONTRIBUTING.
    """

    pass


class TrainingRequest(TrainDataSourceBody, _TrainHyperparameters):
    """UI/orchestrator training job (metadata + same corpus selectors as ``TrainRequest``).

    Tracked runs use the same trainer; ``step_*.pt`` on disk embed ``stoi`` / ``itos``
    / ``chars``. See *Checkpoint vocabulary* in CONTRIBUTING.
    """

    name: str
    model: str
