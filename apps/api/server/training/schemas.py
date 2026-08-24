"""Pydantic models for training HTTP API (dataset manifests, job payloads).

Optional-field defaults on ``TrainRequest`` / ``TrainingRequest`` should match the web UI
constants in ``apps/web/lib/training-defaults.ts`` (``TRAINING_API_DEFAULTS``).

Trainer ``*.soul`` files on disk include ``stoi`` / ``itos`` / ``chars`` so char-LM
eval can decode without vocab warnings; see ``docs/policies/CONTRIBUTING.md``
(*Checkpoint vocabulary*).
"""

from __future__ import annotations

from typing import List, Optional

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

    On-disk trainer ``*.soul`` bundles carry charset maps for fair ``cli.py eval``;
    vocabulary formats are documented under *Checkpoint vocabulary* in CONTRIBUTING.
    """

    epochs: Optional[int] = Field(default=3, ge=1, le=1000)
    batch_size: Optional[int] = Field(default=32, ge=1, le=1024)
    learning_rate: Optional[float] = Field(default=1e-3, gt=0, le=1.0)
    n_embed: Optional[int] = Field(default=128, ge=16, le=1024)
    n_layer: Optional[int] = Field(default=4, ge=1, le=24)
    n_head: Optional[int] = Field(default=4, ge=1, le=64)
    block_size: Optional[int] = Field(default=128, ge=8, le=2048)
    max_steps: Optional[int] = Field(default=None, ge=1)
    log_interval: int = Field(default=10, ge=1, le=50_000)
    eval_interval: int = Field(default=100, ge=1, le=1_000_000)
    dropout: float = Field(default=0.1, ge=0.0, le=0.9)
    weight_decay: float = Field(default=0.01, ge=0.0)
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=10_000)
    max_grad_norm: float = Field(default=1.0, ge=0.0)
    warmup_steps: int = Field(default=100, ge=0, le=1_000_000)
    min_lr: float = Field(default=1e-5, ge=0.0)
    scheduler: str = Field(default="cosine", pattern=r"^(cosine|linear|step|none)$")
    use_lora: bool = False
    lora_rank: int = Field(default=8, ge=1, le=256)
    lora_alpha: int = Field(default=16, ge=1, le=1024)
    checkpoint_dir: str = Field(default="checkpoints", max_length=200)
    checkpoint_interval: int = Field(default=500, ge=1, le=1_000_000)
    save_best_only: bool = False
    max_checkpoints: int = Field(default=5, ge=1, le=100)
    device: Optional[str] = None
    use_compile: bool = False

    @field_validator("device", mode="before")
    @classmethod
    def _normalize_device(cls, value: object) -> Optional[str]:
        if value is None:
            return None
        s = str(value).strip()
        return s if s else None


class TrainRequest(TrainDataSourceBody, _TrainHyperparameters):
    """Train char-level model from ``datasets/<name>/input.txt`` or from a v1 manifest.

    Checkpoints under ``checkpoint_dir`` include charset maps on native ``*.soul``;
    see *Checkpoint vocabulary* in CONTRIBUTING.
    """

    pass


class TrainResolveRequest(TrainDataSourceBody):
    """Preview resolved training file path without starting a job.

    Training endpoints (not this one) write ``*.soul`` with ``stoi`` / ``itos`` /
    ``chars``; see *Checkpoint vocabulary* in CONTRIBUTING.
    """

    pass


class TrainingRequest(TrainDataSourceBody, _TrainHyperparameters):
    """UI/orchestrator training job (metadata + same corpus selectors as ``TrainRequest``).

    Tracked runs use the same trainer; ``*.soul`` on disk embed ``stoi`` / ``itos``
    / ``chars``. See *Checkpoint vocabulary* in CONTRIBUTING.
    """

    name: str
    model: str


class DistillStartRequest(BaseModel):
    """Knowledge distillation: teach a compact student from a larger teacher model.

    The teacher is loaded from the HF model registry (must be loaded in server).
    The student is created as a SloNet LSTM with the given architecture.
    """
    teacher_model: str = Field(default="gpt2", description="Teacher model ID registered in the model server")
    dataset: str = ""
    name: str = "distill-job"
    temperature: float = 4.0
    alpha: float = 0.5
    beta: float = 0.5
    epochs: int = 10
    embed_dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    block_size: int = 64


class LoraFinetuneRequest(BaseModel):
    """LoRA fine-tuning on .slnc models using SloNet numpy autograd (no PyTorch).

    Trains low-rank adapters on top of any model loaded via SloNetChatProvider.
    Adapter is saved as a .npz file alongside the base model.

    Fields map 1:1 to ``domains.training.hf_lora_finetune.HFLoraConfig``.
    """
    model_path: str = Field(description="Path to .slnc model file")
    dataset: str = Field(description="Dataset folder under datasets/")
    name: Optional[str] = None
    rank: int = Field(default=8, ge=1, le=256)
    alpha: float = Field(default=16.0, ge=0.1)
    dropout: float = Field(default=0.0, ge=0.0, le=0.5)
    target_modules: Optional[List[str]] = Field(default=None, description="Modules to apply LoRA to (default: W_q, W_k, W_v, W_o)")
    epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=8, ge=1, le=256)
    block_size: int = Field(default=128, ge=16, le=2048, description="Max sequence length for tokenization")
    learning_rate: float = Field(default=1e-4, gt=0)
    warmup_steps: int = Field(default=10, ge=0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    grad_clip: float = Field(default=1.0, ge=0.0)
    grad_accumulation_steps: int = Field(default=1, ge=1)
    log_interval: int = Field(default=10, ge=1)
    output_dir: str = Field(default="models", description="Directory to save adapter")
    adapter_name: Optional[str] = Field(default=None, description="Custom adapter name (auto-generated if None)")
