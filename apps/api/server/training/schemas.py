"""Pydantic models for training HTTP API (dataset manifests, job payloads).

Optional-field defaults on ``TrainRequest`` / ``TrainingRequest`` should match the web UI
constants in ``apps/web/lib/training-defaults.ts`` (``TRAINING_API_DEFAULTS``).

Trainer ``*.soul`` files on disk include ``stoi`` / ``itos`` / ``chars`` so char-LM
eval can decode without vocab warnings; see ``docs/policies/CONTRIBUTING.md``
(*Checkpoint vocabulary*).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class TrainDatasetRef(BaseModel):
    source: str | None = None
    path: str | None = None
    manifest_uri: str | None = None
    dataset_id: str | None = None
    version: str | None = None


class TrainDataSourceBody(BaseModel):
    dataset: str | None = None
    manifest_uri: str | None = None
    dataset_ref: TrainDatasetRef | None = None

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

    epochs: int | None = Field(default=3, ge=1, le=1000)
    batch_size: int | None = Field(default=32, ge=1, le=1024)
    learning_rate: float | None = Field(default=1e-3, gt=0, le=1.0)
    n_embed: int | None = Field(default=128, ge=16, le=1024)
    n_layer: int | None = Field(default=4, ge=1, le=24)
    n_head: int | None = Field(default=4, ge=1, le=64)
    block_size: int | None = Field(default=128, ge=8, le=2048)
    max_steps: int | None = Field(default=None, ge=1)
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
    device: str | None = None
    use_compile: bool = False

    @field_validator("device", mode="before")
    @classmethod
    def _normalize_device(cls, value: object) -> str | None:
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

    name: str = Field(..., min_length=1, max_length=200)
    model: str = Field(..., min_length=1, max_length=200)


class DistillStartRequest(BaseModel):
    """Knowledge distillation: teach a compact student from a larger teacher model.

    The teacher is loaded from the HF model registry (must be loaded in server).
    The student is created as a SloNet LSTM with the given architecture.
    """

    teacher_model: str = Field(
        default="gpt2", description="Teacher model ID registered in the model server"
    )
    dataset: str = Field(..., min_length=1, max_length=200)
    name: str = Field(default="distill-job", min_length=1, max_length=200)
    temperature: float = Field(default=4.0, ge=0.1, le=20.0)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    beta: float = Field(default=0.5, ge=0.0, le=1.0)
    epochs: int = Field(default=10, ge=1, le=1000)
    embed_dim: int = Field(default=64, ge=16, le=1024)
    n_layers: int = Field(default=2, ge=1, le=24)
    n_heads: int = Field(default=4, ge=1, le=64)
    block_size: int = Field(default=64, ge=8, le=2048)


class LoraFinetuneRequest(BaseModel):
    """LoRA fine-tuning on .slnc models using SloNet numpy autograd (no PyTorch).

    Trains low-rank adapters on top of any model loaded via SloNetChatProvider.
    Adapter is saved as a .npz file alongside the base model.

    Fields map 1:1 to ``domains.training.hf_lora_finetune.HFLoraConfig``.
    """

    model_path: str = Field(description="Path to .slnc model file")
    dataset: str = Field(description="Dataset folder under datasets/")
    name: str | None = None
    rank: int = Field(default=8, ge=1, le=256)
    alpha: float = Field(default=16.0, ge=0.1)
    dropout: float = Field(default=0.0, ge=0.0, le=0.5)
    target_modules: list[str] | None = Field(
        default=None, description="Modules to apply LoRA to (default: W_q, W_k, W_v, W_o)"
    )
    epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=8, ge=1, le=256)
    block_size: int = Field(
        default=128, ge=16, le=2048, description="Max sequence length for tokenization"
    )
    learning_rate: float = Field(default=1e-4, gt=0)
    warmup_steps: int = Field(default=10, ge=0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    grad_clip: float = Field(default=1.0, ge=0.0)
    grad_accumulation_steps: int = Field(default=1, ge=1)
    log_interval: int = Field(default=10, ge=1)
    output_dir: str = Field(default="models", description="Directory to save adapter")
    adapter_name: str | None = Field(
        default=None, description="Custom adapter name (auto-generated if None)"
    )


class VisualTrainRequest(BaseModel):
    """VLM fine-tune on an image-caption dataset."""

    dataset: str
    vision_encoder: str = "slonet"
    llm: str = "gpt2"
    connector_hidden_dim: int = 512
    max_seq_length: int = 128
    stage1_epochs: int = 5
    stage2_epochs: int = 10
    stage1_lr: float = 5e-4
    stage2_lr: float = 2e-4
    batch_size: int = 4
    use_lora: bool = False
    lora_rank: int = 8
    freeze_vision: bool = True
    name: str | None = None


class LoadAdapterRequest(BaseModel):
    """Request to load a LoRA adapter into the running model."""

    adapter_path: str = Field(description="Path to .npz adapter file")
    merge: bool = Field(
        default=False, description="Merge LoRA into base weights for faster inference"
    )


class FromSessionsRequest(BaseModel):
    """Request body for from-sessions training."""

    epochs: int = 5
    learning_rate: float = 3e-4
    batch_size: int = 8
    n_embed: int = 128
    n_layer: int = 4
    n_head: int = 4
    block_size: int = 128
    dropout: float = 0.1
    soul_name: str | None = None
    min_pair_quality: float = 2.0
    max_pairs: int = 500
    checkpoint_name: str | None = None
    session_ids: list[str] | None = None
    experiment_id: str | None = None


class TurboStartRequest(BaseModel):
    """Request body for turbo training."""

    dataset_id: str | None = None
    checkpoint_name: str | None = None
    soul_name: str | None = None
    source_text: str | None = None
    epochs: int = 3
    learning_rate: float = 3e-4
    batch_size: int = 8
    n_embed: int = 128
    n_layer: int = 4
    n_head: int = 4
    block_size: int = 128
    use_lora: bool = False
    lora_rank: int = 8
    experiment_id: str | None = None


class ExportTextRequest(BaseModel):
    """Request body for exporting checkpoint as text."""

    checkpoint: str = Field(description="Checkpoint name")
    max_tokens: int = Field(default=1000, ge=1, le=100000)


class TestWebhookRequest(BaseModel):
    """Request body for testing a webhook."""

    url: str = Field(description="Webhook URL to test")
    secret: str | None = Field(default=None, description="Optional HMAC secret")
