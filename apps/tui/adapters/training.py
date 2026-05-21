"""Local and HTTP training adapters for TUI Phase 2."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from datetime import datetime

import httpx


@dataclass(frozen=True)
class TrainConfig:
    """Canonical training kwargs for LocalTrainAdapter (mirrors CLI train --help)."""

    dataset: str = "shakespeare"
    epochs: int = 3
    batch_size: int = 32
    max_steps: Optional[int] = None
    learning_rate: float = 1e-4
    block_size: int = 128
    n_embed: int = 384
    n_layer: int = 6
    n_head: int = 6
    dropout: float = 0.1
    max_grad_norm: float = 1.0
    log_interval: int = 10
    eval_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 500
    resume_latest: bool = False
    resume_path: Optional[str] = None
    save_stem: Optional[str] = None


@dataclass
class TrainProgress:
    """Streaming progress update from trainer."""

    step: int
    total_steps: Optional[int]
    loss: float
    elapsed_seconds: float
    eta_seconds: Optional[float] = None


@dataclass
class TrainResult:
    """Result after training completes."""

    success: bool
    save_path: Optional[str] = None
    error: Optional[str] = None
    final_step: int = 0


class LocalTrainAdapter:
    """Wraps ``SloughGPTTrainer`` for local training (no API required)."""

    def __init__(self, config: TrainConfig, repo_root: Path):
        self.config = config
        self.repo_root = repo_root.resolve()
        self._result: Optional[TrainResult] = None

    def train(self) -> Iterator[TrainProgress]:
        sys.path.insert(0, str(self.repo_root))

        from config_loader import get_device, load_config, merge_args_with_config
        from domains.training.train_pipeline import SloughGPTTrainer

        import time
        start = time.time()

        config = load_config()
        config = merge_args_with_config(config, self._to_namespace())
        device = get_device(config.device)

        trainer = SloughGPTTrainer(
            data_path=config.data.data_path,
            vocab_size=config.model.vocab_size,
            n_embed=config.model.n_embed,
            n_layer=config.model.n_layer,
            n_head=config.model.n_head,
            block_size=config.model.block_size,
            dropout=config.model.dropout,
            batch_size=config.training.batch_size,
            epochs=config.training.epochs,
            lr=config.training.learning_rate,
            max_steps=config.training.max_steps,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            max_grad_norm=config.training.gradient_clip,
            use_mixed_precision=config.training.use_mixed_precision,
            mixed_precision_dtype=config.training.mixed_precision_dtype,
            checkpoint_dir=config.checkpoint.trainer_dir,
            checkpoint_interval=config.checkpoint.trainer_interval,
            save_best_only=config.checkpoint.save_best_only,
            max_checkpoints=config.checkpoint.max_checkpoints,
            scheduler_type=config.training.scheduler,
            warmup_steps=config.training.warmup_steps,
            min_lr=config.training.min_lr,
            weight_decay=config.training.weight_decay,
            use_lora=config.lora.enabled,
            lora_rank=config.lora.rank,
            lora_alpha=config.lora.alpha,
            soul_name=config.model.name,
            log_interval=config.training.log_interval,
            eval_interval=config.training.eval_interval,
            device=device,
        )

        try:
            for step, loss in trainer.train(
                resume=getattr(self.config, "resume_latest", False),
                resume_path=self.config.resume_path,
            ):
                elapsed = time.time() - start
                yield TrainProgress(
                    step=step,
                    total_steps=config.training.max_steps,
                    loss=loss,
                    elapsed_seconds=elapsed,
                )
        except Exception as e:
            self._result = TrainResult(success=False, error=str(e))
            return

        save_stem = self.config.save_stem or f"{config.model.name}-{config.data.dataset}"
        save_path = f"{config.checkpoint.save_dir}/{save_stem}"

        trainer.save(save_path, format=config.checkpoint.export_format)
        self._result = TrainResult(success=True, save_path=save_path, final_step=step)

    def _to_namespace(self):
        from argparse import Namespace
        return Namespace(
            dataset=self.config.dataset,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            resume=self.config.resume_path,
            resume_latest=self.config.resume_latest,
            save_stem=self.config.save_stem,
        )


@dataclass
class HttpTrainResult:
    """Training result from API."""

    job_id: str
    status: str
    message: Optional[str] = None


class HttpTrainAdapter:
    """HTTP-based training via POST /training/start (mirrors ``cli.py train --api``)."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def start_training(self, config: TrainConfig) -> HttpTrainResult:
        payload = {
            "name": config.save_stem or f"tui-train-{config.dataset}",
            "model": "sloughgpt",
            "dataset": config.dataset,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(f"{self.base_url}/training/start", json=payload)
                if r.status_code == 200:
                    data = r.json()
                    return HttpTrainResult(job_id=data.get("id", ""), status="started")
                return HttpTrainResult(job_id="", status="error", message=r.text)
        except httpx.HTTPError as e:
            return HttpTrainResult(job_id="", status="error", message=str(e))

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{self.base_url}/training/jobs/{job_id}")
                if r.status_code == 200:
                    return r.json()
        except httpx.HTTPError:
            pass
        return None