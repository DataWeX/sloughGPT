"""
Unified Training Pipeline

Composite pipeline that wraps existing trainers through TrainingSequence phases.
Supports SloughGPTTrainer, HFFineTuner, DistillationTrainer, and SloNet.

High-level TrainingStage tracks strategy (pretraining/federated/rlhf),
while TrainingSequence tracks low-level phases (GENERATE_DATA→DISTILL→TRAIN→...).
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from domains.training.trainer_protocol import TrainResult
from domains.training.sequence import (
    TrainingSequence,
    TrainingSequenceState,
    TrainingRunConfig,
    PhaseResult,
    CheckpointFormat,
)

logger = logging.getLogger("man.unified_pipeline")


# =============================================================================
# TrainingStage — High-Level Strategy Stage
# =============================================================================


class TrainingStage(Enum):
    """High-level training pipeline stages (strategy level)."""
    NOT_STARTED = "not_started"
    PRETRAINING = "pretraining"
    FEDERATED = "federated"
    RLHF = "rlhf"
    COMPLETE = "complete"
    FAILED = "failed"


# =============================================================================
# TrainingProgress
# =============================================================================


@dataclass
class TrainingProgress:
    """Real-time training progress shared via SSE or callbacks."""
    phase: str = "idle"
    epoch: int = 0
    total_epochs: int = 0
    step: int = 0
    total_steps: int = 0
    loss: Optional[float] = None
    val_loss: Optional[float] = None
    learning_rate: float = 0.0
    progress_pct: float = 0.0
    status: str = "working"  # working | complete | error | skipped
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _sanitize(val: Any, default: float = 0.0) -> float:
        """Sanitize a value for JSON serialisation: replace Infinity/NaN with default."""
        if val is None:
            return default
        try:
            f = float(val)
            if not math.isfinite(f):
                return default
            return f
        except (TypeError, ValueError, OverflowError):
            return default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "step": self.step,
            "total_steps": self.total_steps,
            "loss": self._sanitize(self.loss),
            "val_loss": self._sanitize(self.val_loss),
            "learning_rate": self.learning_rate,
            "progress_pct": self.progress_pct,
            "status": self.status,
            "message": self.message,
            "metrics": self.metrics,
        }

    def to_sse_event(self, stream_name: str = "auto-train") -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "progress": self.progress_pct,
            "loss": self._sanitize(self.loss),
            "val_loss": self._sanitize(self.val_loss),
            "epoch": self.epoch,
            "step": self.step,
        }
        # Include finish-specific fields from metrics when present
        # Sanitize numeric fields to prevent Infinity/NaN in JSON
        _numeric_metrics = {"final_loss", "total_steps", "elapsed", "epochs"}
        for key in ("final_loss", "model_path", "total_steps", "elapsed", "checkpoint", "cancelled", "epochs", "sensitivity"):
            if key in self.metrics:
                val = self.metrics[key]
                if key in _numeric_metrics and not isinstance(val, str):
                    val = self._sanitize(val)
                data[key] = val
        return {
            "stream": stream_name,
            "phase": self.phase,
            "status": self.status,
            "data": data,
            "meta": {
                "epoch": self.epoch,
                "total_epochs": self.total_epochs,
                "total_steps": self.total_steps,
                "learning_rate": self.learning_rate,
            },
            "message": self.message,
        }


# =============================================================================
# UnifiedTrainingConfig
# =============================================================================


@dataclass
class UnifiedTrainingConfig:
    """Configuration for the unified training pipeline.

    Aggregates settings for all supported trainer types.
    Specific trainer configs (e.g. HFTrainingRequest, TrainerConfig)
    can be passed via the ``trainer_kwargs`` dict.
    """
    # General
    method: str = "auto"  # auto | distill | finetune | turbo | hf
    data_path: str = ""
    dataset_name: str = ""
    output_dir: str = "models/unified-trained"

    # Epochs / steps
    epochs: int = 3
    max_steps: Optional[int] = None
    batch_size: int = 8

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100

    # Distillation
    distill: bool = False
    temperature: float = 4.0
    distill_alpha: float = 0.5
    distill_beta: float = 0.5

    # LoRA
    use_lora: bool = False
    lora_rank: int = 8

    # HF-specific
    hf_model_name: str = ""
    hf_use_lora: bool = True
    hf_lora_rank: int = 8
    hf_max_seq_length: int = 512

    # Turbo-specific
    turbo_model_spec: str = "transformer"
    turbo_vocab_size: int = 1000
    turbo_n_embed: int = 128
    turbo_n_head: int = 4

    # SloNet-specific
    vocab_size: int = 256
    n_embed: int = 256
    n_layer: int = 6
    n_head: int = 8
    block_size: int = 128

    # Soul / personality (used by auto-train route for .soul export)
    soul_name: str = ""
    system_prompt: str = ""

    # Tracking
    save_report_path: str = ""  # empty = no report file written
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 500
    save_best_only: bool = False
    max_checkpoints: int = 5

    # Resume
    resume: bool = False
    resume_path: Optional[str] = None  # explicit .pt path; empty = auto-detect latest

    # Sequence config
    skip_generate: bool = False
    skip_distill: bool = False
    skip_train: bool = False
    skip_evaluate: bool = False
    skip_deploy: bool = False

    # Device
    device: str = "auto"
    use_mixed_precision: bool = True

    # Eval
    eval_every_n_steps: int = 100

    # Extra kwargs forwarded to specific trainers
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# UnifiedTrainingPipeline
# =============================================================================


class UnifiedTrainingPipeline:
    """Composite pipeline orchestration.

    Wraps existing trainers (SloughGPTTrainer, HFFineTuner, DistillationTrainer)
    and drives them through ``TrainingSequence`` phases:

        GENERATE_DATA → DISTILL → TRAIN → EVALUATE → DEPLOY → COMPLETE

    Phases can be skipped via ``TrainingRunConfig``.

    Usage:
        pipeline = UnifiedTrainingPipeline(config)
        result = pipeline.run(on_progress=my_callback)
    """

    _is_training: bool = False
    _cancel_event: Optional[threading.Event] = None
    _pause_event: Optional[threading.Event] = None

    @property
    def is_training(self) -> bool:
        """Whether training is in progress."""
        return self._is_training

    def stop(self) -> None:
        """Request early stopping."""
        self._is_training = False
        if self._cancel_event is not None:
            self._cancel_event.set()

    def __init__(
        self,
        config: Union[UnifiedTrainingConfig, Dict[str, Any]],
        run_config: Optional[TrainingRunConfig] = None,
    ):
        warnings.warn(
            "UnifiedTrainingPipeline is deprecated — use SloughGPTTrainer instead",
            DeprecationWarning, stacklevel=2,
        )
        if isinstance(config, dict):
            config = UnifiedTrainingConfig(**config)
        self.config: UnifiedTrainingConfig = config
        if run_config is None:
            run_config = TrainingRunConfig.defaults()
            run_config.skip_generate = config.skip_generate
            run_config.skip_distill = config.skip_distill
            run_config.skip_train = config.skip_train
            run_config.skip_evaluate = config.skip_evaluate
            run_config.skip_deploy = config.skip_deploy
        self.run_config = run_config
        self.state = TrainingSequenceState()
        self.progress = TrainingProgress()
        self._trainer_instance: Optional[Any] = None
        self._start_time: Optional[float] = None
        self.tracker: Optional[Any] = None

        # Determine effective method
        self._method = self._resolve_method()

    def _resolve_method(self) -> str:
        """Auto-detect training method from config.

        Priority: explicit method > hf > distill > slonet (default).
        Turbo must be set explicitly (method=\"turbo\").
        """
        if self.config.method not in ("auto", ""):
            return self.config.method
        if self.config.hf_model_name:
            return "hf"
        if self.config.distill:
            return "distill"
        if self.config.method == "turbo":
            return "turbo"
        return "slonet"

    def _update_progress(
        self,
        phase: str,
        epoch: int = 0,
        step: int = 0,
        loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        message: str = "",
        status: str = "working",
        metrics: Optional[Dict[str, Any]] = None,
    ):
        self.progress.phase = phase
        self.progress.epoch = epoch
        self.progress.total_epochs = self.config.epochs
        self.progress.step = step
        self.progress.loss = loss
        self.progress.val_loss = val_loss
        self.progress.status = status
        self.progress.message = message
        if metrics:
            self.progress.metrics.update(metrics)

    def _emit_progress(self, on_progress: Optional[Callable[[TrainingProgress], None]]):
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise KeyboardInterrupt("Training cancelled by user")
        if on_progress:
            on_progress(self.progress)

    def run(
        self,
        on_progress: Optional[Callable[[TrainingProgress], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        pause_event: Optional[threading.Event] = None,
    ) -> TrainResult:
        """Run the full pipeline through all enabled phases.

        Args:
            on_progress: Called after each phase transition with current progress.
            cancel_event: Optional threading.Event — if set, pipeline raises
                ``KeyboardInterrupt`` during the next progress emission to abort.
            pause_event: Optional threading.Event — if set, training loop sleeps until cleared.

        Returns:
            TrainResult with status, model_path, final_loss, total_steps, checkpoint, metrics.
        """
        self._cancel_event = cancel_event
        self._pause_event = pause_event
        self._start_time = time.time()
        self._is_training = True
        logger.info("Starting unified pipeline: method=%s, data=%s", self._method, self.config.data_path)

        try:
            return self._run_body(on_progress)
        except KeyboardInterrupt:
            elapsed = time.time() - self._start_time
            self._update_progress(
                "complete",
                status="complete",
                message=f"Training cancelled after {elapsed:.1f}s",
                metrics={"cancelled": True, "elapsed": elapsed},
            )
            self._emit_progress(on_progress)
            logger.info("Pipeline cancelled after %.1fs", elapsed)
            return TrainResult(
                success=False,
                status="cancelled",
                metrics={"cancelled": True, "elapsed": elapsed},
                error="Training cancelled by user",
            )
        finally:
            self._is_training = False

    def _run_body(
        self,
        on_progress: Optional[Callable[[TrainingProgress], None]] = None,
    ) -> TrainResult:
        # --- initialize TrainingStatusTracker ---
        try:
            from domains.training.status import TrainingStatusTracker, CompletionStatus
            self.tracker = TrainingStatusTracker(model_name=self.config.hf_model_name or self._method)
            self.tracker.start_training(
                dataset=self.config.data_path or self.config.dataset_name,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                pretrain_epochs=self.config.epochs,
            )
        except ImportError:
            self.tracker = None

        # --- GENERATE DATA ---
        if not self.run_config.skip_generate:
            self.state.start_phase(TrainingSequence.GENERATE_DATA)
            self._update_progress("generate_data", message="Generating training data...")
            self._emit_progress(on_progress)
            self._run_generate_data()
            self.state.complete_phase(TrainingSequence.GENERATE_DATA)
            if not self._is_training:
                return self._early_exit("Training stopped during data generation")
        else:
            self.state.skip_phase(TrainingSequence.GENERATE_DATA, "Skipped by config")

        # --- DISTILL (not yet implemented — always skip) ---
        self.state.skip_phase(TrainingSequence.DISTILL, "Distillation not implemented")

        # --- TRAIN ---
        if not self.run_config.skip_train:
            self.state.start_phase(TrainingSequence.TRAIN)
            self._update_progress("train", message="Training...")
            self._emit_progress(on_progress)
            train_result = self._run_train(on_progress)
            self.state.complete_phase(
                TrainingSequence.TRAIN,
                metrics={
                    "final_loss": train_result.get("final_loss"),
                    "total_steps": train_result.get("total_steps", 0),
                },
            )
            # Early exit if training produced no data (too few samples)
            if train_result.get("status") == "no_data":
                self._update_progress(
                    "complete",
                    status="error",
                    message=f"Training produced no data — dataset too small ({train_result.get('message', '')})",
                    loss=None,
                    metrics={"error": "no_data", "total_steps": train_result.get("total_steps", 0)},
                )
                self._emit_progress(on_progress)
                return TrainResult(
                    success=False,
                    status="error",
                    error="Training produced no data — dataset too small",
                    final_loss=None,
                    total_steps=0,
                    model_path="",
                    checkpoint_name="",
                    metrics={},
                )

            # Update status tracker
            if self.tracker is not None:
                from domains.training.status import TrainingStage
                final_loss = train_result.get("final_loss")
                if final_loss is not None:
                    self.tracker.update_stage(
                        TrainingStage.PRETRAINING,
                        epoch=self.config.epochs - 1,
                        loss=final_loss,
                    )
                self.tracker.complete_stage(TrainingStage.PRETRAINING)
        else:
            self.state.skip_phase(TrainingSequence.TRAIN, "Skipped by config")
            train_result = {}

        if not self._is_training:
            return self._early_exit("Training stopped")

        # --- EVALUATE ---
        eval_result = {}
        if not self.run_config.skip_evaluate:
            self.state.start_phase(TrainingSequence.EVALUATE)
            self._update_progress("evaluate", message="Evaluating...")
            self._emit_progress(on_progress)
            eval_result = self._run_evaluate()
            self.state.complete_phase(
                TrainingSequence.EVALUATE,
                metrics=eval_result.get("metrics", {}),
            )
            if not self._is_training:
                return self._early_exit("Training stopped during evaluation")
        else:
            self.state.skip_phase(TrainingSequence.EVALUATE, "Skipped by config")

        # --- DEPLOY ---
        deploy_result = {}
        if not self.run_config.skip_deploy:
            self.state.start_phase(TrainingSequence.DEPLOY)
            self._update_progress("deploy", message="Deploying model...", status="working")
            self._emit_progress(on_progress)
            deploy_result = self._run_deploy(train_result)
            self.state.complete_phase(
                TrainingSequence.DEPLOY,
                metrics={"model_path": deploy_result.get("model_path", "")},
            )
        else:
            self.state.skip_phase(TrainingSequence.DEPLOY, "Skipped by config")

        # --- COMPLETE ---
        self.state.current_phase = TrainingSequence.COMPLETE
        elapsed = time.time() - self._start_time
        deploy_path = deploy_result.get("model_path", train_result.get("model_path", ""))
        final_loss = train_result.get("final_loss")
        final_steps = train_result.get("total_steps", 0)
        ckpt = train_result.get("checkpoint", train_result.get("checkpoint_name", ""))
        self._update_progress(
            "complete",
            status="complete",
            message=f"Training complete in {elapsed:.1f}s",
            loss=final_loss,
            metrics={
                "final_loss": final_loss,
                "model_path": deploy_path,
                "total_steps": final_steps,
                "elapsed": elapsed,
                "checkpoint": ckpt,
                "epochs": self.config.epochs,
            },
        )
        self._emit_progress(on_progress)

        # Mark tracker complete & save report
        if self.tracker is not None:
            self.tracker.mark_complete()
            save_path = self.config.save_report_path
            if save_path:
                from pathlib import Path
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                self.tracker.save_report(save_path)

        phase_list = [pr.to_dict() for pr in self.state.phase_results]
        result = TrainResult(
            success=True,
            status="completed",
            final_loss=final_loss,
            total_steps=final_steps,
            model_path=deploy_path,
            checkpoint_name=ckpt,
            method=self._method,
            message=f"Training complete in {elapsed:.1f}s",
            elapsed=elapsed,
            phases=phase_list,
            metrics={
                "phases": phase_list,
                "elapsed": elapsed,
                "message": f"Training complete in {elapsed:.1f}s",
                **(train_result.get("metrics", {}) if isinstance(train_result, dict) else train_result.metrics if hasattr(train_result, "metrics") else {}),
            },
        )
        # Attach tracker report if available (convert enums to values)
        if self.tracker is not None:
            try:
                from dataclasses import asdict
                from enum import Enum

                def _json_safe(obj):
                    if isinstance(obj, Enum):
                        return obj.value
                    if isinstance(obj, dict):
                        return {k: _json_safe(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        return [_json_safe(x) for x in obj]
                    return obj

                report = asdict(self.tracker.get_report())
                result.metrics["tracker_report"] = _json_safe(report)
            except Exception:
                pass
        logger.info("Pipeline complete: %s", result.metrics.get("message", ""))
        return result

    def _early_exit(self, reason: str) -> TrainResult:
        """Return early when training is stopped."""
        elapsed = time.time() - self._start_time
        self._update_progress(
            "complete",
            status="complete",
            message=f"{reason} after {elapsed:.1f}s",
            metrics={"stopped": True, "elapsed": elapsed},
        )
        self._emit_progress(None)
        return TrainResult(
            success=False,
            status="stopped",
            error=reason,
            metrics={"stopped": True, "elapsed": elapsed},
        )

    def _run_generate_data(self):
        """GENERATE_DATA phase: prepare dataset."""
        from domains.training.train_pipeline import prepare_data

        if not self.config.data_path:
            logger.info("No data_path set — checking dataset_name")
            return

        logger.info("Preparing data from %s", self.config.data_path)
        try:
            prepare_data(self.config.data_path, block_size=self.config.block_size)
        except Exception as e:
            logger.warning("Data preparation warning (non-fatal): %s", e)

    def _run_train(
        self,
        on_progress: Optional[Callable[[TrainingProgress], None]] = None,
    ) -> TrainResult:
        """TRAIN phase: dispatch to the appropriate trainer based on method."""
        method = self._method

        if method == "hf":
            return self._train_hf(on_progress)
        elif method == "turbo":
            return self._train_turbo(on_progress)
        else:
            return self._train_slonet(on_progress)

    def _train_hf(
        self,
        on_progress: Optional[Callable[[TrainingProgress], None]] = None,
    ) -> TrainResult:
        """Train via HuggingFace HFFineTuner."""
        from domains.training.hf_finetune import HFFineTuner

        data_path = self.config.data_path or str(
            Path("datasets") / self.config.dataset_name / "input.txt"
        )
        output_dir = self.config.output_dir

        tuner = HFFineTuner(
            model_name=self.config.hf_model_name,
            data_path=data_path,
            output_dir=output_dir,
            use_lora=self.config.hf_use_lora,
            lora_rank=self.config.hf_lora_rank,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            max_seq_length=self.config.hf_max_seq_length,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            **self.config.trainer_kwargs,
        )

        def _hf_progress(info: Dict[str, Any]):
            self._update_progress(
                "train",
                epoch=int(info.get("epoch", 0)),
                step=info.get("step", 0),
                loss=info.get("loss"),
                message=f"Epoch {info.get('epoch', 0):.1f}, loss {info.get('loss', 'N/A')}",
                metrics={"progress_pct": info.get("progress_pct", 0)},
            )
            if on_progress:
                on_progress(self.progress)

        result = tuner.train(on_progress=_hf_progress)
        self._trainer_instance = tuner

        return TrainResult(
            success=result.get("status", "completed") != "error",
            status=result.get("status", "completed"),
            model_path=result.get("model_path", output_dir),
            final_loss=result.get("final_loss"),
            total_steps=result.get("total_steps", 0),
            checkpoint_name=result.get("checkpoint", result.get("checkpoint_name", "")),
            method="hf",
            metrics={},
        )

    def _train_turbo(
        self,
        on_progress: Optional[Callable[[TrainingProgress], None]] = None,
    ) -> TrainResult:
        """Train via SloughGPTTrainer (decoder-only transformer)."""
        from domains.training.train_pipeline import SloughGPTTrainer

        trainer = SloughGPTTrainer(
            data_path=self.config.data_path or self.config.dataset_name or "",
            vocab_size=self.config.turbo_vocab_size,
            n_embed=self.config.turbo_n_embed,
            n_layer=self.config.turbo_n_layer if hasattr(self.config, 'turbo_n_layer') else 2,
            n_head=self.config.turbo_n_head,
            block_size=self.config.turbo_block_size if hasattr(self.config, 'turbo_block_size') else 64,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            lr=self.config.learning_rate,
            checkpoint_dir=self.config.output_dir,
            soul_name=self.config.soul_name,
        )

        step_counter = [0]

        def _turbo_progress(info: Dict[str, Any]):
            step_counter[0] += 1
            metrics: Dict[str, Any] = {"progress_pct": info.get("progress_pct", 0)}
            if "sensitivity" in info:
                metrics["sensitivity"] = info["sensitivity"]
            self._update_progress(
                "train",
                epoch=info.get("epoch", 0),
                step=info.get("step", step_counter[0]),
                loss=info.get("loss"),
                message=f"Step {step_counter[0]}",
                metrics=metrics,
            )
            if on_progress:
                on_progress(self.progress)

        result = trainer.train(on_progress=_turbo_progress)
        self._trainer_instance = trainer

        return TrainResult(
            success=result.get("status", "completed") != "error",
            status=result.get("status", "completed"),
            model_path=result.get("model_path", self.config.output_dir),
            final_loss=result.get("final_loss"),
            total_steps=result.get("total_steps", step_counter[0]),
            checkpoint_name=result.get("checkpoint", result.get("checkpoint_name", "")),
            method="turbo",
            metrics={},
        )

    def _train_slonet(
        self,
        on_progress: Optional[Callable[[TrainingProgress], None]] = None,
    ) -> TrainResult:
        """Train via SloughGPTTrainer (native SloNet / nanoGPT)."""
        from domains.training.train_pipeline import SloughGPTTrainer, TrainerConfig

        trainer_config = TrainerConfig(
            vocab_size=self.config.vocab_size,
            n_embed=self.config.n_embed,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            block_size=self.config.block_size,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            checkpoint_dir=self.config.checkpoint_dir,
            checkpoint_interval=self.config.checkpoint_interval,
            save_best_only=self.config.save_best_only,
            max_checkpoints=self.config.max_checkpoints,
            use_mixed_precision=self.config.use_mixed_precision,
            max_steps=self.config.max_steps,
            warmup_steps=self.config.warmup_steps,
            device="cpu",
            **{k: v for k, v in self.config.trainer_kwargs.items() if k not in ("teacher_model", "student_model", "train_data")},
        )

        data_path = self.config.data_path or self.config.dataset_name
        trainer = SloughGPTTrainer(
            data_path=data_path,
            config=trainer_config,
            epochs=self.config.epochs,
            lr=self.config.learning_rate,
            batch_size=self.config.batch_size,
            checkpoint_dir=self.config.checkpoint_dir,
        )

        step_counter = [0]
        loss_history: List[float] = []

        def _slonet_progress(info: Dict[str, Any]):
            step = info.get("global_step") or info.get("step", 0)
            loss = info.get("train_loss") or info.get("loss")
            epoch = info.get("epoch", 0)
            progress_pct = info.get("progress_percent", 0)
            total_steps = info.get("steps_per_epoch", 0) * self.config.epochs
            lr = info.get("learning_rate", 0.0)
            step_counter[0] = step
            safe_loss = TrainingProgress._sanitize(loss)
            if safe_loss is not None:
                loss_history.append(safe_loss)
            self._update_progress(
                "train",
                epoch=epoch,
                step=step,
                loss=safe_loss,
                metrics={"loss_history": loss_history[-200:]},
                message=f"Step {step}, loss {safe_loss:.4f}" if safe_loss is not None else f"Step {step}",
            )
            self.progress.progress_pct = progress_pct
            self.progress.total_steps = total_steps
            self.progress.learning_rate = lr
            if on_progress:
                on_progress(self.progress)

        # Wrap the progress callback and cancel event
        original_train = trainer.train

        def _patched_train(*args, **kwargs):
            return original_train(
                *args,
                **kwargs,
                on_progress=_slonet_progress,
                cancel_event=self._cancel_event,
                pause_event=self._pause_event,
                resume=self.config.resume,
                resume_path=self.config.resume_path,
            )

        trainer.train = _patched_train
        result = trainer.train()

        self._trainer_instance = trainer

        # Extract final loss from trainer (TrainResult or dict)
        final_loss = None
        if isinstance(result, TrainResult):
            final_loss = result.final_loss or result.best_eval_loss
        elif isinstance(result, dict):
            be = result.get("best_eval_loss")
            fl = result.get("final_loss")
            final_loss = fl if fl is not None else be
        if final_loss is None:
            final_loss = getattr(trainer, "_best_val_loss", None)
        if final_loss is None and loss_history:
            final_loss = loss_history[-1]
        # Sanitize — guard against non-numeric (dict, None, inf, nan)
        final_loss = TrainingProgress._sanitize(final_loss, default=None)

        checkpoint_name = ""
        total_steps = step_counter[0] or getattr(trainer, "global_step", 0)
        # Skip soul export if no steps completed
        if total_steps == 0:
            logger.info("Zero steps completed — skipping soul export")
            return TrainResult(
                success=False,
                status="no_data",
                model_path=self.config.checkpoint_dir,
                total_steps=0,
                metrics={"loss_history": []},
            )
        # Export .soul if soul_name is configured
        if self.config.soul_name and hasattr(trainer, "model") and trainer.model is not None:
            try:
                import time as _time
                from domains.inference import save_soul, SloProfile, PersonalityCore

                soul_name = self.config.soul_name
                ckpt_name = f"{soul_name}_{int(_time.time())}.soul"
                ckpt_path = Path(self.config.checkpoint_dir) / ckpt_name

                safe_loss = TrainingProgress._sanitize(final_loss)
                if not isinstance(safe_loss, (int, float)):
                    safe_loss = 0.0
                soul_profile = SloProfile(
                    name=f"{soul_name}-soul",
                    version="1.0.0",
                    tagline="AI Slo trained via UnifiedTrainingPipeline",
                    description=f"SloNet trained in {total_steps} steps. Loss: {safe_loss:.4f}.",
                    lineage="unified-slp-pt",
                    base_model="slonet-lstm",
                    training_dataset="user-provided",
                    epochs_trained=self.config.epochs,
                    final_train_loss=round(safe_loss, 6),
                    final_val_loss=round(safe_loss, 6),
                    personality=PersonalityCore(),
                    system_prompt=self.config.system_prompt or "You are a helpful assistant.",
                    metadata={
                        "vocab_size": getattr(trainer, "vocab_size", self.config.vocab_size),
                        "n_embed": self.config.n_embed,
                        "n_layer": self.config.n_layer,
                        "n_head": self.config.n_head,
                        "block_size": self.config.block_size,
                        "model_type": "sloughgpt",
                        "total_steps": total_steps,
                        "stoi": getattr(trainer, "stoi", None),
                        "itos": getattr(trainer, "itos", None),
                    },
                )
                save_soul(trainer.model, str(ckpt_path), soul_profile=soul_profile)
                checkpoint_name = ckpt_name
                logger.info("Exported .soul checkpoint: %s", ckpt_name)
            except Exception as e:
                import traceback
                logger.warning("Soul export skipped: %s\n%s", e, traceback.format_exc())

        return TrainResult(
            success=True,
            status="completed",
            final_loss=final_loss,
            total_steps=step_counter[0],
            model_path=self.config.checkpoint_dir,
            checkpoint_name=checkpoint_name,
            method=self._method,
            metrics={"loss_history": loss_history[-200:]},
        )

    def _run_evaluate(self) -> Dict[str, Any]:
        """EVALUATE phase: compute eval metrics."""
        metrics = {}
        if self._trainer_instance is not None:
            try:
                if hasattr(self._trainer_instance, "evaluate"):
                    eval_result = self._trainer_instance.evaluate()
                    if isinstance(eval_result, dict):
                        metrics = eval_result
            except Exception as e:
                logger.warning("Evaluation error (non-fatal): %s", e)
                metrics = {"eval_error": str(e)}
        return {"metrics": metrics}

    def _run_deploy(self, train_result: Dict[str, Any]) -> Dict[str, Any]:
        """DEPLOY phase: export model to output directory."""
        model_path = train_result.get("model_path", self.config.output_dir)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "pipeline": "unified",
            "method": self._method,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "use_lora": self.config.use_lora,
                "distill": self.config.distill,
            },
            "result": {
                "final_loss": train_result.get("final_loss"),
                "total_steps": train_result.get("total_steps", 0),
            },
        }

        manifest_path = output_dir / "pipeline_manifest.json"
        try:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            logger.warning("Could not write manifest: %s", e)

        return {"model_path": str(output_dir)}


# =============================================================================
# Convenience Factory
# =============================================================================


def create_pipeline(
    config: Union[UnifiedTrainingConfig, Dict[str, Any]],
    **kwargs,
) -> UnifiedTrainingPipeline:
    """Create a configured UnifiedTrainingPipeline.

    Args:
        config: Pipeline config (dict or UnifiedTrainingConfig).
        **kwargs: Extra config fields (overrides config dict).

    Returns:
        Configured pipeline instance.
    """
    if isinstance(config, dict):
        config = UnifiedTrainingConfig(**{**config, **kwargs})
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
    return UnifiedTrainingPipeline(config)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TrainingStage",
    "TrainingProgress",
    "UnifiedTrainingConfig",
    "UnifiedTrainingPipeline",
    "create_pipeline",
]
