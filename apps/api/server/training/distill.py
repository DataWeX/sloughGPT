"""Knowledge distillation route — distill.

Extracted from execution.py to keep each module focused.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from domains.shared import find_repo_root
from domains.training.executor import get_training_executor
from fastapi import APIRouter
from schemas.common import raise_error

from .helpers import _finish_job
from .jobs import training_jobs
from .schemas import DistillStartRequest

logger = logging.getLogger("slo")

router = APIRouter(tags=["training-distill"])


@router.post("/training/distill")
async def start_distillation(request: DistillStartRequest):
    """Knowledge distillation: teach a compact SloNet LSTM student from a teacher HF model."""
    import uuid

    job_id = str(uuid.uuid4())[:8]

    datasets_dir = find_repo_root(Path(__file__).resolve()) / "data"
    data_path = datasets_dir / request.dataset
    if not data_path.exists():
        data_path = datasets_dir / f"{request.dataset}.jsonl"
    if not data_path.exists():
        raise_error(f"Dataset not found: {request.dataset}", "E_BAD_REQUEST", status_code=400)
    input_file = data_path / "input.txt" if data_path.is_dir() else data_path
    if data_path.is_dir():
        candidates = [data_path / "input.txt", data_path / "corpus.jsonl", data_path / "train.txt"]
        input_file = next((c for c in candidates if c.exists()), None)
    if not input_file or not Path(input_file).exists():
        raise_error(
            "No training data file (input.txt/corpus.jsonl) in dataset",
            "E_BAD_REQUEST",
            status_code=400,
        )
    data_str = Path(input_file).read_text(encoding="utf-8")
    if not data_str.strip():
        raise_error("Training data is empty", "E_BAD_REQUEST", status_code=400)
    out_stem = request.name or f"distill_{job_id}"
    _REPO_ROOT = find_repo_root(Path(__file__).resolve())
    output_dir = _REPO_ROOT / "models" / "auto-training"
    output_dir.mkdir(parents=True, exist_ok=True)

    job: dict[str, Any] = {
        "id": job_id,
        "name": request.name or f"Distill-{job_id}",
        "type": "distill",
        "status": "queued",
        "progress": 0,
        "epochs": request.epochs,
        "dataset": request.dataset,
        "teacher_model": request.teacher_model,
        "config": request.model_dump(),
    }
    training_jobs[job_id] = job
    cancel_event = threading.Event()
    training_jobs[job_id]["_cancel_event"] = cancel_event

    try:
        from domains.infrastructure.cancel_manager import OpType, get_cancel_manager

        get_cancel_manager().register(
            op_type=OpType.TRAINING,
            label=str(request.name or f"distill-{job_id}"),
            cancel_fn=lambda: cancel_event.set(),
            meta={"job_id": job_id, "method": "distill"},
            op_id=job_id,
        )
        get_cancel_manager().start(job_id)
    except Exception as e:
        logger.warning("CancelManager registration failed for distill %s: %s", job_id, e)

    def _run_distill(job_id_: str = job_id):
        """Background thread that runs distillation."""
        try:
            training_jobs[job_id_]["status"] = "running"
            import random as _random

            import numpy as np

            _random.seed(42)
            np.random.seed(42)

            from domains.infrastructure.model_registry import get_model_registry

            registry = get_model_registry()
            server = registry.get(request.teacher_model) if registry else None

            slonet_provider = None
            teacher_model = None
            teacher_tokenizer = None
            if server is not None:
                teacher_model = server._model_ref
                teacher_tokenizer = getattr(server, "_tokenizer", None)
            else:
                from domains.infrastructure.server_state import get_server_state

                provider = get_server_state().model.get()
                if (
                    provider is not None
                    and getattr(provider, "model_id", None) == request.teacher_model
                ):
                    slonet_provider = provider
                    teacher_model = getattr(provider, "_get_model", lambda: None)()
                    if teacher_model is not None:
                        teacher_tokenizer = provider.tokenize

            if teacher_model is None:
                _finish_job(job_id, "failed", f"Teacher model '{request.teacher_model}' not loaded")
                return

            # Tokenize training data
            if teacher_tokenizer is not None:
                if slonet_provider is not None:
                    tokens = teacher_tokenizer(data_str[:100000])
                else:
                    tokens = teacher_tokenizer.encode(data_str[:100000])
            else:
                tokens = [ord(c) for c in data_str[:100000]]

            # Build vocab from teacher or data
            vocab = sorted(set(tokens))
            stoi = {c: i for i, c in enumerate(vocab)}
            itos = {i: c for c, i in stoi.items()}
            vocab_size = len(stoi)

            from domains.models import SloughGPTModel

            student = SloughGPTModel(
                vocab_size=vocab_size,
                n_embed=request.embed_dim,
                n_layer=request.n_layers,
                n_head=request.n_heads or 4,
                block_size=request.block_size,
            )

            # Prepare data for training
            block_size = request.block_size
            token_ids = [stoi.get(t, 0) for t in tokens]
            inputs_list = []
            targets_list = []
            for i in range(0, len(token_ids) - block_size, block_size // 2):
                x = token_ids[i : i + block_size]
                y = token_ids[i + 1 : i + block_size + 1]
                if len(x) == block_size and len(y) == block_size:
                    inputs_list.append(x)
                    targets_list.append(y)

            if not inputs_list:
                _finish_job(job_id, "failed", "Not enough data for training")
                return

            inputs_np = np.array(inputs_list, dtype=np.int64)
            targets_np = np.array(targets_list, dtype=np.int64)
            n_samples = len(inputs_np)
            batch_size = min(16, n_samples)

            # Create teacher inputs wrapper
            class _TeacherWrapper:
                """Expose a teacher forward pass to the DistillationTrainer."""

                def __init__(self, model, tokenizer, slonet=False):
                    self._model = model
                    self._tokenizer = tokenizer
                    self._slonet = slonet

                def parameters(self):
                    return []

                def eval(self):
                    pass

                def __call__(self, x):
                    import numpy as np

                    if isinstance(x, np.ndarray):
                        if self._slonet:
                            logits_t, _ = self._model.forward(x.astype(np.int64), None)
                            out_np = np.asarray(logits_t.data, dtype=np.float64)[..., :vocab_size]
                            return np.squeeze(out_np, 0) if out_np.shape[0] == 1 else out_np
                        raise RuntimeError("Torch teacher models are not supported — use SloNet")
                    return np.zeros((x.shape[0], vocab_size), dtype=np.float32)

            teacher_wrapper = _TeacherWrapper(
                teacher_model, teacher_tokenizer, slonet=slonet_provider is not None
            )

            from domains.training.distillation import DistillationConfig, DistillationTrainer

            distill_cfg = DistillationConfig(
                temperature=request.temperature,
                alpha=request.alpha,
                beta=request.beta,
            )
            trainer = DistillationTrainer(teacher_wrapper, student, distill_cfg)

            # Training loop
            epoch_losses = []
            for epoch in range(request.epochs):
                if cancel_event.is_set():
                    _finish_job(job_id, "cancelled")
                    return
                indices = list(range(n_samples))
                _random.shuffle(indices)
                epoch_loss = 0.0
                n_batches = 0
                for start in range(0, n_samples, batch_size):
                    batch_idx = indices[start : start + batch_size]
                    bx = inputs_np[batch_idx]
                    by = targets_np[batch_idx]

                    losses = trainer.step(bx, by)
                    batch_loss = losses.get("total_loss", 0.0)
                    epoch_loss += batch_loss
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                epoch_losses.append(avg_loss)

                training_jobs[job_id]["progress"] = int((epoch + 1) / request.epochs * 100)
                training_jobs[job_id]["current_epoch"] = epoch + 1
                training_jobs[job_id]["train_loss"] = avg_loss

            # Save student checkpoint
            safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in out_stem)[:120]
            ckpt_path = output_dir / f"{safe_stem}_distilled.soul"

            from domains.training.slonet import export_to_sou

            export_to_sou(
                student,
                str(ckpt_path),
                metadata={
                    "model_type": "slonet_distill",
                    "teacher": request.teacher_model,
                    "distill_temperature": request.temperature,
                    "epochs": request.epochs,
                    "final_loss": float(epoch_losses[-1]) if epoch_losses else 0.0,
                    "embed_dim": request.embed_dim,
                    "n_layers": request.n_layers,
                    "vocab_size": vocab_size,
                    "stoi": stoi,
                    "itos": itos,
                },
            )

            training_jobs[job_id].update(
                {
                    "progress": 100,
                    "loss": float(epoch_losses[-1]) if epoch_losses else None,
                    "checkpoint": str(ckpt_path),
                    "loss_history": [
                        {"step": i, "value": v, "type": "train"} for i, v in enumerate(epoch_losses)
                    ],
                }
            )
            _finish_job(job_id, "completed")
            logger.info(
                "Distillation complete: %s loss=%.4f",
                ckpt_path,
                epoch_losses[-1] if epoch_losses else 0,
                extra={"tag": "TRAIN"},
            )

        except Exception as e:
            logger.exception("Distillation job %s failed", job_id, extra={"tag": "TRAIN"})
            _finish_job(job_id, "failed", str(e))

    executor = get_training_executor()
    executor.submit(_run_distill, job_id)

    return {
        "status": "queued",
        "job_id": job_id,
        "message": f"Distillation started: teacher={request.teacher_model} epochs={request.epochs}",
    }
