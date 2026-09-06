"""LoRA fine-tuning routes — lora-finetune, load-adapter, unload-adapter.

Extracted from execution.py to keep each module focused.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any

from domains.shared import find_repo_root
from domains.training.executor import get_training_executor
from fastapi import APIRouter, Depends
from infrastructure.auth import require_auth_if_enabled
from schemas.common import raise_error

from .controller import get_training_controller
from .helpers import _finish_job, _run_async
from .jobs import training_jobs
from .schemas import LoadAdapterRequest, LoraFinetuneRequest
from .webhooks import notify_training_event

logger = logging.getLogger("slo")

router = APIRouter(tags=["training-lora"])


@router.post("/training/lora-finetune")
async def start_lora_finetune(
    request: LoraFinetuneRequest, auth_user: dict = Depends(require_auth_if_enabled)
):
    """LoRA fine-tuning on .slnc models using SloNet numpy autograd (no PyTorch)."""
    import uuid

    job_id = f"lora_{uuid.uuid4().hex[:8]}"

    # Validate model path
    model_path = Path(request.model_path)
    if not model_path.is_file():
        repo_root = find_repo_root(Path(__file__).resolve())
        alt_path = repo_root / "models" / request.model_path
        if alt_path.is_file():
            model_path = alt_path
        else:
            raise_error(
                f"Model not found: {request.model_path}. Provide a .slnc file path.",
                "E_BAD_REQUEST",
                status_code=400,
            )
    # Validate dataset
    repo_root = find_repo_root(Path(__file__).resolve())
    datasets_dir = repo_root / "datasets"
    data_dir = datasets_dir / request.dataset
    data_path = None
    if data_dir.is_dir():
        for candidate in ["input.txt", "corpus.jsonl", "train.txt"]:
            p = data_dir / candidate
            if p.is_file():
                data_path = p
                break
    if data_path is None:
        raise_error(
            f"Dataset not found: {request.dataset}. Use POST /datasets/import/local first.",
            "E_BAD_REQUEST",
            status_code=400,
        )
    model_stem = model_path.stem
    dataset_name = request.dataset.strip() if request.dataset else data_path.stem

    # Create job record — matches TrainingJob interface
    job: dict[str, Any] = {
        "id": job_id,
        "name": request.name or f"LoRA-{dataset_name}-r{request.rank}",
        "model": model_stem,
        "dataset": dataset_name,
        "data_path": str(data_path),
        "status": "running",
        "progress": 0,
        "epochs": request.epochs,
        "current_epoch": 0,
        "global_step": 0,
        "total_steps": 0,
        "steps_per_sec": None,
        "eta_s": None,
        "elapsed_s": None,
        "loss": None,
        "train_loss": None,
        "eval_loss": None,
        "loss_history": [],
        "rank": request.rank,
        "alpha": request.alpha,
        "error": None,
        "result": None,
        "checkpoint": None,
    }
    training_jobs[job_id] = job

    # Update global training controller
    controller = get_training_controller()
    controller.start(job_id, request.name or f"LoRA-{dataset_name}")

    # Audit trail
    try:
        from infrastructure.auth import get_audit_logger

        get_audit_logger().log(
            "training.start",
            resource=dataset_name,
            detail="lora",
            extra={
                "job_id": job_id,
                "model": model_stem,
                "rank": request.rank,
                "epochs": request.epochs,
            },
        )
    except Exception as e:
        logger.warning("Audit log failed for LoRA training start %s: %s", job_id, e)

    # Webhook notification
    try:

        async def notify_async():
            await notify_training_event(
                "training.started",
                {
                    "job_id": job_id,
                    "job_name": request.name or f"LoRA-{dataset_name}",
                    "dataset": dataset_name,
                    "epochs": request.epochs,
                    "method": "lora",
                    "rank": request.rank,
                },
            )

        asyncio.create_task(notify_async())
    except Exception as e:
        logger.debug("LoRA training webhook notification failed: %s", e, extra={"tag": "TRAIN"})

    # Cancel event for stop support
    cancel_event = threading.Event()
    training_jobs[job_id]["_cancel_event"] = cancel_event

    # Register with runtime for consistency
    try:
        from .runtime import get_training_runtime

        get_training_runtime().register(
            job_id, training_jobs[job_id], cancel_event, request.model_dump()
        )
    except Exception as e:
        logger.warning("Training runtime registration failed for %s: %s", job_id, e)

    try:
        from domains.infrastructure.cancel_manager import OpType, get_cancel_manager

        get_cancel_manager().register(
            op_type=OpType.TRAINING,
            label=str(request.model_dump().get("dataset") or job_id),
            cancel_fn=lambda: cancel_event.set(),
            meta={"job_id": job_id, "method": "lora"},
            op_id=job_id,
        )
        get_cancel_manager().start(job_id)
    except Exception as e:
        logger.warning("CancelManager registration failed for LoRA %s: %s", job_id, e)

    def run_lora_finetune(job_id_: str = job_id):
        try:
            from domains.training.hf_lora_finetune import HFLoraConfig, HFLoraTrainer

            start_time = time.time()

            def on_progress(info: dict[str, Any]) -> None:
                rec = training_jobs.get(job_id)
                if not rec:
                    return
                elapsed = time.time() - start_time
                rec["current_epoch"] = int(info.get("epoch", rec.get("current_epoch", 0)))
                rec["global_step"] = int(info.get("step", rec.get("global_step", 0)))
                loss = info.get("loss")
                if loss is not None:
                    rec["train_loss"] = float(loss)
                    rec["loss"] = float(loss)
                    rec.setdefault("loss_history", []).append(
                        {
                            "step": rec["global_step"],
                            "value": float(loss),
                            "type": "train",
                        }
                    )
                rec["elapsed_s"] = elapsed
                step = rec["global_step"]
                if step > 0 and elapsed > 0:
                    rec["steps_per_sec"] = step / elapsed
                    epochs_left = max(
                        0,
                        (rec.get("epochs", 1) - rec["current_epoch"])
                        / max(rec["current_epoch"], 1),
                    )
                    rec["eta_s"] = elapsed * epochs_left
                # Compute progress from epochs
                total_epochs = rec.get("epochs", 1)
                if total_epochs > 0:
                    rec["progress"] = min(99, int((rec["current_epoch"] / total_epochs) * 100))

            config = HFLoraConfig(
                model_path=str(model_path),
                data_path=str(data_path),
                rank=request.rank,
                alpha=request.alpha,
                dropout=request.dropout,
                target_modules=request.target_modules,
                epochs=request.epochs,
                batch_size=request.batch_size,
                block_size=request.block_size,
                learning_rate=request.learning_rate,
                warmup_steps=request.warmup_steps,
                weight_decay=request.weight_decay,
                grad_clip=request.grad_clip,
                grad_accumulation_steps=request.grad_accumulation_steps,
                log_interval=request.log_interval,
                output_dir=request.output_dir,
                adapter_name=request.adapter_name,
                _cancel_event=cancel_event,
            )

            trainer = HFLoraTrainer(config)
            result = trainer.train(on_progress=on_progress)

            if cancel_event.is_set():
                _finish_job(job_id, "cancelled")
                training_jobs[job_id]["progress"] = 0
                get_training_controller().complete()
                return

            training_jobs[job_id].update(
                {
                    "progress": 100,
                    "current_epoch": result.epochs_completed or request.epochs,
                    "loss": result.final_loss,
                    "train_loss": result.final_loss,
                    "result": {
                        "adapter_path": result.model_path,
                        "total_steps": result.total_steps,
                        "final_loss": result.final_loss,
                        "epochs_completed": result.epochs_completed,
                    },
                    "checkpoint": result.model_path,
                }
            )
            _finish_job(job_id, "completed")

            # Sync runtime
            try:
                from .runtime import get_training_runtime

                get_training_runtime().sync(job_id)
            except Exception as e:
                logger.warning("Training runtime sync failed for %s: %s", job_id, e)

            get_training_controller().complete()

            logger.info(
                "LoRA fine-tune complete: %s loss=%.4f",
                result.model_path,
                result.final_loss or 0,
                extra={"tag": "TRAIN"},
            )

            # Webhook notification
            try:
                _run_async(
                    notify_training_event(
                        "training.completed",
                        {
                            "job_id": job_id,
                            "job_name": request.name or f"LoRA-{dataset_name}",
                            "dataset": dataset_name,
                            "final_loss": result.final_loss,
                            "adapter_path": result.model_path,
                        },
                    )
                )
            except Exception as e:
                logger.warning("LoRA training completion webhook failed for %s: %s", job_id, e)

        except Exception as exc:
            logger.exception("LoRA fine-tune job %s failed", job_id, extra={"tag": "TRAIN"})
            _finish_job(job_id, "failed", str(exc))
            get_training_controller().complete()

    executor = get_training_executor()
    executor.submit(run_lora_finetune, job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"LoRA fine-tune started: rank={request.rank} epochs={request.epochs} dataset={dataset_name}",
    }


def _resolve_adapter_path(raw: str) -> Path:
    """Resolve adapter path from user input, checking repo-relative paths."""
    p = Path(raw)
    if p.is_file():
        return p
    repo_root = find_repo_root(Path(__file__).resolve())
    for base in [repo_root / "models", repo_root / "data" / "user_adapters"]:
        alt = base / raw
        if alt.is_file():
            return alt
    raise_error(f"Adapter not found: {raw}", "E_BAD_REQUEST", status_code=400)


@router.post("/training/load-adapter")
async def load_adapter(request: LoadAdapterRequest):
    """Load a LoRA adapter into the currently running model for inference."""
    adapter_path = _resolve_adapter_path(request.adapter_path)

    # Find the ProcessGuard — stored in the models controller (adopted during autoload)
    from domains.infrastructure.server_state import get_server_state

    provider = get_server_state().model.get()
    if provider is None:
        raise_error("No model loaded — load a model first", "E_BAD_REQUEST", status_code=400)

    try:
        from domains.training.lora import load_lora_adapter

        load_lora_adapter(provider, str(adapter_path), merge=request.merge)
    except Exception as e:
        raise_error(f"Failed to load adapter: {e}", "E_BAD_REQUEST", status_code=400)

    return {
        "status": "loaded",
        "adapter_path": str(adapter_path),
        "merged": request.merge,
        "message": f"Adapter loaded: {adapter_path.name}",
    }


@router.post("/training/unload-adapter")
async def unload_adapter():
    """Unload the active LoRA adapter, reverting to base weights."""
    from domains.infrastructure.server_state import get_server_state

    provider = get_server_state().model.get()
    if provider is None:
        raise_error("No model loaded", "E_BAD_REQUEST", status_code=400)

    try:
        from domains.training.lora import unload_lora_adapter

        unload_lora_adapter(provider)
    except Exception as e:
        raise_error(f"Failed to unload adapter: {e}", "E_BAD_REQUEST", status_code=400)

    return {"status": "unloaded", "message": "LoRA adapter unloaded"}
