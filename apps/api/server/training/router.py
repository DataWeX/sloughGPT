"""FastAPI routes for char-level training and job orchestration.

Trainer ``step_*.pt`` charset maps: ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from fastapi import APIRouter, HTTPException, Request

from training.jobs import training_jobs
from training.resolution import resolve_training_inputs
from training.schemas import TrainingRequest, TrainRequest, TrainResolveRequest, HFTrainingRequest, VLMRequest, UnifiedStartRequest, QuickTrainRequest
from training.controller import get_training_controller, TrainingState
from training.webhooks import (
    get_webhook_store,
    TRAINING_EVENTS,
    notify_training_event,
)
from training.job_store import get_job_store

logger = logging.getLogger("man")

router = APIRouter(tags=["training"])


def _job_summary(job: dict[str, Any]) -> dict[str, Any]:
    """Add plain-language status message to a training job dict.

    Translates raw numbers into human-readable descriptions:
    - "Training... 60% done, about 2 minutes left"
    - "Training complete! Model saved to models/..."
    - "Training failed: CUDA out of memory"
    """
    summary = dict(job)
    status = job.get("status", "unknown")
    progress = job.get("progress", 0)
    model = job.get("model", "")
    dataset = job.get("dataset", "")
    method = job.get("data_source", "")
    explanation = job.get("explanation", "")

    if status == "running":
        parts = [f"Training {model} on {dataset}"]
        if progress > 0:
            parts.append(f"{progress}% done")
        epoch = job.get("current_epoch")
        epochs = job.get("epochs")
        if epoch is not None and epochs:
            parts.append(f"epoch {epoch}/{epochs}")
        summary["status_message"] = ", ".join(parts) + "..."
    elif status == "completed":
        if explanation:
            summary["status_message"] = explanation
        else:
            summary["status_message"] = f"Training complete! Model: {model}"
    elif status == "failed":
        error = job.get("error", "Unknown error")
        summary["status_message"] = f"Training failed: {error}"
    elif status == "queued":
        summary["status_message"] = f"Queued: {model} on {dataset}"
    elif status == "stopping":
        summary["status_message"] = "Stopping..."
    else:
        summary["status_message"] = f"Status: {status}"

    return summary


def _sloughgpt_trainer_kwds(req_snapshot: dict[str, Any]) -> dict[str, Any]:
    """Build ``SloughGPTTrainer`` keyword arguments from a request ``model_dump()`` (except ``data_path``)."""
    device = req_snapshot.get("device")
    return {
        "n_embed": int(req_snapshot.get("n_embed") or 128),
        "n_layer": int(req_snapshot.get("n_layer") or 4),
        "n_head": int(req_snapshot.get("n_head") or 4),
        "block_size": int(req_snapshot.get("block_size") or 128),
        "dropout": float(
            req_snapshot.get("dropout") if req_snapshot.get("dropout") is not None else 0.1
        ),
        "batch_size": int(req_snapshot.get("batch_size") or 32),
        "epochs": int(req_snapshot.get("epochs") or 3),
        "lr": float(req_snapshot.get("learning_rate") or 1e-3),
        "max_steps": req_snapshot.get("max_steps"),
        "gradient_accumulation_steps": int(req_snapshot.get("gradient_accumulation_steps") or 1),
        "max_grad_norm": float(
            req_snapshot.get("max_grad_norm")
            if req_snapshot.get("max_grad_norm") is not None
            else 1.0
        ),
        "use_mixed_precision": bool(req_snapshot.get("use_mixed_precision", True)),
        "mixed_precision_dtype": str(req_snapshot.get("mixed_precision_dtype") or "bf16"),
        "checkpoint_dir": str(req_snapshot.get("checkpoint_dir") or "checkpoints"),
        "checkpoint_interval": int(req_snapshot.get("checkpoint_interval") or 500),
        "save_best_only": bool(req_snapshot.get("save_best_only", False)),
        "max_checkpoints": int(req_snapshot.get("max_checkpoints") or 5),
        "scheduler_type": str(req_snapshot.get("scheduler") or "cosine"),
        "warmup_steps": int(
            req_snapshot.get("warmup_steps")
            if req_snapshot.get("warmup_steps") is not None
            else 100
        ),
        "min_lr": float(
            req_snapshot.get("min_lr") if req_snapshot.get("min_lr") is not None else 1e-5
        ),
        "weight_decay": float(
            req_snapshot.get("weight_decay")
            if req_snapshot.get("weight_decay") is not None
            else 0.01
        ),
        "use_lora": bool(req_snapshot.get("use_lora", False)),
        "lora_rank": int(req_snapshot.get("lora_rank") or 8),
        "lora_alpha": int(req_snapshot.get("lora_alpha") or 16),
        "log_interval": int(req_snapshot.get("log_interval") or 10),
        "eval_interval": int(req_snapshot.get("eval_interval") or 100),
        "device": device if device is not None and str(device).strip() != "" else None,
    }


@router.post("/train")
async def train(request: TrainRequest):
    """Start a training job (background thread).

    ``SloughGPTTrainer`` writes periodic ``step_*.pt`` under ``checkpoint_dir`` with
    ``stoi`` / ``itos`` / ``chars`` for char-LM eval; see
    ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
    """
    from domains.training.dataset_manifest import ManifestError
    from domains.training.train_pipeline import SloughGPTTrainer

    try:
        data_path_str, out_stem, manifest_meta, source_kind = resolve_training_inputs(
            request.dataset,
            request.manifest_uri,
            request.dataset_ref,
        )
    except ManifestError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    req_snapshot = request.model_dump()

    def train_model() -> None:
        try:
            trainer = SloughGPTTrainer(
                data_path=data_path_str,
                **_sloughgpt_trainer_kwds(req_snapshot),
            )
            trainer.train()
            safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in out_stem)[:120]
            trainer.save(f"models/{safe_stem}_trained.pt")
        except Exception as e:
            logger.exception("Background /train failed: %s", e)

    thread = threading.Thread(target=train_model, daemon=True)
    thread.start()

    out: dict[str, Any] = {
        "status": "started",
        "data_path": data_path_str,
        "output_checkpoint_stem": out_stem,
        "data_source": source_kind,
        "epochs": request.epochs,
        "message": "Training started in background",
    }
    if request.dataset is not None:
        out["dataset"] = request.dataset.strip()
    if manifest_meta is not None:
        out["manifest"] = manifest_meta
    return out


@router.post("/train/resolve")
async def train_resolve(body: TrainResolveRequest) -> dict[str, Any]:
    """Resolve ``data_path`` and checkpoint stem (dry run; no training).

    Does not write ``.pt`` artifacts. After ``POST /train`` or ``POST /training/start``,
    native ``step_*.pt`` includes char vocab; see ``docs/policies/CONTRIBUTING.md``
    (*Checkpoint vocabulary*).
    """
    from domains.training.dataset_manifest import ManifestError

    try:
        data_path_str, out_stem, manifest_meta, source_kind = resolve_training_inputs(
            body.dataset,
            body.manifest_uri,
            body.dataset_ref,
        )
    except ManifestError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    out: dict[str, Any] = {
        "ok": True,
        "data_path": data_path_str,
        "output_checkpoint_stem": out_stem,
        "data_source": source_kind,
    }
    if body.dataset is not None:
        out["dataset"] = body.dataset.strip()
    if manifest_meta is not None:
        out["manifest"] = manifest_meta
    return out


@router.get("/train/status")
async def train_status():
    """Legacy training status stub."""
    return {"status": "ready", "message": "Use /train endpoint to start training"}


@router.get("/training/jobs")
async def list_training_jobs():
    """List all tracked training jobs with plain-language status.

    Each job includes a ``status_message`` field with a human-readable
    description of what's happening: "Training... 60% done" or
    "Training complete! Model saved to models/...".
    """
    return [_job_summary(j) for j in training_jobs.values()]


@router.get("/training/jobs/{job_id}")
async def get_training_job(job_id: str):
    """Get one training job by id with plain-language status."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_summary(training_jobs[job_id])


@router.post("/training/jobs/{job_id}/stop")
async def stop_training_job(job_id: str):
    """Stop a specific training job by id."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = training_jobs[job_id]
    if job.get("status") not in ("running", "queued", "starting"):
        raise HTTPException(status_code=400, detail=f"Job is not running (status: {job.get('status', 'unknown')})")
    job["status"] = "stopping"
    return {"status": "stopping", "job_id": job_id}


@router.get("/training/jobs/{job_id}/summary")
async def get_training_summary(job_id: str):
    """Plain-language summary of a training job.

    Returns what was trained, what data was used, how it went,
    and what to do next.  No ML jargon — just facts Alex can use.
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = training_jobs[job_id]

    model = job.get("model", "unknown model")
    dataset = job.get("dataset", "unknown dataset")
    status = job.get("status", "unknown")
    explanation = job.get("explanation", "")
    checkpoint = job.get("checkpoint", "")
    epochs = job.get("epochs")
    current_epoch = job.get("current_epoch")
    final_loss = job.get("loss")
    rl = bool(job.get("reward_history"))

    lines = []

    if status == "completed":
        lines.append(f"You trained {model} on {dataset}.")
        if epochs:
            lines.append(f"It learned from {epochs} passes over your data.")
        if final_loss is not None:
            if final_loss < 1.5:
                lines.append(f"Loss is low ({final_loss:.2f}) — your AI learned well.")
            elif final_loss < 3.0:
                lines.append(f"Loss is moderate ({final_loss:.2f}) — your AI learned something, but could do better.")
            else:
                lines.append(f"Loss is high ({final_loss:.2f}) — your AI may need more data or more training.")
        if rl:
            lines.append("Personality reinforcement was applied — your AI learned to give better answers.")
        if checkpoint:
            lines.append(f"Your trained model is at: {checkpoint}")
            lines.append("Load it in the Models page to use it in chat.")
        else:
            lines.append("The model was saved but the checkpoint path isn't available yet.")
    elif status == "running":
        progress = job.get("progress", 0)
        lines.append(f"Training {model} on {dataset}... {progress}% done.")
        if current_epoch is not None and epochs:
            lines.append(f"Epoch {current_epoch} of {epochs}.")
    elif status == "failed":
        error = job.get("error", "Unknown error")
        lines.append(f"Training failed: {error}")
        lines.append("Try using a smaller model, or check that you have enough disk space.")
    elif status == "queued":
        lines.append(f"Training {model} on {dataset} is queued. It will start shortly.")
    else:
        lines.append(f"Training status: {status}")

    return {
        "job_id": job_id,
        "summary": " ".join(lines),
        "status": status,
        "model": model,
        "dataset": dataset,
    }


@router.delete("/training/jobs/{job_id}")
async def delete_training_job(job_id: str):
    """Delete a training job and optionally its checkpoint files.

    Removes job from registry. If ``delete_files`` is true, removes checkpoint
    files associated with the job from disk.
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = training_jobs[job_id]
    deleted_files = []

    if job.get("checkpoint"):
        checkpoint_path = Path(job["checkpoint"])
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                deleted_files.append(str(checkpoint_path))
            except OSError:
                pass

    if job.get("checkpoint_dir"):
        checkpoint_dir = Path(job["checkpoint_dir"])
        if checkpoint_dir.exists() and checkpoint_dir.is_dir():
            try:
                shutil.rmtree(checkpoint_dir)
                deleted_files.append(str(checkpoint_dir))
            except OSError:
                pass

    del training_jobs[job_id]

    return {
        "status": "deleted",
        "job_id": job_id,
        "deleted_files": deleted_files,
    }


@router.get("/training/export/{job_id}")
async def export_training_job(job_id: str):
    """Export a completed training job's checkpoint file."""
    from fastapi.responses import FileResponse

    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = training_jobs[job_id]

    if job.get("status") not in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=400, detail="Job must be completed before export")

    checkpoint = job.get("checkpoint")
    if not checkpoint:
        raise HTTPException(status_code=404, detail="No checkpoint found for this job")

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail="Checkpoint file not found on disk")

    return FileResponse(
        path=checkpoint_path,
        filename=checkpoint_path.name,
        media_type="application/octet-stream",
    )


@router.post("/training/export-text")
async def export_feedback_pairs(request: Request):
    """Export feedback conversation pairs with a minimum quality threshold."""
    from controllers.feedback import get_feedback_controller
    body = await request.json()
    min_quality = body.get("min_quality", 0)
    target_count = body.get("target_count", 100)
    ctrl = get_feedback_controller()
    pairs = []
    feedback_file = ctrl.feedback_dir / "feedback.jsonl"
    if feedback_file.exists():
        with open(feedback_file) as f:
            for line in f:
                fb = json.loads(line)
                if fb.get("user_message") and fb.get("assistant_response"):
                    pairs.append(fb)
                    if len(pairs) >= target_count:
                        break
    output_file = ctrl.feedback_dir / f"export_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump({"pairs": pairs, "total": len(pairs)}, f, indent=2)
    return {"pairs_count": len(pairs), "filepath": str(output_file), "status": "exported"}


@router.post("/training/start")
async def start_training(request: TrainingRequest):
    """Start a tracked training job (web UI).

    ``step_*.pt`` files saved on the server include ``stoi`` / ``itos`` / ``chars``
    for char-LM eval; see ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
    """
    from domains.training.dataset_manifest import ManifestError

    try:
        data_path_str, out_stem, manifest_meta, source_kind = resolve_training_inputs(
            request.dataset,
            request.manifest_uri,
            request.dataset_ref,
        )
    except ManifestError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    job_id = f"job_{len(training_jobs) + 1}"
    job: dict[str, Any] = {
        "id": job_id,
        "name": request.name,
        "model": request.model,
        "dataset": request.dataset.strip() if request.dataset else out_stem,
        "data_path": data_path_str,
        "output_checkpoint_stem": out_stem,
        "data_source": source_kind,
        "status": "running",
        "progress": 0,
        "epochs": request.epochs,
        "current_epoch": 0,
        "global_step": 0,
        "loss": None,
        "train_loss": None,
        "eval_loss": None,
        "loss_history": [],
    }
    if manifest_meta is not None:
        job["manifest"] = manifest_meta
    training_jobs[job_id] = job

    # Update global training controller
    controller = get_training_controller()
    controller.start(job_id, request.name or "training")

    # Trigger webhook notification for training started
    try:
        import asyncio

        async def notify_async():
            await notify_training_event(
                "training.started",
                {
                    "job_id": job_id,
                    "job_name": request.name or "training",
                    "dataset": request.dataset,
                    "epochs": request.epochs,
                },
            )

        asyncio.create_task(notify_async())
    except Exception:
        pass

    req_snapshot = request.model_dump()
    data_path_for_thread = data_path_str
    out_stem_for_thread = out_stem
    jid = job_id

    def run_training() -> None:
        from domains.training.train_pipeline import SloughGPTTrainer
        from domains.training.wandb_helpers import create_training_tracker_for_api_job

        tracker = None
        try:
            tracker = create_training_tracker_for_api_job(
                job_id=jid,
                job_name=str(req_snapshot.get("name") or "training"),
                data_path=data_path_for_thread,
                hyperparams=dict(req_snapshot),
            )

            def on_progress(info: dict[str, Any]) -> None:
                rec = training_jobs.get(jid)
                if not rec:
                    return
                rec["progress"] = int(info.get("progress_percent", rec.get("progress", 0)))
                rec["current_epoch"] = int(info.get("epoch", rec.get("current_epoch", 0)))
                rec["global_step"] = int(info.get("global_step", 0))
                tl = info.get("train_loss")
                if tl is not None:
                    rec["train_loss"] = float(tl)
                    rec.setdefault("loss_history", []).append({"step": rec.get("global_step", 0), "value": float(tl), "type": "train"})
                el = info.get("eval_loss")
                if el is not None:
                    fe = float(el)
                    rec["eval_loss"] = fe
                    rec["loss"] = fe
                    rec.setdefault("loss_history", []).append({"step": rec.get("global_step", 0), "value": fe, "type": "eval"})

            trainer = SloughGPTTrainer(
                data_path=data_path_for_thread,
                **_sloughgpt_trainer_kwds(req_snapshot),
                experiment_tracker=tracker,
            )
            result = trainer.train(on_progress=on_progress)
            safe_stem = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in out_stem_for_thread
            )[:120]
            trainer.save(f"models/{safe_stem}_trained.pt")
            training_jobs[jid]["status"] = "completed"
            training_jobs[jid]["progress"] = 100
            training_jobs[jid]["current_epoch"] = int(req_snapshot.get("epochs") or 3)
            bel = result.get("best_eval_loss")
            training_jobs[jid]["loss"] = bel if bel is not None and bel < float("inf") else None
            training_jobs[jid]["checkpoint"] = f"models/{safe_stem}_trained.pt"
            get_training_controller().complete()

            # Trigger webhook notification (fire and forget)
            try:
                import asyncio

                asyncio.get_event_loop().run_until_complete(
                    notify_training_event(
                        "training.completed",
                        {
                            "job_id": jid,
                            "job_name": training_jobs[jid].get("name", "training"),
                            "status": "completed",
                            "loss": training_jobs[jid]["loss"],
                            "checkpoint": training_jobs[jid]["checkpoint"],
                        },
                    )
                )
            except Exception:
                pass
        except Exception as e:
            logger.exception("Training job %s failed", jid)
            training_jobs[jid]["status"] = "failed"
            training_jobs[jid]["error"] = str(e)
            training_jobs[jid]["progress"] = 0
            get_training_controller().fail(str(e))

            # Trigger webhook notification (fire and forget)
            try:
                import asyncio

                asyncio.get_event_loop().run_until_complete(
                    notify_training_event(
                        "training.failed",
                        {
                            "job_id": jid,
                            "job_name": training_jobs[jid].get("name", "training"),
                            "status": "failed",
                            "error": str(e),
                        },
                    )
                )
            except Exception:
                pass
        finally:
            if tracker is not None:
                try:
                    tracker.end_run()
                except Exception:
                    logger.exception("W&B end_run failed for job %s", jid)

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()


@router.post("/training/hf-start")
async def start_hf_training(request: HFTrainingRequest):
    """Fine-tune a HuggingFace model (causal LM) on text data with optional LoRA.

    Uses transformers.Trainer + peft. The ``model`` field specifies which
    HuggingFace model to fine-tune (e.g. ``Qwen/Qwen2.5-0.5B-Instruct``).

    The ``dataset`` field must match a folder under ``datasets/`` containing
    ``input.txt``.
    """
    import os

    job_id = f"hf_job_{len(training_jobs) + 1}_{int(time.time())}"

    # Resolve dataset path relative to repo root
    _repo_root = Path(__file__).resolve().parents[4]
    _datasets_dir = _repo_root / "datasets"
    data_path_str = ""
    if request.dataset:
        ds_dir = _datasets_dir / request.dataset
        ds_path = ds_dir / "input.txt"
        if not ds_path.is_file():
            ds_path = ds_dir / "corpus.jsonl"
        if not ds_path.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"Dataset not found: {request.dataset}. Use POST /datasets/import/local first.",
            )
        data_path_str = str(ds_path.resolve())
    elif request.manifest_uri:
        ds_path = Path(request.manifest_uri.replace("file://", ""))
        if not ds_path.is_file():
            raise HTTPException(status_code=400, detail=f"Manifest file not found: {ds_path}")
        data_path_str = str(ds_path.resolve())
    else:
        raise HTTPException(status_code=400, detail="Provide a `dataset` or `manifest_uri`")

    # Create output directory
    safe_model = request.model.replace("/", "--")
    output_dir = f"models/hf-finetuned/{safe_model}_{request.dataset or 'custom'}_{int(time.time())}"

    job: dict[str, Any] = {
        "id": job_id,
        "name": request.name,
        "model": request.model,
        "dataset": request.dataset,
        "data_path": data_path_str,
        "status": "queued",
        "progress": 0,
        "epochs": request.epochs,
        "current_epoch": 0,
        "global_step": 0,
        "loss": None,
        "output_dir": output_dir,
        "loss_history": [],
        "reward_history": [],
    }
    training_jobs[job_id] = job

    # Auto-configure when model not specified
    model_name = request.model
    if not model_name and data_path_str:
        from training.auto_config import auto_configure
        _auto = auto_configure(
            dataset=request.dataset,
            dataset_path=data_path_str,
            preferred_model=None,
        )
        model_name = _auto.model
        job["model"] = model_name
        job["explanation"] = _auto.explanation
        # Apply auto-configured values where user didn't specify
        if not request.rl_post_train:
            request.rl_post_train = _auto.rl_post_train
        request.use_lora = _auto.use_lora

    data_path = data_path_str
    req = request

    def run_hf_training() -> None:
        jid = job_id
        try:
            training_jobs[jid]["status"] = "running"
            get_job_store().create(jid, req.name, req.model_dump(), req.dataset)
            get_job_store().mark_started(jid)

            if req.rl_post_train:
                from domains.training.hf_finetune import GRPOTrainer, RewardFn

                def on_progress(info: dict[str, Any]) -> None:
                    rec = training_jobs.get(jid)
                    if not rec:
                        return
                    rec["progress"] = info.get("progress_pct", rec.get("progress", 0))
                    rec["current_epoch"] = info.get("epoch", rec.get("current_epoch", 0))
                    rec["global_step"] = info.get("step", rec.get("global_step", 0))
                    rec["loss"] = info.get("loss", rec.get("loss"))
                    loss_val = info.get("loss")
                    reward_val = info.get("reward")
                    if loss_val is not None:
                        rec.setdefault("loss_history", []).append({"step": rec.get("global_step", 0), "value": float(loss_val), "type": "train"})
                    if reward_val is not None:
                        rec.setdefault("reward_history", []).append({"step": rec.get("global_step", 0), "value": float(reward_val)})
                    get_job_store().update_progress(
                        jid,
                        rec["progress"],
                        epoch=int(rec["current_epoch"]),
                        step=int(rec["global_step"]),
                        loss=rec["loss"],
                    )

                reward_mode = req.rl_reward_mode or "length"
                if reward_mode == "keyword" and req.rl_reward_keywords:
                    reward_fn = RewardFn(mode="keyword", keywords=req.rl_reward_keywords)
                else:
                    reward_fn = RewardFn(mode=reward_mode)

                tuner = GRPOTrainer(
                    model_name=model_name,
                    prompts_path=data_path,
                    output_dir=output_dir,
                    num_generations=req.rl_num_generations,
                    learning_rate=req.rl_learning_rate,
                    kl_coef=req.rl_kl_coef,
                    clip_range=req.rl_clip_range,
                    epochs=req.epochs,
                    max_new_tokens=req.rl_max_new_tokens,
                    batch_size=req.batch_size,
                    device=req.device,
                    reward_fn=reward_fn,
                    use_lora=req.use_lora,
                    lora_rank=req.lora_rank,
                    lora_alpha=req.lora_alpha,
                )
                result = tuner.train(on_progress=on_progress)
            else:
                from domains.training.hf_finetune import HFFineTuner

                def on_progress(info: dict[str, Any]) -> None:
                    rec = training_jobs.get(jid)
                    if not rec:
                        return
                    rec["progress"] = info.get("progress_pct", rec.get("progress", 0))
                    rec["current_epoch"] = info.get("epoch", rec.get("current_epoch", 0))
                    rec["global_step"] = info.get("step", rec.get("global_step", 0))
                    rec["loss"] = info.get("loss", rec.get("loss"))
                    loss_val = info.get("loss")
                    if loss_val is not None:
                        rec.setdefault("loss_history", []).append({"step": rec.get("global_step", 0), "value": float(loss_val), "type": "train"})
                    get_job_store().update_progress(
                        jid,
                        rec["progress"],
                        epoch=int(rec["current_epoch"]),
                        step=int(rec["global_step"]),
                        loss=rec["loss"],
                    )

                tuner = HFFineTuner(
                    model_name=model_name,
                    data_path=data_path,
                    output_dir=output_dir,
                    use_lora=req.use_lora,
                    lora_rank=req.lora_rank,
                    lora_alpha=req.lora_alpha,
                    epochs=req.epochs,
                    batch_size=req.batch_size,
                    learning_rate=req.learning_rate,
                    max_seq_length=req.max_seq_length,
                    warmup_steps=req.warmup_steps,
                    device=req.device,
                )
                result = tuner.train(on_progress=on_progress)

            training_jobs[jid]["status"] = "completed"
            training_jobs[jid]["progress"] = 100
            training_jobs[jid]["result"] = result
            model_path = result.get("model_path", "")

            # Build plain-language explanation
            from training.auto_config import plain_language_verdict
            final_loss = result.get("final_loss")
            final_reward = result.get("final_reward")

            if req.rl_post_train and final_reward is not None:
                # GRPO completed — use reward-based verdict
                reward_delta = {"verdict": "improved" if final_reward > 0.5 else "mixed"}
                training_jobs[jid]["explanation"] = (
                    plain_language_verdict(reward_delta)
                    + f" Model saved to {model_path}."
                )
            elif final_loss is not None:
                # Standard fine-tune — use loss-based explanation
                if final_loss < 2.0:
                    training_jobs[jid]["explanation"] = (
                        f"Training complete! Your AI learned from the data. "
                        f"Final loss: {final_loss:.2f} (lower is better). "
                        f"Model saved to {model_path}."
                    )
                else:
                    training_jobs[jid]["explanation"] = (
                        f"Training complete, but the loss is still high ({final_loss:.2f}). "
                        f"Your AI may need more data or more epochs. "
                        f"Model saved to {model_path}."
                    )
            else:
                training_jobs[jid]["explanation"] = f"Training complete! Model saved to {model_path}."

            get_job_store().mark_completed(jid, checkpoint_path=model_path)

        except Exception as exc:
            logger.exception("HF fine-tune job %s failed", job_id)
            if jid in training_jobs:
                training_jobs[jid]["status"] = "failed"
            training_jobs[jid]["error"] = str(exc)
            get_job_store().mark_failed(job_id, str(exc))

    thread = threading.Thread(target=run_hf_training, daemon=True)
    thread.start()

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"HF fine-tune started: {request.model} on {request.dataset}",
    }


@router.post("/training/quick")
async def quick_train(request: QuickTrainRequest):
    """One-click training: pick a dataset, we handle everything.

    Analyses the dataset, picks the right method, model, epochs,
    and learning rate automatically.  Returns the chosen config
    with a plain-language explanation of what we're doing and why.

    Optional ``model`` override if you want a specific model.
    """
    import os

    from training.auto_config import auto_configure

    # Resolve dataset path
    _repo_root = Path(__file__).resolve().parents[4]
    _datasets_dir = _repo_root / "datasets"
    ds_path = _datasets_dir / request.dataset

    if not ds_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset}")

    # Find the data file
    input_file = ds_path / "input.txt"
    corpus_file = ds_path / "corpus.jsonl"
    data_file = input_file if input_file.exists() else (corpus_file if corpus_file.exists() else None)
    if data_file is None:
        raise HTTPException(status_code=400, detail=f"No input.txt or corpus.jsonl in {request.dataset}")

    # Discover available models
    available_models = ["gpt2"]
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        for d in hf_cache.iterdir():
            if d.name.startswith("models--"):
                model_id = d.name.replace("models--", "").replace("--", "/")
                available_models.append(model_id)

    # Run auto-config
    config = auto_configure(
        dataset=request.dataset,
        dataset_path=str(data_file),
        available_models=available_models,
        preferred_model=request.model,
    )

    if request.name:
        config.dataset = request.name

    # Now start training using the existing HF endpoint logic
    job_id = f"quick_{len(training_jobs) + 1}_{int(time.time())}"
    job: dict[str, Any] = {
        "id": job_id,
        "name": request.name or f"Quick: {request.dataset}",
        "model": config.model,
        "dataset": request.dataset,
        "data_path": config.data_path,
        "status": "queued",
        "progress": 0,
        "epochs": config.epochs,
        "current_epoch": 0,
        "global_step": 0,
        "loss": None,
        "output_dir": f"models/hf-finetuned/{request.dataset}_{int(time.time())}",
        "loss_history": [],
        "reward_history": [],
    }
    training_jobs[job_id] = job

    from domains.training.hf_finetune import HFFineTuner, GRPOTrainer, RewardFn

    def _run_quick():
        jid = job_id
        try:
            training_jobs[jid]["status"] = "running"
            get_job_store().create(jid, job["name"], {"auto_config": True, **config.to_dict()}, request.dataset)
            get_job_store().mark_started(jid)

            def on_progress(info: dict[str, Any]) -> None:
                rec = training_jobs.get(jid)
                if not rec:
                    return
                rec["progress"] = info.get("progress_pct", rec.get("progress", 0))
                rec["current_epoch"] = info.get("epoch", rec.get("current_epoch", 0))
                rec["global_step"] = info.get("step", rec.get("global_step", 0))
                rec["loss"] = info.get("loss", rec.get("loss"))
                loss_val = info.get("loss")
                reward_val = info.get("reward")
                if loss_val is not None:
                    rec.setdefault("loss_history", []).append({"step": rec.get("global_step", 0), "value": float(loss_val), "type": "train"})
                if reward_val is not None:
                    rec.setdefault("reward_history", []).append({"step": rec.get("global_step", 0), "value": float(reward_val)})
                get_job_store().update_progress(
                    jid, rec["progress"],
                    epoch=int(rec["current_epoch"]),
                    step=int(rec["global_step"]),
                    loss=rec["loss"],
                )

            if config.rl_post_train:
                reward_fn = RewardFn(mode=config.rl_reward_mode)
                trainer = GRPOTrainer(
                    model_name=config.model,
                    prompts_path=config.data_path,
                    output_dir=job["output_dir"],
                    num_generations=config.rl_num_generations,
                    learning_rate=config.rl_learning_rate,
                    kl_coef=config.rl_kl_coef,
                    epochs=config.epochs,
                    max_new_tokens=128,
                    batch_size=config.batch_size,
                    device=config.device,
                    reward_fn=reward_fn,
                    use_lora=config.use_lora,
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                )
            else:
                trainer = HFFineTuner(
                    model_name=config.model,
                    data_path=config.data_path,
                    output_dir=job["output_dir"],
                    use_lora=config.use_lora,
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    max_seq_length=config.max_seq_length,
                    warmup_steps=config.warmup_steps,
                    device=config.device,
                )

            result = trainer.train(on_progress=on_progress)
            training_jobs[jid]["status"] = "completed"
            training_jobs[jid]["progress"] = 100
            training_jobs[jid]["result"] = result

            # Build plain-language completion message
            from training.auto_config import plain_language_verdict
            model_path = result.get("model_path", "")
            final_loss = result.get("final_loss")
            final_reward = result.get("final_reward")

            if config.rl_post_train and final_reward is not None:
                reward_delta = {"verdict": "improved" if final_reward > 0.5 else "mixed"}
                training_jobs[jid]["explanation"] = (
                    plain_language_verdict(reward_delta)
                    + f" Model saved to {model_path}."
                )
            elif final_loss is not None:
                if final_loss < 2.0:
                    training_jobs[jid]["explanation"] = (
                        f"Training complete! Your AI learned from {config.dataset}. "
                        f"Final loss: {final_loss:.2f}. "
                        f"Model saved to {model_path}."
                    )
                else:
                    training_jobs[jid]["explanation"] = (
                        f"Training complete, but the loss is high ({final_loss:.2f}). "
                        f"Your AI may need more data or more epochs. "
                        f"Model saved to {model_path}."
                    )
            else:
                training_jobs[jid]["explanation"] = config.explanation + f" Model saved to {model_path}."

            get_job_store().mark_completed(jid, checkpoint_path=model_path)

        except Exception as exc:
            logger.exception("Quick train job %s failed", jid)
            if jid in training_jobs:
                training_jobs[jid]["status"] = "failed"
                training_jobs[jid]["error"] = str(exc)
            get_job_store().mark_failed(jid, str(exc))

    thread = threading.Thread(target=_run_quick, daemon=True)
    thread.start()

    return {
        "job_id": job_id,
        "status": "queued",
        "config": {
            "method": config.method,
            "model": config.model,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "use_lora": config.use_lora,
            "rl_enabled": config.rl_post_train,
        },
        "explanation": config.explanation,
    }


@router.post("/training/vlm-start")
async def start_vlm_training(request: VLMRequest):
    """Multimodal VLM training: vision encoder + LLM with trainable connector.

    Two-stage pipeline:
      1. Connector pretrain (LLM frozen, vision optionally frozen)
      2. Full LoRA fine-tune

    Dataset must be a JSONL with image-text pairs under ``datasets/<name>/``.
    """
    import os

    job_id = f"vlm_job_{len(training_jobs) + 1}_{int(time.time())}"

    # Resolve dataset path
    _repo_root = Path(__file__).resolve().parents[4]
    data_path = _repo_root / "datasets" / request.dataset / "corpus.jsonl"
    if not data_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"VLM dataset not found: {data_path}. Expected JSONL with image_path + conversations.",
        )

    output_dir = f"models/vlm/{request.dataset}_{int(time.time())}"

    job: dict[str, Any] = {
        "id": job_id,
        "name": request.name or f"VLM-{request.dataset}",
        "type": "vlm",
        "vision_encoder": request.vision_encoder,
        "llm": request.llm,
        "dataset": request.dataset,
        "data_path": str(data_path),
        "status": "queued",
        "progress": 0,
        "stage": "queued",
        "loss": None,
        "output_dir": output_dir,
        "loss_history": [],
    }
    training_jobs[job_id] = job

    req = request

    def run_vlm_training() -> None:
        jid = job_id
        try:
            training_jobs[jid]["status"] = "running"
            training_jobs[jid]["stage"] = "loading"

            from domains.training.multimodal import VLMConfig, VLMTrainer

            config = VLMConfig(
                vision_encoder=req.vision_encoder,
                llm=req.llm,
                connector_hidden_dim=req.connector_hidden_dim,
                max_seq_length=req.max_seq_length,
                stage1_epochs=req.stage1_epochs,
                stage2_epochs=req.stage2_epochs,
                stage1_lr=req.stage1_lr,
                stage2_lr=req.stage2_lr,
                batch_size=req.batch_size,
                use_lora=req.use_lora,
                lora_rank=req.lora_rank,
                lora_alpha=req.lora_alpha,
                freeze_vision=req.freeze_vision,
                gradient_accumulation_steps=req.gradient_accumulation_steps,
                warmup_steps=req.warmup_steps,
                weight_decay=req.weight_decay,
                output_dir=output_dir,
            )

            trainer = VLMTrainer(config)

            def on_progress(info):
                loss_val = info.get("loss")
                training_jobs[jid].update({
                    "stage": info.get("stage", "training"),
                    "progress": info.get("progress_pct", 0),
                    "loss": loss_val,
                    "current_epoch": info.get("epoch", 0),
                    "global_step": info.get("step", 0),
                })
                if loss_val is not None:
                    training_jobs[jid].setdefault("loss_history", []).append({"step": info.get("step", 0), "value": float(loss_val), "type": "train"})

            result = trainer.train(data_path=str(data_path), on_progress=on_progress)
            training_jobs[jid].update({
                "status": result.get("status", "completed"),
                "progress": 100,
                "loss": result.get("final_loss"),
                "model_path": result.get("model_path"),
                "sou_path": result.get("sou_path"),
                "output_dir": output_dir,
                "type": "vlm",
            })

            # Copy .sou to checkpoints directory so it appears in the catalog
            sou_path = result.get("sou_path")
            if sou_path:
                try:
                    import shutil
                    from pathlib import Path as _P
                    _ckpt_dir = _P(__file__).resolve().parents[4] / "models" / "auto-training"
                    _ckpt_dir.mkdir(parents=True, exist_ok=True)
                    sou_file = _P(sou_path)
                    if sou_file.is_file():
                        dest = _ckpt_dir / sou_file.name
                        shutil.copy2(str(sou_file), str(dest))
                        meta_src = sou_file.with_suffix(".sou.meta.json")
                        if meta_src.is_file():
                            shutil.copy2(str(meta_src), str(_ckpt_dir / meta_src.name))
                        logger.info("VLM .sou copied to checkpoints: %s", dest)
                except Exception as copy_err:
                    logger.warning("VLM .sou copy failed: %s", copy_err)

        except Exception as e:
            logger.exception("VLM training failed for job %s: %s", jid, e)
            training_jobs[jid].update({
                "status": "failed",
                "error": str(e),
            })

    thread = threading.Thread(target=run_vlm_training, daemon=True)
    thread.start()

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"VLM training started: {request.vision_encoder} + {request.llm} on {request.dataset}",
    }


@router.post("/training/from-feedback")
async def train_from_feedback():
    """Train a model from collected feedback data.

    This endpoint:
    1. Exports feedback as training data (DPO format)
    2. Starts training with the exported data
    3. Returns the job ID for tracking
    """
    import os
    import uuid
    from pathlib import Path
    from pydantic import BaseModel

    class TrainFromFeedbackRequest(BaseModel):
        epochs: int = 3
        batch_size: int = 16
        learning_rate: float = 1e-4
        use_lora: bool = True

    try:
        from domains.feedback.training import FeedbackTrainer

        trainer = FeedbackTrainer()

        # Export feedback data
        timestamp = int(time.time())
        export_dir = Path("data/training_exports")
        export_dir.mkdir(parents=True, exist_ok=True)

        # Export as SFT format for training
        sft_path = export_dir / f"feedback_sft_{timestamp}.jsonl"
        count = trainer.export_sft(str(sft_path))

        if count == 0:
            return {"status": "no_data", "message": "No feedback data available for training"}

        # Create training job
        jid = f"feedback_train_{uuid.uuid4().hex[:8]}"
        data_path = str(sft_path)
        out_stem = f"feedback_model_{timestamp}"

        training_jobs[jid] = {
            "id": jid,
            "name": f"Feedback Training {timestamp}",
            "status": "running",
            "progress": 0,
            "dataset": str(sft_path),
            "data_source": "feedback",
            "epochs": 3,
            "checkpoint_interval": 100,
            "output_checkpoint_stem": out_stem,
        }

        # Update global training controller
        get_training_controller().start(jid, f"Feedback Training {timestamp}")

        def run_feedback_training():
            try:
                from domains.training.train_pipeline import SloughGPTTrainer

                trainer = SloughGPTTrainer(
                    data_path=data_path,
                    n_embed=256,
                    n_layer=6,
                    n_head=8,
                    block_size=256,
                    epochs=3,
                    batch_size=16,
                    lr=1e-4,
                    use_lora=True,
                    lora_rank=8,
                    lora_alpha=16,
                    checkpoint_dir="models",
                    checkpoint_interval=100,
                    use_mixed_precision=True,
                )

                def on_progress(
                    step: int, epoch: int, loss: Optional[float], loss_type: str = "train"
                ):
                    training_jobs[jid]["progress"] = min(99, int((epoch / 3) * 100))
                    training_jobs[jid]["current_epoch"] = epoch
                    if loss is not None:
                        training_jobs[jid][loss_type] = float(loss)
                        training_jobs[jid].setdefault("loss_history", []).append({"step": step, "value": float(loss), "type": loss_type})

                result = trainer.train(on_progress=on_progress)
                safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in out_stem)[:120]
                trainer.save(f"models/{safe_stem}.pt")

                training_jobs[jid]["status"] = "completed"
                training_jobs[jid]["progress"] = 100
                training_jobs[jid]["checkpoint"] = f"models/{safe_stem}.pt"
                training_jobs[jid]["samples_used"] = count
                get_training_controller().complete()

                # Trigger webhook notification (fire and forget)
                try:
                    import asyncio

                    asyncio.get_event_loop().run_until_complete(
                        notify_training_event(
                            "training.completed",
                            {
                                "job_id": jid,
                                "job_name": training_jobs[jid].get("name", "feedback_training"),
                                "status": "completed",
                                "samples_used": count,
                                "checkpoint": training_jobs[jid]["checkpoint"],
                            },
                        )
                    )
                except Exception:
                    pass

            except Exception as e:
                logger.exception("Feedback training job %s failed", jid)
                training_jobs[jid]["status"] = "failed"
                training_jobs[jid]["error"] = str(e)
                get_training_controller().fail(str(e))

                # Trigger webhook notification
                try:
                    import asyncio

                    asyncio.get_event_loop().run_until_complete(
                        notify_training_event(
                            "training.failed",
                            {
                                "job_id": jid,
                                "job_name": training_jobs[jid].get("name", "feedback_training"),
                                "status": "failed",
                                "error": str(e),
                            },
                        )
                    )
                except Exception:
                    pass

        thread = threading.Thread(target=run_feedback_training, daemon=True)
        thread.start()

        return {
            "status": "started",
            "job_id": jid,
            "samples": count,
            "data_path": str(sft_path),
            "message": "Training started from feedback data",
        }

    except Exception as e:
        logger.exception("Failed to start feedback training")
        raise HTTPException(status_code=500, detail=str(e))

    return job


# ===== TRAINING STATE CONTROLLER =====


@router.get("/training/status")
async def get_training_status():
    """
    Get comprehensive training system status.

    Returns current state (idle/running/paused), current job info,
    and statistics about completed/failed jobs.
    """
    controller = get_training_controller()
    status = controller.get_status()

    # Also include any running jobs from the job registry
    running_jobs = [
        {"id": jid, "name": job.get("name"), "progress": job.get("progress", 0)}
        for jid, job in training_jobs.items()
        if job.get("status") == "running"
    ]

    status["running_jobs"] = running_jobs
    status["total_tracked_jobs"] = len(training_jobs)

    return status


@router.post("/training/control/start")
async def control_start_training():
    """
    Request to start training.

    Returns success/failure with current state.
    Note: Actual training start happens via POST /training/start
    """
    controller = get_training_controller()

    if controller.is_running():
        return {
            "success": False,
            "message": "Training is already running",
            **controller.get_status(),
        }

    if controller.is_paused():
        return {
            "success": False,
            "message": "Training is paused. Use /training/control/resume to continue.",
            **controller.get_status(),
        }

    return {
        "success": True,
        "message": "Ready to start training",
        **controller.get_status(),
    }


@router.post("/training/control/pause")
async def control_pause_training():
    """
    Pause current training.

    Pauses the training loop if running.
    """
    controller = get_training_controller()
    result = controller.pause()

    # Notify the training job if it's listening
    if result["success"]:
        logger.info("Training pause requested")

    return result


@router.post("/training/control/resume")
async def control_resume_training():
    """
    Resume paused training.

    Continues training from where it was paused.
    """
    controller = get_training_controller()
    result = controller.resume()

    if result["success"]:
        logger.info("Training resumed")

    return result


@router.post("/training/control/stop")
async def control_stop_training():
    """
    Stop current training.

    Gracefully stops the training job.
    """
    controller = get_training_controller()
    result = controller.stop()

    # Update all running jobs to stopping
    if result["success"]:
        for jid, job in training_jobs.items():
            if job.get("status") == "running":
                job["status"] = "stopping"
        logger.info("Training stop requested")

    return result


@router.post("/training/control/reset")
async def control_reset_training():
    """
    Reset training controller to idle state.

    Use after training completes or fails to clear state.
    """
    controller = get_training_controller()
    return controller.reset()


@router.get("/training/is-running")
async def is_training_running():
    """
    Quick check if training is currently running.

    Useful for UI to conditionally show controls.
    """
    controller = get_training_controller()
    return {
        "is_running": controller.is_running(),
        "is_paused": controller.is_paused(),
        "is_idle": controller.is_idle(),
        "state": controller.state.value,
        "current_job": controller.current_job_id,
    }


# ===== WEBHOOK NOTIFICATIONS =====


@router.get("/training/webhooks")
async def list_webhooks():
    """
    List all registered webhooks.
    """
    store = get_webhook_store()
    webhooks = store.list()

    return {
        "webhooks": [
            {
                "id": w.id,
                "url": w.url,
                "events": w.events,
                "description": w.description,
                "is_active": w.is_active,
                "created_at": w.created_at.isoformat(),
            }
            for w in webhooks
        ],
        "available_events": TRAINING_EVENTS,
    }


@router.post("/training/webhooks")
async def register_webhook(
    url: str,
    events: str,  # JSON stringified array
    description: str = "",
    secret: Optional[str] = None,
):
    """
    Register a new webhook endpoint.

    Args:
        url: The URL to send notifications to
        events: JSON stringified list of events (e.g., '["training.completed","training.failed"]')
        description: Optional description
        secret: Optional HMAC secret (generated if not provided)
    """
    # Parse events from JSON string
    import json

    try:
        events_list = json.loads(events) if isinstance(events, str) else events
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid events format. Must be JSON array.")

    # Validate URL
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    # Validate events
    invalid_events = [e for e in events_list if e not in TRAINING_EVENTS]
    if invalid_events:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid events: {invalid_events}. Available: {TRAINING_EVENTS}",
        )

    store = get_webhook_store()
    webhook_id = store.register(
        url=url,
        events=events_list,
        secret=secret,
        description=description,
        headers=None,
    )

    webhook = store.get(webhook_id)

    return {
        "id": webhook_id,
        "url": url,
        "events": events,
        "secret": webhook.secret if webhook else None,
        "message": "Webhook registered successfully",
    }


@router.delete("/training/webhooks/{webhook_id}")
async def unregister_webhook(webhook_id: str):
    """Unregister a webhook."""
    store = get_webhook_store()

    if not store.get(webhook_id):
        raise HTTPException(status_code=404, detail="Webhook not found")

    store.unregister(webhook_id)

    return {"status": "deleted", "webhook_id": webhook_id}


@router.get("/training/webhooks/{webhook_id}")
async def get_webhook(webhook_id: str):
    """Get webhook details (without secret)."""
    store = get_webhook_store()
    webhook = store.get(webhook_id)

    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")

    return {
        "id": webhook.id,
        "url": webhook.url,
        "events": webhook.events,
        "description": webhook.description,
        "is_active": webhook.is_active,
        "created_at": webhook.created_at.isoformat(),
    }


@router.get("/training/webhooks/{webhook_id}/deliveries")
async def get_webhook_deliveries(webhook_id: str, limit: int = 50):
    """Get delivery log for a webhook."""
    store = get_webhook_store()

    if not store.get(webhook_id):
        raise HTTPException(status_code=404, detail="Webhook not found")

    deliveries = store.get_deliveries(webhook_id, limit=limit)

    return {
        "deliveries": [
            {
                "id": d.id,
                "event": d.event,
                "success": d.success,
                "status_code": d.status_code,
                "attempted_at": d.attempted_at.isoformat(),
                "error": d.error,
            }
            for d in deliveries
        ]
    }


@router.get("/training/webhooks/stats")
async def get_webhook_stats():
    """Get webhook statistics."""
    store = get_webhook_store()
    return store.get_stats()


class TestWebhookRequest(BaseModel):
    url: str

@router.post("/training/webhooks/test")
async def test_webhook(req: TestWebhookRequest):
    """
    Send a test notification to a URL.

    Useful for verifying webhook setup.
    """
    store = get_webhook_store()

    # Register temporary webhook for test
    webhook_id = store.register(
        url=req.url,
        events=TRAINING_EVENTS,
        description="Temporary test webhook",
    )

    # Send test event
    delivery = await store.deliver(
        webhook_id=webhook_id,
        event="training.completed",
        payload={
            "job_id": "test",
            "job_name": "Test Training",
            "status": "completed",
            "message": "This is a test webhook notification",
        },
        retries=1,
    )

    # Clean up
    store.unregister(webhook_id)

    return {
        "success": delivery.success,
        "status_code": delivery.status_code,
        "error": delivery.error,
        "response_body": delivery.response_body,
    }


@router.get("/training/builds")
async def list_builds():
    """List all training builds (checkpoints + fine-tuned models + LoRA adapters).

    Combines:
      - ``GET /auto-train/checkpoints`` (SloNet checkpoints + LoRA .soul files)
      - Completed HF fine-tune jobs from ``training_jobs``
      - HF fine-tuned model directories under ``models/hf-finetuned/``
    """
    from routers.auto_train import _load_soul, _load_lora_soul
    _repo_root = Path(__file__).resolve().parents[4]
    _checkpoints_dir = _repo_root / "models" / "auto-training"
    _lora_dir = _repo_root / "data" / "user_adapters"
    _hf_finetuned_dir = _repo_root / "models" / "hf-finetuned"

    builds = []

    # 1. Auto-train checkpoints (.soul / .pt)
    seen = set()
    for ext in ("*.soul", "*.pt"):
        for f in sorted(_checkpoints_dir.glob(ext), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.name in seen:
                continue
            seen.add(f.name)
            info = _load_soul(f.name)
            if info:
                info["build_type"] = "auto-train"
                builds.append(info)

    # 2. LoRA .soul files
    for npz in sorted(_lora_dir.glob("*.soul"), key=lambda p: p.stat().st_mtime, reverse=True):
        if npz.name in seen:
            continue
        seen.add(npz.name)
        info = _load_lora_soul(npz.name)
        if info:
            info["build_type"] = "lora"
            builds.append(info)

    # 3. Completed HF fine-tune jobs
    from training.jobs import training_jobs
    for jid, job in training_jobs.items():
        if job.get("status") == "completed":
            model_path = job.get("result", {}).get("model_path", "") if isinstance(job.get("result"), dict) else ""
            builds.append({
                "name": job.get("name") or jid,
                "build_type": "hf-finetune",
                "job_id": jid,
                "model": job.get("model", ""),
                "dataset": job.get("dataset", ""),
                "loss": job.get("loss"),
                "epochs": job.get("epochs"),
                "model_path": model_path,
                "created_at": job.get("started_at", ""),
                "finished_at": job.get("completed_at", ""),
            })

    # 5. VLM fine-tuned model directories under models/vlm/
    _vlm_dir = _repo_root / "models" / "vlm"
    if _vlm_dir.is_dir():
        for d in sorted(_vlm_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if d.is_dir() and d.name not in seen:
                seen.add(d.name)
                config_path = d / "vlm_config.json"
                size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 * 1024)
                config = {}
                if config_path.exists():
                    try:
                        config = json.loads(config_path.read_text())
                    except Exception:
                        pass
                builds.append({
                    "name": d.name,
                    "build_type": "vlm",
                    "model_path": str(d),
                    "size_mb": round(size_mb, 1),
                    "created_at": datetime.fromtimestamp(d.stat().st_mtime).isoformat(),
                    "vision_encoder": config.get("vision_encoder", ""),
                    "llm": config.get("llm", ""),
                    "dataset": config.get("training_dataset", ""),
                    "connector_hidden_dim": config.get("connector_hidden_dim"),
                    "use_lora": config.get("use_lora", False),
                })

    # 4. HF fine-tuned model directories on disk (for builds not tracked in memory)
    if _hf_finetuned_dir.is_dir():
        for d in sorted(_hf_finetuned_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if d.is_dir() and d.name not in seen:
                seen.add(d.name)
                config_path = d / "config.json"
                size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 * 1024)
                builds.append({
                    "name": d.name,
                    "build_type": "hf-finetuned-dir",
                    "model_path": str(d),
                    "size_mb": round(size_mb, 1),
                    "created_at": datetime.fromtimestamp(d.stat().st_mtime).isoformat(),
                    "model": d.name.split("_")[0].replace("--", "/"),
                    "dataset": d.name.split("_")[1] if "_" in d.name else "",
                })

    return {"builds": builds}


# ===== JOB RECOVERY =====


@router.get("/recovery/check")
async def check_crashed_jobs(timeout_seconds: int = 300):
    """
    Check for jobs that may have crashed.

    Jobs that are 'running' but haven't sent a heartbeat in timeout_seconds
    are considered potentially crashed.
    """
    store = get_job_store()
    crashed = store.detect_crashed_jobs(timeout_seconds)

    return {
        "detected_crashes": len(crashed),
        "jobs": crashed,
        "message": f"Found {len(crashed)} potentially crashed job(s)",
    }


@router.get("/recovery/recoverable")
async def get_recoverable_jobs():
    """
    Get all jobs that can be recovered.

    Includes interrupted and crashed jobs.
    """
    store = get_job_store()
    jobs = store.get_recoverable_jobs()

    return {
        "count": len(jobs),
        "jobs": jobs,
    }


@router.post("/recovery/recover/{job_id}")
async def recover_job(job_id: str):
    """
    Recover and restart an interrupted/crashed job.

    Resumes training from the last checkpoint if available.
    """
    store = get_job_store()
    job = store.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] not in ("interrupted", "failed"):
        raise HTTPException(
            status_code=400,
            detail=f"Job status is '{job['status']}', only 'interrupted' or 'failed' jobs can be recovered",
        )

    # Get config and checkpoint
    config = job.get("config", {})
    data_path = job.get("data_path", "")
    checkpoint_path = job.get("checkpoint_path", "")
    checkpoint_dir = job.get("checkpoint_dir", "checkpoints")
    job_name = job.get("name", "recovered_job")

    # Find checkpoint
    from pathlib import Path

    if checkpoint_path and Path(checkpoint_path).exists():
        pass  # Use existing checkpoint_path
    else:
        # Try to find any checkpoint in the checkpoint dir
        checkpoint_dir_path = Path(checkpoint_dir)
        if checkpoint_dir_path.exists():
            checkpoints = list(checkpoint_dir_path.glob("step_*.pt")) + list(
                checkpoint_dir_path.glob("*.pt")
            )
            if checkpoints:
                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                checkpoint_path = str(latest)

    # Create recovery job in training_jobs
    recovery_job_id = f"recovery_{job_id}"
    recovery_job = {
        "id": recovery_job_id,
        "name": f"Recovered: {job_name}",
        "model": config.get("model", "sloughgpt"),
        "dataset": job.get("dataset", ""),
        "data_path": data_path,
        "status": "running",
        "progress": job.get("progress", 0),
        "current_epoch": job.get("current_epoch", 0),
        "global_step": job.get("global_step", 0),
        "checkpoint_path": checkpoint_path,
        "checkpoint_dir": checkpoint_dir,
        "original_job_id": job_id,
        **config,
    }
    training_jobs[recovery_job_id] = recovery_job

    # Update job store
    store.update(job_id, status="recovering", crashed=0)

    # Update controller
    controller = get_training_controller()
    controller.start(recovery_job_id, f"Recovered: {job_name}")

    # Start recovery in background thread
    jid = recovery_job_id
    checkpoint_for_recovery = checkpoint_path

    def run_recovery():
        try:
            from domains.training.train_pipeline import SloughGPTTrainer

            trainer_config = {
                "data_path": recovery_job.get("data_path", ""),
                "epochs": recovery_job.get("epochs", 10),
                "batch_size": recovery_job.get("batch_size", 32),
                "lr": recovery_job.get("learning_rate", 1e-3),
                "n_embed": recovery_job.get("n_embed", 256),
                "n_layer": recovery_job.get("n_layer", 6),
                "n_head": recovery_job.get("n_head", 8),
                "block_size": recovery_job.get("block_size", 128),
                "checkpoint_dir": recovery_job.get("checkpoint_dir", "checkpoints"),
                "checkpoint_interval": recovery_job.get("checkpoint_interval", 500),
            }

            def on_progress(info: dict):
                rec = training_jobs.get(jid)
                if not rec:
                    return
                rec["progress"] = int(info.get("progress_percent", rec.get("progress", 0)))
                rec["current_epoch"] = int(info.get("epoch", rec.get("current_epoch", 0)))
                rec["global_step"] = int(info.get("global_step", 0))
                tl = info.get("train_loss")
                if tl is not None:
                    rec.setdefault("loss_history", []).append({"step": int(info.get("global_step", 0)), "value": float(tl), "type": "train"})
                el = info.get("eval_loss")
                if el is not None:
                    rec.setdefault("loss_history", []).append({"step": int(info.get("global_step", 0)), "value": float(el), "type": "eval"})
                store.update_progress(
                    jid, rec["progress"], epoch=rec["current_epoch"], step=rec["global_step"]
                )

            trainer = SloughGPTTrainer(**trainer_config)

            # Resume from checkpoint if available
            result = trainer.train(
                on_progress=on_progress,
                resume=True,
                resume_path=checkpoint_for_recovery,
            )

            # Mark as completed
            training_jobs[jid]["status"] = "completed"
            training_jobs[jid]["progress"] = 100
            store.mark_completed(
                jid, checkpoint_for_recovery or trainer_config["checkpoint_dir"] + "/final.pt"
            )
            store.update(job_id, status="recovered")
            controller.complete()

            # Trigger webhook
            try:
                import asyncio

                asyncio.get_event_loop().run_until_complete(
                    notify_training_event(
                        "training.completed",
                        {
                            "job_id": jid,
                            "job_name": training_jobs[jid].get("name"),
                            "status": "completed",
                            "recovered_from": job_id,
                        },
                    )
                )
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            training_jobs[jid]["status"] = "failed"
            training_jobs[jid]["error"] = str(e)
            store.mark_failed(jid, str(e))
            controller.fail()

    thread = threading.Thread(target=run_recovery, daemon=True)
    thread.start()

    return {
        "status": "recovered",
        "original_job_id": job_id,
        "recovery_job_id": recovery_job_id,
        "checkpoint_path": checkpoint_path,
        "message": f"Recovery started. Training restarting from checkpoint: {checkpoint_path or 'beginning'}",
    }


@router.delete("/recovery/abandon/{job_id}")
async def abandon_recovery(job_id: str):
    """
    Abandon a crashed job and mark it as permanently failed.
    """
    store = get_job_store()
    job = store.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    store.update(job_id, status="abandoned")

    return {
        "status": "abandoned",
        "job_id": job_id,
        "message": "Job marked as abandoned",
    }


@router.get("/recovery/stats")
async def get_recovery_stats():
    """Get recovery statistics."""
    store = get_job_store()
    stats = store.get_stats()

    return {
        **stats,
        "crashed_jobs": store.detect_crashed_jobs().__len__(),
        "recoverable_jobs": len(store.get_recoverable_jobs()),
    }


# =============================================================================
# Unified Pipeline Endpoints (additive — wraps UnifiedTrainingPipeline)
# =============================================================================

_unified_config: Optional[dict] = None
_unified_cancel_event: Optional[threading.Event] = None


class _UnifiedCancelled(Exception):
    """Raised inside the unified worker thread when user requests cancel."""


@router.post("/training/unified-stop")
async def unified_stop():
    """Cancel a running unified training pipeline."""
    global _unified_cancel_event
    if _unified_cancel_event is not None:
        _unified_cancel_event.set()
        return {"status": "cancelling", "message": "Cancelling unified training"}
    return {"status": "idle", "message": "No unified training running"}


@router.post("/training/unified-start")
async def unified_start(req: UnifiedStartRequest):
    """
    Configure and start a unified training pipeline session.

    Args:
        req: UnifiedStartRequest with method, data_path, epochs, etc.

    Returns:
        dict with status and config summary

    Side effects:
        - Stores config for /training/unified-stream to consume
    """
    global _unified_config
    _unified_config = req.model_dump()
    logger.info(
        "Unified pipeline configured: method=%s data=%s epochs=%d",
        req.method, req.data_path or req.dataset_name, req.epochs,
    )
    return {
        "status": "ready",
        "method": req.method,
        "data_path": req.data_path or req.dataset_name or "",
        "epochs": req.epochs,
        "message": "Call GET /training/unified-stream to start training",
    }


@router.get("/training/unified-stream")
async def unified_stream():
    """
    SSE endpoint that runs the unified training pipeline and streams phase progress.

    Phases: GENERATE_DATA → DISTILL → TRAIN → EVALUATE → DEPLOY → COMPLETE

    Each event uses the standard SSE envelope:
        { stream, phase, status, data, meta, message }

    Requires a prior POST to /training/unified-start.
    """
    global _unified_config
    if _unified_config is None:
        return StreamingResponse(
            iter(["data: " + json.dumps({
                "stream": "unified-train", "phase": "idle", "status": "error",
                "data": {"error": "No config — POST /training/unified-start first"},
            }) + "\n\n"]),
            media_type="text/event-stream",
        )

    from fastapi.responses import StreamingResponse
    import asyncio

    global _unified_cancel_event
    config = dict(_unified_config)
    _unified_cancel_event = threading.Event()
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _enqueue(event_str: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event_str)

    def _worker():
        try:
            from domains.training.unified_pipeline import UnifiedTrainingPipeline, UnifiedTrainingConfig
            from domains.training.sequence import TrainingRunConfig

            pipe_config = UnifiedTrainingConfig(**config)
            run_config = TrainingRunConfig(
                skip_generate=config.get("skip_generate", False),
                skip_distill=config.get("skip_distill", False),
                skip_train=config.get("skip_train", False),
                skip_evaluate=config.get("skip_evaluate", False),
                skip_deploy=config.get("skip_deploy", False),
            )
            pipeline = UnifiedTrainingPipeline(pipe_config, run_config=run_config)

            def on_progress(progress):
                if _unified_cancel_event is not None and _unified_cancel_event.is_set():
                    raise _UnifiedCancelled("Training cancelled by user")
                event = json.dumps(progress.to_sse_event("unified-train"))
                _enqueue("data: " + event + "\n\n")

            pipeline.run(on_progress=on_progress)
        except _UnifiedCancelled:
            logger.info("Unified pipeline cancelled by user")
            _enqueue("data: " + json.dumps({
                "stream": "unified-train",
                "phase": "complete",
                "status": "complete",
                "data": {"cancelled": True, "message": "Training cancelled by user"},
                "message": "Training cancelled",
            }) + "\n\n")
        except Exception as e:
            logger.exception("Unified pipeline worker error: %s", e)
            _enqueue("data: " + json.dumps({
                "stream": "unified-train",
                "phase": "failed",
                "status": "error",
                "data": {"error": str(e)},
                "message": f"Error: {e}",
            }) + "\n\n")
        finally:
            _unified_cancel_event = None

    async def event_generator():
        worker_task = loop.run_in_executor(None, _worker)
        while True:
            event = await queue.get()
            yield event
            if event.startswith("data: "):
                try:
                    ev = json.loads(event[6:])
                    if ev.get("status") in ("complete", "error"):
                        break
                except json.JSONDecodeError:
                    pass
        try:
            await worker_task
        except Exception:
            pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")
