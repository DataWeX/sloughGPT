"""Legacy training endpoints — /train, /train/resolve.

Kept for backward compatibility. New clients should use /training/start.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from fastapi import APIRouter

from schemas.common import raise_error

from .jobs import training_jobs
from .resolution import resolve_training_inputs
from .helpers import _finish_job, _sloughgpt_trainer_kwds
from domains.training.executor import get_training_executor

logger = logging.getLogger("slo")

router = APIRouter(tags=["training-legacy"])


@router.post("/train")
async def train(request):
    """Start a training job (background thread).

    ``SloughGPTTrainer`` writes periodic ``<dataset>_<timestamp>.soul`` checkpoints under
    ``checkpoint_dir`` with ``stoi`` / ``itos`` / ``chars`` for char-LM eval; see
    ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
    """
    from domains.training.dataset_manifest import ManifestError
    from domains.training.train_pipeline import SloughGPTTrainer
    from .schemas import TrainRequest

    request = TrainRequest(**request) if isinstance(request, dict) else request

    try:
        data_path_str, out_stem, manifest_meta, source_kind = resolve_training_inputs(
            request.dataset,
            request.manifest_uri,
            request.dataset_ref,
        )
    except ManifestError as e:
        raise_error(str(e), "E_BAD_REQUEST", status_code=400)

    req_snapshot = request.model_dump()

    def train_model(job_id: str) -> None:
        try:
            training_jobs[job_id]["status"] = "running"
            trainer = SloughGPTTrainer(
                data_path=data_path_str,
                **_sloughgpt_trainer_kwds(req_snapshot),
            )
            if not cancel_event.is_set():
                trainer.train(cancel_event=cancel_event)
            else:
                _finish_job(job_id, "cancelled", "Cancelled before start")
                return
            safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in out_stem)[:120]
            trainer.save(f"models/{safe_stem}_trained.soul")
            _finish_job(job_id, "completed")
            training_jobs[job_id]["checkpoint"] = f"models/{safe_stem}_trained.soul"
        except Exception as e:
            logger.exception("Background /train failed: %s", e, extra={"tag": "TRAIN"})
            _finish_job(job_id, "failed", str(e))

    executor = get_training_executor()
    job_id = f"train_{int(time.time())}"
    cancel_event = threading.Event()
    training_jobs[job_id] = {
        "status": "queued",
        "data_path": data_path_str,
        "output_checkpoint_stem": out_stem,
        "epochs": request.epochs,
        "_cancel_event": cancel_event,
    }

    # Register with CancelManager
    try:
        from domains.infrastructure.cancel_manager import get_cancel_manager, OpType
        _mgr = get_cancel_manager()
        op_id = _mgr.register(
            op_type=OpType.TRAINING,
            label=f"train:{out_stem}",
            cancel_fn=lambda: cancel_event.set(),
        )
        _mgr.start(op_id)
        training_jobs[job_id]["_cancel_manager_op_id"] = op_id
    except Exception as exc:
        logger.warning("CancelManager registration failed for %s: %s", job_id, exc)

    executor.submit(train_model, job_id)

    out: dict[str, Any] = {
        "status": "started",
        "job_id": job_id,
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
async def train_resolve(body) -> dict[str, Any]:
    """Resolve ``data_path`` and checkpoint stem (dry run; no training)."""
    from domains.training.dataset_manifest import ManifestError
    from .schemas import TrainResolveRequest

    body = TrainResolveRequest(**body) if isinstance(body, dict) else body

    try:
        data_path_str, out_stem, manifest_meta, source_kind = resolve_training_inputs(
            body.dataset,
            body.manifest_uri,
            body.dataset_ref,
        )
    except ManifestError as e:
        raise_error(str(e), "E_BAD_REQUEST", status_code=400)

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
