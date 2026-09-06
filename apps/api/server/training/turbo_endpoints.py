"""Unified turbo and from-sessions endpoints.

Delegates to domains.training.service for state and logic.
"""

from __future__ import annotations

import asyncio
import logging
import threading

from fastapi import APIRouter
from schemas.common import classify_and_raise, safe_audit_log, success_response

from .schemas import FromSessionsRequest, TurboStartRequest

logger = logging.getLogger("slo")

router = APIRouter(tags=["training-turbo"])


@router.get("/training/turbo/status")
async def get_turbo_status():
    from domains.training.service import get_turbo_status

    return get_turbo_status()


@router.post("/training/from-sessions-start")
async def start_from_sessions_unified(request: dict):
    """Start from-sessions training."""
    try:
        req = FromSessionsRequest(**request)

        from domains.training.service import _state, start_from_sessions_training

        config = {
            "epochs": req.epochs,
            "learning_rate": req.learning_rate,
            "batch_size": req.batch_size,
            "n_embed": req.n_embed,
            "n_layer": req.n_layer,
            "n_head": req.n_head,
            "block_size": req.block_size,
            "dropout": req.dropout,
            "soul_name": req.soul_name,
            "min_pair_quality": req.min_pair_quality,
            "max_pairs": req.max_pairs,
            "checkpoint_name": req.checkpoint_name,
            "session_ids": req.session_ids,
            "experiment_id": req.experiment_id,
        }
        built_config = start_from_sessions_training(_state, config)
        safe_audit_log(
            "training.start",
            resource=req.soul_name or "from-sessions",
            detail="from-sessions",
            session_ids=len(req.session_ids) if req.session_ids else 0,
            epochs=req.epochs,
        )
        return success_response(data=built_config, message="Training started")

    except Exception as e:
        classify_and_raise(e, source="training.start_from_sessions")


@router.post("/training/turbo-start")
async def start_turbo_training_unified(request: dict):
    """Start turbo training."""
    try:
        req = TurboStartRequest(**request)

        from domains.training.service import run_turbo_worker, start_turbo_training

        config = req.model_dump()
        job_info = await asyncio.to_thread(start_turbo_training, config)

        # Run in background thread
        threading.Thread(
            target=run_turbo_worker,
            args=(config,),
            name=f"turbo-train-{job_info['job_id']}",
            daemon=True,
        ).start()

        logger.info(
            "Turbo training started: job_id=%s data=%s",
            job_info["job_id"],
            job_info["data_path"],
            extra={"tag": "TRAIN"},
        )
        return success_response(
            data={
                "status": "started",
                "job_id": job_info["job_id"],
                "message": "Turbo training started",
            }
        )

    except Exception as e:
        classify_and_raise(e, source="training.start_turbo")
