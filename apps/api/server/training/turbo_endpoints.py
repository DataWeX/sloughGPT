"""Unified turbo and from-sessions endpoints.

Delegates to domain.training._internal.service for state and logic.
"""

from __future__ import annotations

import asyncio
import logging
import threading

from fastapi import APIRouter
from schemas.common import classify_and_raise, raise_error, safe_audit_log, success_response

from .schemas import FromSessionsRequest, TurboStartRequest

logger = logging.getLogger("slo")

router = APIRouter(tags=["training-turbo"])


@router.get("/training/turbo/status")
async def get_turbo_status():
    from domain.training._internal.service import get_turbo_status

    return get_turbo_status()


@router.post("/training/from-sessions-start")
async def start_from_sessions_unified(req: FromSessionsRequest):
    """Start from-sessions training."""
    try:
        from domain.training._internal.service import _state, start_from_sessions_training

        # Pre-flight validation
        if req.epochs < 1:
            raise_error(422, "epochs must be >= 1", source="training.from_sessions")
        if req.learning_rate <= 0:
            raise_error(422, "learning_rate must be > 0", source="training.from_sessions")
        if req.batch_size < 1:
            raise_error(422, "batch_size must be >= 1", source="training.from_sessions")
        if req.n_embed < 16:
            raise_error(422, "n_embed must be >= 16", source="training.from_sessions")
        if req.n_layer < 1:
            raise_error(422, "n_layer must be >= 1", source="training.from_sessions")
        if req.n_head < 1:
            raise_error(422, "n_head must be >= 1", source="training.from_sessions")
        if req.n_embed % req.n_head != 0:
            raise_error(
                422,
                f"n_embed ({req.n_embed}) must be divisible by n_head ({req.n_head})",
                source="training.from_sessions",
            )
        if req.block_size < 8:
            raise_error(422, "block_size must be >= 8", source="training.from_sessions")

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
async def start_turbo_training_unified(req: TurboStartRequest):
    """Start turbo training."""
    try:
        from domain.shared import find_repo_root
        from domain.training._internal.service import run_turbo_worker, start_turbo_training

        # Pre-flight validation
        if req.source_text and len(req.source_text.strip()) < 200:
            raise_error(
                f"Source text too short for training ({len(req.source_text.strip())} chars, minimum 200)",
                "E_BAD_REQUEST",
                status_code=400,
            )

        if req.source_text and not req.dataset_id:
            from .resolution import materialize_source_text

            try:
                req.dataset_id = materialize_source_text(req.source_text, "turbo-paste")
            except ValueError as e:
                raise_error(str(e), "E_BAD_REQUEST", status_code=400)

        if req.dataset_id:
            from pathlib import Path

            from domain.training._internal.cache_tags import resolve_in_cache

            from .resolution import resolve_legacy_corpus_path

            repo_root = find_repo_root(Path(__file__).resolve())
            ds_path = repo_root / "data" / req.dataset_id
            if not ds_path.exists():
                ds_path = repo_root / "data" / f"{req.dataset_id}.jsonl"
            data_file = None
            if ds_path.is_file():
                data_file = ds_path
            elif ds_path.is_dir():
                # Shared legacy lookup: datasets/{id}/input.txt, data/{id},
                # data/datasets/{id} (corpus.jsonl, input.txt, train.txt,
                # text.txt, *.txt, *.jsonl).
                data_file = resolve_legacy_corpus_path(req.dataset_id)
            if data_file is None and not resolve_in_cache(req.dataset_id):
                raise_error(
                    f"Dataset not found: {req.dataset_id}",
                    "E_BAD_REQUEST",
                    status_code=400,
                )
            if data_file is not None:
                if data_file.stat().st_size < 100:
                    raise_error(
                        f"Dataset file too small ({data_file.stat().st_size} bytes, minimum 100)",
                        "E_BAD_REQUEST",
                        status_code=400,
                    )

        config = req.model_dump()
        job_info = await asyncio.to_thread(start_turbo_training, config)

        # Register with CancelManager for cancellation support
        cancel_event = threading.Event()
        try:
            from domain.infrastructure.cancel_manager import OpType, get_cancel_manager

            get_cancel_manager().register(
                op_type=OpType.TRAINING,
                label=f"turbo:{job_info['job_id']}",
                cancel_fn=lambda: cancel_event.set(),
                meta={"job_id": job_info["job_id"], "method": "turbo"},
                op_id=job_info["job_id"],
            )
            get_cancel_manager().start(job_info["job_id"])
        except Exception as exc:
            logger.warning(
                "CancelManager registration failed for turbo %s: %s", job_info["job_id"], exc
            )

        # Run via executor pool for proper tracking
        from domain.training._internal.executor import get_training_executor

        executor = get_training_executor()

        # NOTE: the executor calls fn(job_id, ...) — _run must accept it.
        def _run(_job_id: str) -> None:
            try:
                run_turbo_worker(config)
            except Exception as exc:
                logger.exception(
                    "Turbo training job %s failed", job_info["job_id"], extra={"tag": "TRAIN"}
                )
                # Backstop: run_turbo_worker handles its own errors, but if it
                # raises before that, surface the cause instead of leaving the
                # job stuck until the heartbeat watchdog fires.
                try:
                    from domain.training._internal.state import _turbo_pause_event
                    from domain.training._internal.turbo import _turbo_lock, _turbo_state

                    with _turbo_lock:
                        _turbo_state["status"] = "error"
                        _turbo_state["error"] = str(exc) or "Turbo training failed"
                        _turbo_state["paused"] = False
                    _turbo_pause_event.clear()
                except Exception:
                    logger.debug("Failed to mark turbo job as error", exc_info=True)

        executor.submit(_run, job_info["job_id"])

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
