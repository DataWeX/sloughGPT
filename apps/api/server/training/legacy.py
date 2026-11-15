"""Legacy training endpoints — /train, /train/resolve.

Kept for backward compatibility. New clients should use /training/start.

Descriptor-projected (card 20261104_151036): ``/train/resolve`` is one
``ToolSpec`` contract → ``create_router`` projection (raw wire,
``envelope=False`` — bare payload); its untyped ``body`` param is declared as
a **required query param**, exactly what hand-built FastAPI published.

**Listed exception — ``/train`` stays hand-built** (brief's exception clause;
baseline entry retained as ``training/legacy.py: 1``). Hand-built FastAPI
publishes its untyped ``request`` as a required query member the handler can
never consume (a JSON-body POST has always 422'd here; a ``?request=`` string
dies at ``TrainRequest(...)`` — no working invocation exists), and the
projection machinery rightly **reserves the name ``request``** for raw-Request
injection: `_check_handler_accepts` refuses to declare it as contract data.
Declaring it under any other name would rename the wire parameter (breaking
URL/param parity), and publishing nothing would drop the 422 every real POST
receives today (breaking runtime parity) — so this site is not projectable
without a contract redesign, which is out of the card's projection bar.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from fastapi import APIRouter
from infrastructure.contract import RouteSpec, create_router
from schemas.common import raise_error

from domain.agents import ToolParam, ToolSpec
from domain.training import get_training_executor

from .helpers import _finish_job, _sloughgpt_trainer_kwds
from .jobs import training_jobs
from .resolution import resolve_training_inputs

logger = logging.getLogger("slo")

router = APIRouter(tags=["training-legacy"])


@router.post("/train")  # retained hand-built exception — see module docstring
async def train(request):
    """Start a training job (background thread).

    ``SloughGPTTrainer`` writes periodic ``<dataset>_<timestamp>.soul`` checkpoints under
    ``checkpoint_dir`` with ``stoi`` / ``itos`` / ``chars`` for char-LM eval; see
    ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
    """
    from domain.training import ManifestError, SloughGPTTrainer

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
                **_sloughgpt_trainer_kwds(req_snapshot, data_path=data_path_str),
            )
            if not cancel_event.is_set():
                trainer.train(cancel_event=cancel_event)
            else:
                _finish_job(job_id, "cancelled", "Cancelled before start")
                return
            safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in out_stem)[:120]
            soul_path = f"models/{safe_stem}_trained.soul"
            trainer.save(soul_path)
            try:
                from domain.infrastructure import soul_to_slnc

                soul_to_slnc(soul_path, soul_path.replace(".soul", ".slnc"))
            except Exception:
                pass
            try:
                from domain.inference import SloNetChatProvider
                from domain.models import register_provider, setup_providers

                prov = SloNetChatProvider.from_soul(soul_path, model_id=safe_stem)
                register_provider("slonet-native", prov)
                register_provider("default", prov)
                try:
                    setup_providers(force_default=True)
                except Exception:
                    pass
            except Exception:
                pass
            _finish_job(job_id, "completed")
            training_jobs[job_id]["checkpoint"] = soul_path
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
        from domain.infrastructure.cancel_manager import OpType, get_cancel_manager

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


async def train_resolve(body) -> dict[str, Any]:
    """Resolve ``data_path`` and checkpoint stem (dry run; no training)."""
    from domain.training import ManifestError

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


# ── descriptor projection (card 20261104_151036) ────────────────────────
#
# One ToolSpec per capability; create_router derives verb, auth, validation,
# envelope and OpenAPI metadata. RouteSpec.summary carries the hand-built
# operation title (FastAPI derives it from the route name as
# ``route.name.replace("_", " ").title()``); spec.description is the handler
# docstring verbatim, so summary/description publish byte-identically.

# ``/train`` is the listed exception (module docstring): the projection
# reserves the param name ``request`` for raw-Request injection, so its
# hand-built declaration stays — baseline entry retained.

SPEC_LEGACY_RESOLVE = ToolSpec(
    name="training.legacy.resolve",
    description="Resolve ``data_path`` and checkpoint stem (dry run; no training).",
    parameters=[ToolParam("body", "string", "", True)],
    execute=train_resolve,
    idempotent=False,
    version="1",
)

router.include_router(
    create_router(
        SPEC_LEGACY_RESOLVE,
        RouteSpec(
            path="/train/resolve",
            input="query",  # hand-built published ``body`` as a required query member
            envelope=False,
            summary="Train Resolve",
        ),
        train_resolve,
    )
)
