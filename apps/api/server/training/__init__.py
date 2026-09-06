"""Training API: schemas, corpus resolution, job store, and HTTP router."""

from .jobs import training_jobs
from .resolution import resolve_training_inputs

try:
    from .router import router
except ImportError:
    import logging

    logging.getLogger("slo.training").warning("Training router failed to import", exc_info=True)
    from fastapi import APIRouter

    router = APIRouter()

from .schemas import (
    DistillStartRequest,
    ExportTextRequest,
    FromSessionsRequest,
    LoadAdapterRequest,
    LoraFinetuneRequest,
    TestWebhookRequest,
    TrainDatasetRef,
    TrainDataSourceBody,
    TrainingRequest,
    TrainRequest,
    TrainResolveRequest,
    TurboStartRequest,
    VisualTrainRequest,
)

__all__ = [
    "router",
    "training_jobs",
    "resolve_training_inputs",
    "TrainDatasetRef",
    "TrainDataSourceBody",
    "TrainRequest",
    "TrainResolveRequest",
    "TrainingRequest",
    "DistillStartRequest",
    "LoraFinetuneRequest",
    "VisualTrainRequest",
    "LoadAdapterRequest",
    "FromSessionsRequest",
    "TurboStartRequest",
    "ExportTextRequest",
    "TestWebhookRequest",
]
