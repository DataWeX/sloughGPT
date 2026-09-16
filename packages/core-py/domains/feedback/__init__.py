"""Backward-compatibility shim — imports from the new ``domain.feedback`` package."""

# Allow submodule access (domains.X.Y) for test mocking
import importlib as _importlib

from domain.feedback import (
    Feedback,
    FeedbackDB,
    Message,
    MetaWeightManager,
    MetaWeights,
    ResponseTracker,
    SimilarPattern,
    get_feedback_db,
    get_meta_weight_manager,
    get_response_tracker,
)
from domain.feedback._internal.workflow import get_feedback_workflow


def __getattr__(name):
    try:
        return _importlib.import_module(f"domain.{name}")
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        return _importlib.import_module(f"domain.feedback._internal.{name}")
    except (ImportError, ModuleNotFoundError):
        pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FeedbackDB",
    "get_feedback_db",
    "Message",
    "Feedback",
    "SimilarPattern",
    "MetaWeightManager",
    "MetaWeights",
    "get_meta_weight_manager",
    "ResponseTracker",
    "get_response_tracker",
    "get_feedback_workflow",
]
