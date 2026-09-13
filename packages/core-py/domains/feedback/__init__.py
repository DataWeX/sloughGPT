"""Backward-compatibility shim — imports from the new ``domain.feedback`` package."""

from domain.feedback import (
    FeedbackDB,
    get_feedback_db,
    Message,
    Feedback,
    SimilarPattern,
    MetaWeightManager,
    MetaWeights,
    get_meta_weight_manager,
    ResponseTracker,
    get_response_tracker,
)
from domain.feedback._internal.workflow import get_feedback_workflow

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
