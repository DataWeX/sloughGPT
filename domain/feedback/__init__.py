"""feedback — Feedback storage, meta-weight learning, response tracking.

Public API:
    FeedbackDB, get_feedback_db, Message, Feedback, SimilarPattern
    MetaWeightManager, MetaWeights, get_meta_weight_manager
    ResponseTracker, get_response_tracker
    get_feedback_workflow
"""

from domain.feedback._internal.database import (
    FeedbackDB,
    get_feedback_db,
    Message,
    Feedback,
    SimilarPattern,
)
from domain.feedback._internal.meta_weights import (
    MetaWeightManager,
    MetaWeights,
    get_meta_weight_manager,
)
from domain.feedback._internal.response_tracker import (
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
