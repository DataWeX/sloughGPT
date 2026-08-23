"""
Feedback Router - MVC View layer
"""
import logging
from fastapi import APIRouter, Query

from pydantic import BaseModel, Field
from schemas.feedback import FeedbackRequest, FeedbackResponse, FeedbackStats, ConversationCreate, ConversationUpdate, ConversationResponse
from schemas.common import raise_error, success_response, classify_and_raise, safe_audit_log
from controllers.feedback import get_feedback_controller

logger = logging.getLogger("slo.api.feedback")


class WorkflowFeedbackRequest(BaseModel):
    """Schema for workflow feedback recording."""
    conversation_id: str = Field(..., max_length=256)
    rating: str = Field(..., pattern=r'^(thumbs_up|thumbs_down|neutral)$')
    assistant_response: str = Field('', max_length=10000)
    user_message: str = Field('', max_length=10000)


class FeedbackRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/feedback", tags=["feedback"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/workflow-record", self.record_feedback_workflow, methods=["POST"])
        self.router.add_api_route("", self.record_feedback, methods=["POST"], response_model=FeedbackResponse)
        self.router.add_api_route("/stats/summary", self.get_feedback_stats, methods=["GET"], response_model=FeedbackStats)
        self.router.add_api_route("/conversations", self.create_conversation, methods=["POST"], response_model=ConversationResponse)
        self.router.add_api_route("/conversations", self.list_conversations, methods=["GET"], response_model=list[ConversationResponse])
        self.router.add_api_route("/conversations/{conv_id}", self.get_conversation, methods=["GET"], response_model=ConversationResponse)
        self.router.add_api_route("/conversations/{conv_id}", self.update_conversation, methods=["PATCH"], response_model=ConversationResponse)
        self.router.add_api_route("/conversations/{conv_id}", self.delete_conversation, methods=["DELETE"])
        self.router.add_api_route("/{message_id}", self.get_feedback, methods=["GET"])

    async def record_feedback_workflow(self, req: WorkflowFeedbackRequest) -> dict:
        """Record user feedback (workflow variant used by frontend feedback store)."""
        try:
            from controllers.feedback import get_feedback_controller
            ctrl = get_feedback_controller()
            feedback = ctrl.record_feedback(
                message_id=req.conversation_id,
                rating=req.rating,
                session_id=req.conversation_id,
                message_content=req.assistant_response,
                user_message=req.user_message,
                assistant_response=req.assistant_response,
            )
            safe_audit_log("feedback.record_workflow", resource=req.conversation_id, detail=f"rating={req.rating}")
            return success_response(data={
                "feedback_id": feedback.get("feedback_id", ""),
                "workflow_active": True,
            }, message="recorded")
        except Exception as e:
            logger.error("Failed to record workflow feedback (conversation=%s): %s", req.conversation_id, e)
            raise

    async def record_feedback(self, req: FeedbackRequest) -> dict:
        """Record user feedback and pipe into learning systems."""
        try:
            ctrl = get_feedback_controller()
            feedback = ctrl.record_feedback(
                message_id=req.message_id,
                rating=req.rating,
                session_id=req.session_id,
                message_content=req.message_content,
                user_message=getattr(req, 'user_message', None),
                assistant_response=getattr(req, 'assistant_response', None),
            )
            safe_audit_log("feedback.record", resource=req.message_id, detail=f"rating={req.rating}")
            return FeedbackResponse(**feedback)
        except Exception as e:
            logger.error("Failed to record feedback (message=%s): %s", req.message_id, e)
            raise

    async def get_feedback_stats(self) -> dict:
        """Retrieve aggregate feedback statistics across all conversations."""
        try:
            ctrl = get_feedback_controller()
            stats = ctrl.get_stats()
            return FeedbackStats(**stats)
        except Exception as e:
            classify_and_raise(e, source="feedback.get_stats")

    async def create_conversation(self, req: ConversationCreate) -> dict:
        """Create a new conversation to associate feedback with."""
        try:
            ctrl = get_feedback_controller()
            conv = ctrl.create_conversation(
                name=req.name,
                session_id=req.session_id,
            )
            safe_audit_log("feedback.conversation_create", resource=getattr(conv, "id", "unknown"), detail=f"name={req.name}")
            return conv
        except Exception as e:
            classify_and_raise(e, source="feedback.create_conversation")

    async def list_conversations(
        self,
        limit: int = Query(default=50, ge=1, le=1000, description="Maximum number of conversations to return"),
    ) -> dict:
        """List all conversations sorted by most recent first."""
        try:
            ctrl = get_feedback_controller()
            return ctrl.list_conversations(limit=limit)
        except Exception as e:
            classify_and_raise(e, source="feedback.list_conversations")

    async def get_conversation(self, conv_id: str) -> dict:
        """Retrieve a single conversation by its unique ID."""
        try:
            ctrl = get_feedback_controller()
            conv = ctrl.get_conversation(conv_id)
            if not conv:
                raise_error("Conversation not found", "E_NOT_FOUND", status_code=404)
            return conv
        except Exception as e:
            classify_and_raise(e, source="feedback.get_conversation")

    async def update_conversation(self, conv_id: str, req: ConversationUpdate) -> dict:
        """Update a conversation's metadata (name, session_id, etc.)."""
        try:
            ctrl = get_feedback_controller()
            conv = ctrl.update_conversation(conv_id, req.model_dump(exclude_unset=True))
            if not conv:
                raise_error("Conversation not found", "E_NOT_FOUND", status_code=404)
            safe_audit_log("feedback.conversation_update", resource=conv_id)
            return conv
        except Exception as e:
            classify_and_raise(e, source="feedback.update_conversation")

    async def delete_conversation(self, conv_id: str) -> dict:
        """Delete a conversation and its associated feedback records."""
        try:
            ctrl = get_feedback_controller()
            ctrl.delete_conversation(conv_id)
            safe_audit_log("feedback.conversation_delete", resource=conv_id)
            return success_response(data={"status": "deleted", "id": conv_id})
        except Exception as e:
            classify_and_raise(e, source="feedback.delete_conversation")

    async def get_feedback(self, message_id: str) -> dict:
        """Get feedback for a message."""
        try:
            ctrl = get_feedback_controller()
            feedback = ctrl.get_feedback(message_id)
            if not feedback:
                raise_error("Feedback not found", "E_NOT_FOUND", status_code=404)
            return feedback
        except Exception as e:
            classify_and_raise(e, source="feedback.get_feedback")


router = FeedbackRouter().router
