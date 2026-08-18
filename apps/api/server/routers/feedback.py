"""
Feedback Router - MVC View layer
"""
from fastapi import APIRouter, HTTPException, Query, Request

from pydantic import BaseModel, Field
from schemas.feedback import FeedbackRequest, FeedbackResponse, FeedbackStats, ConversationCreate, ConversationUpdate, ConversationResponse
from schemas.common import success_response
from controllers.feedback import get_feedback_controller


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
        return success_response(data={
            "feedback_id": feedback.get("feedback_id", ""),
            "workflow_active": True,
        }, message="recorded")

    async def record_feedback(self, req: FeedbackRequest) -> dict:
        """Record user feedback and pipe into learning systems."""
        ctrl = get_feedback_controller()
        feedback = ctrl.record_feedback(
            message_id=req.message_id,
            rating=req.rating,
            session_id=req.session_id,
            message_content=req.message_content,
            user_message=getattr(req, 'user_message', None),
            assistant_response=getattr(req, 'assistant_response', None),
        )
        return FeedbackResponse(**feedback)

    async def get_feedback_stats(self) -> dict:
        """Retrieve aggregate feedback statistics across all conversations.

        Returns thumbs_up/thumbs_down counts, average rating, and other
        summary metrics computed by the FeedbackController.

        Returns:
            FeedbackStats with aggregate feedback metrics.
        """
        ctrl = get_feedback_controller()
        stats = ctrl.get_stats()
        return FeedbackStats(**stats)

    async def create_conversation(self, req: ConversationCreate) -> dict:
        """Create a new conversation to associate feedback with.

        Registers a conversation record that groups related feedback
        entries together for analysis.

        Args:
            req: ConversationCreate with name (required) and session_id (optional).

        Returns:
            ConversationResponse with the new conversation's id and metadata.

        Side effects:
            - Persists the conversation record to the feedback store.
        """
        ctrl = get_feedback_controller()
        conv = ctrl.create_conversation(
            name=req.name,
            session_id=req.session_id,
        )
        return conv

    async def list_conversations(
        self,
        limit: int = Query(default=50, ge=1, le=1000, description="Maximum number of conversations to return"),
    ) -> dict:
        """List all conversations sorted by most recent first.

        Args:
            limit: Maximum number of conversations to return (1-1000, default 50).

        Returns:
            List of ConversationResponse objects with id, name, and metadata.
        """
        ctrl = get_feedback_controller()
        return ctrl.list_conversations(limit=limit)

    async def get_conversation(self, conv_id: str) -> dict:
        """Retrieve a single conversation by its unique ID.

        Args:
            conv_id: The conversation identifier.

        Returns:
            ConversationResponse with conversation details.

        Raises:
            404 if the conversation is not found.
        """
        ctrl = get_feedback_controller()
        conv = ctrl.get_conversation(conv_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conv

    async def update_conversation(self, conv_id: str, req: ConversationUpdate) -> dict:
        """Update a conversation's metadata (name, session_id, etc.).

        Args:
            conv_id: The conversation identifier.
            req: ConversationUpdate with optional fields to update.

        Returns:
            ConversationResponse with the updated conversation.

        Raises:
            404 if the conversation is not found.
        """
        ctrl = get_feedback_controller()
        conv = ctrl.update_conversation(conv_id, req.model_dump(exclude_unset=True))
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conv

    async def delete_conversation(self, conv_id: str) -> dict:
        """Delete a conversation and its associated feedback records.

        Args:
            conv_id: The conversation identifier.

        Returns:
            Dict with status "deleted" and the conversation id.

        Side effects:
            - Removes the conversation and all linked feedback from the store.
        """
        ctrl = get_feedback_controller()
        ctrl.delete_conversation(conv_id)
        return {"status": "deleted", "id": conv_id}

    async def get_feedback(self, message_id: str) -> dict:
        """Get feedback for a message"""
        ctrl = get_feedback_controller()
        feedback = ctrl.get_feedback(message_id)
        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback not found")
        return feedback


router = FeedbackRouter().router
