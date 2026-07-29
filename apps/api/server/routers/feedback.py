"""
Feedback Router - MVC View layer
"""
from fastapi import APIRouter, HTTPException, Request

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

    async def record_feedback_workflow(self, req: WorkflowFeedbackRequest):
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

    async def record_feedback(self, req: FeedbackRequest):
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

    async def get_feedback_stats(self):
        """Get feedback statistics"""
        ctrl = get_feedback_controller()
        stats = ctrl.get_stats()
        return FeedbackStats(**stats)

    async def create_conversation(self, req: ConversationCreate):
        """Create a new conversation"""
        ctrl = get_feedback_controller()
        conv = ctrl.create_conversation(
            name=req.name,
            session_id=req.session_id,
        )
        return conv

    async def list_conversations(self, limit: int = 50):
        """List all conversations"""
        ctrl = get_feedback_controller()
        return ctrl.list_conversations(limit=limit)

    async def get_conversation(self, conv_id: str):
        """Get a conversation by ID"""
        ctrl = get_feedback_controller()
        conv = ctrl.get_conversation(conv_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conv

    async def update_conversation(self, conv_id: str, req: ConversationUpdate):
        """Update a conversation"""
        ctrl = get_feedback_controller()
        conv = ctrl.update_conversation(conv_id, req.model_dump(exclude_unset=True))
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conv

    async def delete_conversation(self, conv_id: str):
        """Delete a conversation"""
        ctrl = get_feedback_controller()
        ctrl.delete_conversation(conv_id)
        return {"status": "deleted", "id": conv_id}

    async def get_feedback(self, message_id: str):
        """Get feedback for a message"""
        ctrl = get_feedback_controller()
        feedback = ctrl.get_feedback(message_id)
        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback not found")
        return feedback


router = FeedbackRouter().router
