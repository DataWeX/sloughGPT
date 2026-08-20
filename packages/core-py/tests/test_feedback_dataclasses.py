"""Tests for domains.feedback — MessageData, MessageFeedback, ResponseLog, MetaWeights, HealthSnapshot."""

from domains.feedback.message_feedback import MessageData, MessageFeedback
from domains.feedback.response_tracker import ResponseLog
from domains.feedback.meta_weights import MetaWeights
from domains.feedback.model_health import HealthSnapshot
from domains.feedback.database import Message, Feedback, SimilarPattern


class TestMessageData:
    def test_fields(self):
        md = MessageData(role="user", content="hello")
        assert md.role == "user"
        assert md.content == "hello"


class TestMessageFeedback:
    def test_record_feedback(self):
        mf = MessageFeedback()
        mf.record_feedback("msg1", "thumbs_up")
        fb = mf.get_feedback("msg1")
        assert fb is not None
        assert fb["rating"] == "thumbs_up"

    def test_get_feedback_missing(self):
        mf = MessageFeedback()
        assert mf.get_feedback("nonexistent") is None

    def test_store_and_get_session_context(self):
        mf = MessageFeedback()
        msgs = [MessageData(role="user", content="hi")]
        mf.store_session_context("s1", msgs)
        ctx = mf.get_session_context("s1")
        assert ctx is not None
        assert len(ctx) == 1

    def test_clear_session_context(self):
        mf = MessageFeedback()
        mf.store_session_context("s1", [MessageData(role="user", content="x")])
        mf.clear_session_context("s1")
        assert mf.get_session_context("s1") is None

    def test_record_regeneration(self):
        mf = MessageFeedback()
        regen = mf.record_regeneration("orig_msg", "new_msg")
        assert regen["original_message_id"] == "orig_msg"
        assert regen["new_message_id"] == "new_msg"

    def test_get_stats(self):
        mf = MessageFeedback()
        stats = mf.get_stats()
        assert "total_feedback" in stats
        assert stats["total_feedback"] == 0

    def test_list_conversations(self):
        mf = MessageFeedback()
        result = mf.list_conversations()
        assert isinstance(result, list)


class TestResponseLog:
    def test_fields(self):
        rl = ResponseLog(
            timestamp="2024-01-01", user_message="hi", assistant_response="hello",
            model="gpt2", temperature=0.7, max_tokens=100, session_id="s1",
            user_id="u1", tokens_generated=5, duration_ms=100.0,
        )
        assert rl.user_message == "hi"
        assert rl.tokens_generated == 5


class TestMetaWeights:
    def test_defaults(self):
        mw = MetaWeights()
        assert mw.temperature == 0.7
        assert mw.repetition_penalty == 1.15
        assert mw.top_p == 0.85
        assert mw.top_k == 40


class TestHealthSnapshot:
    def test_fields(self):
        hs = HealthSnapshot(
            timestamp=1.0, perplexity=5.0, loss=1.5, num_sentences=10,
        )
        assert hs.perplexity == 5.0
        assert hs.loss == 1.5
        assert hs.num_sentences == 10


class TestDatabaseDataclasses:
    def test_message(self):
        m = Message(
            id="m1", conversation_id="c1", role="user", content="hello",
        )
        assert m.id == "m1"

    def test_feedback(self):
        f = Feedback(
            id="f1", message_id="m1", rating="thumbs_up",
        )
        assert f.rating == "thumbs_up"

    def test_similar_pattern(self):
        sp = SimilarPattern(
            content="hello", rating="thumbs_up", similarity=0.9, pattern_type="exact",
        )
        assert sp.similarity == 0.9
