"""Tests for domains.feedback — MessageData, MessageFeedback, ResponseLog, MetaWeights, HealthSnapshot."""

import dataclasses
import threading
import numpy as np
import pytest

from domains.feedback.message_feedback import MessageData, MessageFeedback, get_message_feedback, _feedback_instance
from domains.feedback.response_tracker import ResponseLog
from domains.feedback.meta_weights import MetaWeights
from domains.feedback.model_health import HealthSnapshot
from domains.feedback.database import Message, Feedback, SimilarPattern


class TestMessageData:
    def test_fields(self):
        md = MessageData(role="user", content="hello")
        assert md.role == "user"
        assert md.content == "hello"

    def test_assistant_role(self):
        md = MessageData(role="assistant", content="hi there")
        assert md.role == "assistant"
        assert md.content == "hi there"

    def test_empty_content(self):
        md = MessageData(role="user", content="")
        assert md.content == ""

    def test_equality(self):
        md1 = MessageData(role="user", content="hello")
        md2 = MessageData(role="user", content="hello")
        assert md1 == md2

    def test_inequality(self):
        md1 = MessageData(role="user", content="hello")
        md2 = MessageData(role="user", content="world")
        assert md1 != md2

    def test_repr(self):
        md = MessageData(role="user", content="test")
        r = repr(md)
        assert "user" in r
        assert "test" in r

    def test_special_characters(self):
        md = MessageData(role="user", content="<script>alert('x')</script>")
        assert md.content == "<script>alert('x')</script>"

    def test_long_content(self):
        long = "a" * 10000
        md = MessageData(role="user", content=long)
        assert len(md.content) == 10000

    def test_default_fields(self):
        md = MessageData(role="user", content="hi")
        assert md.role == "user"
        assert md.content == "hi"

    def test_dataclass_fields(self):
        fields = {f.name for f in dataclasses.fields(MessageData)}
        assert fields == {"role", "content"}

    def test_not_hashable(self):
        md = MessageData(role="user", content="hello")
        with pytest.raises(TypeError):
            hash(md)

    def test_not_equal_to_non_message(self):
        md = MessageData(role="user", content="hello")
        assert md != "hello"
        assert md != 42

    def test_role_any_string(self):
        for role in ("system", "tool", "function", "custom"):
            md = MessageData(role=role, content="x")
            assert md.role == role

    def test_content_multiline(self):
        content = "line1\nline2\nline3"
        md = MessageData(role="user", content=content)
        assert md.content.count("\n") == 2

    def test_copy_semantics(self):
        md = MessageData(role="user", content="hello")
        import dataclasses
        md2 = dataclasses.replace(md, content="world")
        assert md.content == "hello"
        assert md2.content == "world"


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

    def test_record_feedback_with_session_id(self):
        mf = MessageFeedback()
        fb = mf.record_feedback("msg1", "thumbs_up", session_id="s1")
        assert fb["session_id"] == "s1"

    def test_record_feedback_with_context(self):
        mf = MessageFeedback()
        fb = mf.record_feedback("msg1", "thumbs_up", context="some context")
        assert fb["context"] == "some context"

    def test_record_feedback_context_truncated(self):
        mf = MessageFeedback()
        long_ctx = "x" * 2000
        fb = mf.record_feedback("msg1", "thumbs_up", context=long_ctx)
        assert len(fb["context"]) == 1000

    def test_record_feedback_has_timestamp(self):
        mf = MessageFeedback()
        fb = mf.record_feedback("msg1", "thumbs_up")
        assert "timestamp" in fb
        assert "T" in fb["timestamp"]

    def test_record_feedback_replaces_existing(self):
        mf = MessageFeedback()
        mf.record_feedback("msg1", "thumbs_down")
        mf.record_feedback("msg1", "thumbs_up")
        fb = mf.get_feedback("msg1")
        assert fb["rating"] == "thumbs_up"

    def test_record_regeneration_with_session(self):
        mf = MessageFeedback()
        regen = mf.record_regeneration("orig", "new", session_id="s1")
        assert regen["session_id"] == "s1"
        assert "timestamp" in regen

    def test_get_stats_with_data(self):
        mf = MessageFeedback()
        mf.record_feedback("m1", "thumbs_up")
        mf.record_feedback("m2", "thumbs_down")
        mf.record_feedback("m3", "thumbs_up")
        stats = mf.get_stats()
        assert stats["total_feedback"] == 3
        assert stats["thumbs_up"] == 2
        assert stats["thumbs_down"] == 1

    def test_get_stats_regenerations(self):
        mf = MessageFeedback()
        mf.record_regeneration("o1", "n1")
        mf.record_regeneration("o2", "n2")
        stats = mf.get_stats()
        assert stats["total_regenerations"] == 2

    def test_get_stats_active_sessions(self):
        mf = MessageFeedback()
        mf.store_session_context("s1", [MessageData(role="u", content="a")])
        mf.store_session_context("s2", [MessageData(role="u", content="b")])
        stats = mf.get_stats()
        assert stats["active_sessions"] == 2

    def test_list_conversations_with_data(self):
        mf = MessageFeedback()
        mf.store_session_context("s1", [MessageData(role="u", content="a")] * 3)
        result = mf.list_conversations()
        assert len(result) == 1
        assert result[0]["session_id"] == "s1"
        assert result[0]["message_count"] == 3

    def test_clear_nonexistent_session(self):
        mf = MessageFeedback()
        mf.clear_session_context("nonexistent")
        assert mf.get_session_context("nonexistent") is None

    def test_store_session_context_overwrites(self):
        mf = MessageFeedback()
        mf.store_session_context("s1", [MessageData(role="u", content="old")])
        mf.store_session_context("s1", [MessageData(role="u", content="new")])
        ctx = mf.get_session_context("s1")
        assert ctx[0].content == "new"

    def test_feedback_no_context_key(self):
        mf = MessageFeedback()
        fb = mf.record_feedback("msg1", "thumbs_up")
        assert "context" not in fb

    def test_context_boundary_exactly_1000(self):
        mf = MessageFeedback()
        ctx = "x" * 1000
        fb = mf.record_feedback("msg1", "thumbs_up", context=ctx)
        assert len(fb["context"]) == 1000

    def test_context_boundary_1001(self):
        mf = MessageFeedback()
        ctx = "x" * 1001
        fb = mf.record_feedback("msg1", "thumbs_up", context=ctx)
        assert len(fb["context"]) == 1000

    def test_empty_context_not_stored(self):
        mf = MessageFeedback()
        fb = mf.record_feedback("msg1", "thumbs_up", context="")
        assert "context" not in fb

    def test_none_session_id(self):
        mf = MessageFeedback()
        fb = mf.record_feedback("msg1", "thumbs_up")
        assert fb["session_id"] is None

    def test_many_feedback_messages(self):
        mf = MessageFeedback()
        for i in range(100):
            mf.record_feedback(f"msg{i}", "thumbs_up" if i % 2 == 0 else "thumbs_down")
        stats = mf.get_stats()
        assert stats["total_feedback"] == 100
        assert stats["thumbs_up"] == 50
        assert stats["thumbs_down"] == 50

    def test_many_sessions(self):
        mf = MessageFeedback()
        for i in range(50):
            mf.store_session_context(f"s{i}", [MessageData(role="u", content=f"msg{i}")])
        stats = mf.get_stats()
        assert stats["active_sessions"] == 50

    def test_clear_one_of_many_sessions(self):
        mf = MessageFeedback()
        mf.store_session_context("s1", [MessageData(role="u", content="a")])
        mf.store_session_context("s2", [MessageData(role="u", content="b")])
        mf.clear_session_context("s1")
        assert mf.get_session_context("s1") is None
        assert mf.get_session_context("s2") is not None

    def test_regeneration_overwrites_same_original(self):
        mf = MessageFeedback()
        mf.record_regeneration("orig", "new1")
        mf.record_regeneration("orig", "new2")
        stats = mf.get_stats()
        assert stats["total_regenerations"] == 1

    def test_feedback_message_content_param_accepted(self):
        mf = MessageFeedback()
        fb = mf.record_feedback("msg1", "thumbs_up", message_content="great response")
        assert fb["message_id"] == "msg1"
        assert fb["rating"] == "thumbs_up"

    def test_feedback_with_none_context(self):
        mf = MessageFeedback()
        fb = mf.record_feedback("msg1", "thumbs_up", context=None)
        assert "context" not in fb

    def test_thread_safety_concurrent_record(self):
        mf = MessageFeedback()
        errors = []

        def record(i):
            try:
                mf.record_feedback(f"msg{i}", "thumbs_up")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(mf._feedback) == 20


class TestMessageFeedbackSingleton:
    def test_get_message_feedback_singleton(self):
        import domains.feedback.message_feedback as mod
        old = mod._feedback_instance
        try:
            mod._feedback_instance = None
            a = get_message_feedback()
            b = get_message_feedback()
            assert a is b
            assert isinstance(a, MessageFeedback)
        finally:
            mod._feedback_instance = old

    def test_singleton_not_none_after_first_call(self):
        import domains.feedback.message_feedback as mod
        old = mod._feedback_instance
        try:
            mod._feedback_instance = None
            result = get_message_feedback()
            assert result is not None
        finally:
            mod._feedback_instance = old

    def test_singleton_returns_same_type(self):
        import domains.feedback.message_feedback as mod
        old = mod._feedback_instance
        try:
            mod._feedback_instance = None
            result = get_message_feedback()
            assert type(result) is MessageFeedback
        finally:
            mod._feedback_instance = old

    def test_singleton_preserves_state_across_calls(self):
        import domains.feedback.message_feedback as mod
        old = mod._feedback_instance
        try:
            mod._feedback_instance = None
            a = get_message_feedback()
            a.record_feedback("msg1", "thumbs_up")
            b = get_message_feedback()
            fb = b.get_feedback("msg1")
            assert fb is not None
            assert fb["rating"] == "thumbs_up"
        finally:
            mod._feedback_instance = old

    def test_singleton_resets_cleanly(self):
        import domains.feedback.message_feedback as mod
        old = mod._feedback_instance
        try:
            mod._feedback_instance = None
            a = get_message_feedback()
            a.record_feedback("msg1", "thumbs_up")
            mod._feedback_instance = None
            b = get_message_feedback()
            assert b.get_feedback("msg1") is None
        finally:
            mod._feedback_instance = old


class TestResponseLog:
    def test_fields(self):
        rl = ResponseLog(
            timestamp="2024-01-01", user_message="hi", assistant_response="hello",
            model="gpt2", temperature=0.7, max_tokens=100, session_id="s1",
            user_id="u1", tokens_generated=5, duration_ms=100.0,
        )
        assert rl.user_message == "hi"
        assert rl.tokens_generated == 5

    def test_default_optional_fields(self):
        rl = ResponseLog(
            timestamp="t", user_message="u", assistant_response="a",
            model="m", temperature=0.5, max_tokens=64, session_id="s",
            user_id="u", tokens_generated=0, duration_ms=0.0,
        )
        assert rl.has_images is False
        assert rl.context_tokens == 0
        assert rl.eval_scores is None

    def test_with_images(self):
        rl = ResponseLog(
            timestamp="t", user_message="u", assistant_response="a",
            model="m", temperature=0.5, max_tokens=64, session_id="s",
            user_id="u", tokens_generated=10, duration_ms=50.0, has_images=True,
        )
        assert rl.has_images is True

    def test_with_eval_scores(self):
        scores = {"fluency": 0.9, "relevance": 0.8}
        rl = ResponseLog(
            timestamp="t", user_message="u", assistant_response="a",
            model="m", temperature=0.5, max_tokens=64, session_id="s",
            user_id="u", tokens_generated=10, duration_ms=50.0,
            eval_scores=scores,
        )
        assert rl.eval_scores == scores

    def test_all_fields_assignable(self):
        rl = ResponseLog(
            timestamp="ts", user_message="um", assistant_response="ar",
            model="md", temperature=1.0, max_tokens=512, session_id="sid",
            user_id="uid", tokens_generated=100, duration_ms=999.9,
            has_images=True, context_tokens=50,
        )
        assert rl.context_tokens == 50
        assert rl.duration_ms == 999.9

    def test_repr(self):
        rl = ResponseLog(
            timestamp="t", user_message="u", assistant_response="a",
            model="m", temperature=0.5, max_tokens=64, session_id="s",
            user_id="u", tokens_generated=10, duration_ms=50.0,
        )
        r = repr(rl)
        assert "ResponseLog" in r

    def test_equality(self):
        kw = dict(timestamp="t", user_message="u", assistant_response="a",
                  model="m", temperature=0.5, max_tokens=64, session_id="s",
                  user_id="u", tokens_generated=10, duration_ms=50.0)
        rl1 = ResponseLog(**kw)
        rl2 = ResponseLog(**kw)
        assert rl1 == rl2

    def test_inequality_different_model(self):
        rl1 = ResponseLog(timestamp="t", user_message="u", assistant_response="a",
                          model="gpt2", temperature=0.5, max_tokens=64, session_id="s",
                          user_id="u", tokens_generated=10, duration_ms=50.0)
        rl2 = ResponseLog(timestamp="t", user_message="u", assistant_response="a",
                          model="gpt4", temperature=0.5, max_tokens=64, session_id="s",
                          user_id="u", tokens_generated=10, duration_ms=50.0)
        assert rl1 != rl2

    def test_inequality_different_tokens(self):
        rl1 = ResponseLog(timestamp="t", user_message="u", assistant_response="a",
                          model="m", temperature=0.5, max_tokens=64, session_id="s",
                          user_id="u", tokens_generated=10, duration_ms=50.0)
        rl2 = ResponseLog(timestamp="t", user_message="u", assistant_response="a",
                          model="m", temperature=0.5, max_tokens=64, session_id="s",
                          user_id="u", tokens_generated=20, duration_ms=50.0)
        assert rl1 != rl2

    def test_eval_scores_empty_dict(self):
        rl = ResponseLog(timestamp="t", user_message="u", assistant_response="a",
                         model="m", temperature=0.5, max_tokens=64, session_id="s",
                         user_id="u", tokens_generated=0, duration_ms=0.0,
                         eval_scores={})
        assert rl.eval_scores == {}

    def test_eval_scores_multiple_keys(self):
        scores = {"fluency": 0.9, "relevance": 0.8, "coherence": 0.7, "safety": 0.95}
        rl = ResponseLog(timestamp="t", user_message="u", assistant_response="a",
                         model="m", temperature=0.5, max_tokens=64, session_id="s",
                         user_id="u", tokens_generated=0, duration_ms=0.0,
                         eval_scores=scores)
        assert len(rl.eval_scores) == 4

    def test_dataclass_fields(self):
        fields = {f.name for f in dataclasses.fields(ResponseLog)}
        expected = {"timestamp", "user_message", "assistant_response", "model",
                    "temperature", "max_tokens", "session_id", "user_id",
                    "tokens_generated", "duration_ms", "has_images",
                    "context_tokens", "eval_scores"}
        assert fields == expected

    def test_tokens_generated_zero(self):
        rl = ResponseLog(timestamp="t", user_message="u", assistant_response="a",
                         model="m", temperature=0.5, max_tokens=64, session_id="s",
                         user_id="u", tokens_generated=0, duration_ms=0.0)
        assert rl.tokens_generated == 0

    def test_duration_ms_negative(self):
        rl = ResponseLog(timestamp="t", user_message="u", assistant_response="a",
                         model="m", temperature=0.5, max_tokens=64, session_id="s",
                         user_id="u", tokens_generated=0, duration_ms=-1.0)
        assert rl.duration_ms == -1.0

    def test_max_tokens_large(self):
        rl = ResponseLog(timestamp="t", user_message="u", assistant_response="a",
                         model="m", temperature=0.5, max_tokens=100000, session_id="s",
                         user_id="u", tokens_generated=0, duration_ms=0.0)
        assert rl.max_tokens == 100000

    def test_temperature_boundary_zero(self):
        rl = ResponseLog(timestamp="t", user_message="u", assistant_response="a",
                         model="m", temperature=0.0, max_tokens=64, session_id="s",
                         user_id="u", tokens_generated=0, duration_ms=0.0)
        assert rl.temperature == 0.0

    def test_temperature_boundary_two(self):
        rl = ResponseLog(timestamp="t", user_message="u", assistant_response="a",
                         model="m", temperature=2.0, max_tokens=64, session_id="s",
                         user_id="u", tokens_generated=0, duration_ms=0.0)
        assert rl.temperature == 2.0


class TestMetaWeights:
    def test_defaults(self):
        mw = MetaWeights()
        assert mw.temperature == 0.7
        assert mw.repetition_penalty == 1.15
        assert mw.top_p == 0.85
        assert mw.top_k == 40

    def test_custom_values(self):
        mw = MetaWeights(temperature=1.0, repetition_penalty=1.0, top_p=0.9, top_k=50)
        assert mw.temperature == 1.0
        assert mw.repetition_penalty == 1.0
        assert mw.top_p == 0.9
        assert mw.top_k == 50

    def test_optional_defaults(self):
        mw = MetaWeights()
        assert mw.length_penalty == 1.0
        assert mw.style_bias == 0.0
        assert mw.confidence_boost == 0.0

    def test_style_bias_range(self):
        mw = MetaWeights(style_bias=-0.5)
        assert mw.style_bias == -0.5
        mw2 = MetaWeights(style_bias=1.0)
        assert mw2.style_bias == 1.0

    def test_repr(self):
        mw = MetaWeights()
        r = repr(mw)
        assert "MetaWeights" in r
        assert "0.7" in r

    def test_dataclass_fields(self):
        fields = {f.name for f in dataclasses.fields(MetaWeights)}
        assert "temperature" in fields
        assert "repetition_penalty" in fields
        assert "top_p" in fields
        assert "top_k" in fields
        assert "length_penalty" in fields
        assert "style_bias" in fields
        assert "confidence_boost" in fields

    def test_equality(self):
        mw1 = MetaWeights()
        mw2 = MetaWeights()
        assert mw1 == mw2

    def test_inequality(self):
        mw1 = MetaWeights(temperature=0.7)
        mw2 = MetaWeights(temperature=0.9)
        assert mw1 != mw2

    def test_all_float_fields(self):
        for f in dataclasses.fields(MetaWeights):
            if f.name == "top_k":
                assert f.type is int
            else:
                assert f.type is float

    def test_copy_via_replace(self):
        mw1 = MetaWeights()
        mw2 = dataclasses.replace(mw1, temperature=1.2)
        assert mw1.temperature == 0.7
        assert mw2.temperature == 1.2

    def test_style_bias_full_range(self):
        for val in (-1.0, -0.5, 0.0, 0.5, 1.0):
            mw = MetaWeights(style_bias=val)
            assert mw.style_bias == val

    def test_confidence_boost_positive(self):
        mw = MetaWeights(confidence_boost=0.5)
        assert mw.confidence_boost == 0.5

    def test_confidence_boost_negative(self):
        mw = MetaWeights(confidence_boost=-0.3)
        assert mw.confidence_boost == -0.3

    def test_top_k_zero(self):
        mw = MetaWeights(top_k=0)
        assert mw.top_k == 0

    def test_top_k_large(self):
        mw = MetaWeights(top_k=1000)
        assert mw.top_k == 1000

    def test_length_penalty_below_one(self):
        mw = MetaWeights(length_penalty=0.5)
        assert mw.length_penalty == 0.5

    def test_length_penalty_above_one(self):
        mw = MetaWeights(length_penalty=2.0)
        assert mw.length_penalty == 2.0

    def test_temperature_zero(self):
        mw = MetaWeights(temperature=0.0)
        assert mw.temperature == 0.0

    def test_temperature_high(self):
        mw = MetaWeights(temperature=3.0)
        assert mw.temperature == 3.0

    def test_top_p_boundary(self):
        mw = MetaWeights(top_p=0.0)
        assert mw.top_p == 0.0
        mw2 = MetaWeights(top_p=1.0)
        assert mw2.top_p == 1.0


class TestHealthSnapshot:
    def test_fields(self):
        hs = HealthSnapshot(
            timestamp=1.0, perplexity=5.0, loss=1.5, num_sentences=10,
        )
        assert hs.perplexity == 5.0
        assert hs.loss == 1.5
        assert hs.num_sentences == 10

    def test_timestamp(self):
        hs = HealthSnapshot(timestamp=100.0, perplexity=3.0, loss=0.5, num_sentences=5)
        assert hs.timestamp == 100.0

    def test_zero_values(self):
        hs = HealthSnapshot(timestamp=0.0, perplexity=0.0, loss=0.0, num_sentences=0)
        assert hs.perplexity == 0.0
        assert hs.num_sentences == 0

    def test_high_values(self):
        hs = HealthSnapshot(timestamp=99999.0, perplexity=1000.0, loss=10.0, num_sentences=10000)
        assert hs.perplexity == 1000.0

    def test_repr(self):
        hs = HealthSnapshot(timestamp=1.0, perplexity=5.0, loss=1.5, num_sentences=10)
        r = repr(hs)
        assert "HealthSnapshot" in r

    def test_dataclass_fields(self):
        fields = {f.name for f in dataclasses.fields(HealthSnapshot)}
        assert fields == {"timestamp", "perplexity", "loss", "num_sentences"}

    def test_equality(self):
        kw = dict(timestamp=1.0, perplexity=5.0, loss=1.5, num_sentences=10)
        hs1 = HealthSnapshot(**kw)
        hs2 = HealthSnapshot(**kw)
        assert hs1 == hs2

    def test_inequality(self):
        hs1 = HealthSnapshot(timestamp=1.0, perplexity=5.0, loss=1.5, num_sentences=10)
        hs2 = HealthSnapshot(timestamp=2.0, perplexity=5.0, loss=1.5, num_sentences=10)
        assert hs1 != hs2

    def test_float_precision(self):
        hs = HealthSnapshot(timestamp=1.123456789, perplexity=2.987654321, loss=0.111111, num_sentences=1)
        assert abs(hs.perplexity - 2.987654321) < 1e-6

    def test_copy_via_replace(self):
        hs1 = HealthSnapshot(timestamp=1.0, perplexity=5.0, loss=1.5, num_sentences=10)
        hs2 = dataclasses.replace(hs1, perplexity=10.0)
        assert hs1.perplexity == 5.0
        assert hs2.perplexity == 10.0

    def test_inequality_different_loss(self):
        hs1 = HealthSnapshot(timestamp=1.0, perplexity=5.0, loss=1.5, num_sentences=10)
        hs2 = HealthSnapshot(timestamp=1.0, perplexity=5.0, loss=2.0, num_sentences=10)
        assert hs1 != hs2

    def test_inequality_different_num_sentences(self):
        hs1 = HealthSnapshot(timestamp=1.0, perplexity=5.0, loss=1.5, num_sentences=10)
        hs2 = HealthSnapshot(timestamp=1.0, perplexity=5.0, loss=1.5, num_sentences=20)
        assert hs1 != hs2

    def test_negative_timestamp(self):
        hs = HealthSnapshot(timestamp=-1.0, perplexity=5.0, loss=1.5, num_sentences=10)
        assert hs.timestamp == -1.0

    def test_negative_perplexity(self):
        hs = HealthSnapshot(timestamp=1.0, perplexity=-1.0, loss=1.5, num_sentences=10)
        assert hs.perplexity == -1.0

    def test_negative_loss(self):
        hs = HealthSnapshot(timestamp=1.0, perplexity=5.0, loss=-0.5, num_sentences=10)
        assert hs.loss == -0.5

    def test_large_num_sentences(self):
        hs = HealthSnapshot(timestamp=1.0, perplexity=5.0, loss=1.5, num_sentences=1_000_000)
        assert hs.num_sentences == 1_000_000

    def test_all_fields_same(self):
        hs1 = HealthSnapshot(timestamp=0.0, perplexity=0.0, loss=0.0, num_sentences=0)
        hs2 = HealthSnapshot(timestamp=0.0, perplexity=0.0, loss=0.0, num_sentences=0)
        assert hs1 == hs2

    def test_dataclass_field_types(self):
        fields = {f.name: f.type for f in dataclasses.fields(HealthSnapshot)}
        assert fields["timestamp"] is float
        assert fields["perplexity"] is float
        assert fields["loss"] is float
        assert fields["num_sentences"] is int


class TestDatabaseDataclasses:
    def test_message(self):
        m = Message(id="m1", conversation_id="c1", role="user", content="hello")
        assert m.id == "m1"

    def test_feedback(self):
        f = Feedback(id="f1", message_id="m1", rating="thumbs_up")
        assert f.rating == "thumbs_up"

    def test_similar_pattern(self):
        sp = SimilarPattern(content="hello", rating="thumbs_up", similarity=0.9, pattern_type="exact")
        assert sp.similarity == 0.9

    def test_message_optional_embedding(self):
        m = Message(id="m1", conversation_id="c1", role="user", content="hi")
        assert m.embedding is None
        assert m.created_at is None

    def test_message_with_embedding(self):
        emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        m = Message(id="m1", conversation_id="c1", role="user", content="hi", embedding=emb)
        assert m.embedding is not None

    def test_feedback_optional_fields(self):
        f = Feedback(id="f1", message_id="m1", rating="thumbs_up")
        assert f.quality_score is None
        assert f.created_at is None

    def test_feedback_with_quality_score(self):
        f = Feedback(id="f1", message_id="m1", rating="thumbs_up", quality_score=0.95)
        assert f.quality_score == 0.95

    def test_similar_pattern_all_types(self):
        for pt in ("exact", "semantic", "keyword_match", "message"):
            sp = SimilarPattern(content="c", rating="thumbs_up", similarity=0.5, pattern_type=pt)
            assert sp.pattern_type == pt

    def test_message_repr(self):
        m = Message(id="m1", conversation_id="c1", role="user", content="hi")
        r = repr(m)
        assert "m1" in r

    def test_feedback_repr(self):
        f = Feedback(id="f1", message_id="m1", rating="thumbs_up")
        r = repr(f)
        assert "thumbs_up" in r

    def test_similar_pattern_repr(self):
        sp = SimilarPattern(content="hello", rating="thumbs_up", similarity=0.9, pattern_type="exact")
        r = repr(sp)
        assert "0.9" in r

    def test_message_dataclass_fields(self):
        fields = {f.name for f in dataclasses.fields(Message)}
        assert fields == {"id", "conversation_id", "role", "content", "embedding", "created_at"}

    def test_feedback_dataclass_fields(self):
        fields = {f.name for f in dataclasses.fields(Feedback)}
        assert fields == {"id", "message_id", "rating", "quality_score", "created_at"}

    def test_similar_pattern_dataclass_fields(self):
        fields = {f.name for f in dataclasses.fields(SimilarPattern)}
        assert fields == {"content", "rating", "similarity", "pattern_type"}

    def test_message_equality(self):
        m1 = Message(id="m1", conversation_id="c1", role="user", content="hello")
        m2 = Message(id="m1", conversation_id="c1", role="user", content="hello")
        assert m1 == m2

    def test_message_inequality(self):
        m1 = Message(id="m1", conversation_id="c1", role="user", content="hello")
        m2 = Message(id="m2", conversation_id="c1", role="user", content="hello")
        assert m1 != m2

    def test_feedback_equality(self):
        f1 = Feedback(id="f1", message_id="m1", rating="thumbs_up")
        f2 = Feedback(id="f1", message_id="m1", rating="thumbs_up")
        assert f1 == f2

    def test_feedback_inequality(self):
        f1 = Feedback(id="f1", message_id="m1", rating="thumbs_up")
        f2 = Feedback(id="f1", message_id="m1", rating="thumbs_down")
        assert f1 != f2

    def test_similar_pattern_equality(self):
        sp1 = SimilarPattern(content="a", rating="thumbs_up", similarity=0.9, pattern_type="exact")
        sp2 = SimilarPattern(content="a", rating="thumbs_up", similarity=0.9, pattern_type="exact")
        assert sp1 == sp2

    def test_message_with_large_embedding(self):
        emb = np.random.randn(384).astype(np.float32)
        m = Message(id="m1", conversation_id="c1", role="user", content="hi", embedding=emb)
        assert m.embedding.shape == (384,)

    def test_feedback_quality_score_zero(self):
        f = Feedback(id="f1", message_id="m1", rating="thumbs_up", quality_score=0.0)
        assert f.quality_score == 0.0

    def test_feedback_quality_score_one(self):
        f = Feedback(id="f1", message_id="m1", rating="thumbs_up", quality_score=1.0)
        assert f.quality_score == 1.0

    def test_similar_pattern_zero_similarity(self):
        sp = SimilarPattern(content="a", rating="thumbs_up", similarity=0.0, pattern_type="exact")
        assert sp.similarity == 0.0

    def test_similar_pattern_one_similarity(self):
        sp = SimilarPattern(content="a", rating="thumbs_up", similarity=1.0, pattern_type="exact")
        assert sp.similarity == 1.0

    def test_message_long_content(self):
        content = "x" * 10000
        m = Message(id="m1", conversation_id="c1", role="user", content=content)
        assert len(m.content) == 10000

    def test_message_assistant_role(self):
        m = Message(id="m1", conversation_id="c1", role="assistant", content="hi")
        assert m.role == "assistant"

    def test_message_system_role(self):
        m = Message(id="m1", conversation_id="c1", role="system", content="you are helpful")
        assert m.role == "system"

    def test_feedback_rating_variations(self):
        for rating in ("thumbs_up", "thumbs_down", "neutral", "star_5"):
            f = Feedback(id="f1", message_id="m1", rating=rating)
            assert f.rating == rating

    def test_message_with_created_at(self):
        m = Message(id="m1", conversation_id="c1", role="user", content="hi", created_at="2024-01-01T00:00:00Z")
        assert m.created_at == "2024-01-01T00:00:00Z"

    def test_feedback_with_created_at(self):
        f = Feedback(id="f1", message_id="m1", rating="thumbs_up", created_at="2024-01-01T00:00:00Z")
        assert f.created_at == "2024-01-01T00:00:00Z"

    def test_similar_pattern_negative_similarity(self):
        sp = SimilarPattern(content="a", rating="thumbs_up", similarity=-0.5, pattern_type="exact")
        assert sp.similarity == -0.5
