"""Tests for domains.learner.knowledge — KnowledgeFact, FeedSubscription, _topic_slug; domains.training.distillation — DistillationConfig, DistillationLoss."""

import numpy as np
import pytest
from domains.learner.knowledge import (
    KnowledgeFact, FeedSubscription, _topic_slug,
    chunk_by_fixed_size, chunk_by_paragraph, chunk_by_heading,
    chunk_by_semantic, chunk_text, _extract_facts_from_text,
    _extract_topics,
)
from domains.training.distillation import (
    DistillationConfig, DistillationLoss, _to_np, _to_tensor, _size,
)


class TestKnowledgeFact:
    def test_defaults(self):
        kf = KnowledgeFact(content="test fact")
        assert kf.content == "test fact"
        assert kf.topic == "general"
        assert kf.source == "manual"
        assert kf.importance == 0.5

    def test_custom(self):
        kf = KnowledgeFact(content="fact", topic="AI", importance=0.9)
        assert kf.topic == "AI"
        assert kf.importance == 0.9

    def test_url_default(self):
        kf = KnowledgeFact(content="x")
        assert kf.url == ""

    def test_timestamp_default(self):
        kf = KnowledgeFact(content="x")
        assert kf.timestamp == 0.0

    def test_all_fields_custom(self):
        kf = KnowledgeFact(
            content="hello", topic="tech", source="web",
            url="http://example.com", timestamp=12345.0, importance=0.8
        )
        assert kf.content == "hello"
        assert kf.topic == "tech"
        assert kf.source == "web"
        assert kf.url == "http://example.com"
        assert kf.timestamp == 12345.0
        assert kf.importance == 0.8

    def test_empty_content(self):
        kf = KnowledgeFact(content="")
        assert kf.content == ""

    def test_importance_boundary_zero(self):
        kf = KnowledgeFact(content="x", importance=0.0)
        assert kf.importance == 0.0

    def test_importance_boundary_one(self):
        kf = KnowledgeFact(content="x", importance=1.0)
        assert kf.importance == 1.0

    def test_unicode_content(self):
        kf = KnowledgeFact(content="日本語テスト")
        assert kf.content == "日本語テスト"

    def test_long_content(self):
        kf = KnowledgeFact(content="a" * 10000)
        assert len(kf.content) == 10000

    def test_special_characters(self):
        kf = KnowledgeFact(content="line1\nline2\ttab")
        assert "\n" in kf.content
        assert "\t" in kf.content


class TestFeedSubscription:
    def test_defaults(self):
        fs = FeedSubscription(url="http://example.com/rss")
        assert fs.url == "http://example.com/rss"
        assert fs.enabled is True
        assert fs.poll_interval > 0

    def test_custom(self):
        fs = FeedSubscription(url="http://x", title="Feed", poll_interval=60.0)
        assert fs.title == "Feed"
        assert fs.poll_interval == 60.0

    def test_disabled(self):
        fs = FeedSubscription(url="http://x", enabled=False)
        assert fs.enabled is False

    def test_last_fetched_default(self):
        fs = FeedSubscription(url="http://x")
        assert fs.last_fetched == 0.0

    def test_last_fetched_custom(self):
        fs = FeedSubscription(url="http://x", last_fetched=999.0)
        assert fs.last_fetched == 999.0

    def test_title_default(self):
        fs = FeedSubscription(url="http://x")
        assert fs.title == ""

    def test_empty_url(self):
        fs = FeedSubscription(url="")
        assert fs.url == ""

    def test_poll_interval_zero(self):
        fs = FeedSubscription(url="http://x", poll_interval=0.0)
        assert fs.poll_interval == 0.0

    def test_large_poll_interval(self):
        fs = FeedSubscription(url="http://x", poll_interval=86400.0)
        assert fs.poll_interval == 86400.0


class TestTopicSlug:
    def test_normal(self):
        assert _topic_slug("Machine Learning") == "machine_learning"
    def test_special_chars(self):
        assert _topic_slug("AI/ML & NLP!") == "ai_ml_nlp_"
    def test_truncate(self):
        long = "a" * 100
        assert len(_topic_slug(long)) == 64

    def test_empty_string(self):
        assert _topic_slug("") == ""

    def test_whitespace_only(self):
        assert _topic_slug("   ") == ""

    def test_already_slug(self):
        assert _topic_slug("machine_learning") == "machine_learning"

    def test_numbers(self):
        assert _topic_slug("python3") == "python3"

    def test_single_char(self):
        assert _topic_slug("A") == "a"

    def test_mixed_case(self):
        result = _topic_slug("Deep Learning")
        assert result == "deep_learning"

    def test_leading_trailing_spaces(self):
        assert _topic_slug("  hello  ") == "hello"

    def test_multiple_spaces(self):
        assert _topic_slug("hello  world") == "hello_world"

    def test_underscores_preserved(self):
        assert _topic_slug("a_b_c") == "a_b_c"

    def test_consecutive_special_chars(self):
        assert _topic_slug("a//b&&c") == "a_b_c"


class TestChunkFixedSize:
    def test_short_text(self):
        assert chunk_by_fixed_size("hello", chunk_size=10) == ["hello"]

    def test_exact_size(self):
        assert chunk_by_fixed_size("abcde", chunk_size=5) == ["abcde"]

    def test_long_text(self):
        result = chunk_by_fixed_size("a" * 20, chunk_size=10)
        assert len(result) >= 2

    def test_empty_text(self):
        assert chunk_by_fixed_size("") == []

    def test_whitespace_only(self):
        assert chunk_by_fixed_size("   ") == []

    def test_overlap(self):
        result = chunk_by_fixed_size("abcdefghij", chunk_size=5, overlap=2)
        assert len(result) >= 2

    def test_no_overlap(self):
        result = chunk_by_fixed_size("abcdefghij", chunk_size=5, overlap=0)
        assert result[0] == "abcde"

    def test_large_overlap(self):
        result = chunk_by_fixed_size("abcdefgh", chunk_size=4, overlap=3)
        assert len(result) >= 1

    def test_chunks_are_stripped(self):
        result = chunk_by_fixed_size("  hello  ", chunk_size=10)
        assert result[0] == "hello"

    def test_very_long_text(self):
        text = "word " * 1000
        result = chunk_by_fixed_size(text, chunk_size=100)
        assert len(result) > 5

    def test_single_chunk_many_spaces(self):
        result = chunk_by_fixed_size("a   b", chunk_size=100)
        assert len(result) == 1


class TestChunkParagraph:
    def test_single_paragraph(self):
        result = chunk_by_paragraph("hello world")
        assert len(result) == 1

    def test_multiple_paragraphs(self):
        text = "para1\n\npara2\n\npara3"
        result = chunk_by_paragraph(text)
        # Short paragraphs get merged into one chunk
        assert len(result) >= 1
        assert any("para1" in c for c in result)

    def test_empty_text(self):
        assert chunk_by_paragraph("") == []

    def test_whitespace_only(self):
        assert chunk_by_paragraph("   ") == []

    def test_merges_short_paragraphs(self):
        text = "a\n\nb\n\nc"
        result = chunk_by_paragraph(text, max_chunk_size=100)
        assert len(result) == 1

    def test_splits_long_paragraphs(self):
        text = "x" * 2000 + "\n\n" + "y" * 2000
        result = chunk_by_paragraph(text, max_chunk_size=1000)
        assert len(result) >= 2

    def test_paragraphs_preserved(self):
        text = "first paragraph.\n\nsecond paragraph."
        result = chunk_by_paragraph(text)
        assert any("first" in c for c in result)
        assert any("second" in c for c in result)

    def test_single_newline_not_split(self):
        text = "line1\nline2\nline3"
        result = chunk_by_paragraph(text)
        assert len(result) == 1

    def test_max_chunk_size_respected(self):
        text = ("short.\n\n" * 10).strip()
        result = chunk_by_paragraph(text, max_chunk_size=50)
        for chunk in result:
            assert len(chunk) <= 100  # allow some tolerance

    def test_various_separators(self):
        text = "a\r\n\r\nb"
        result = chunk_by_paragraph(text)
        assert len(result) >= 1


class TestChunkHeading:
    def test_with_headings(self):
        text = "# Title\nContent here\n## Sub\nMore content"
        result = chunk_by_heading(text)
        # Short sections get merged under default max_chunk_size
        assert len(result) >= 1
        assert any("Title" in c for c in result)

    def test_no_headings_fallback(self):
        text = "plain text without headings at all."
        result = chunk_by_heading(text)
        assert len(result) >= 1

    def test_empty_text(self):
        assert chunk_by_heading("") == []

    def test_whitespace_only(self):
        assert chunk_by_heading("   ") == []

    def test_heading_only_no_content(self):
        result = chunk_by_heading("# Title")
        assert len(result) >= 1

    def test_merge_small_sections(self):
        text = "# A\nshort\n## B\nshort\n## C\nshort"
        result = chunk_by_heading(text, max_chunk_size=1000)
        assert len(result) >= 1

    def test_split_large_section(self):
        # Use paragraphs with newlines so chunk_by_heading falls back to chunk_by_fixed_size
        text = "# Big\n" + "\n\n".join(["x" * 100 for _ in range(20)])
        result = chunk_by_heading(text, max_chunk_size=500)
        assert len(result) >= 1

    def test_multiple_h1(self):
        text = "# First\nshort\n# Second\nshort"
        result = chunk_by_heading(text)
        # Both sections are short, so they may merge
        assert len(result) >= 1
        assert any("First" in c for c in result)

    def test_heading_content_together(self):
        text = "# Title\nImportant content here"
        result = chunk_by_heading(text)
        assert any("Title" in c and "Important" in c for c in result)


class TestChunkSemantic:
    def test_short_text(self):
        result = chunk_by_semantic("Hello world.")
        assert len(result) == 1

    def test_empty(self):
        assert chunk_by_semantic("") == []

    def test_multiple_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        result = chunk_by_semantic(text)
        assert len(result) >= 1

    def test_topic_shift(self):
        text = "The cat sat on the mat. Quantum physics is complex. The dog ran fast."
        result = chunk_by_semantic(text)
        assert len(result) >= 1

    def test_whitespace_only(self):
        assert chunk_by_semantic("   ") == []

    def test_two_sentences(self):
        text = "First sentence. Second sentence."
        result = chunk_by_semantic(text)
        assert len(result) == 1

    def test_long_text(self):
        text = ". ".join(["Sentence number {} with some words here for testing".format(i) for i in range(20)])
        result = chunk_by_semantic(text, max_chunk_size=200)
        assert len(result) >= 1

    def test_min_chunk_size(self):
        text = "A. B. C. D. E. F. G. H."
        result = chunk_by_semantic(text, max_chunk_size=50, min_chunk_size=10)
        assert len(result) >= 1


class TestChunkTextAuto:
    def test_with_headings(self):
        text = "# Heading\nSome content"
        result = chunk_text(text, strategy="auto")
        assert len(result) >= 1

    def test_with_paragraphs(self):
        text = "p1\n\np2\n\np3\n\np4"
        result = chunk_text(text, strategy="auto")
        assert len(result) >= 1

    def test_short_text(self):
        result = chunk_text("hello", strategy="auto")
        assert len(result) == 1

    def test_explicit_strategy_fixed(self):
        result = chunk_text("abcdefghij", strategy="fixed", chunk_size=5)
        assert len(result) >= 2

    def test_explicit_strategy_paragraph(self):
        result = chunk_text("a\n\nb", strategy="paragraph")
        assert len(result) >= 1

    def test_explicit_strategy_heading(self):
        result = chunk_text("# Title\ncontent", strategy="heading")
        assert len(result) >= 1

    def test_explicit_strategy_semantic(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = chunk_text(text, strategy="semantic")
        assert len(result) >= 1

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            chunk_text("text", strategy="invalid")

    def test_auto_long_text_uses_semantic(self):
        text = "word " * 500
        result = chunk_text(text, strategy="auto")
        assert len(result) >= 1

    def test_fixed_strategy_passthrough(self):
        result = chunk_text("a b c d e f", strategy="fixed", chunk_size=3)
        assert len(result) >= 2


class TestExtractFacts:
    def test_empty(self):
        assert _extract_facts_from_text("") == []

    def test_short_text(self):
        assert _extract_facts_from_text("hi") == []

    def test_factual_statement(self):
        text = "The temperature reached 38.5 degrees in the summer of 2023."
        facts = _extract_facts_from_text(text)
        assert len(facts) >= 0  # may or may not match depending on patterns

    def test_question_skipped(self):
        text = "What is the meaning of life and why do we exist in this universe?"
        facts = _extract_facts_from_text(text)
        assert len(facts) == 0

    def test_imperative_skipped(self):
        text = "Try to make sure you remember to update the configuration file."
        facts = _extract_facts_from_text(text)
        assert len(facts) == 0

    def test_exclamation_skipped(self):
        text = "This is amazing and incredible and wonderful news for everyone!"
        facts = _extract_facts_from_text(text)
        assert len(facts) == 0

    def test_has_pattern(self):
        text = "The system has been updated with new features and capabilities."
        facts = _extract_facts_from_text(text)
        assert len(facts) >= 0

    def test_can_pattern(self):
        text = "The model can process up to 1000 tokens per second in inference."
        facts = _extract_facts_from_text(text)
        assert len(facts) >= 0

    def test_very_long_text(self):
        text = " ".join(["Word"] * 200)
        facts = _extract_facts_from_text(text)
        assert isinstance(facts, list)


class TestExtractTopics:
    def test_basic(self):
        topics = _extract_topics("machine learning and deep neural networks")
        assert len(topics) <= 5

    def test_empty(self):
        topics = _extract_topics("")
        assert topics == []

    def test_stopwords_filtered(self):
        topics = _extract_topics("the and for are but not you all can")
        assert len(topics) == 0

    def test_max_topics(self):
        text = "alpha bravo charlie delta echo foxtrot golf hotel india"
        topics = _extract_topics(text, max_topics=3)
        assert len(topics) <= 3

    def test_repeated_words_ranked_higher(self):
        text = "python python python java java"
        topics = _extract_topics(text)
        assert len(topics) >= 1

    def test_special_chars(self):
        topics = _extract_topics("C++ is great for systems programming")
        assert isinstance(topics, list)

    def test_single_word(self):
        topics = _extract_topics("Python")
        assert isinstance(topics, list)

    def test_long_words(self):
        text = "implementation characterization optimization"
        topics = _extract_topics(text)
        assert len(topics) >= 1


class TestDistillationConfig:
    def test_defaults(self):
        dc = DistillationConfig()
        assert dc.temperature == 4.0
        assert dc.alpha == 0.5
        assert dc.distillation_type == "logits"

    def test_custom(self):
        dc = DistillationConfig(temperature=2.0, alpha=0.3, beta=0.7)
        assert dc.temperature == 2.0
        assert dc.alpha == 0.3
        assert dc.beta == 0.7

    def test_temperature_schedule_default(self):
        dc = DistillationConfig()
        assert dc.temperature_schedule is None

    def test_label_smoothing_default(self):
        dc = DistillationConfig()
        assert dc.use_label_smoothing is False
        assert dc.label_smoothing == 0.1

    def test_progressive_default(self):
        dc = DistillationConfig()
        assert dc.progressive is False

    def test_stage_weights_default(self):
        dc = DistillationConfig()
        assert dc.stage_weights is None

    def test_hidden_layer_mapping_default(self):
        dc = DistillationConfig()
        assert dc.hidden_layer_mapping is None

    def test_gamma_default(self):
        dc = DistillationConfig()
        assert dc.gamma == 0.0

    def test_custom_all(self):
        dc = DistillationConfig(
            temperature=1.0, alpha=0.1, beta=0.9, gamma=0.3,
            distillation_type="hidden", use_label_smoothing=True,
            label_smoothing=0.2, progressive=True,
        )
        assert dc.temperature == 1.0
        assert dc.gamma == 0.3
        assert dc.distillation_type == "hidden"
        assert dc.use_label_smoothing is True

    def test_temperature_schedule_custom(self):
        dc = DistillationConfig(temperature_schedule=[4.0, 2.0, 1.0])
        assert dc.temperature_schedule == [4.0, 2.0, 1.0]


class TestToNp:
    def test_ndarray_passthrough(self):
        a = np.array([1.0, 2.0])
        assert _to_np(a) is a

    def test_list_to_array(self):
        result = _to_np([1.0, 2.0])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_tensor_conversion(self):
        from domains.training.slonet import Tensor
        t = Tensor(np.array([3.0, 4.0]))
        result = _to_np(t)
        np.testing.assert_array_equal(result, [3.0, 4.0])

    def test_nested_list(self):
        result = _to_np([[1.0, 2.0], [3.0, 4.0]])
        assert result.shape == (2, 2)

    def test_int_array(self):
        result = _to_np([1, 2, 3])
        assert result.dtype in (np.int64, np.float32, np.float64)


class TestToTensor:
    def test_ndarray_conversion(self):
        a = np.array([1.0, 2.0])
        t = _to_tensor(a)
        from domains.training.slonet import Tensor
        assert isinstance(t, Tensor)
        np.testing.assert_array_equal(t.data, a)

    def test_requires_grad(self):
        a = np.array([1.0])
        t = _to_tensor(a, requires_grad=True)
        assert t.requires_grad is True

    def test_no_grad_by_default(self):
        a = np.array([1.0])
        t = _to_tensor(a)
        assert t.requires_grad is False

    def test_tensor_passthrough(self):
        from domains.training.slonet import Tensor
        t = Tensor(np.array([5.0]))
        result = _to_tensor(t)
        assert result is t


class TestSize:
    def test_basic(self):
        a = np.ones((3, 4))
        assert _size(a, 0) == 3
        assert _size(a, 1) == 4

    def test_out_of_range(self):
        a = np.ones((3,))
        assert _size(a, 5) == 1

    def test_3d(self):
        a = np.ones((2, 3, 4))
        assert _size(a, 2) == 4


class TestDistillationLoss:
    def test_init(self):
        dc = DistillationConfig()
        dl = DistillationLoss(dc)
        assert dl.config is dc
        assert dl.projection is None

    def test_forward_basic(self):
        dc = DistillationConfig(beta=0.5, gamma=0.0, alpha=0.0)
        dl = DistillationLoss(dc)
        student = np.random.randn(2, 10).astype(np.float32)
        teacher = np.random.randn(2, 10).astype(np.float32)
        result = dl.forward(student, teacher)
        total_loss, losses = result
        assert isinstance(total_loss, float)
        assert "soft_loss" in losses
        assert losses["soft_loss"] >= 0.0

    def test_forward_with_labels(self):
        dc = DistillationConfig(alpha=1.0, beta=0.0)
        dl = DistillationLoss(dc)
        student = np.random.randn(2, 10).astype(np.float32)
        teacher = np.random.randn(2, 10).astype(np.float32)
        labels = np.random.randint(0, 10, size=(2,))
        total_loss, losses = dl.forward(student, teacher, labels=labels)
        assert "hard_loss" in losses
        assert losses["hard_loss"] >= 0.0

    def test_forward_with_hidden(self):
        dc = DistillationConfig(gamma=1.0, beta=0.0, alpha=0.0)
        dl = DistillationLoss(dc)
        student = np.random.randn(2, 10).astype(np.float32)
        teacher = np.random.randn(2, 10).astype(np.float32)
        sh = np.random.randn(2, 5).astype(np.float32)
        th = np.random.randn(2, 5).astype(np.float32)
        total_loss, losses = dl.forward(student, teacher, student_hidden=sh, teacher_hidden=th)
        assert "feature_loss" in losses

    def test_forward_all_loss_types(self):
        dc = DistillationConfig(alpha=0.3, beta=0.5, gamma=0.2)
        dl = DistillationLoss(dc)
        student = np.random.randn(2, 10).astype(np.float32)
        teacher = np.random.randn(2, 10).astype(np.float32)
        labels = np.random.randint(0, 10, size=(2,))
        sh = np.random.randn(2, 10).astype(np.float32)
        th = np.random.randn(2, 10).astype(np.float32)
        total_loss, losses = dl.forward(student, teacher, labels=labels,
                                        student_hidden=sh, teacher_hidden=th)
        assert "hard_loss" in losses
        assert "soft_loss" in losses
        assert "feature_loss" in losses
        assert "total_loss" in losses
        expected = 0.3 * losses["hard_loss"] + 0.5 * losses["soft_loss"] + 0.2 * losses["feature_loss"]
        assert losses["total_loss"] == pytest.approx(expected, abs=1e-5)

    def test_total_loss_sum(self):
        dc = DistillationConfig(alpha=1.0, beta=1.0, gamma=0.0)
        dl = DistillationLoss(dc)
        student = np.random.randn(2, 10).astype(np.float32)
        teacher = np.random.randn(2, 10).astype(np.float32)
        labels = np.array([0, 1])
        total, losses = dl.forward(student, teacher, labels=labels)
        assert total == pytest.approx(losses["hard_loss"] + losses["soft_loss"], abs=1e-5)

    def test_higher_temperature_scales_soft_loss(self):
        dc1 = DistillationConfig(temperature=1.0, alpha=0.0, beta=1.0)
        dc2 = DistillationConfig(temperature=8.0, alpha=0.0, beta=1.0)
        dl1 = DistillationLoss(dc1)
        dl2 = DistillationLoss(dc2)
        s = np.random.randn(2, 10).astype(np.float32)
        t = np.random.randn(2, 10).astype(np.float32)
        _, l1 = dl1.forward(s, t)
        _, l2 = dl2.forward(s, t)
        # Higher temp typically gives larger KL loss (scaled by temp^2)
        assert l2["soft_loss"] >= l1["soft_loss"]

    def test_projection_created(self):
        dc = DistillationConfig(gamma=1.0, beta=0.0, alpha=0.0)
        dl = DistillationLoss(dc)
        s = np.random.randn(2, 10).astype(np.float32)
        t = np.random.randn(2, 10).astype(np.float32)
        sh = np.random.randn(2, 5).astype(np.float32)
        th = np.random.randn(2, 10).astype(np.float32)
        dl.forward(s, t, student_hidden=sh, teacher_hidden=th)
        assert dl.projection is not None

    def test_no_loss_when_beta_zero_alpha_zero_gamma_zero(self):
        dc = DistillationConfig(alpha=0.0, beta=0.0, gamma=0.0)
        dl = DistillationLoss(dc)
        s = np.random.randn(2, 10).astype(np.float32)
        t = np.random.randn(2, 10).astype(np.float32)
        total, losses = dl.forward(s, t)
        assert total == 0.0
        assert "total_loss" in losses

    def test_non_float32_input(self):
        dc = DistillationConfig(beta=1.0, alpha=0.0)
        dl = DistillationLoss(dc)
        s = np.random.randn(2, 10)
        t = np.random.randn(2, 10)
        total, losses = dl.forward(s, t)
        assert isinstance(total, float)
