"""Tests for domains.learner.knowledge — KnowledgeFact, FeedSubscription, _topic_slug; domains.training.distillation — DistillationConfig, DistillationLoss."""

import numpy as np
from domains.learner.knowledge import KnowledgeFact, FeedSubscription, _topic_slug
from domains.training.distillation import DistillationConfig, DistillationLoss


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


class TestTopicSlug:
    def test_normal(self):
        assert _topic_slug("Machine Learning") == "machine_learning"
    def test_special_chars(self):
        assert _topic_slug("AI/ML & NLP!") == "ai_ml_nlp_"
    def test_truncate(self):
        long = "a" * 100
        assert len(_topic_slug(long)) == 64


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
