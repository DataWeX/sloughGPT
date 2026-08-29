"""Tests for domains.feedback.training — TrainingExample, DPOPair."""

from domains.feedback.training import TrainingExample, DPOPair


class TestTrainingExample:
    def test_fields(self):
        te = TrainingExample(prompt="hello", response="hi", rating="positive")
        assert te.prompt == "hello"
        assert te.response == "hi"
        assert te.rating == "positive"
        assert te.quality_score is None

    def test_with_quality(self):
        te = TrainingExample(prompt="q", response="a", rating="good", quality_score=0.9)
        assert te.quality_score == 0.9


class TestDPOPair:
    def test_fields(self):
        dp = DPOPair(chosen="good answer", rejected="bad answer", prompt="question")
        assert dp.chosen == "good answer"
        assert dp.rejected == "bad answer"
        assert dp.prompt == "question"
