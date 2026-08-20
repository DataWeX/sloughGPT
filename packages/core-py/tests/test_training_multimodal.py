"""Tests for domains.feedback.training — TrainingExample, DPOPair; domains.multimodal.engine — MultimodalOutput; domains.multimodal.speech — TranscriptionResult."""

from domains.feedback.training import TrainingExample, DPOPair
from domains.multimodal.engine import MultimodalOutput
from domains.multimodal.speech import TranscriptionResult


class TestTrainingExample:
    def test_fields(self):
        te = TrainingExample(prompt="hi", response="hello", rating="good")
        assert te.prompt == "hi"
        assert te.response == "hello"
        assert te.quality_score is None

    def test_custom(self):
        te = TrainingExample(prompt="q", response="a", rating="bad", quality_score=0.3)
        assert te.quality_score == 0.3


class TestDPOPair:
    def test_fields(self):
        dp = DPOPair(chosen="yes", rejected="no", prompt="do it?")
        assert dp.chosen == "yes"
        assert dp.rejected == "no"
        assert dp.prompt == "do it?"


class TestMultimodalOutput:
    def test_fields(self):
        mo = MultimodalOutput(text="hello", confidence=0.95)
        assert mo.text == "hello"
        assert mo.confidence == 0.95


class TestTranscriptionResult:
    def test_fields(self):
        tr = TranscriptionResult(text="hello world", confidence=0.9, language="en")
        assert tr.text == "hello world"
        assert tr.confidence == 0.9
