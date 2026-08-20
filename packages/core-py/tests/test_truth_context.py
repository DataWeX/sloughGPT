"""Tests for domains.infrastructure.truth_labeler — LabelResult, TruthLabeler; domains.infrastructure.context_core — ContextLayer, ContextFrame."""

from domains.infrastructure.truth_labeler import LabelResult, TruthLabeler
from domains.infrastructure.context_core import ContextLayer, ContextFrame


class TestLabelResult:
    def test_fields(self):
        lr = LabelResult(label="question", confidence=0.9, reason="has question mark", scores={"question": 0.9})
        assert lr.label == "question"
        assert lr.confidence == 0.9

    def test_to_dict(self):
        lr = LabelResult(label="statement", confidence=0.8, reason="declarative", scores={"statement": 0.8})
        d = lr.to_dict()
        assert isinstance(d, dict)
        assert d["label"] == "statement"
        assert d["confidence"] == 0.8


class TestTruthLabeler:
    def test_init(self):
        tl = TruthLabeler()
        assert len(tl._rules) == 7

    def test_label_question(self):
        tl = TruthLabeler()
        result = tl.label("What is the meaning of life?")
        assert "question" in result.label or "interrogative" in result.label
        assert result.confidence > 0

    def test_label_statement(self):
        tl = TruthLabeler()
        result = tl.label("The sky is blue and the grass is green.")
        assert isinstance(result.label, str)
        assert result.confidence > 0

    def test_label_empty(self):
        tl = TruthLabeler()
        result = tl.label("")
        assert isinstance(result.label, str)


class TestContextLayer:
    def test_fields(self):
        cl = ContextLayer(layer_type="session", content="hello", tokens=1, source="user", timestamp="t")
        assert cl.layer_type == "session"
        assert cl.content == "hello"
        assert cl.tokens == 1
        assert cl.priority == 1.0


class TestContextFrame:
    def test_fields(self):
        cl = ContextLayer(layer_type="session", content="hi", tokens=1, source="u", timestamp="t")
        cf = ContextFrame(id="f1", system_prompt="sys", layers=[cl], total_tokens=10, max_tokens=100, created_at="t")
        assert cf.id == "f1"
        assert cf.system_prompt == "sys"
        assert cf.total_tokens == 10

    def test_to_prompt(self):
        cl = ContextLayer(layer_type="session", content="hello", tokens=1, source="u", timestamp="t")
        cf = ContextFrame(id="f1", system_prompt="sys", layers=[cl], total_tokens=10, max_tokens=100, created_at="t")
        prompt = cf.to_prompt()
        assert isinstance(prompt, str)
        assert "sys" in prompt
