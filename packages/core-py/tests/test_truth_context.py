"""Tests for domains.infrastructure.truth_labeler — LabelResult, TruthLabeler; domains.infrastructure.context_core — ContextLayer, ContextFrame."""

from domains.infrastructure.truth_labeler import (
    LabelResult, TruthLabeler,
    get_truth_labeler, reset_truth_labeler,
    _rule_interrogative, _rule_directive, _rule_descriptive,
    _rule_analytical, _rule_procedural, _rule_conceptual, _rule_factual,
)
from domains.infrastructure.context_core import ContextLayer, ContextFrame


# ── LabelResult ──────────────────────────────────────────────────────

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

    def test_to_dict_has_all_keys(self):
        lr = LabelResult(label="x", confidence=0.5, reason="r", scores={"x": 0.5})
        d = lr.to_dict()
        assert set(d.keys()) == {"label", "confidence", "reason", "scores"}

    def test_scores_is_dict(self):
        lr = LabelResult(label="x", confidence=0.5, reason="r", scores={"x": 0.5})
        assert isinstance(lr.scores, dict)


# ── TruthLabeler init ────────────────────────────────────────────────

class TestTruthLabelerInit:
    def test_init(self):
        tl = TruthLabeler()
        assert len(tl._rules) == 7

    def test_singleton(self):
        reset_truth_labeler()
        tl1 = get_truth_labeler()
        tl2 = get_truth_labeler()
        assert tl1 is tl2
        reset_truth_labeler()


# ── Question labeling ────────────────────────────────────────────────

class TestLabelInterrogative:
    def test_label_question(self):
        tl = TruthLabeler()
        result = tl.label("What is the meaning of life?")
        assert "question" in result.label or "interrogative" in result.label
        assert result.confidence > 0

    def test_question_mark(self):
        tl = TruthLabeler()
        result = tl.label("Is this correct?")
        assert result.label == "interrogative"
        assert result.confidence >= 0.5

    def test_question_word_how(self):
        tl = TruthLabeler()
        result = tl.label("How does this work")
        assert result.label == "interrogative"

    def test_question_word_why(self):
        tl = TruthLabeler()
        result = tl.label("Why is the sky blue")
        assert result.label == "interrogative"

    def test_question_word_when(self):
        tl = TruthLabeler()
        result = tl.label("When will this be ready")
        assert result.label == "interrogative"

    def test_question_word_where(self):
        tl = TruthLabeler()
        result = tl.label("Where are the files")
        assert result.label == "interrogative"

    def test_question_word_who(self):
        tl = TruthLabeler()
        result = tl.label("Who wrote this code")
        assert result.label == "interrogative"

    def test_question_word_which(self):
        tl = TruthLabeler()
        result = tl.label("Which option should I choose")
        assert result.label == "interrogative"

    def test_question_word_is(self):
        tl = TruthLabeler()
        result = tl.label("Is this the right approach")
        assert result.label == "interrogative"

    def test_question_word_does(self):
        tl = TruthLabeler()
        result = tl.label("Does this work")
        assert result.label == "interrogative"

    def test_question_word_can(self):
        tl = TruthLabeler()
        result = tl.label("Can you help me")
        assert result.label == "interrogative"


# ── Directive labeling ───────────────────────────────────────────────

class TestLabelDirective:
    def test_label_directive(self):
        tl = TruthLabeler()
        result = tl.label("Please run the tests")
        assert "directive" in result.label
        assert result.confidence > 0

    def test_imperative_run(self):
        tl = TruthLabeler()
        result = tl.label("run the build")
        assert result.label == "directive"

    def test_imperative_create(self):
        tl = TruthLabeler()
        result = tl.label("create a new file")
        assert result.label == "directive"

    def test_imperative_build(self):
        tl = TruthLabeler()
        result = tl.label("build the project")
        assert result.label == "directive"

    def test_imperative_write(self):
        tl = TruthLabeler()
        result = tl.label("write a test for this")
        assert result.label == "directive"

    def test_imperative_delete(self):
        tl = TruthLabeler()
        result = tl.label("delete the old cache")
        assert result.label == "directive"

    def test_imperative_fix(self):
        tl = TruthLabeler()
        result = tl.label("fix the bug in login")
        assert result.label == "directive"

    def test_directive_you_should(self):
        tl = TruthLabeler()
        result = tl.label("you should update the config")
        assert result.label == "directive"

    def test_directive_you_must(self):
        tl = TruthLabeler()
        result = tl.label("you must save before closing")
        assert result.label == "directive"


# ── Statement / factual labeling ─────────────────────────────────────

class TestLabelFactual:
    def test_label_statement(self):
        tl = TruthLabeler()
        result = tl.label("The sky is blue and the grass is green.")
        assert isinstance(result.label, str)
        assert result.confidence > 0

    def test_factual_contains(self):
        tl = TruthLabeler()
        result = tl.label("This function contains a bug")
        assert result.label == "factual"

    def test_factual_measures(self):
        tl = TruthLabeler()
        result = tl.label("The file measures 1024 bytes")
        assert result.label == "factual"

    def test_factual_has(self):
        tl = TruthLabeler()
        result = tl.label("This module has three classes")
        assert result.label == "factual"

    def test_factual_will(self):
        tl = TruthLabeler()
        result = tl.label("This will take about 5 minutes")
        assert result.label == "factual"


# ── Descriptive labeling ─────────────────────────────────────────────

class TestLabelDescriptive:
    def test_descriptive_the(self):
        tl = TruthLabeler()
        result = tl.label("The server handles requests efficiently")
        assert result.label in ("descriptive", "factual")

    def test_descriptive_declarative(self):
        tl = TruthLabeler()
        result = tl.label("This function returns a boolean")
        assert result.label in ("descriptive", "factual")

    def test_descriptive_it(self):
        tl = TruthLabeler()
        result = tl.label("It works as expected")
        assert result.label in ("descriptive", "factual")


# ── Analytical labeling ─────────────────────────────────────────────

class TestLabelAnalytical:
    def test_analytical_because(self):
        tl = TruthLabeler()
        result = tl.label("This fails because the config is wrong")
        assert result.label == "analytical"

    def test_analytical_therefore(self):
        tl = TruthLabeler()
        result = tl.label("The test passed, therefore the fix works")
        assert result.label == "analytical"

    def test_analytical_however(self):
        tl = TruthLabeler()
        result = tl.label("However, this approach has limitations")
        assert result.label == "analytical"

    def test_analytical_thus(self):
        tl = TruthLabeler()
        result = tl.label("Thus we can conclude the algorithm is correct")
        assert result.label == "analytical"

    def test_analytical_complex_sentence(self):
        tl = TruthLabeler()
        result = tl.label("The system works because the tests pass")
        assert result.label == "analytical"


# ── Procedural labeling ─────────────────────────────────────────────

class TestLabelProcedural:
    def test_procedural_numbered_steps(self):
        tl = TruthLabeler()
        result = tl.label("1. Install dependencies. 2. Run migrations. 3. Start server.")
        assert result.label == "procedural"

    def test_procedural_step_keyword(self):
        tl = TruthLabeler()
        result = tl.label("First, configure the database. Then run the app.")
        assert result.label == "procedural"

    def test_procedural_next(self):
        tl = TruthLabeler()
        result = tl.label("Next, install the required packages")
        assert result.label == "procedural"

    def test_procedural_finally(self):
        tl = TruthLabeler()
        result = tl.label("Finally, restart the service")
        assert result.label == "procedural"


# ── Conceptual labeling ─────────────────────────────────────────────

class TestLabelConceptual:
    def test_conceptual_definition_pattern(self):
        tl = TruthLabeler()
        result = tl.label("Recursion is a technique that calls itself")
        assert result.label == "conceptual"

    def test_conceptual_abstract_marker(self):
        tl = TruthLabeler()
        result = tl.label("The concept of polymorphism allows flexibility")
        assert result.label == "conceptual"

    def test_conceptual_theory(self):
        tl = TruthLabeler()
        result = tl.label("This theory explains the behavior")
        assert result.label == "conceptual"

    def test_conceptual_principle(self):
        tl = TruthLabeler()
        result = tl.label("The principle of least surprise guides design")
        assert result.label == "conceptual"


# ── Empty / edge cases ──────────────────────────────────────────────

class TestLabelEdgeCases:
    def test_label_empty(self):
        tl = TruthLabeler()
        result = tl.label("")
        assert isinstance(result.label, str)
        assert result.confidence == 0.0

    def test_label_whitespace(self):
        tl = TruthLabeler()
        result = tl.label("   ")
        assert isinstance(result.label, str)

    def test_label_single_word(self):
        tl = TruthLabeler()
        result = tl.label("hello")
        assert isinstance(result.label, str)

    def test_label_exclamation(self):
        tl = TruthLabeler()
        result = tl.label("Great job!")
        assert isinstance(result.label, str)

    def test_label_batch(self):
        tl = TruthLabeler()
        results = tl.label_batch(["What?", "Run this.", "The sky is blue."])
        assert len(results) == 3
        for r in results:
            assert isinstance(r, LabelResult)
            assert r.confidence >= 0


# ── Rule functions directly ──────────────────────────────────────────

class TestRulesDirect:
    def test_rule_interrogative_qmark(self):
        scores, reasons = _rule_interrogative("Is this right?")
        assert scores.get("interrogative", 0) > 0

    def test_rule_interrogative_word(self):
        scores, reasons = _rule_interrogative("What is this")
        assert scores.get("interrogative", 0) > 0

    def test_rule_interrogative_no_match(self):
        scores, reasons = _rule_interrogative("The sky is blue")
        assert scores.get("interrogative", 0) == 0

    def test_rule_directive_match(self):
        scores, reasons = _rule_directive("run the tests")
        assert scores.get("directive", 0) > 0

    def test_rule_directive_no_match(self):
        scores, reasons = _rule_directive("the sky is blue")
        assert scores.get("directive", 0) == 0

    def test_rule_descriptive_match(self):
        scores, reasons = _rule_descriptive("the server is running")
        assert scores.get("descriptive", 0) > 0

    def test_rule_analytical_match(self):
        scores, reasons = _rule_analytical("because it failed")
        assert scores.get("analytical", 0) > 0

    def test_rule_procedural_match(self):
        scores, reasons = _rule_procedural("1. do this 2. do that")
        assert scores.get("procedural", 0) > 0

    def test_rule_conceptual_match(self):
        scores, reasons = _rule_conceptual("recursion is a concept")
        assert scores.get("conceptual", 0) > 0

    def test_rule_factual_match(self):
        scores, reasons = _rule_factual("the function contains a bug")
        assert scores.get("factual", 0) > 0


# ── ContextLayer ─────────────────────────────────────────────────────

class TestContextLayer:
    def test_fields(self):
        cl = ContextLayer(layer_type="session", content="hello", tokens=1, source="user", timestamp="t")
        assert cl.layer_type == "session"
        assert cl.content == "hello"
        assert cl.tokens == 1
        assert cl.priority == 1.0

    def test_default_priority(self):
        cl = ContextLayer(layer_type="memory", content="x", tokens=5, source="s", timestamp="t")
        assert cl.priority == 1.0

    def test_custom_priority(self):
        cl = ContextLayer(layer_type="rag", content="x", tokens=5, source="s", timestamp="t", priority=0.7)
        assert cl.priority == 0.7

    def test_layer_types(self):
        for lt in ("session", "memory", "rag", "system"):
            cl = ContextLayer(layer_type=lt, content="x", tokens=1, source="s", timestamp="t")
            assert cl.layer_type == lt


# ── ContextFrame ─────────────────────────────────────────────────────

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

    def test_to_prompt_includes_layers(self):
        cl = ContextLayer(layer_type="session", content="user msg", tokens=1, source="u", timestamp="t")
        cf = ContextFrame(id="f1", system_prompt="sys", layers=[cl], total_tokens=10, max_tokens=100, created_at="t")
        prompt = cf.to_prompt()
        assert "user msg" in prompt

    def test_to_prompt_sorted_by_priority(self):
        cl1 = ContextLayer(layer_type="rag", content="rag stuff", tokens=1, source="s", timestamp="t", priority=0.5)
        cl2 = ContextLayer(layer_type="session", content="session stuff", tokens=1, source="s", timestamp="t", priority=1.0)
        cf = ContextFrame(id="f1", system_prompt="sys", layers=[cl1, cl2], total_tokens=10, max_tokens=100, created_at="t")
        prompt = cf.to_prompt()
        idx_session = prompt.index("session stuff")
        idx_rag = prompt.index("rag stuff")
        assert idx_session < idx_rag

    def test_to_prompt_no_layers(self):
        cf = ContextFrame(id="f1", system_prompt="sys", layers=[], total_tokens=0, max_tokens=100, created_at="t")
        prompt = cf.to_prompt()
        assert "sys" in prompt

    def test_max_tokens(self):
        cf = ContextFrame(id="f1", system_prompt="sys", layers=[], total_tokens=0, max_tokens=2048, created_at="t")
        assert cf.max_tokens == 2048

    def test_created_at(self):
        cf = ContextFrame(id="f1", system_prompt="sys", layers=[], total_tokens=0, max_tokens=100, created_at="2025-01-01")
        assert cf.created_at == "2025-01-01"
