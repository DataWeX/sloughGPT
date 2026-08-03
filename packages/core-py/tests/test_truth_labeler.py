"""Tests for TruthLabeler — rule-based first-glance text labeling."""

import pytest

from domains.infrastructure.truth_labeler import (
    LabelResult,
    TruthLabeler,
    get_truth_labeler,
)


class TestLabelResult:
    def test_to_dict(self):
        r = LabelResult(label="factual", confidence=0.8, reason="declarative",
                        scores={"factual": 0.8})
        d = r.to_dict()
        assert d["label"] == "factual"
        assert d["confidence"] == pytest.approx(0.8)
        assert d["reason"] == "declarative"
        assert d["scores"]["factual"] == pytest.approx(0.8)


class TestTruthLabelerEmpty:
    def test_empty_text(self):
        r = TruthLabeler().label("")
        assert r.label == "descriptive"
        assert r.confidence == 0.0
        assert r.reason == "empty text"
        assert r.scores == {}

    def test_whitespace_text(self):
        r = TruthLabeler().label("   \n\t ")
        assert r.label == "descriptive"
        assert r.confidence == 0.0

    def test_label_strips_input(self):
        r = TruthLabeler().label("  why is the sky blue?  ")
        assert r.label == "interrogative"


class TestTruthLabelerInterrogative:
    def test_ends_with_question_mark(self):
        r = TruthLabeler().label("What is the capital of France?")
        assert r.label == "interrogative"
        assert r.confidence >= 0.5

    def test_starts_with_question_word(self):
        r = TruthLabeler().label("How does photosynthesis work")
        assert r.label == "interrogative"

    def test_question_word_has_reason(self):
        r = TruthLabeler().label("Why did the war start?")
        assert r.reason == "ends with ?"


class TestTruthLabelerDirective:
    def test_please_prefix(self):
        r = TruthLabeler().label("please send me the report")
        assert r.label == "directive"

    def test_imperative_verb(self):
        r = TruthLabeler().label("run the tests now")
        assert r.label == "directive"

    def test_imperative_verb_not_directive_start(self):
        r = TruthLabeler().label("verify the build passes")
        assert r.label == "directive"
        assert r.scores["directive"] == 0.6
        assert "verify" in r.reason

    def test_you_should_prefix(self):
        r = TruthLabeler().label("you should check the logs")
        assert r.label == "directive"


class TestTruthLabelerDescriptive:
    def test_the_prefix(self):
        r = TruthLabeler().label("the cat is sitting on the mat")
        assert r.label == "descriptive"

    def test_this_prefix(self):
        r = TruthLabeler().label("this room has high ceilings")
        assert r.label == "descriptive"


class TestTruthLabelerAnalytical:
    def test_because_marker(self):
        r = TruthLabeler().label("Prices rose because demand increased")
        assert r.label == "analytical"

    def test_however_marker(self):
        r = TruthLabeler().label("The plan was sound, however execution failed")
        assert r.label == "analytical"

    def test_complex_sentence(self):
        r = TruthLabeler().label("One factor, two factors, three factors influence growth")
        assert r.label == "analytical"


class TestTruthLabelerProcedural:
    def test_numbered_steps(self):
        r = TruthLabeler().label("1. Open the file 2. Edit it 3. Save")
        assert r.label == "procedural"

    def test_step_keyword(self):
        r = TruthLabeler().label("first, gather ingredients then mix them")
        assert r.label == "procedural"


class TestTruthLabelerConceptual:
    def test_definition_pattern(self):
        r = TruthLabeler().label("A neural network is a system that learns from data")
        assert r.label == "conceptual"

    def test_x_is_y(self):
        r = TruthLabeler().label("Gravity is a fundamental force")
        assert r.label == "conceptual"

    def test_abstract_marker(self):
        r = TruthLabeler().label("the concept of justice")
        assert r.label == "conceptual"


class TestTruthLabelerFactual:
    def test_factual_indicator(self):
        r = TruthLabeler().label("Water contains hydrogen and oxygen")
        assert r.label == "factual"

    def test_default_fallback(self):
        r = TruthLabeler().label("abcdefghijklmnop!")
        assert r.label == "factual"
        assert r.confidence == pytest.approx(0.1)

    def test_confidence_in_range(self):
        for text in [
            "The sky is blue",
            "what time is it?",
            "please wait",
            "this is a test sentence",
            "a b c d",
        ]:
            r = TruthLabeler().label(text)
            assert 0.0 <= r.confidence <= 1.0


class TestTruthLabelerBatch:
    def test_batch_matches_individual(self):
        l = TruthLabeler()
        texts = ["The sky is blue", "why?", "run the command"]
        batch = l.label_batch(texts)
        assert len(batch) == 3
        for text, r in zip(texts, batch):
            single = l.label(text)
            assert r.label == single.label
            assert r.confidence == pytest.approx(single.confidence)


class TestTruthLabelerScores:
    def test_all_labels_present(self):
        r = TruthLabeler().label("something to classify here")
        assert set(r.scores.keys()) == {
            "factual", "conceptual", "procedural", "interrogative",
            "descriptive", "directive", "analytical",
        }

    def test_scores_sum_positive(self):
        r = TruthLabeler().label("how do servers work?")
        assert sum(r.scores.values()) > 0


class TestGetTruthLabeler:
    def test_singleton(self):
        assert get_truth_labeler() is get_truth_labeler()
        assert isinstance(get_truth_labeler(), TruthLabeler)
