"""Tests for domains.feedback.training — TrainingExample, DPOPair; domains.multimodal.engine — MultimodalOutput; domains.multimodal.speech — TranscriptionResult."""

import pytest
from domains.feedback.training import TrainingExample, DPOPair
from domains.multimodal.engine import MultimodalOutput
from domains.multimodal.speech import TranscriptionResult


# ── TrainingExample ──────────────────────────────────────────────────────


class TestTrainingExample:
    def test_fields(self):
        te = TrainingExample(prompt="hi", response="hello", rating="good")
        assert te.prompt == "hi"
        assert te.response == "hello"
        assert te.quality_score is None

    def test_custom(self):
        te = TrainingExample(prompt="q", response="a", rating="bad", quality_score=0.3)
        assert te.quality_score == 0.3

    def test_empty_prompt(self):
        te = TrainingExample(prompt="", response="r", rating="good")
        assert te.prompt == ""

    def test_empty_response(self):
        te = TrainingExample(prompt="q", response="", rating="good")
        assert te.response == ""

    def test_empty_rating(self):
        te = TrainingExample(prompt="q", response="r", rating="")
        assert te.rating == ""

    def test_long_prompt(self):
        long = "x" * 10000
        te = TrainingExample(prompt=long, response="r", rating="good")
        assert len(te.prompt) == 10000

    def test_quality_score_zero(self):
        te = TrainingExample(prompt="q", response="r", rating="good", quality_score=0.0)
        assert te.quality_score == 0.0

    def test_quality_score_one(self):
        te = TrainingExample(prompt="q", response="r", rating="good", quality_score=1.0)
        assert te.quality_score == 1.0

    def test_quality_score_negative(self):
        te = TrainingExample(prompt="q", response="r", rating="good", quality_score=-0.5)
        assert te.quality_score == -0.5

    def test_unicode_content(self):
        te = TrainingExample(prompt="日本語", response="回答", rating="good")
        assert te.prompt == "日本語"
        assert te.response == "回答"

    def test_equality(self):
        te1 = TrainingExample(prompt="q", response="r", rating="good", quality_score=0.8)
        te2 = TrainingExample(prompt="q", response="r", rating="good", quality_score=0.8)
        assert te1 == te2

    def test_inequality(self):
        te1 = TrainingExample(prompt="q", response="r", rating="good")
        te2 = TrainingExample(prompt="q", response="x", rating="good")
        assert te1 != te2

    def test_repr(self):
        te = TrainingExample(prompt="q", response="r", rating="good")
        assert "TrainingExample" in repr(te)

    def test_default_quality_score_is_none(self):
        te = TrainingExample(prompt="a", response="b", rating="c")
        assert te.quality_score is None

    def test_many_examples(self):
        examples = [
            TrainingExample(prompt=f"q{i}", response=f"r{i}", rating="good")
            for i in range(100)
        ]
        assert len(examples) == 100
        assert examples[0].prompt == "q0"
        assert examples[99].response == "r99"


# ── DPOPair ──────────────────────────────────────────────────────────────


class TestDPOPair:
    def test_fields(self):
        dp = DPOPair(chosen="yes", rejected="no", prompt="do it?")
        assert dp.chosen == "yes"
        assert dp.rejected == "no"
        assert dp.prompt == "do it?"

    def test_empty_chosen(self):
        dp = DPOPair(chosen="", rejected="no", prompt="q")
        assert dp.chosen == ""

    def test_empty_rejected(self):
        dp = DPOPair(chosen="yes", rejected="", prompt="q")
        assert dp.rejected == ""

    def test_empty_prompt(self):
        dp = DPOPair(chosen="yes", rejected="no", prompt="")
        assert dp.prompt == ""

    def test_same_chosen_rejected(self):
        dp = DPOPair(chosen="same", rejected="same", prompt="q")
        assert dp.chosen == dp.rejected

    def test_unicode(self):
        dp = DPOPair(chosen="はい", rejected="いいえ", prompt="質問")
        assert dp.chosen == "はい"
        assert dp.rejected == "いいえ"

    def test_long_content(self):
        long = "word " * 500
        dp = DPOPair(chosen=long, rejected="short", prompt="q")
        assert len(dp.chosen) > 2000

    def test_equality(self):
        dp1 = DPOPair(chosen="y", rejected="n", prompt="q")
        dp2 = DPOPair(chosen="y", rejected="n", prompt="q")
        assert dp1 == dp2

    def test_inequality(self):
        dp1 = DPOPair(chosen="y", rejected="n", prompt="q")
        dp2 = DPOPair(chosen="x", rejected="n", prompt="q")
        assert dp1 != dp2

    def test_repr(self):
        dp = DPOPair(chosen="y", rejected="n", prompt="q")
        assert "DPOPair" in repr(dp)

    def test_many_pairs(self):
        pairs = [
            DPOPair(chosen=f"c{i}", rejected=f"r{i}", prompt=f"p{i}")
            for i in range(50)
        ]
        assert len(pairs) == 50


# ── MultimodalOutput ─────────────────────────────────────────────────────


class TestMultimodalOutput:
    def test_fields(self):
        mo = MultimodalOutput(text="hello", confidence=0.95)
        assert mo.text == "hello"
        assert mo.confidence == 0.95

    def test_empty_text(self):
        mo = MultimodalOutput(text="", confidence=0.0)
        assert mo.text == ""

    def test_zero_confidence(self):
        mo = MultimodalOutput(text="x", confidence=0.0)
        assert mo.confidence == 0.0

    def test_max_confidence(self):
        mo = MultimodalOutput(text="x", confidence=1.0)
        assert mo.confidence == 1.0

    def test_high_confidence(self):
        mo = MultimodalOutput(text="a", confidence=0.9999)
        assert mo.confidence == 0.9999

    def test_unicode(self):
        mo = MultimodalOutput(text="画像は猫です", confidence=0.85)
        assert mo.text == "画像は猫です"

    def test_long_text(self):
        long = "description " * 200
        mo = MultimodalOutput(text=long, confidence=0.7)
        assert len(mo.text) > 2000

    def test_equality(self):
        m1 = MultimodalOutput(text="a", confidence=0.8)
        m2 = MultimodalOutput(text="a", confidence=0.8)
        assert m1 == m2

    def test_inequality(self):
        m1 = MultimodalOutput(text="a", confidence=0.8)
        m2 = MultimodalOutput(text="b", confidence=0.8)
        assert m1 != m2

    def test_repr(self):
        mo = MultimodalOutput(text="test", confidence=0.9)
        r = repr(mo)
        assert "MultimodalOutput" in r

    def test_special_characters(self):
        mo = MultimodalOutput(text="<html>&amp;</html>", confidence=0.5)
        assert mo.text == "<html>&amp;</html>"

    def test_newlines(self):
        mo = MultimodalOutput(text="line1\nline2\nline3", confidence=0.6)
        assert "\n" in mo.text


# ── TranscriptionResult ──────────────────────────────────────────────────


class TestTranscriptionResult:
    def test_fields(self):
        tr = TranscriptionResult(text="hello world", confidence=0.9, language="en")
        assert tr.text == "hello world"
        assert tr.confidence == 0.9

    def test_language(self):
        tr = TranscriptionResult(text="bonjour", confidence=0.85, language="fr")
        assert tr.language == "fr"

    def test_duration_default(self):
        tr = TranscriptionResult(text="t", confidence=0.7, language="en")
        assert tr.duration is None

    def test_duration_set(self):
        tr = TranscriptionResult(text="t", confidence=0.7, language="en", duration=4.5)
        assert tr.duration == 4.5

    def test_is_valid_default(self):
        tr = TranscriptionResult(text="t", confidence=0.7, language="en")
        assert tr.is_valid is True

    def test_is_valid_false(self):
        tr = TranscriptionResult(text="", confidence=0.0, language="en", is_valid=False)
        assert tr.is_valid is False

    def test_empty_text(self):
        tr = TranscriptionResult(text="", confidence=0.0, language="en")
        assert tr.text == ""

    def test_unicode(self):
        tr = TranscriptionResult(text="안녕하세요", confidence=0.92, language="ko")
        assert tr.text == "안녕하세요"

    def test_equality(self):
        t1 = TranscriptionResult(text="hi", confidence=0.9, language="en")
        t2 = TranscriptionResult(text="hi", confidence=0.9, language="en")
        assert t1 == t2

    def test_inequality(self):
        t1 = TranscriptionResult(text="a", confidence=0.9, language="en")
        t2 = TranscriptionResult(text="b", confidence=0.9, language="en")
        assert t1 != t2

    def test_repr(self):
        tr = TranscriptionResult(text="x", confidence=0.5, language="en")
        assert "TranscriptionResult" in repr(tr)

    def test_many_languages(self):
        for lang in ["en", "fr", "de", "es", "it", "pt", "ja", "ko", "zh", "ar"]:
            tr = TranscriptionResult(text="t", confidence=0.9, language=lang)
            assert tr.language == lang
