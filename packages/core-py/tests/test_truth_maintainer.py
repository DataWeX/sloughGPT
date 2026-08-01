"""Tests for TruthMaintainer — post-epoch self-retrain on misclassified texts."""

import numpy as np
import pytest

from domains.infrastructure.anchor_store import MeaningTags
from domains.infrastructure.truth_labeler import LabelResult
from domains.infrastructure.truth_maintainer import (
    TruthMaintainer,
    _encode_tokens,
    get_truth_maintainer,
)


def _tag_store():
    """Small MeaningTags with factual/interrogative reference vectors."""
    store = MeaningTags(dimension=8)
    factual = np.zeros(8)
    factual[0] = 1.0
    interrogative = np.zeros(8)
    interrogative[1] = 1.0
    store.add("factual", factual)
    store.add("interrogative", interrogative)
    return store


class _FakeLabeler:
    """Deterministic labeler for isolating the confidence filter."""

    def __init__(self, label, confidence):
        self._label = label
        self._confidence = confidence

    def label(self, text):
        return LabelResult(
            label=self._label,
            confidence=self._confidence,
            reason="fake",
            scores={self._label: self._confidence},
        )


class TestFindMisclassified:
    def test_detects_disagreement(self):
        store = _tag_store()
        m = TruthMaintainer(labeler=_FakeLabeler("factual", 0.9))
        texts = ["whatever"]
        embeddings = np.stack([store.get("interrogative")])
        found = m.find_misclassified(texts, embeddings, store)
        assert len(found) == 1
        assert found[0]["index"] == 0
        assert found[0]["text"] == "whatever"
        assert found[0]["rule_label"] == "factual"
        assert found[0]["embed_label"] == "interrogative"

    def test_agreement_not_misclassified(self):
        store = _tag_store()
        m = TruthMaintainer(labeler=_FakeLabeler("factual", 0.9))
        embeddings = np.stack([store.get("factual")])
        found = m.find_misclassified(["whatever"], embeddings, store)
        assert found == []

    def test_low_confidence_excluded(self):
        store = _tag_store()
        m = TruthMaintainer(labeler=_FakeLabeler("factual", 0.3))
        embeddings = np.stack([store.get("interrogative")])
        found = m.find_misclassified(["whatever"], embeddings, store)
        assert found == []

    def test_boundary_confidence_included(self):
        store = _tag_store()
        m = TruthMaintainer(labeler=_FakeLabeler("factual", 0.4))
        embeddings = np.stack([store.get("interrogative")])
        found = m.find_misclassified(["whatever"], embeddings, store)
        assert len(found) == 0

    def test_empty_inputs(self):
        store = _tag_store()
        m = TruthMaintainer()
        assert m.find_misclassified([], np.zeros((0, 8)), store) == []

    def test_multiple_misclassified(self):
        store = _tag_store()
        m = TruthMaintainer(labeler=_FakeLabeler("factual", 0.8))
        texts = ["a", "b", "c"]
        embeddings = np.stack([store.get("interrogative"), store.get("factual"), store.get("interrogative")])
        found = m.find_misclassified(texts, embeddings, store)
        assert [f["index"] for f in found] == [0, 2]

    def test_fields_present(self):
        store = _tag_store()
        m = TruthMaintainer(labeler=_FakeLabeler("factual", 0.8))
        embeddings = np.stack([store.get("interrogative")])
        found = m.find_misclassified(["text"], embeddings, store)
        assert set(found[0].keys()) == {"index", "text", "rule_label", "embed_label", "confidence"}


class TestGenerateCorrectivePairs:
    def _scenario(self, store):
        texts = ["Why is the sky blue?", "How are you?", "Water is wet", "Fire is hot"]
        embeddings = np.stack([store.get("factual") for _ in texts])
        return texts, embeddings

    def test_generates_triplets(self):
        store = _tag_store()
        m = TruthMaintainer()
        texts, embeddings = self._scenario(store)
        mis = m.find_misclassified(texts, embeddings, store)
        assert len(mis) == 2
        queries, positives, negatives = m.generate_corrective_pairs(mis, texts, embeddings, store)
        assert len(queries) == len(positives) == len(negatives) == 2
        assert set(queries) == {"Why is the sky blue?", "How are you?"}
        assert positives[0] == "How are you?"
        assert negatives[0] in texts and negatives[0] != queries[0]

    def test_max_pairs_limit(self):
        store = _tag_store()
        m = TruthMaintainer()
        texts, embeddings = self._scenario(store)
        mis = m.find_misclassified(texts, embeddings, store)
        queries, positives, negatives = m.generate_corrective_pairs(mis, texts, embeddings, store, max_pairs=1)
        assert len(queries) == len(positives) == len(negatives) == 1

    def test_no_positive_skips_pair(self):
        store = _tag_store()
        m = TruthMaintainer()
        texts = ["Why?", "Water is wet"]
        embeddings = np.stack([store.get("factual"), store.get("factual")])
        mis = m.find_misclassified(texts, embeddings, store)
        assert len(mis) == 1
        queries, positives, negatives = m.generate_corrective_pairs(mis, texts, embeddings, store)
        assert queries == []
        assert positives == []
        assert negatives == []

    def test_no_misclassified_returns_empty(self):
        store = _tag_store()
        m = TruthMaintainer()
        texts, embeddings = self._scenario(store)
        queries, positives, negatives = m.generate_corrective_pairs([], texts, embeddings, store)
        assert (queries, positives, negatives) == ([], [], [])


class TestApplyCorrection:
    def test_empty_queries_returns_zero(self):
        m = TruthMaintainer()
        loss = m.apply_correction(None, [], [], [], None, {"<unk>": 0})
        assert loss == 0.0

    def test_runs_and_updates_params(self):
        from domains.training.slonet import Tensor

        rng = np.random.default_rng(7)
        params = [Tensor(rng.standard_normal((1, 4)).astype(np.float64), requires_grad=True)]

        class FakeEncoder:
            def parameters(self):
                return params

            def forward(self, ids):
                x = Tensor(np.asarray(ids, dtype=np.float64).mean(axis=1, keepdims=True))
                return x @ params[0]

        def encode_fn(text, max_len):
            ids = np.zeros(max_len, dtype=np.int64)
            # query pulled toward +, away from -
            ids[0] = {"q": 5, "p": -4, "n": 1}[text]
            return ids

        m = TruthMaintainer()
        before = params[0].data.copy()
        loss = m.apply_correction(
            FakeEncoder(),
            ["q", "q"],
            ["p", "p"],
            ["n", "n"],
            None,
            {},
            encode_fn=encode_fn,
            max_seq_len=8,
        )
        assert isinstance(loss, float)
        assert loss >= 0.0
        assert not np.allclose(before, params[0].data)

    def test_fallback_tokenizer_path(self):
        from domains.training.slonet import Tensor

        rng = np.random.default_rng(3)
        params = [Tensor(rng.standard_normal((1, 4)).astype(np.float64), requires_grad=True)]

        class FakeEncoder:
            def parameters(self):
                return params

            def forward(self, ids):
                x = Tensor(np.asarray(ids, dtype=np.float64).mean(axis=1, keepdims=True))
                return x @ params[0]

        m = TruthMaintainer()
        vocab = {"q": 5, "p": -4, "n": 1}
        loss = m.apply_correction(
            FakeEncoder(),
            ["q", "q"],
            ["p", "p"],
            ["n", "n"],
            None,
            vocab,
            max_seq_len=8,
        )
        assert isinstance(loss, float)


class TestEncodeTokens:
    def test_pads_to_max_len(self):
        ids = _encode_tokens("hello world", {"hello": 1, "world": 2}, max_len=6)
        assert ids.shape == (6,)
        assert ids[:2].tolist() == [1, 2]
        assert ids[2:].tolist() == [0, 0, 0, 0]

    def test_unknown_token_uses_unk(self):
        ids = _encode_tokens("hello nope", {"hello": 1, "<unk>": 9}, max_len=4)
        assert ids[1] == 9

    def test_truncates_long_text(self):
        ids = _encode_tokens("a b c d e f g", {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7}, max_len=3)
        assert ids.tolist() == [1, 2, 3]

    def test_lowercases(self):
        ids = _encode_tokens("HELLO", {"hello": 5, "<unk>": 0}, max_len=4)
        assert ids[0] == 5


class TestGetTruthMaintainer:
    def test_singleton(self):
        assert get_truth_maintainer() is get_truth_maintainer()
        assert isinstance(get_truth_maintainer(), TruthMaintainer)
