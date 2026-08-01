"""Tests for MorphTokenizer — own BPE + morphological analysis."""

import pytest
from domains.infrastructure.morph_tokenizer import MorphTokenizer
from domains.infrastructure.safetensors_loader import _find_safetensors, _get_model_dir

QWEN2_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def _is_cached(model_id: str) -> bool:
    """Whether a model's tokenizer is present in the local cache.

    Uses the same resolution the loader uses (searching both the standard HF
    cache and the flat project-local models/hf-cache/hub layout).
    """
    return (_get_model_dir(model_id) / "tokenizer.json").exists()


class TestMorphTokenizerGPT2:
    """GPT-2 tokenizer tests (byte_level=True)."""

    @pytest.fixture
    def tok(self):
        if not _is_cached("gpt2"):
            pytest.skip("gpt2 not cached locally")
        return MorphTokenizer.from_pretrained("gpt2")

    def test_vocab_size(self, tok):
        assert tok.vocab_size == 50257

    def test_byte_level_detected(self, tok):
        assert tok.byte_level is True

    def test_encode_simple(self, tok):
        ids = tok.encode("Hello")
        assert isinstance(ids, list)
        assert len(ids) >= 1
        assert all(isinstance(i, int) for i in ids)

    def test_encode_decode_roundtrip(self, tok):
        text = "The capital of France is Paris."
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_encode_decode_with_spaces(self, tok):
        text = "Hello world, how are you?"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_encode_single_char(self, tok):
        ids = tok.encode("a")
        assert len(ids) >= 1

    def test_encode_empty(self, tok):
        ids = tok.encode("")
        assert ids == []

    def test_tokenize_returns_strings(self, tok):
        tokens = tok.tokenize("Hello world")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_tokenize_containsĠ(self, tok):
        tokens = tok.tokenize("Hello world")
        assert any("Ġ" in t for t in tokens)

    def test_eos_token_id(self, tok):
        assert tok.eos_token_id == 50256

    def test_inv_vocab(self, tok):
        assert tok.inv_vocab[15496] == "Hello"


class TestMorphTokenizerQwen2:
    """Qwen2 tokenizer tests (byte_level=True, bfloat16 weights)."""

    @pytest.fixture
    def tok(self):
        if not _is_cached(QWEN2_ID):
            pytest.skip(f"{QWEN2_ID} not cached locally")
        return MorphTokenizer.from_pretrained(QWEN2_ID)

    def test_vocab_size(self, tok):
        assert tok.vocab_size > 100000

    def test_byte_level_detected(self, tok):
        assert tok.byte_level is True

    def test_encode_decode_roundtrip(self, tok):
        text = "The capital of France is Paris."
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_encode_knowledge(self, tok):
        ids = tok.encode("knowledge")
        assert len(ids) >= 1

    def test_encode_chinese(self, tok):
        text = "你好世界"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_tokenize(self, tok):
        tokens = tok.tokenize("The capital of France")
        assert len(tokens) >= 3

    def test_eos_token_id(self, tok):
        assert tok.eos_token_id > 0


class TestMorphologicalAnalysis:
    """Linguistic rule-based morphological tests."""

    @pytest.fixture
    def tok(self):
        if not _is_cached(QWEN2_ID):
            pytest.skip(f"{QWEN2_ID} not cached locally")
        return MorphTokenizer.from_pretrained(QWEN2_ID)

    def test_decompose_unhappiness(self, tok):
        assert tok.decompose("unhappiness") == ["un", "happy", "ness"]

    def test_decompose_running(self, tok):
        result = tok.decompose("running")
        assert "run" in result

    def test_decompose_cats(self, tok):
        result = tok.decompose("cats")
        assert "cat" in result

    def test_decompose_simple(self, tok):
        result = tok.decompose("hello")
        assert len(result) >= 1

    def test_stem_running(self, tok):
        assert tok.stem("running") == "run"

    def test_stem_cats(self, tok):
        assert tok.stem("cats") == "cat"

    def test_stem_better(self, tok):
        assert tok.stem("better") == "good"

    def test_stem_was(self, tok):
        assert tok.stem("was") == "be"

    def test_stem_went(self, tok):
        assert tok.stem("went") == "go"

    def test_root_distance_same(self, tok):
        assert tok.root_distance("running", "running") == 0

    def test_root_distance_same_root(self, tok):
        assert tok.root_distance("running", "ran") == 1

    def test_root_distance_unrelated(self, tok):
        assert tok.root_distance("cats", "dog") == 3

    def test_generate_forms_run(self, tok):
        forms = tok.generate_forms("run")
        assert "runs" in forms
        assert "running" in forms
        assert "ran" in forms

    def test_generate_forms_happy(self, tok):
        forms = tok.generate_forms("happy")
        assert "happiness" in forms
        assert "unhappy" in forms

    def test_find_related(self, tok):
        related = tok.find_related("running")
        assert "run" in related or "runs" in related

    def test_decompose_caching(self, tok):
        r1 = tok.decompose("unhappiness")
        r2 = tok.decompose("unhappiness")
        assert r1 is r2

    def test_vocabulary_coverage(self, tok):
        words = ["the", "cat", "sat", "on", "a", "mat"]
        cov = tok.vocabulary_coverage(words)
        assert 0.0 <= cov <= 1.0

    def test_morphological_diversity(self, tok):
        words = ["running", "jumping", "swimming", "walking"]
        div = tok.morphological_diversity(words)
        assert 0.0 < div <= 1.0
