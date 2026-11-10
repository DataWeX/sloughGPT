"""Tests for MorphTokenizer — own BPE + morphological analysis."""

import pytest

from domain.infrastructure._internal.model_resolver import get_model_dir
from domain.infrastructure._internal.morph_tokenizer import MorphTokenizer

QWEN2_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def _is_cached(model_id: str) -> bool:
    """Whether a model's tokenizer is present in the local cache.

    Uses the same resolution the loader uses (searching both the standard HF
    cache and the flat project-local models/hf-cache/hub layout).
    """
    return (get_model_dir(model_id) / "tokenizer.json").exists()


class TestMorphTokenizerGPT2:
    """GPT-2 tokenizer tests (byte_level=True)."""

    @pytest.fixture
    def tok(self):
        if not _is_cached("gpt2"):
            pytest.skip("gpt2 not cached locally")
        return MorphTokenizer.from_pretrained("gpt2")

    def test_vocab_size(self, tok):
        assert tok.vocab_size == 50257

    def test_chat_stop_ids_plain_eos_only(self, tok):
        stop = tok.chat_stop_ids()
        assert stop == (tok.eos_token_id,)

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

    def test_chat_stop_ids_includes_im_end(self, tok):
        stop = tok.chat_stop_ids()
        assert 50256 in stop
        assert tok.added_tokens["<|im_end|>"] in stop
        assert tok.added_tokens["<|endoftext|>"] in stop

    def test_chat_stop_ids_ordered_tuple(self, tok):
        stop = tok.chat_stop_ids()
        assert isinstance(stop, tuple)
        assert stop == tuple(sorted(stop))


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

    def test_stem_ies_suffix(self, tok):
        assert tok.stem("cries") == "crie"
        assert tok.stem("spies") == "spie"

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


class TestMorphTokenizerLocalDir:
    def test_loads_from_local_directory(self, tmp_path):
        import json

        tok_data = {
            "model": {
                "vocab": {"a": 0, "b": 1, "c": 2, " ": 3},
                "merges": [],
                "eos_token_id": 3,
            },
            "pre_tokenizer": {"type": "Whitespace"},
        }
        (tmp_path / "tokenizer.json").write_text(json.dumps(tok_data))
        tok = MorphTokenizer.from_pretrained(str(tmp_path))
        assert tok.vocab_size == 4
        assert tok.encode("abc") == [0, 1, 2]


class TestByteFallbackDecodeStreaming:
    """Single-token streaming decode must preserve ▁→space (card 053).

    Full-sequence decode suppresses the first token's ▁ (SentencePiece
    convention) — that must stay. Single-token decode (slonet_provider chat
    stream decodes one id per call) must NOT drop the ▁, or streamed words
    glue together.
    """

    @pytest.fixture
    def tok(self):
        return MorphTokenizer(
            vocab={"▁Hello": 10, "▁world": 11, "<0x41>": 12},
            merges=[],
            byte_fallback=True,
            byte_level=False,
            eos_token_id=0,
        )

    def test_single_token_decode_preserves_space(self, tok):
        assert tok.decode([10]) == " Hello"

    def test_single_byte_token_decodes(self, tok):
        assert tok.decode([12]) == "A"

    def test_full_sequence_first_space_suppressed(self, tok):
        assert tok.decode([10, 11]) == "Hello world"

    def test_stream_concat_matches_full_decode_after_start_strip(self, tok):
        ids = [10, 11]
        streamed = "".join(tok.decode([i]) for i in ids)
        assert streamed.lstrip(" ") == tok.decode(ids)


class TestBPEMergeComplexity:
    """`_bpe_encode` must be O(n log n), not O(n²) (card 20261102_001616).

    The pre-fix merge loop rescanned EVERY adjacent pair and rebuilt the whole
    list on each merge. Measured: 1024 chars -> 172ms, 3120 -> 1.65s,
    6000 -> 6.18s (fitted exponent 2.02), which is a large slice of /chat's
    time-to-first-token for any non-trivial prompt.
    """

    @pytest.fixture
    def tok(self):
        if not _is_cached(QWEN2_ID):
            pytest.skip(f"{QWEN2_ID} not cached locally")
        return MorphTokenizer.from_pretrained(QWEN2_ID)

    @staticmethod
    def _reference_bpe(bpe_ranks, tokens):
        """The pre-fix algorithm, copied verbatim — differential oracle.

        Kept here (not in production code) so the optimized merge can be
        proved byte-identical to the behaviour the shipped tokenizer had.
        """
        tokens = list(tokens)
        while len(tokens) > 1:
            best_pair = None
            best_rank = float("inf")
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = bpe_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = (i, pair)
            if best_pair is None or best_rank == float("inf"):
                break
            i, (a, b) = best_pair
            tokens = tokens[:i] + [a + b] + tokens[i + 2 :]
        return tokens

    def test_merge_is_byte_identical_to_reference(self, tok):
        """Randomised differential test: optimized == pre-fix, exactly."""
        import random

        assert hasattr(tok, "_bpe_merge"), "MorphTokenizer._bpe_merge missing"
        rng = random.Random(20261109)
        symbols = sorted({t for pair in tok.bpe_ranks for t in pair})
        assert symbols, "tokenizer declares no merges"

        for trial in range(80):
            n = rng.randint(2, 90)
            init = [rng.choice(symbols) for _ in range(n)]
            got = tok._bpe_merge(list(init))
            want = self._reference_bpe(tok.bpe_ranks, init)
            assert got == want, f"trial {trial} n={n}: {got!r} != {want!r}"

    def test_merge_handles_no_mergeable_pairs(self, tok):
        """Pairs absent from bpe_ranks must be returned untouched."""
        assert hasattr(tok, "_bpe_merge"), "MorphTokenizer._bpe_merge missing"
        sentinel = ["__no_such_pair_a__", "__no_such_pair_b__", "__no_such_pair_c__"]
        assert tok._bpe_merge(list(sentinel)) == sentinel
        assert tok._bpe_merge([]) == []
        assert tok._bpe_merge(["only"]) == ["only"]

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_merge_small_inputs_byte_identical(self, tok, n):
        """Lock the crossover region (n=2..6) explicitly.

        The randomised oracle draws n from randint(2, 90), so at 80 trials it
        may never land on the smallest sizes — yet that is precisely where the
        heap + linked-list setup costs more than the O(n²) scan it replaces
        (measured 1.62x slower at n=2, break-even by n=5). Correctness there
        must not rest on a dice roll.
        """
        import random

        rng = random.Random(20261109 + n)
        symbols = sorted({t for pair in tok.bpe_ranks for t in pair})
        for _ in range(25):
            init = [rng.choice(symbols) for _ in range(n)]
            got = tok._bpe_merge(list(init))
            want = self._reference_bpe(tok.bpe_ranks, init)
            assert got == want, f"n={n}: {got!r} != {want!r}"

    def test_encode_scales_subquadratically(self, tok):
        """4x the input must not cost ~16x the time (O(n²) signature).

        A ratio threshold (rather than a wall-clock budget) keeps this honest
        across machines: O(n²) -> ~16x, O(n log n) -> ~5x.
        """
        import time

        base = "The quick brown fox jumps over the lazy dog. "
        small = (base * 36)[:1500]
        large = (base * 140)[:6000]

        tok.encode(small)  # warm numba / flatten weights
        tok.encode(large)

        t0 = time.perf_counter()
        tok.encode(small)
        t_small = time.perf_counter() - t0

        t0 = time.perf_counter()
        tok.encode(large)
        t_large = time.perf_counter() - t0

        ratio = t_large / max(t_small, 1e-9)
        assert ratio < 8, (
            f"encode took {t_large * 1000:.0f}ms for 6000 chars vs "
            f"{t_small * 1000:.0f}ms for 1500 = {ratio:.1f}x for 4x input "
            "(quadratic signature)"
        )
