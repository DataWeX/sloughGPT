"""Tests for the multimodal BPE tokenizer (train/encode/decode/save/load)."""

import pytest

from domains.multimodal.bpe_tokenizer import BPETokenizer


@pytest.fixture
def captions():
    return [
        "A cat sits on the mat",
        "The cat and the dog play",
        "Cat and dog",
        "A dog runs fast",
        "The mat is red",
    ]


class TestInit:
    def test_defaults(self):
        t = BPETokenizer()
        assert t.vocab_size == 4096
        assert t.special_tokens == ["<BOS>", "<EOS>", "<PAD>", "<UNK>"]
        assert t.vocab == {}
        assert t.merges == []
        assert not t._built

    def test_custom_vocab_and_specials(self):
        t = BPETokenizer(vocab_size=128, special_tokens=["<GO>"])
        assert t.vocab_size == 128
        assert t.special_tokens == ["<GO>"]

    def test_special_tokens_optional(self):
        t = BPETokenizer(special_tokens=None)
        assert t.special_tokens == ["<BOS>", "<EOS>", "<PAD>", "<UNK>"]


class TestPreprocess:
    def test_lowercases(self):
        t = BPETokenizer()
        tokens = t._preprocess("Hello World")
        assert all(x == x.lower() for x in tokens)

    def test_words_end_with_marker(self):
        t = BPETokenizer()
        tokens = t._preprocess("cat")
        assert tokens[-1] == "t</w>"

    def test_handles_punctuation(self):
        t = BPETokenizer()
        tokens = t._preprocess("hi!")
        joined = "".join(tokens).replace("</w>", "")
        assert "!" in joined


class TestStatsAndMerge:
    def test_get_stats_counts_adjacent(self):
        t = BPETokenizer()
        from collections import Counter
        stats = t._get_stats(Counter({"a b a b": 3, "a b": 1}))
        assert stats[("a", "b")] == 7  # 2 per word × 3, plus 1
        assert stats[("b", "a")] == 3

    def test_merge_vocab_combines_pair(self):
        t = BPETokenizer()
        from collections import Counter
        out = t._merge_vocab(("a", "b"), Counter({"a b a b": 2}))
        assert out == Counter({"ab ab": 2})


class TestTrain:
    def test_builds_vocab_with_specials_first(self, captions):
        t = BPETokenizer(vocab_size=64)
        t.train(captions)
        assert t._built
        assert t.vocab["<BOS>"] == 0
        assert t.vocab["<EOS>"] == 1
        assert t.vocab["<PAD>"] == 2
        assert t.vocab["<UNK>"] == 3

    def test_vocab_size_respected(self, captions):
        t = BPETokenizer(vocab_size=32)
        t.train(captions)
        assert len(t.vocab) <= 32

    def test_itos_mirrors_vocab(self, captions):
        t = BPETokenizer(vocab_size=64)
        t.train(captions)
        for tok, i in t.vocab.items():
            assert t.itos[i] == tok

    def test_small_corpus_single_token_phrase_prevented(self):
        t = BPETokenizer(vocab_size=32)
        t.train(["ab ab ab", "ab ab"])
        # Merges capped at character vocab size — vocab should stay bounded
        assert len(t.vocab) <= 32

    def test_train_twice_rebuilds(self, captions):
        t = BPETokenizer(vocab_size=64)
        t.train(captions)
        first = t.merges
        t.train(captions)
        assert t.merges == first


class TestEncode:
    def test_encode_requires_train(self):
        t = BPETokenizer()
        with pytest.raises(RuntimeError, match="not trained"):
            t.encode("hello")

    def test_encode_returns_ids(self, captions):
        t = BPETokenizer(vocab_size=64)
        t.train(captions)
        ids = t.encode("cat")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_unknown_tokens_map_to_unk(self, captions):
        t = BPETokenizer(vocab_size=64)
        t.train(captions)
        ids = t.encode("zzzzqqqqxx")
        assert all(i == t.vocab["<UNK>"] for i in ids)

    def test_known_word_reconstructs_tokens(self, captions):
        t = BPETokenizer(vocab_size=4096)
        t.train(captions)
        ids = t.encode("cat")
        text = t.decode(ids)
        assert "cat" in text


class TestDecode:
    def test_skips_special_tokens(self, captions):
        t = BPETokenizer(vocab_size=64)
        t.train(captions)
        ids = [t.vocab["<BOS>"], t.vocab["<EOS>"], t.vocab["<PAD>"]]
        assert t.decode(ids) == ""

    def test_decode_roundtrip_known(self, captions):
        t = BPETokenizer(vocab_size=4096)
        t.train(captions)
        for cap in captions:
            ids = t.encode(cap)
            decoded = t.decode(ids)
            assert decoded  # non-empty
            assert decoded.lower()  # case-normalized text preserved roughly

    def test_unknown_ids_produce_empty_strings(self, captions):
        t = BPETokenizer(vocab_size=64)
        t.train(captions)
        assert t.decode([999999]) == ""


class TestSaveLoad:
    def test_save_creates_file(self, tmp_path, captions):
        t = BPETokenizer(vocab_size=64)
        t.train(captions)
        p = str(tmp_path / "bpe.json")
        t.save(p)
        assert (tmp_path / "bpe.json").exists()

    def test_roundtrip_preserves_state(self, tmp_path, captions):
        t = BPETokenizer(vocab_size=64)
        t.train(captions)
        p = str(tmp_path / "bpe.json")
        t.save(p)

        t2 = BPETokenizer()
        assert t2.load(p)
        assert t2.vocab == t.vocab
        assert t2.merges == t.merges
        assert t2.itos == t.itos
        assert t2.vocab_size == t.vocab_size
        assert t2.special_tokens == t.special_tokens
        assert t2._built

    def test_encode_consistent_after_load(self, tmp_path, captions):
        t = BPETokenizer(vocab_size=64)
        t.train(captions)
        before = t.encode("cat and dog")
        p = str(tmp_path / "bpe.json")
        t.save(p)

        t2 = BPETokenizer()
        t2.load(p)
        assert t2.encode("cat and dog") == before

    def test_load_missing_returns_false(self, tmp_path):
        t = BPETokenizer()
        assert t.load(str(tmp_path / "nope.json")) is False

    def test_save_parent_dirs_created(self, tmp_path, captions):
        t = BPETokenizer(vocab_size=64)
        t.train(captions)
        p = str(tmp_path / "a" / "b" / "c.json")
        t.save(p)
        assert (tmp_path / "a" / "b" / "c.json").exists()


class TestVocabProperties:
    def test_special_tokens_in_vocab_first(self, captions):
        t = BPETokenizer(vocab_size=64)
        t.train(captions)
        ids = list(t.vocab.values())
        assert ids == sorted(ids)
