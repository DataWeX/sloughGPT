"""Tests for the content-addressed token cache (prepare_data warm path)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from domain.training._internal.token_cache import (
    DEFAULT_CACHE_ROOT,
    cache_paths,
    file_sha256,
    store_cache,
    tokenizer_fingerprint,
    try_load,
)

TEXT = (
    "The mitochondria is the powerhouse of the cell. It converts nutrients "
    "into ATP through oxidative phosphorylation, a process studied extensively "
    "in molecular biology since the mid twentieth century."
)


class _FakeTok:
    """Minimal SloBPE-compatible stub: stoi/itos/encode/vocab_size."""

    def __init__(self, mapping: dict[str, int]):
        self.stoi = dict(mapping)
        self.itos = {v: k for k, v in mapping.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, **_kw) -> list[int]:
        return [self.stoi.get(c, 1) for c in text]


def _write_text(tmp: Path, body: str = TEXT) -> Path:
    p = tmp / "corpus.txt"
    p.write_text(body, encoding="utf-8")
    return p


class TestKeying:
    def test_file_sha_changes_on_edit(self, tmp_path: Path):
        p = _write_text(tmp_path)
        before = file_sha256(p)
        p.write_text(TEXT + " more words.", encoding="utf-8")
        assert file_sha256(p) != before

    def test_tokenizer_fp_changes_on_swap(self):
        a = tokenizer_fingerprint(_FakeTok({"a": 0, "b": 1}))
        b = tokenizer_fingerprint(_FakeTok({"a": 0, "c": 1}))
        assert a != b

    def test_char_mode_uses_stable_fp(self, tmp_path: Path):
        p = _write_text(tmp_path)
        bin1, meta1 = cache_paths(p, "char", "nochar", tmp_path / "cache")
        bin2, meta2 = cache_paths(p, "char", "nochar", tmp_path / "cache")
        assert (bin1, meta1) == (bin2, meta2)


class TestRoundtrip:
    def test_char_roundtrip(self, tmp_path: Path):
        src = _write_text(tmp_path)
        root = tmp_path / "cache"
        stoi = {c: i for i, c in enumerate(sorted(set(TEXT)))}
        itos = dict(enumerate(sorted(set(TEXT))))
        data = np.asarray([stoi[c] for c in TEXT], dtype=np.int64)
        out = store_cache(src, data, len(stoi), stoi, itos, "char", cache_root=root)
        assert out is not None
        hit = try_load(src, "char", cache_root=root)
        assert hit is not None
        cached, vocab, c_stoi, c_itos = hit
        assert vocab == len(stoi)
        assert np.array_equal(np.asarray(cached), data)
        assert c_stoi == stoi
        assert {int(k): v for k, v in c_itos.items()} == itos

    def test_bpe_roundtrip_omits_stoi(self, tmp_path: Path):
        src = _write_text(tmp_path)
        root = tmp_path / "cache"
        tok = _FakeTok({c: i for i, c in enumerate(sorted(set(TEXT)))})
        data = np.asarray(tok.encode(TEXT), dtype=np.int64)
        store_cache(
            src,
            data,
            tok.vocab_size,
            tok.stoi,
            tok.itos,
            "bpe",
            tokenizer=tok,
            cache_root=root,
        )
        hit = try_load(src, "bpe", tok, cache_root=root)
        assert hit is not None
        cached, vocab, c_stoi, _ = hit
        assert vocab == tok.vocab_size
        assert np.array_equal(np.asarray(cached), data)
        # bpe meta does not persist stoi — it reloads from the tokenizer.
        assert c_stoi == {}

    def test_compact_dtype_on_disk(self, tmp_path: Path):
        src = _write_text(tmp_path)
        root = tmp_path / "cache"
        stoi = {c: i for i, c in enumerate(sorted(set(TEXT)))}
        data = np.asarray([stoi[c] for c in TEXT], dtype=np.int64)
        bin_path, meta_path = store_cache(src, data, len(stoi), stoi, {}, "char", cache_root=root)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["dtype"] == "uint16"
        assert bin_path.stat().st_size == len(TEXT) * 2  # uint16, not int64


class TestInvalidation:
    def test_source_edit_misses(self, tmp_path: Path):
        src = _write_text(tmp_path)
        root = tmp_path / "cache"
        stoi = {c: i for i, c in enumerate(sorted(set(TEXT)))}
        data = np.asarray([stoi[c] for c in TEXT], dtype=np.int64)
        store_cache(src, data, len(stoi), stoi, {}, "char", cache_root=root)
        src.write_text(TEXT + " Edited corpus body.", encoding="utf-8")
        assert try_load(src, "char", cache_root=root) is None

    def test_tokenizer_swap_misses(self, tmp_path: Path):
        src = _write_text(tmp_path)
        root = tmp_path / "cache"
        tok_a = _FakeTok({"a": 0, "b": 1})
        tok_b = _FakeTok({"a": 0, "z": 1})
        data = np.asarray([0, 1], dtype=np.int64)
        store_cache(
            src,
            data,
            2,
            tok_a.stoi,
            tok_a.itos,
            "bpe",
            tokenizer=tok_a,
            cache_root=root,
        )
        assert try_load(src, "bpe", tok_b, cache_root=root) is None
        assert try_load(src, "bpe", tok_a, cache_root=root) is not None


class TestVocabOverflow:
    def test_vocab_above_65535_uses_uint32(self, tmp_path: Path):
        src = _write_text(tmp_path)
        root = tmp_path / "cache"
        vocab = 70_000
        data = np.asarray([0, 65535, 69999, 12], dtype=np.int64)
        out = store_cache(src, data, vocab, {}, {}, "char", cache_root=root)
        assert out is not None
        bin_path, meta_path = out
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["dtype"] == "uint32"
        assert bin_path.stat().st_size == 4 * 4  # 4 tokens x 4 bytes
        hit = try_load(src, "char", cache_root=root)
        assert hit is not None
        assert np.array_equal(np.asarray(hit[0]), data)
        assert hit[1] == vocab


class TestAtomicity:
    def test_no_tmp_left_and_meta_arrives(self, tmp_path: Path):
        src = _write_text(tmp_path)
        root = tmp_path / "cache"
        stoi = {c: i for i, c in enumerate(sorted(set(TEXT)))}
        data = np.asarray([stoi[c] for c in TEXT], dtype=np.int64)
        store_cache(src, data, len(stoi), stoi, {}, "char", cache_root=root)
        leftovers = [p for p in root.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []
        metas = list(root.glob("*.meta.json"))
        bins = list(root.glob("*.bin"))
        assert len(metas) == 1 and len(bins) == 1

    def test_truncated_bin_is_miss_not_crash(self, tmp_path: Path):
        src = _write_text(tmp_path)
        root = tmp_path / "cache"
        stoi = {c: i for i, c in enumerate(sorted(set(TEXT)))}
        data = np.asarray([stoi[c] for c in TEXT], dtype=np.int64)
        bin_path, _meta = store_cache(src, data, len(stoi), stoi, {}, "char", cache_root=root)
        bin_path.write_bytes(bin_path.read_bytes()[:-4])  # truncate
        assert try_load(src, "char", cache_root=root) is None

    def test_missing_meta_is_miss(self, tmp_path: Path):
        src = _write_text(tmp_path)
        root = tmp_path / "cache"
        root.mkdir(parents=True, exist_ok=True)
        assert try_load(src, "char", cache_root=root) is None


class TestPrepareDataHook:
    """End-to-end through prepare_data: cold populates, warm hits."""

    def test_cold_then_warm_char(self, tmp_path: Path, monkeypatch):
        from domain.training._internal.train_pipeline import prepare_data

        # validate_training_data requires >=200 chars
        src = tmp_path / "corpus.txt"
        src.write_text((TEXT + "\n") * 2, encoding="utf-8")
        cache_root = tmp_path / "cache"
        monkeypatch.setattr("domain.training._internal.token_cache.DEFAULT_CACHE_ROOT", cache_root)
        d1, v1, s1, i1 = prepare_data(str(src))
        assert (cache_root).exists() and list(cache_root.glob("*.bin"))
        d2, v2, s2, i2 = prepare_data(str(src))
        assert v1 == v2 and s1 == s2 and i1 == i2
        assert np.array_equal(np.asarray(d2), d1)

    def test_int64_batch_from_compact_cache(self, tmp_path: Path, monkeypatch):
        from domain.training._internal.train_pipeline import (
            SloughGPTTrainer,
            TrainerConfig,
            prepare_data,
        )

        # need a longer corpus for batching
        body = (TEXT + "\n") * 50
        src = tmp_path / "corpus.txt"
        src.write_text(body, encoding="utf-8")
        cache_root = tmp_path / "cache"
        monkeypatch.setattr("domain.training._internal.token_cache.DEFAULT_CACHE_ROOT", cache_root)
        prepare_data(str(src))  # cold → populates cache

        cfg = TrainerConfig(block_size=8, batch_size=2)
        trainer = object.__new__(SloughGPTTrainer)  # bypass __init__ (model build)
        trainer.config = cfg
        data, _vocab, _s, _i = prepare_data(str(src))  # warm → memmap uint16
        trainer.train_data = data
        trainer.val_data = data[:16]
        trainer._experience_sampler = None
        trainer._feed_sampler = None
        x, y = trainer.get_batch("train")
        assert x.dtype == np.int64 and y.dtype == np.int64


def test_default_root_is_gitignored(tmp_path: Path):
    """Cache root lives under runs/ which is gitignored (no accidental commits)."""
    assert DEFAULT_CACHE_ROOT.parts[0] == "runs"
