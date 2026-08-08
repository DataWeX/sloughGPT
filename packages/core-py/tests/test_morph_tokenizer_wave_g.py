"""Wave G: MorphTokenizer coverage — from_pretrained layout edge cases,
synthetic BPE modes (byte_level / byte_fallback / char-level), chat template
rendering, and morphology edge paths. No model cache required."""

import json

import pytest

from domains.infrastructure.morph_tokenizer import MorphTokenizer


def _make_tok(vocab=None, merges=(), **kw):
    defaults = dict(eos_token_id=99)
    defaults.update(kw)
    return MorphTokenizer(vocab=vocab or {}, merges=list(merges), **defaults)


def _write_tokenizer_json(snap_dir, vocab, merges=(), eos=0, pre=None, dec=None,
                          post=None, added=(), chat_template=None,
                          vocab_as_list=False):
    if vocab_as_list:
        raw_vocab = [[t, i] for i, t in enumerate(vocab)]
    else:
        raw_vocab = {t: i for i, t in enumerate(vocab)}
    data = {"model": {"vocab": raw_vocab, "merges": list(merges),
                      "eos_token_id": eos}}
    if pre is not None:
        data["pre_tokenizer"] = pre
    if dec is not None:
        data["decoder"] = dec
    if post is not None:
        data["post_processor"] = post
    if added:
        data["added_tokens"] = [{"content": t, "id": i} for t, i in added]
    if chat_template:
        data["chat_template"] = chat_template
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "tokenizer.json").write_text(json.dumps(data))


class TestFromPretrainedLayouts:
    """from_pretrained cache-layout and tokenizer.json parsing edge cases."""

    def test_snapshot_layout(self, tmp_path, monkeypatch):
        snap = tmp_path / "hub" / "models--wavetest" / "snapshots" / "snap1"
        _write_tokenizer_json(snap, ["a", "b"], merges=["a b"], eos=5)
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        tok = MorphTokenizer.from_pretrained("wavetest")
        assert tok.vocab == {"a": 0, "b": 1}
        assert tok.merges == [("a", "b")]
        assert tok.eos_token_id == 5
        assert tok.model_id == "wavetest"

    def test_flat_layout(self, tmp_path, monkeypatch):
        base = tmp_path / "hub" / "models--wavetest"
        (base / "snapshots").mkdir(parents=True)
        _write_tokenizer_json(base, ["a"], eos=0)
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        tok = MorphTokenizer.from_pretrained("wavetest")
        assert tok.vocab == {"a": 0}

    def test_missing_raises(self, tmp_path, monkeypatch):
        (tmp_path / "hub").mkdir(parents=True)
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            MorphTokenizer.from_pretrained("wavetest")

    def test_vocab_list_of_pairs(self, tmp_path, monkeypatch):
        snap = tmp_path / "hub" / "models--wavetest" / "snapshots" / "snap1"
        _write_tokenizer_json(snap, ["a", "b"], vocab_as_list=True)
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        tok = MorphTokenizer.from_pretrained("wavetest")
        assert tok.vocab == {"a": 0, "b": 1}

    def test_pretok_bytelevel_detection(self, tmp_path, monkeypatch):
        snap = tmp_path / "hub" / "models--wavetest" / "snapshots" / "snap1"
        _write_tokenizer_json(snap, ["a"], pre={"type": "ByteLevel"})
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        tok = MorphTokenizer.from_pretrained("wavetest")
        assert tok.byte_level is True

    def test_decoder_bytefallback_detection(self, tmp_path, monkeypatch):
        snap = tmp_path / "hub" / "models--wavetest" / "snapshots" / "snap1"
        _write_tokenizer_json(snap, ["a"], dec={"decoders": [{"type": "ByteFallback"}]})
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        tok = MorphTokenizer.from_pretrained("wavetest")
        assert tok.byte_fallback is True
        assert tok.byte_level is False

    def test_eos_token_list(self, tmp_path, monkeypatch):
        snap = tmp_path / "hub" / "models--wavetest" / "snapshots" / "snap1"
        _write_tokenizer_json(snap, ["a"], eos=[5])
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        tok = MorphTokenizer.from_pretrained("wavetest")
        assert tok.eos_token_id == 5

    def test_post_processor_end_token_id(self, tmp_path, monkeypatch):
        snap = tmp_path / "hub" / "models--wavetest" / "snapshots" / "snap1"
        _write_tokenizer_json(snap, ["a"], eos=1, post={"end_token_id": 200})
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        tok = MorphTokenizer.from_pretrained("wavetest")
        assert tok.eos_token_id == 200

    def test_added_tokens_parsed(self, tmp_path, monkeypatch):
        snap = tmp_path / "hub" / "models--wavetest" / "snapshots" / "snap1"
        _write_tokenizer_json(snap, ["a"], added=[("<|im_start|>", 7), ("<|im_end|>", 8)])
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        tok = MorphTokenizer.from_pretrained("wavetest")
        assert tok.added_tokens == {"<|im_start|>": 7, "<|im_end|>": 8}

    def test_chat_template_from_tokenizer_json(self, tmp_path, monkeypatch):
        snap = tmp_path / "hub" / "models--wavetest" / "snapshots" / "snap1"
        _write_tokenizer_json(snap, ["a"], chat_template="<|im_start|>tpl")
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        tok = MorphTokenizer.from_pretrained("wavetest")
        assert tok._chat_template == "<|im_start|>tpl"
        assert tok._chat_template_jinja is True

    def test_config_at_snapshots_dir(self, tmp_path, monkeypatch):
        snap = tmp_path / "hub" / "models--wavetest" / "snapshots" / "snap1"
        _write_tokenizer_json(snap, ["a"])
        cfg = snap.parent / "tokenizer_config.json"
        cfg.write_text(json.dumps({"chat_template": "<|im_start|>snapshots-tpl"}))
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        tok = MorphTokenizer.from_pretrained("wavetest")
        assert tok._chat_template == "<|im_start|>snapshots-tpl"

    def test_config_nested_snapshot_search(self, tmp_path, monkeypatch):
        snap = tmp_path / "hub" / "models--wavetest" / "snapshots" / "snap1"
        _write_tokenizer_json(snap, ["a"])
        nested = tmp_path / "hub" / "models--wavetest" / "snapshots" / "snapshots" / "snap2"
        nested.mkdir(parents=True)
        (nested / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "<|im_start|>nested-tpl"}))
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        tok = MorphTokenizer.from_pretrained("wavetest")
        assert tok._chat_template == "<|im_start|>nested-tpl"

    def test_config_invalid_json_tolerated(self, tmp_path, monkeypatch):
        snap = tmp_path / "hub" / "models--wavetest" / "snapshots" / "snap1"
        _write_tokenizer_json(snap, ["a"])
        (snap / "tokenizer_config.json").write_text("not { json")
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        tok = MorphTokenizer.from_pretrained("wavetest")
        assert tok._chat_template is None

    def test_byte_chars_in_vocab_detect_byte_level(self, tmp_path, monkeypatch):
        snap = tmp_path / "hub" / "models--wavetest" / "snapshots" / "snap1"
        _write_tokenizer_json(snap, ["\xc3"])
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        tok = MorphTokenizer.from_pretrained("wavetest")
        assert tok.byte_level is True

    def test_project_local_cache_candidate(self, tmp_path, monkeypatch):
        from pathlib import Path
        import shutil

        project_dir = (Path(__file__).resolve().parents[3]
                       / "models" / "hf-cache" / "hub" / "models--wavetestproj")
        project_dir.mkdir(parents=True, exist_ok=True)
        try:
            _write_tokenizer_json(project_dir / "snapshots" / "snap1", ["a"])
            (tmp_path / "hub").mkdir(parents=True)
            monkeypatch.setenv("HF_HOME", str(tmp_path))
            tok = MorphTokenizer.from_pretrained("wavetestproj")
            assert tok.vocab == {"a": 0}
        finally:
            shutil.rmtree(project_dir, ignore_errors=True)


class TestSyntheticBpeModes:
    """Synthetic BPE encode/decode across byte_level, byte_fallback, char."""

    def test_no_added_tokens_fast_path(self):
        tok = _make_tok(vocab={"a": 1})
        assert tok._added_token_patterns == []
        assert tok.encode("a") == [1]

    def test_encode_added_tokens_split(self):
        tok = _make_tok(vocab={"a": 1, "b": 2}, added_tokens={"<|im_start|>": 7})
        assert tok.encode("a<|im_start|>b") == [1, 7, 2]

    def test_decode_added_token(self):
        tok = _make_tok(vocab={"a": 1}, added_tokens={"<|im_start|>": 7})
        assert tok.decode([1, 7]) == "a<|im_start|>"

    def test_encode_byte_level_in_vocab(self):
        tok = _make_tok(vocab={"a": 1, "b": 2, "ab": 3}, merges=[("a", "b")],
                        byte_level=True)
        assert tok.encode("ab") == [3]

    def test_encode_byte_level_fallback(self):
        tok = _make_tok(vocab={"a": 1, "b": 2}, merges=[("a", "b")], byte_level=True)
        assert tok.encode("ab") == [1, 2]

    def test_encode_char_level(self):
        tok = _make_tok(vocab={"h": 1, "i": 2})
        assert tok.encode("hi") == [1, 2]

    def test_encode_char_level_unknown_eos(self):
        tok = _make_tok(vocab={"h": 1})
        assert tok.encode("hx") == [1, 99]

    def test_encode_byte_fallback_normalize_and_hex(self):
        tok = _make_tok(
            vocab={"\u2581": 10, "h": 11, "i": 12, "<0x68>": 20, "<0x69>": 21},
            merges=[("h", "i")], byte_fallback=True)
        ids = tok.encode("hi")
        assert 20 in ids and 21 in ids
        assert 10 in ids

    def test_encode_byte_fallback_unknown_hex_eos(self):
        tok = _make_tok(vocab={"\u2581": 10, "h": 11, "i": 12},
                        merges=[("h", "i")], byte_fallback=True)
        ids = tok.encode("hi")
        assert ids == [10, 99, 99]

    def test_decode_byte_fallback_mixed(self):
        tok = _make_tok(
            vocab={"<0x48>": 1, "<0xGG>": 2, "\u2581word": 3, "word": 4},
            byte_fallback=True)
        assert tok.decode([1, 2, 3, 4, 3]) == "H<0xGG>wordword word"

    def test_decode_char_level(self):
        tok = _make_tok(vocab={"a": 1, "b": 2})
        assert tok.decode([1, 2]) == "ab"

    def test_tokenize_char_level(self):
        tok = _make_tok(vocab={"h": 1, "i": 2})
        assert tok.tokenize("hi") == ["h", "i"]


class TestChatTemplate:
    """apply_chat_template fallback + _render_chat_template branches."""

    def test_empty_messages(self):
        tok = _make_tok()
        assert tok.apply_chat_template([]) == ""
        assert tok.apply_chat_template(None) == ""

    def test_fallback_without_template(self):
        tok = _make_tok()
        out = tok.apply_chat_template([
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ])
        assert out == "System: S\nUser: U\nAssistant: A\nAssistant:"

    def test_fallback_defaults_role_and_content(self):
        tok = _make_tok()
        out = tok.apply_chat_template([{}, {"content": "x"}])
        assert out == "User: \nUser: x\nAssistant:"

    def test_render_im_start_with_system(self):
        tok = _make_tok()
        tok._chat_template = "<|im_start|>{role}{content}<|im_end|>"
        out = tok.apply_chat_template([
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ])
        assert out == ("<|im_start|>system\nS<|im_end|>\n"
                       "<|im_start|>user\nU<|im_end|>\n"
                       "<|im_start|>assistant\n")

    def test_render_im_start_without_system(self):
        tok = _make_tok()
        tok._chat_template = "<|im_start|>{role}{content}<|im_end|>"
        out = tok.apply_chat_template([{"role": "user", "content": "U"}])
        assert out == ("<|im_start|>user\nU<|im_end|>\n"
                       "<|im_start|>assistant\n")

    def test_render_generic_loop(self):
        tok = _make_tok()
        tok._chat_template = "{% for message in messages %}{% endfor %}"
        out = tok.apply_chat_template([{"role": "user", "content": "U"}])
        assert out == "user\nU\nassistant\n"

    def test_render_generic_loop_with_system(self):
        tok = _make_tok()
        tok._chat_template = "{% for message in messages %}{% endfor %}"
        out = tok.apply_chat_template([
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ])
        assert out == "system\nS\n\nuser\nU\nassistant\n"

    def test_render_default_chatml(self):
        tok = _make_tok()
        tok._chat_template = "plain template"
        out = tok.apply_chat_template([{"role": "user", "content": "U"}])
        assert out == "<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"

    def test_render_default_chatml_with_system(self):
        tok = _make_tok()
        tok._chat_template = "plain template"
        out = tok.apply_chat_template([
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ])
        assert out == ("<|im_start|>system\nS<|im_end|>\n"
                       "<|im_start|>user\nU<|im_end|>\n"
                       "<|im_start|>assistant\n")


class TestMorphologyEdges:
    """Morphology paths not exercised by the cached-model suite."""

    def test_decompose_multi_char_stem_form(self):
        tok = _make_tok()
        result = tok.decompose("formation")
        assert "ation" in result
        assert "formate" in result

    def test_stem_cache_hit(self):
        tok = _make_tok()
        assert tok.stem("happily") == "happi"
        assert tok.stem("happily") == "happi"

    def test_stem_fallback_loop(self):
        tok = _make_tok()
        assert tok.stem("abbed") == "abb"

    def test_stem_root_in_forms(self):
        tok = _make_tok()
        assert tok.stem("unran") == "run"

    def test_stem_resolved_root_in_forms(self):
        tok = _make_tok()
        assert tok.stem("reader") == "read"

    def test_generate_forms_e_root(self):
        tok = _make_tok()
        forms = tok.generate_forms("love")
        assert "loving" in forms

    def test_root_distance_related(self):
        tok = _make_tok()
        assert tok.root_distance("cat", "bat") == 2

    def test_are_related(self):
        tok = _make_tok()
        assert tok.are_related("running", "ran") is True
        assert tok.are_related("cat", "dog") is False

    def test_decompose_batch(self):
        tok = _make_tok()
        out = tok.decompose_batch(["cats", "running"])
        assert "cat" in out["cats"]
        assert "run" in out["running"]

    def test_stem_batch(self):
        tok = _make_tok()
        out = tok.stem_batch(["running", "cats"])
        assert out == {"running": "run", "cats": "cat"}

    def test_build_root_index(self):
        tok = _make_tok()
        index = tok.build_root_index(["running", "runs", "cats"])
        assert index == {"run": ["running", "runs"], "cat": ["cats"]}

    def test_chat_stop_ids_no_eos(self):
        tok = _make_tok(eos_token_id=None)
        assert tok.chat_stop_ids() == ()

    def test_chat_stop_ids_added_marker(self):
        tok = _make_tok(added_tokens={"<|im_end|>": 42})
        assert tok.chat_stop_ids() == (42, 99)

    def test_stem_suffix_with_stem_form(self):
        tok = _make_tok()
        assert tok.stem("preation") == "preation"

    def test_generate_forms_irregular_root(self):
        tok = _make_tok()
        forms = tok.generate_forms("run")
        assert "ran" in forms
        assert "running" in forms

    def test_find_related_excludes_self(self):
        tok = _make_tok()
        related = tok.find_related("cats")
        assert "cats" not in related
        assert "cat" in related

    def test_vocabulary_coverage_mixed(self):
        tok = _make_tok(vocab={"hello": 1, "world": 2})
        assert tok.vocabulary_coverage(["hello", "nope"]) == 0.5

    def test_vocabulary_coverage_empty(self):
        tok = _make_tok()
        assert tok.vocabulary_coverage([]) == 0.0

    def test_morphological_diversity_shared_root(self):
        tok = _make_tok()
        assert tok.morphological_diversity(["running", "runs"]) == 0.5

    def test_morphological_diversity_empty(self):
        tok = _make_tok()
        assert tok.morphological_diversity([]) == 0.0
