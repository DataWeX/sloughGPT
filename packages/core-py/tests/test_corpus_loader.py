"""Tests for domain.training._internal.corpus_loader — file & structured-data ingestion."""

from __future__ import annotations

import json

import pytest

from domain.training._internal.corpus_loader import (
    DEFAULT_PATTERNS,
    STRUCTURED_SUFFIXES,
    is_structured,
    load_corpus,
    load_corpus_dir,
    load_corpus_file,
    render_messages,
)


class TestTextFiles:
    def test_txt_passthrough_byte_identical(self, tmp_path):
        p = tmp_path / "a.txt"
        content = "line one\nline two\twith tab\n\nend"
        p.write_text(content, encoding="utf-8")
        assert load_corpus_file(p) == [content]

    def test_unknown_extension_treated_as_text(self, tmp_path):
        p = tmp_path / "run.log"
        p.write_text("log entry 12345 uniquechars", encoding="utf-8")
        docs = load_corpus_file(p)
        assert len(docs) == 1 and docs[0].startswith("log entry")

    def test_empty_file_skipped(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        assert load_corpus_file(p) == []

    def test_whitespace_only_file_skipped(self, tmp_path):
        p = tmp_path / "ws.txt"
        p.write_text("   \n\t\n", encoding="utf-8")
        assert load_corpus_file(p) == []


class TestRenderMessages:
    def test_pair_matches_chat_convention(self):
        doc = render_messages(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        )
        assert doc == "User: Hi\nAssistant: Hello\n\n"

    def test_system_first(self):
        doc = render_messages([{"role": "system", "content": "be brief"}])
        assert doc == "System: be brief\n\n"

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError, match="role"):
            render_messages([{"role": "tool", "content": "x"}])

    def test_empty_messages_raises(self):
        with pytest.raises(ValueError, match="messages"):
            render_messages([])


class TestJsonl:
    def test_messages_record_renders(self, tmp_path):
        p = tmp_path / "conv.jsonl"
        rec = {
            "messages": [
                {"role": "user", "content": "Say hi"},
                {"role": "assistant", "content": "hi"},
            ]
        }
        p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        assert load_corpus_file(p) == ["User: Say hi\nAssistant: hi\n\n"]

    def test_text_key_flattened(self, tmp_path):
        p = tmp_path / "t.jsonl"
        p.write_text(json.dumps({"text": "alpha beta"}) + "\n", encoding="utf-8")
        assert load_corpus_file(p) == ["alpha beta"]

    def test_input_output_keys_flattened_in_order(self, tmp_path):
        p = tmp_path / "io.jsonl"
        p.write_text(json.dumps({"input": "question", "output": "answer"}) + "\n", encoding="utf-8")
        assert load_corpus_file(p) == ["question\nanswer"]

    def test_record_without_text_keys_dumped(self, tmp_path):
        p = tmp_path / "r.jsonl"
        obj = {"a": 1, "b": "two"}
        p.write_text(json.dumps(obj) + "\n", encoding="utf-8")
        assert load_corpus_file(p) == [json.dumps(obj, ensure_ascii=False)]

    def test_bad_line_skipped_valid_kept(self, tmp_path):
        p = tmp_path / "mixed.jsonl"
        p.write_text(
            "{not json\n" + json.dumps({"text": "good"}) + "\n",
            encoding="utf-8",
        )
        assert load_corpus_file(p) == ["good"]

    def test_multiple_records_one_doc_each(self, tmp_path):
        p = tmp_path / "multi.jsonl"
        p.write_text(
            json.dumps({"text": "first"}) + "\n" + json.dumps({"text": "second"}) + "\n",
            encoding="utf-8",
        )
        assert load_corpus_file(p) == ["first", "second"]

    def test_top_level_string_line(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text('"just a string"\n', encoding="utf-8")
        assert load_corpus_file(p) == ["just a string"]


class TestJson:
    def test_list_of_records(self, tmp_path):
        p = tmp_path / "arr.json"
        p.write_text(
            json.dumps([{"text": "one"}, {"text": "two"}]),
            encoding="utf-8",
        )
        assert load_corpus_file(p) == ["one", "two"]

    def test_object_with_messages_renders(self, tmp_path):
        p = tmp_path / "obj.json"
        p.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        assert load_corpus_file(p) == ["User: q\nAssistant: a\n\n"]

    def test_object_without_text_keys_dumped(self, tmp_path):
        p = tmp_path / "plain.json"
        obj = {"k": "v", "n": 2}
        p.write_text(json.dumps(obj), encoding="utf-8")
        assert load_corpus_file(p) == [json.dumps(obj, ensure_ascii=False)]


class TestCsv:
    def test_rows_render_as_header_value_lines(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text("name,age\nalice,30\nbob,40\n", encoding="utf-8")
        docs = load_corpus_file(p)
        assert docs == ["name: alice\nage: 30", "name: bob\nage: 40"]

    def test_empty_rows_skipped(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text("a,b\n\n1,2\n", encoding="utf-8")
        assert load_corpus_file(p) == ["a: 1\nb: 2"]


class TestDirectory:
    def test_default_patterns_collect_mixed_formats(self, tmp_path):
        (tmp_path / "a.txt").write_text("plain text doc uniquechars", encoding="utf-8")
        (tmp_path / "b.py").write_text("def f(): return 42", encoding="utf-8")
        (tmp_path / "c.md").write_text("# heading markdown doc", encoding="utf-8")
        (tmp_path / "d.jsonl").write_text(
            json.dumps({"text": "jsonl doc"}) + "\n", encoding="utf-8"
        )
        (tmp_path / "e.csv").write_text("col\nval\n", encoding="utf-8")
        (tmp_path / "f.bin").write_bytes(b"\x00\x01\x02 binary")
        docs = load_corpus_dir(str(tmp_path))
        joined = "\n".join(docs)
        assert "plain text doc" in joined
        assert "return 42" in joined
        assert "heading" in joined
        assert "jsonl doc" in joined
        assert "col: val" in joined
        assert len(docs) == 5

    def test_explicit_patterns_filter(self, tmp_path):
        (tmp_path / "a.txt").write_text("keep me uniquechars", encoding="utf-8")
        (tmp_path / "b.jsonl").write_text(json.dumps({"text": "drop me"}) + "\n", encoding="utf-8")
        docs = load_corpus_dir(str(tmp_path), patterns=("*.txt",))
        assert docs == ["keep me uniquechars"]

    def test_non_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested doc uniquechars", encoding="utf-8")
        (tmp_path / "top.txt").write_text("top doc uniquechars", encoding="utf-8")
        docs = load_corpus_dir(str(tmp_path), recursive=False)
        assert docs == ["top doc uniquechars"]

    def test_deterministic_order(self, tmp_path):
        (tmp_path / "z.txt").write_text("zzz uniquechars", encoding="utf-8")
        (tmp_path / "a.txt").write_text("aaa uniquechars", encoding="utf-8")
        docs = load_corpus_dir(str(tmp_path))
        assert docs[0].startswith("aaa")

    def test_missing_dir_raises(self):
        with pytest.raises(ValueError, match="Not a directory"):
            load_corpus_dir("/nonexistent/dir/xyz")

    def test_empty_dir_returns_no_docs(self, tmp_path):
        assert load_corpus_dir(str(tmp_path)) == []


class TestDispatch:
    def test_is_structured(self):
        assert is_structured("a.jsonl") and is_structured("b.json") and is_structured("c.csv")
        assert not is_structured("d.txt")
        assert ".jsonl" in STRUCTURED_SUFFIXES

    def test_load_corpus_file_spec(self, tmp_path):
        p = tmp_path / "one.txt"
        p.write_text("single file doc", encoding="utf-8")
        assert load_corpus(str(p)) == ["single file doc"]

    def test_load_corpus_dir_spec(self, tmp_path):
        (tmp_path / "x.txt").write_text("in dir doc", encoding="utf-8")
        assert load_corpus(str(tmp_path)) == ["in dir doc"]

    def test_load_corpus_missing_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, ValueError)):
            load_corpus(str(tmp_path / "nope.txt"))

    def test_default_patterns_shape(self):
        assert "*.txt" in DEFAULT_PATTERNS and "*.jsonl" in DEFAULT_PATTERNS


class TestTokenizerTrainFromDirectoryDefault:
    def test_manager_default_patterns_train_from_jsonl_only(self, tmp_path):
        from domain.training._internal.tokenizer_manager import TokenizerManager

        rec = {
            "messages": [
                {"role": "user", "content": "hello corpus world"},
                {"role": "assistant", "content": "greetings traveler"},
            ]
        }
        (tmp_path / "conv.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
        mgr = TokenizerManager()
        stats = mgr.train_from_directory(str(tmp_path), vocab_size=128, min_frequency=1)
        assert mgr.is_trained() is True
        assert stats["vocab_size"] > 0

    def test_slobpe_default_patterns_train_from_jsonl_only(self, tmp_path):
        from domain.training._internal.tokenizer import SloBPE

        (tmp_path / "d.jsonl").write_text(
            json.dumps({"text": "alpha beta gamma"}) + "\n", encoding="utf-8"
        )
        tok = SloBPE.train_from_directory(str(tmp_path), vocab_size=64)
        assert tok is not None

    def test_slounigram_default_patterns_train_from_jsonl_only(self, tmp_path):
        from domain.training._internal.tokenizer import SloUnigram

        (tmp_path / "d.jsonl").write_text(
            json.dumps({"text": "alpha beta gamma"}) + "\n", encoding="utf-8"
        )
        tok = SloUnigram.train_from_directory(str(tmp_path), vocab_size=64)
        assert tok is not None
