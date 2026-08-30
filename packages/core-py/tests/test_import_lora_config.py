"""Tests for domains.training.data_import — ImportResult and related logic;
domains.training.hf_lora_finetune — HFLoraConfig, _LoRADataset."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from domains.training.data_import import (
    ImportResult,
    RepoImporter,
    DEFAULT_IGNORES,
    DataImporter,
    GitHubSearch,
    BooksSearch,
)
from domains.training.hf_lora_finetune import HFLoraConfig, _LoRADataset


# ---------------------------------------------------------------------------
# ImportResult — dataclass basics
# ---------------------------------------------------------------------------
class TestImportResultFields:
    def test_success_fields(self):
        ir = ImportResult(
            success=True, name="test", source="url",
            files_imported=5, total_chars=1000, output_path="/tmp/out",
        )
        assert ir.success is True
        assert ir.files_imported == 5
        assert ir.error is None

    def test_error_fields(self):
        ir = ImportResult(
            success=False, name="test", source="url",
            files_imported=0, total_chars=0, output_path="", error="timeout",
        )
        assert ir.success is False
        assert ir.error == "timeout"

    def test_name_field(self):
        ir = ImportResult(
            success=True, name="my_data", source="s",
            files_imported=0, total_chars=0, output_path="",
        )
        assert ir.name == "my_data"

    def test_source_field(self):
        ir = ImportResult(
            success=True, name="n", source="https://example.com",
            files_imported=0, total_chars=0, output_path="",
        )
        assert ir.source == "https://example.com"

    def test_output_path_field(self):
        ir = ImportResult(
            success=True, name="n", source="s",
            files_imported=0, total_chars=0, output_path="/data/out.jsonl",
        )
        assert ir.output_path == "/data/out.jsonl"

    def test_total_chars_field(self):
        ir = ImportResult(
            success=True, name="n", source="s",
            files_imported=0, total_chars=9999, output_path="",
        )
        assert ir.total_chars == 9999

    def test_error_default_none(self):
        ir = ImportResult(
            success=True, name="n", source="s",
            files_imported=0, total_chars=0, output_path="",
        )
        assert ir.error is None

    def test_zero_files(self):
        ir = ImportResult(
            success=True, name="n", source="s",
            files_imported=0, total_chars=0, output_path="",
        )
        assert ir.files_imported == 0

    def test_large_files_imported(self):
        ir = ImportResult(
            success=True, name="n", source="s",
            files_imported=10000, total_chars=0, output_path="",
        )
        assert ir.files_imported == 10000

    def test_negative_chars(self):
        ir = ImportResult(
            success=True, name="n", source="s",
            files_imported=0, total_chars=-1, output_path="",
        )
        assert ir.total_chars == -1

    def test_empty_name(self):
        ir = ImportResult(
            success=True, name="", source="s",
            files_imported=0, total_chars=0, output_path="",
        )
        assert ir.name == ""

    def test_empty_source(self):
        ir = ImportResult(
            success=True, name="n", source="",
            files_imported=0, total_chars=0, output_path="",
        )
        assert ir.source == ""

    def test_error_long_message(self):
        long_err = "x" * 10000
        ir = ImportResult(
            success=False, name="n", source="s",
            files_imported=0, total_chars=0, output_path="", error=long_err,
        )
        assert len(ir.error) == 10000

    def test_success_with_error_set(self):
        ir = ImportResult(
            success=True, name="n", source="s",
            files_imported=5, total_chars=100, output_path="/out",
            error="ignored",
        )
        assert ir.success is True
        assert ir.error == "ignored"


# ---------------------------------------------------------------------------
# ImportResult — equality and repr
# ---------------------------------------------------------------------------
class TestImportResultBehavior:
    def test_equality(self):
        a = ImportResult(True, "n", "s", 1, 2, "/o")
        b = ImportResult(True, "n", "s", 1, 2, "/o")
        assert a == b

    def test_inequality(self):
        a = ImportResult(True, "n", "s", 1, 2, "/o")
        b = ImportResult(False, "n", "s", 1, 2, "/o")
        assert a != b

    def test_repr_contains_fields(self):
        ir = ImportResult(True, "test", "src", 3, 100, "/path")
        r = repr(ir)
        assert "test" in r
        assert "src" in r

    def test_hash(self):
        ir = ImportResult(True, "n", "s", 0, 0, "")
        d = ir.__dict__.copy()
        assert isinstance(d["success"], bool)


# ---------------------------------------------------------------------------
# ImportResult — as dict-like access via dataclass
# ---------------------------------------------------------------------------
class TestImportResultAccess:
    def test_attribute_access(self):
        ir = ImportResult(True, "abc", "url", 10, 500, "/out", None)
        assert ir.success is True
        assert ir.name == "abc"
        assert ir.source == "url"
        assert ir.files_imported == 10
        assert ir.total_chars == 500
        assert ir.output_path == "/out"
        assert ir.error is None

    def test_unpack(self):
        ir = ImportResult(True, "n", "s", 0, 0, "")
        assert ir.success is True
        assert ir.name == "n"
        assert ir.source == "s"


# ---------------------------------------------------------------------------
# DEFAULT_IGNORES — set constants
# ---------------------------------------------------------------------------
class TestDefaultIgnores:
    def test_is_set(self):
        assert isinstance(DEFAULT_IGNORES, set)

    def test_contains_common_dirs(self):
        assert ".git" in DEFAULT_IGNORES
        assert "node_modules" in DEFAULT_IGNORES
        assert "__pycache__" in DEFAULT_IGNORES
        assert ".venv" in DEFAULT_IGNORES
        assert "venv" in DEFAULT_IGNORES

    def test_contains_build_dirs(self):
        assert "dist" in DEFAULT_IGNORES
        assert "build" in DEFAULT_IGNORES

    def test_contains_cache_dirs(self):
        assert ".pytest_cache" in DEFAULT_IGNORES
        assert ".mypy_cache" in DEFAULT_IGNORES

    def test_size(self):
        assert len(DEFAULT_IGNORES) >= 10


# ---------------------------------------------------------------------------
# RepoImporter — directory creation
# ---------------------------------------------------------------------------
class TestRepoImporterInit:
    def test_creates_cache_dir(self, tmp_path):
        cache = tmp_path / "repos"
        ri = RepoImporter(cache_dir=str(cache))
        assert cache.exists()

    def test_default_cache_dir(self):
        ri = RepoImporter()
        assert ri.cache_dir.exists()


# ---------------------------------------------------------------------------
# RepoImporter — branch name validation
# ---------------------------------------------------------------------------
class TestRepoImporterBranchValidation:
    def test_valid_branch_names(self, tmp_path):
        import re
        ri = RepoImporter(cache_dir=str(tmp_path / "c"))
        assert re.match(r'^[a-zA-Z0-9_\-/.]+$', "main")
        assert re.match(r'^[a-zA-Z0-9_\-/.]+$', "feature/x")
        assert re.match(r'^[a-zA-Z0-9_\-/.]+$', "release/1.0")

    def test_invalid_branch_chars(self):
        import re
        assert not re.match(r'^[a-zA-Z0-9_\-/.]+$', "branch with space")
        assert not re.match(r'^[a-zA-Z0-9_\-/.]+$', "branch;rm -rf /")


# ---------------------------------------------------------------------------
# RepoImporter — _detect_language
# ---------------------------------------------------------------------------
class TestRepoImporterDetectLanguage:
    def _detect(self, path_str, content=""):
        ri = RepoImporter(cache_dir="/tmp/_test_ri")
        return ri._detect_language(Path(path_str), content)

    def test_python(self):
        assert self._detect("main.py") == "python"

    def test_javascript(self):
        assert self._detect("app.js") == "javascript"

    def test_typescript(self):
        assert self._detect("index.ts") == "typescript"

    def test_markdown(self):
        assert self._detect("README.md") == "markdown"

    def test_json(self):
        assert self._detect("config.json") == "json"

    def test_yaml(self):
        assert self._detect("config.yaml") == "yaml"

    def test_yml(self):
        assert self._detect("config.yml") == "yaml"

    def test_html(self):
        assert self._detect("index.html") == "html"

    def test_css(self):
        assert self._detect("style.css") == "css"

    def test_rust(self):
        assert self._detect("main.rs") == "rust"

    def test_go(self):
        assert self._detect("main.go") == "go"

    def test_java(self):
        assert self._detect("Main.java") == "java"

    def test_c(self):
        assert self._detect("main.c") == "c"

    def test_cpp(self):
        assert self._detect("main.cpp") == "cpp"

    def test_ruby(self):
        assert self._detect("app.rb") == "ruby"

    def test_php(self):
        assert self._detect("index.php") == "php"

    def test_sql(self):
        assert self._detect("query.sql") == "sql"

    def test_lua(self):
        assert self._detect("script.lua") == "lua"

    def test_swift(self):
        assert self._detect("App.swift") == "swift"

    def test_kotlin(self):
        assert self._detect("Main.kt") == "kotlin"

    def test_dart(self):
        assert self._detect("main.dart") == "dart"

    def test_julia(self):
        assert self._detect("main.jl") == "julia"

    def test_xml(self):
        assert self._detect("data.xml") == "xml"

    def test_csv(self):
        assert self._detect("data.csv") == "csv"

    def test_toml(self):
        assert self._detect("config.toml") == "toml"

    def test_ini(self):
        assert self._detect("config.ini") == "ini"

    def test_dockerfile(self):
        assert self._detect("Dockerfile") == "dockerfile"

    def test_makefile(self):
        assert self._detect("Makefile") == "makefile"

    def test_scss(self):
        assert self._detect("style.scss") == "scss"

    def test_vue(self):
        assert self._detect("App.vue") == "vue"

    def test_svelte(self):
        assert self._detect("App.svelte") == "svelte"

    def test_elixir(self):
        assert self._detect("app.ex") == "elixir"

    def test_haskell(self):
        assert self._detect("Main.hs") == "haskell"

    def test_clojure(self):
        assert self._detect("core.clj") == "clojure"

    def test_ocaml(self):
        assert self._detect("main.ml") == "ocaml"

    def test_fsharp(self):
        assert self._detect("Program.fs") == "fsharp"

    def test_perl(self):
        assert self._detect("script.pl") == "perl"

    def test_r(self):
        assert self._detect("analysis.R") == "r"

    def test_scala(self):
        assert self._detect("Main.scala") == "scala"

    def test_unknown_extension(self):
        assert self._detect("file.xyz") == "text"

    def test_pyw(self):
        assert self._detect("script.pyw") == "python"

    def test_mjs(self):
        assert self._detect("module.mjs") == "javascript"

    def test_cjs(self):
        assert self._detect("module.cjs") == "javascript"

    def test_tsx(self):
        assert self._detect("App.tsx") == "typescript"

    def test_jsx(self):
        assert self._detect("App.jsx") == "javascript"

    def test_h(self):
        assert self._detect("header.h") == "c"

    def test_hpp(self):
        assert self._detect("header.hpp") == "cpp"

    def test_cc(self):
        assert self._detect("main.cc") == "cpp"

    def test_kts(self):
        assert self._detect("build.kts") == "kotlin"

    def test_nim(self):
        assert self._detect("main.nim") == "nim"

    def test_crystal(self):
        assert self._detect("main.cr") == "crystal"

    def test_d(self):
        assert self._detect("main.d") == "d"

    def test_asm(self):
        assert self._detect("boot.asm") == "assembly"

    def test_readme_special(self):
        assert self._detect("README.rst") == "restructuredtext"

    def test_bash(self):
        assert self._detect("script.bash") == "shell"

    def test_zsh(self):
        assert self._detect("script.zsh") == "shell"

    def test_text(self):
        assert self._detect("notes.txt") == "text"

    def test_latex(self):
        assert self._detect("paper.tex") == "latex"

    def test_org(self):
        assert self._detect("notes.org") == "org"

    def test_content_python(self):
        result = self._detect("unknown", "def main():\n    pass\nimport os\nprint('hi')\nclass Foo:\n    pass\nif __name__")
        assert result == "python"

    def test_content_javascript(self):
        result = self._detect("unknown", "function main() {\n  const x = 1;\n  let y = 2;\n  var z = 3;\n}\nexport default")
        assert result == "javascript"

    def test_content_typescript(self):
        result = self._detect("unknown", "interface Foo {\n  x: string;\n  y: number;\n  z: boolean;\n}")
        assert result == "typescript"

    def test_content_html(self):
        result = self._detect("unknown", "<!DOCTYPE html>\n<html>\n<head></head>\n<body></body>\n</html>")
        assert result == "html"

    def test_content_css(self):
        result = self._detect("unknown", "body {\n  margin: 0;\n  padding: 0;\n  color: red;\n}")
        assert result == "css"

    def test_content_json(self):
        result = self._detect("unknown", '{"key": "value", "num": 42}')
        assert result == "json"

    def test_content_yaml(self):
        result = self._detect("unknown", "---\nkey: value\n  nested: true\n  other: 123")
        assert result == "yaml"

    def test_content_shell(self):
        result = self._detect("unknown", "#!/bin/bash\necho hello\nexport FOO=bar\ncd /tmp")
        assert result == "shell"

    def test_content_sql(self):
        result = self._detect("unknown", "SELECT * FROM users WHERE id = 1")
        assert result == "sql"

    def test_content_go(self):
        result = self._detect("unknown", "package main\nimport (\n    \"fmt\"\n)\nfunc main()")
        assert result == "go"

    def test_content_rust(self):
        result = self._detect("unknown", "fn main() {\n    let mut x = 5;\n    impl Foo {\n        pub fn new()")
        assert result == "rust"

    def test_content_java(self):
        result = self._detect("unknown", "public class Main {\n    private int x;\n    public static void main(String[] args)")
        assert result == "java"

    def test_content_cpp(self):
        result = self._detect("unknown", "#include <iostream>\nstd::cout << \"hi\"\ncin >> x")
        assert result == "cpp"

    def test_content_c(self):
        result = self._detect("unknown", "#include <stdio.h>\nint main()\nprintf(\"hi\")")
        assert result == "c"

    def test_content_markdown(self):
        result = self._detect("unknown", "# Title\n## Sub\n- item\n* item\n```code\n[link](url)\n**bold**")
        assert result == "markdown"

    def test_content_fallback_text(self):
        result = self._detect("unknown", "just some random text without patterns")
        assert result == "text"

    def test_content_empty(self):
        result = self._detect("unknown", "")
        assert result == "text"

    def test_mime_python(self):
        assert self._detect("script.py") == "python"

    def test_mime_html(self):
        assert self._detect("page.htm") == "html"


# ---------------------------------------------------------------------------
# RepoImporter — _iter_files
# ---------------------------------------------------------------------------
class TestRepoImporterIterFiles:
    def test_iter_python_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x=1")
        (tmp_path / "b.js").write_text("y=2")
        ri = RepoImporter(cache_dir=str(tmp_path / "c"))
        files = list(ri._iter_files(tmp_path, set(), [".py"]))
        assert len(files) == 1
        assert files[0].name == "a.py"

    def test_iter_ignores(self, tmp_path):
        (tmp_path / "a.py").write_text("x=1")
        sub = tmp_path / "__pycache__"
        sub.mkdir()
        (sub / "b.py").write_text("y=2")
        ri = RepoImporter(cache_dir=str(tmp_path / "c"))
        files = list(ri._iter_files(tmp_path, {"__pycache__"}, [".py"]))
        assert len(files) == 1

    def test_iter_skips_hidden(self, tmp_path):
        (tmp_path / "a.py").write_text("x=1")
        (tmp_path / ".hidden.py").write_text("secret")
        ri = RepoImporter(cache_dir=str(tmp_path / "c"))
        files = list(ri._iter_files(tmp_path, set(), [".py"]))
        assert len(files) == 1
        assert files[0].name == "a.py"

    def test_iter_all_extensions(self, tmp_path):
        (tmp_path / "a.py").write_text("x=1")
        (tmp_path / "b.txt").write_text("hello")
        ri = RepoImporter(cache_dir=str(tmp_path / "c"))
        files = list(ri._iter_files(tmp_path, set(), None))
        assert len(files) == 2


# ---------------------------------------------------------------------------
# RepoImporter — export_to_corpus
# ---------------------------------------------------------------------------
class TestRepoImporterExport:
    def test_export_single_file(self, tmp_path):
        (tmp_path / "code.py").write_text("print('hi')")
        ri = RepoImporter(cache_dir=str(tmp_path / "c"))
        out = tmp_path / "out.jsonl"
        count = ri.export_to_corpus(tmp_path, str(out), extensions=[".py"])
        assert count == 1
        assert out.exists()

    def test_export_jsonl_format(self, tmp_path):
        (tmp_path / "a.py").write_text("x=1")
        ri = RepoImporter(cache_dir=str(tmp_path / "c"))
        out = tmp_path / "out.jsonl"
        ri.export_to_corpus(tmp_path, str(out), extensions=[".py"])
        with open(out) as f:
            record = json.loads(f.readline())
        assert "path" in record
        assert "content" in record
        assert "size" in record
        assert "language" in record

    def test_export_max_files(self, tmp_path):
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text(f"x={i}")
        ri = RepoImporter(cache_dir=str(tmp_path / "c"))
        out = tmp_path / "out.jsonl"
        count = ri.export_to_corpus(tmp_path, str(out), extensions=[".py"], max_files=2)
        assert count == 2

    def test_export_max_bytes_truncates(self, tmp_path):
        (tmp_path / "big.py").write_text("x" * 10000)
        ri = RepoImporter(cache_dir=str(tmp_path / "c"))
        out = tmp_path / "out.jsonl"
        ri.export_to_corpus(tmp_path, str(out), extensions=[".py"], max_bytes=50)
        with open(out) as f:
            record = json.loads(f.readline())
        assert record["size"] <= 50

    def test_export_creates_parent_dirs(self, tmp_path):
        (tmp_path / "a.py").write_text("x=1")
        ri = RepoImporter(cache_dir=str(tmp_path / "c"))
        out = tmp_path / "deep" / "nested" / "out.jsonl"
        count = ri.export_to_corpus(tmp_path, str(out), extensions=[".py"])
        assert count == 1
        assert out.exists()

    def test_export_skips_binary(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        ri = RepoImporter(cache_dir=str(tmp_path / "c"))
        out = tmp_path / "out.jsonl"
        count = ri.export_to_corpus(tmp_path, str(out), extensions=[".png"])
        assert count == 0

    def test_export_language_detection(self, tmp_path):
        (tmp_path / "main.py").write_text("def main(): pass")
        ri = RepoImporter(cache_dir=str(tmp_path / "c"))
        out = tmp_path / "out.jsonl"
        ri.export_to_corpus(tmp_path, str(out), extensions=[".py"])
        with open(out) as f:
            record = json.loads(f.readline())
        assert record["language"] == "python"


# ---------------------------------------------------------------------------
# DataImporter — local import
# ---------------------------------------------------------------------------
class TestDataImporterLocal:
    def test_import_single_file(self, tmp_path):
        src = tmp_path / "input.py"
        src.write_text("print('hello')")
        di = DataImporter(output_dir=str(tmp_path / "out"))
        result = di.import_from_local(str(src), "test_ds")
        assert result.success is True
        assert result.files_imported == 1
        assert result.total_chars > 0

    def test_import_directory(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("x=1")
        (src / "b.py").write_text("y=2")
        di = DataImporter(output_dir=str(tmp_path / "out"))
        result = di.import_from_local(str(src), "test_ds")
        assert result.success is True
        assert result.files_imported == 2

    def test_import_nonexistent_path(self, tmp_path):
        di = DataImporter(output_dir=str(tmp_path / "out"))
        result = di.import_from_local("/no/such/path", "test_ds")
        assert result.files_imported == 0
        assert result.total_chars == 0

    def test_import_empty_directory(self, tmp_path):
        src = tmp_path / "empty"
        src.mkdir()
        di = DataImporter(output_dir=str(tmp_path / "out"))
        result = di.import_from_local(str(src), "test_ds")
        assert result.success is True
        assert result.files_imported == 0


# ---------------------------------------------------------------------------
# BooksSearch — _is_isbn
# ---------------------------------------------------------------------------
class TestBooksSearchISBN:
    def test_isbn10_valid(self):
        bs = BooksSearch()
        assert bs._is_isbn("0321125215") == "isbn:0321125215"

    def test_isbn13_valid(self):
        bs = BooksSearch()
        assert bs._is_isbn("9780321125217") == "isbn:9780321125217"

    def test_isbn_with_dashes(self):
        bs = BooksSearch()
        assert bs._is_isbn("0-321-12521-5") == "isbn:0321125215"

    def test_isbn_with_spaces(self):
        bs = BooksSearch()
        assert bs._is_isbn("0 321 12521 5") == "isbn:0321125215"

    def test_not_isbn_too_short(self):
        bs = BooksSearch()
        assert bs._is_isbn("12345") is None

    def test_not_isbn_too_long(self):
        bs = BooksSearch()
        assert bs._is_isbn("12345678901234") is None

    def test_not_isbn_letters(self):
        bs = BooksSearch()
        assert bs._is_isbn("abc1234567") is None

    def test_not_isbn_empty(self):
        bs = BooksSearch()
        assert bs._is_isbn("") is None

    def test_isbn_with_underscores(self):
        bs = BooksSearch()
        assert bs._is_isbn("0_321_12521_5") == "isbn:0321125215"


# ---------------------------------------------------------------------------
# BooksSearch — _sanitize_query
# ---------------------------------------------------------------------------
class TestBooksSearchSanitize:
    def test_spaces_encoded(self):
        bs = BooksSearch()
        result = bs._sanitize_query("hello world")
        assert " " not in result

    def test_special_chars_encoded(self):
        bs = BooksSearch()
        result = bs._sanitize_query("foo&bar=baz")
        assert "&" not in result
        assert "=" not in result

    def test_stripped(self):
        bs = BooksSearch()
        result = bs._sanitize_query("  hello  ")
        assert result == "hello"

    def test_empty(self):
        bs = BooksSearch()
        result = bs._sanitize_query("")
        assert result == ""


# ---------------------------------------------------------------------------
# GitHubSearch — _sanitize_query
# ---------------------------------------------------------------------------
class TestGitHubSearchSanitize:
    def test_query_lowercased(self):
        gs = GitHubSearch()
        result = gs._sanitize_query("Python Code")
        assert result == "python%20code"

    def test_special_chars(self):
        gs = GitHubSearch()
        result = gs._sanitize_query("c++ code")
        assert "%2B" in result or "+" in result

    def test_stripped(self):
        gs = GitHubSearch()
        result = gs._sanitize_query("  hello  ")
        assert result == "hello"

    def test_empty(self):
        gs = GitHubSearch()
        result = gs._sanitize_query("")
        assert result == ""


# ---------------------------------------------------------------------------
# HFLoraConfig — defaults
# ---------------------------------------------------------------------------
class TestHFLoraConfigDefaults:
    def test_default_rank(self):
        hc = HFLoraConfig()
        assert hc.rank == 8

    def test_default_alpha(self):
        hc = HFLoraConfig()
        assert hc.alpha == 16.0

    def test_default_epochs(self):
        hc = HFLoraConfig()
        assert hc.epochs == 3

    def test_default_batch_size(self):
        hc = HFLoraConfig()
        assert hc.batch_size == 8

    def test_default_model_path(self):
        hc = HFLoraConfig()
        assert hc.model_path == ""

    def test_default_data_path(self):
        hc = HFLoraConfig()
        assert hc.data_path == ""

    def test_default_dropout(self):
        hc = HFLoraConfig()
        assert hc.dropout == 0.0

    def test_default_target_modules(self):
        hc = HFLoraConfig()
        assert hc.target_modules == ["W_q", "W_k", "W_v", "W_o"]

    def test_default_block_size(self):
        hc = HFLoraConfig()
        assert hc.block_size == 128

    def test_default_learning_rate(self):
        hc = HFLoraConfig()
        assert hc.learning_rate == 1e-4

    def test_default_weight_decay(self):
        hc = HFLoraConfig()
        assert hc.weight_decay == 0.01

    def test_default_warmup_steps(self):
        hc = HFLoraConfig()
        assert hc.warmup_steps == 0

    def test_default_grad_clip(self):
        hc = HFLoraConfig()
        assert hc.grad_clip == 1.0

    def test_default_grad_accumulation_steps(self):
        hc = HFLoraConfig()
        assert hc.grad_accumulation_steps == 1

    def test_default_output_dir(self):
        hc = HFLoraConfig()
        assert hc.output_dir == "models"

    def test_default_log_interval(self):
        hc = HFLoraConfig()
        assert hc.log_interval == 10

    def test_default_progress_callback(self):
        hc = HFLoraConfig()
        assert hc.progress_callback is None

    def test_default_cancel_event(self):
        hc = HFLoraConfig()
        assert hc._cancel_event is None


# ---------------------------------------------------------------------------
# HFLoraConfig — custom values
# ---------------------------------------------------------------------------
class TestHFLoraConfigCustom:
    def test_custom_rank(self):
        hc = HFLoraConfig(rank=16)
        assert hc.rank == 16

    def test_custom_alpha(self):
        hc = HFLoraConfig(alpha=32.0)
        assert hc.alpha == 32.0

    def test_custom_epochs(self):
        hc = HFLoraConfig(epochs=10)
        assert hc.epochs == 10

    def test_custom_batch_size(self):
        hc = HFLoraConfig(batch_size=32)
        assert hc.batch_size == 32

    def test_custom_model_path(self):
        hc = HFLoraConfig(model_path="my/model.slnc")
        assert hc.model_path == "my/model.slnc"

    def test_custom_data_path(self):
        hc = HFLoraConfig(data_path="data/train.txt")
        assert hc.data_path == "data/train.txt"

    def test_custom_dropout(self):
        hc = HFLoraConfig(dropout=0.1)
        assert hc.dropout == 0.1

    def test_custom_target_modules(self):
        hc = HFLoraConfig(target_modules=["W_q", "W_v"])
        assert hc.target_modules == ["W_q", "W_v"]

    def test_custom_learning_rate(self):
        hc = HFLoraConfig(learning_rate=5e-4)
        assert hc.learning_rate == 5e-4

    def test_custom_weight_decay(self):
        hc = HFLoraConfig(weight_decay=0.05)
        assert hc.weight_decay == 0.05

    def test_custom_output_dir(self):
        hc = HFLoraConfig(output_dir="/tmp/adapters")
        assert hc.output_dir == "/tmp/adapters"

    def test_custom_log_interval(self):
        hc = HFLoraConfig(log_interval=50)
        assert hc.log_interval == 50


# ---------------------------------------------------------------------------
# HFLoraConfig — adapter_name auto-generation
# ---------------------------------------------------------------------------
class TestHFLoraConfigAdapterName:
    def test_auto_adapter_name(self):
        hc = HFLoraConfig(model_path="gpt2.safetensors")
        assert "gpt2" in hc.adapter_name

    def test_auto_adapter_name_includes_rank(self):
        hc = HFLoraConfig(model_path="gpt2.safetensors", rank=16)
        assert "r16" in hc.adapter_name

    def test_auto_adapter_name_slnc(self):
        hc = HFLoraConfig(model_path="models/my_model.slnc")
        assert "my_model" in hc.adapter_name

    def test_explicit_adapter_name(self):
        hc = HFLoraConfig(adapter_name="custom_name")
        assert hc.adapter_name == "custom_name"

    def test_auto_adapter_name_no_path(self):
        hc = HFLoraConfig(model_path="")
        assert "lora_r" in hc.adapter_name

    def test_auto_adapter_name_complex_path(self):
        hc = HFLoraConfig(model_path="/very/deep/path/to/model_v2.slnc")
        assert "model_v2" in hc.adapter_name


# ---------------------------------------------------------------------------
# _LoRADataset — dataset behavior
# ---------------------------------------------------------------------------
class TestLoRADataset:
    def test_length(self):
        data = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        ds = _LoRADataset(data, block_size=3)
        assert len(ds) == 2  # 5 - 3

    def test_length_short_data(self):
        data = np.array([1, 2], dtype=np.int64)
        ds = _LoRADataset(data, block_size=5)
        assert len(ds) == 0

    def test_getitem(self):
        data = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        ds = _LoRADataset(data, block_size=3)
        x, y = ds[0]
        np.testing.assert_array_equal(x, [10, 20, 30])
        np.testing.assert_array_equal(y, [20, 30, 40])

    def test_getitem_second(self):
        data = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        ds = _LoRADataset(data, block_size=3)
        x, y = ds[1]
        np.testing.assert_array_equal(x, [20, 30, 40])
        np.testing.assert_array_equal(y, [30, 40, 50])

    def test_converts_list_to_array(self):
        ds = _LoRADataset([1, 2, 3, 4, 5], block_size=3)
        assert isinstance(ds.data, np.ndarray)

    def test_empty_data(self):
        ds = _LoRADataset(np.array([], dtype=np.int64), block_size=3)
        assert len(ds) == 0

    def test_block_size_one(self):
        data = np.array([1, 2, 3], dtype=np.int64)
        ds = _LoRADataset(data, block_size=1)
        assert len(ds) == 2
        x, y = ds[0]
        assert x[0] == 1
        assert y[0] == 2

    def test_block_size_equals_data(self):
        data = np.array([1, 2, 3], dtype=np.int64)
        ds = _LoRADataset(data, block_size=3)
        assert len(ds) == 0

    def test_large_dataset(self):
        data = np.arange(1000, dtype=np.int64)
        ds = _LoRADataset(data, block_size=128)
        assert len(ds) == 872

    def test_shapes(self):
        data = np.arange(50, dtype=np.int64)
        ds = _LoRADataset(data, block_size=16)
        x, y = ds[0]
        assert x.shape == (16,)
        assert y.shape == (16,)


# ---------------------------------------------------------------------------
# ImportResult — serialisation round-trip
# ---------------------------------------------------------------------------
class TestImportResultSerialisation:
    def test_to_dict_via_dataclass(self):
        ir = ImportResult(True, "n", "s", 5, 100, "/out", None)
        d = {
            "success": ir.success,
            "name": ir.name,
            "source": ir.source,
            "files_imported": ir.files_imported,
            "total_chars": ir.total_chars,
            "output_path": ir.output_path,
            "error": ir.error,
        }
        assert d["success"] is True
        assert d["name"] == "n"
        assert d["error"] is None

    def test_from_dict_reconstruction(self):
        d = {"success": True, "name": "x", "source": "y", "files_imported": 1,
             "total_chars": 50, "output_path": "/o", "error": None}
        ir = ImportResult(**d)
        assert ir.name == "x"
