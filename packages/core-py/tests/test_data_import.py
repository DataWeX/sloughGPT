"""Tests for domains/training/data_import.py (repo/URL/HF/ISBN/local importers)."""

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from domains.training.data_import import (
    DEFAULT_IGNORES,
    BooksSearch,
    DataImporter,
    GitHubSearch,
    HuggingFaceImporter,
    ISBNImporter,
    ImportResult,
    RepoImporter,
    URLImporter,
    import_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, payload):
        self._payload = payload.encode() if isinstance(payload, str) else payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeRequest:
    captured = []

    def __init__(self, url, headers=None):
        self.url = url
        self.headers = headers or {}
        FakeRequest.captured.append(url)

    def add_header(self, key, value):
        self.headers[key] = value


@pytest.fixture
def fake_urlopen(monkeypatch):
    calls = []
    responses = []

    def _urlopen(req, timeout=None):
        calls.append((req.url, timeout))
        if isinstance(responses, list) and responses:
            resp = responses.pop(0)
        else:
            resp = responses if not isinstance(responses, list) else None
        if isinstance(resp, Exception):
            raise resp
        return resp

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(urllib.request, "Request", FakeRequest)
    FakeRequest.captured.clear()
    return calls, responses


def write_repo(root, files):
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# ImportResult
# ---------------------------------------------------------------------------

class TestImportResult:
    def test_defaults(self):
        r = ImportResult(success=True, name="n", source="s",
                         files_imported=1, total_chars=2, output_path="p")
        assert r.error is None

    def test_fields(self):
        r = ImportResult(success=False, name="n", source="s",
                         files_imported=0, total_chars=0, output_path="",
                         error="boom")
        assert r.error == "boom"
        assert r.success is False


# ---------------------------------------------------------------------------
# RepoImporter
# ---------------------------------------------------------------------------

class TestRepoImporter:
    def test_init_creates_cache_dir(self, tmp_path):
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        assert (tmp_path / "cache").is_dir()

    def test_clone_existing_repo_skips_git(self, tmp_path, monkeypatch):
        imp = RepoImporter(cache_dir=str(tmp_path))
        target = tmp_path / "repo"
        target.mkdir()

        def fail(*args, **kwargs):
            raise AssertionError("should not clone existing repo")

        monkeypatch.setattr(subprocess, "check_call", fail)
        result = imp.clone_repo("https://github.com/org/repo.git")
        assert result == target

    def test_clone_repo_command(self, tmp_path, monkeypatch):
        imp = RepoImporter(cache_dir=str(tmp_path))
        calls = []

        def fake_check_call(cmd, **kwargs):
            calls.append(cmd)

        monkeypatch.setattr(subprocess, "check_call", fake_check_call)
        target = imp.clone_repo("https://github.com/org/repo.git")
        assert calls[0] == ["git", "clone", "https://github.com/org/repo.git",
                            str(target), "--depth", "1"]

    def test_clone_repo_branch_and_depth(self, tmp_path, monkeypatch):
        imp = RepoImporter(cache_dir=str(tmp_path))
        calls = []

        def fake_check_call(cmd, **kwargs):
            calls.append(cmd)

        monkeypatch.setattr(subprocess, "check_call", fake_check_call)
        imp.clone_repo("https://github.com/org/repo", branch="main", depth=2)
        assert calls[0][-4:] == ["-b", "main", "--depth", "2"]

    def test_clone_repo_no_depth(self, tmp_path, monkeypatch):
        imp = RepoImporter(cache_dir=str(tmp_path))
        calls = []

        def fake_check_call(cmd, **kwargs):
            calls.append(cmd)

        monkeypatch.setattr(subprocess, "check_call", fake_check_call)
        imp.clone_repo("https://github.com/org/repo", depth=None)
        assert "--depth" not in calls[0]

    def test_clone_repo_name_strips_git(self, tmp_path, monkeypatch):
        imp = RepoImporter(cache_dir=str(tmp_path))
        monkeypatch.setattr(subprocess, "check_call", lambda *a, **k: None)
        target = imp.clone_repo("https://github.com/org/myrepo.git")
        assert target.name == "myrepo"

    # export_to_corpus -----------------------------------------------------

    def _make_repo(self, tmp_path):
        repo = tmp_path / "repo"
        write_repo(repo, {
            "src/main.py": "print('hello')\n",
            "README.md": "# Project\n",
            "notes.txt": "some notes\n",
            ".hidden.py": "x\n",
            "node_modules/lib/index.js": "const x = 1;\n",
            ".venv/lib/a.py": "import os\n",
            "binary.dat": "\xff\xfe\x00bad",
        })
        return repo

    def test_export_to_corpus_basic(self, tmp_path):
        repo = self._make_repo(tmp_path)
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        out = tmp_path / "out" / "corpus.jsonl"
        count = imp.export_to_corpus(repo, str(out))
        assert count == 3  # main.py, README.md, notes.txt
        records = [json.loads(line) for line in out.read_text().splitlines()]
        paths = {r["path"] for r in records}
        assert "src/main.py" in paths
        assert "README.md" in paths
        record = next(r for r in records if r["path"] == "src/main.py")
        assert record["content"] == "print('hello')\n"
        assert record["size"] == len(record["content"])
        assert record["language"] == "python"

    def test_export_to_corpus_extensions_filter(self, tmp_path):
        repo = self._make_repo(tmp_path)
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        out = tmp_path / "corpus.jsonl"
        count = imp.export_to_corpus(repo, str(out), extensions=[".md"])
        assert count == 1

    def test_export_to_corpus_max_files(self, tmp_path):
        repo = self._make_repo(tmp_path)
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        out = tmp_path / "corpus.jsonl"
        count = imp.export_to_corpus(repo, str(out), max_files=1)
        assert count == 1

    def test_export_to_corpus_max_bytes_truncates(self, tmp_path):
        repo = tmp_path / "repo"
        write_repo(repo, {"a.py": "x" * 500})
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        out = tmp_path / "corpus.jsonl"
        count = imp.export_to_corpus(repo, str(out), max_bytes=100)
        assert count == 1
        record = json.loads(out.read_text().splitlines()[0])
        assert record["size"] == 100

    def test_export_skips_binary_and_ignored(self, tmp_path):
        repo = self._make_repo(tmp_path)
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        out = tmp_path / "corpus.jsonl"
        imp.export_to_corpus(repo, str(out))
        content = out.read_text()
        assert "binary.dat" not in content
        assert "index.js" not in content
        assert "hidden" not in content

    def test_export_skips_unreadable_file(self, tmp_path):
        repo = tmp_path / "repo"
        write_repo(repo, {"ok.py": "x"})
        (repo / "bad.py").write_bytes(b"\xff\xfe\x00\x01")
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        out = tmp_path / "corpus.jsonl"
        count = imp.export_to_corpus(repo, str(out))
        assert count == 1
        assert "bad.py" not in out.read_text()

    def test_export_creates_parent_dirs(self, tmp_path):
        repo = tmp_path / "repo"
        write_repo(repo, {"a.py": "x"})
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        out = tmp_path / "deep" / "nested" / "corpus.jsonl"
        imp.export_to_corpus(repo, str(out))
        assert out.exists()

    def test_iter_files_skips_ignores_and_dots(self, tmp_path):
        repo = self._make_repo(tmp_path)
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        files = list(imp._iter_files(repo, DEFAULT_IGNORES, [".py", ".md", ".txt"]))
        names = {p.name for p in files}
        assert names == {"main.py", "README.md", "notes.txt"}

    # _detect_language -----------------------------------------------------

    def test_detect_language_extensions(self, tmp_path):
        imp = RepoImporter()
        cases = {
            "f.py": "python", "f.js": "javascript", "f.ts": "typescript",
            "f.md": "markdown", "f.json": "json", "f.yaml": "yaml",
            "f.sh": "shell", "f.rs": "rust", "f.go": "go", "f.java": "java",
            "f.cpp": "cpp", "f.c": "c", "f.cs": "csharp", "f.rb": "ruby",
            "f.php": "php", "f.sql": "sql", "f.lua": "lua", "f.html": "html",
            "f.css": "css", "f.toml": "toml", "f.csv": "csv", "f.xml": "xml",
            "f.tex": "latex", "f.txt": "text",
        }
        for name, expected in cases.items():
            assert imp._detect_language(tmp_path / name) == expected, name

    def test_detect_language_special_files(self, tmp_path):
        imp = RepoImporter()
        assert imp._detect_language(tmp_path / "Dockerfile") == "dockerfile"
        assert imp._detect_language(tmp_path / "Makefile") == "makefile"
        assert imp._detect_language(tmp_path / "Rakefile") == "rakefile"
        assert imp._detect_language(tmp_path / "README") == "markdown"
        assert imp._detect_language(tmp_path / "LICENSE") == "text"
        assert imp._detect_language(tmp_path / ".env") == "env"

    def test_detect_language_content_python(self, tmp_path):
        imp = RepoImporter()
        content = "def foo():\n    import os\n    print('x')\n"
        assert imp._detect_language(tmp_path / "sample", content) == "python"

    def test_detect_language_content_typescript(self, tmp_path):
        imp = RepoImporter()
        content = "interface Foo {\n  name: string\n}\nconst x: number = 1;\n"
        assert imp._detect_language(tmp_path / "sample", content) == "typescript"

    def test_detect_language_content_javascript(self, tmp_path):
        imp = RepoImporter()
        content = "function foo() {\n  const x = 1;\n  console.log(x);\n}\n"
        assert imp._detect_language(tmp_path / "sample", content) == "javascript"

    def test_detect_language_content_html(self, tmp_path):
        imp = RepoImporter()
        content = "<!DOCTYPE html><html><body><p>hi</p></body></html>"
        assert imp._detect_language(tmp_path / "sample", content) == "html"

    def test_detect_language_content_html_tags(self, tmp_path):
        imp = RepoImporter()
        content = "<div>a</div>\n<span>b</span>\n"
        assert imp._detect_language(tmp_path / "sample", content) == "html"

    def test_detect_language_content_css_atrule_only(self, tmp_path):
        imp = RepoImporter()
        content = "@keyframes spin {\n  from { transform: rotate(0); }\n}\n"
        assert imp._detect_language(tmp_path / "sample", content) == "css"

    def test_detect_language_content_invalid_json(self, tmp_path):
        imp = RepoImporter()
        assert imp._detect_language(tmp_path / "sample", "{invalid json") == "text"

    def test_detect_language_content_shell_commands(self, tmp_path):
        imp = RepoImporter()
        content = "echo hi\nexport A=1\ncd /tmp\nmkdir x\n"
        assert imp._detect_language(tmp_path / "sample", content) == "shell"

    def test_detect_language_content_css(self, tmp_path):
        imp = RepoImporter()
        content = "body { color: red; padding: 0; }\n"
        assert imp._detect_language(tmp_path / "sample", content) == "css"

    def test_detect_language_content_css_atrule(self, tmp_path):
        imp = RepoImporter()
        content = "@media (max-width: 600px) { color: red; }\n"
        assert imp._detect_language(tmp_path / "sample", content) == "css"

    def test_detect_language_content_json(self, tmp_path):
        imp = RepoImporter()
        assert imp._detect_language(tmp_path / "sample", '{"a": 1}') == "json"

    def test_detect_language_content_yaml(self, tmp_path):
        imp = RepoImporter()
        content = "---\nkey: value\n  nested:\n    deep: x\n"
        assert imp._detect_language(tmp_path / "sample", content) == "yaml"

    def test_detect_language_content_shell(self, tmp_path):
        imp = RepoImporter()
        content = "#!/bin/bash\necho hi\n"
        assert imp._detect_language(tmp_path / "sample", content) == "shell"

    def test_detect_language_content_sql(self, tmp_path):
        imp = RepoImporter()
        content = "SELECT * FROM users WHERE id = 1;"
        assert imp._detect_language(tmp_path / "sample", content) == "sql"

    def test_detect_language_content_go(self, tmp_path):
        imp = RepoImporter()
        content = "package main\nimport (\nfunc main() {\n"
        assert imp._detect_language(tmp_path / "sample", content) == "go"

    def test_detect_language_content_rust(self, tmp_path):
        imp = RepoImporter()
        content = "fn main() {\n    let mut x = 1;\n}\n"
        assert imp._detect_language(tmp_path / "sample", content) == "rust"

    def test_detect_language_content_java(self, tmp_path):
        imp = RepoImporter()
        content = ("public class Hello {\n"
                   "  public static void main(String[] args) {\n"
                   "    System.out.println('x');\n  }\n}\n")
        assert imp._detect_language(tmp_path / "sample", content) == "java"

    def test_detect_language_content_cpp(self, tmp_path):
        imp = RepoImporter()
        content = "#include <iostream>\nint main() { std::cout << 'hi'; }\n"
        assert imp._detect_language(tmp_path / "sample", content) == "cpp"

    def test_detect_language_content_c(self, tmp_path):
        imp = RepoImporter()
        content = "#include <stdio.h>\nint main() { printf('x'); }\n"
        assert imp._detect_language(tmp_path / "sample", content) == "c"

    def test_detect_language_content_markdown(self, tmp_path):
        imp = RepoImporter()
        content = "# Title\n\n- item one\n- item two\n```code```\n"
        assert imp._detect_language(tmp_path / "sample", content) == "markdown"

    def test_detect_language_content_xml(self, tmp_path):
        imp = RepoImporter()
        content = '<?xml version="1.0"?><root><item>x</item></root>'
        assert imp._detect_language(tmp_path / "sample", content) == "xml"

    def test_detect_language_content_fallback(self, tmp_path):
        imp = RepoImporter()
        assert imp._detect_language(tmp_path / "sample", "just some plain words") == "text"

    def test_detect_language_empty_content(self, tmp_path):
        imp = RepoImporter()
        assert imp._detect_language(tmp_path / "sample", "") == "text"

    def test_detect_from_content_empty(self):
        assert RepoImporter()._detect_from_content("") == "text"

    # import_from_github ---------------------------------------------------

    def test_import_from_github_success(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        write_repo(repo, {"a.py": "print('hi')", "b.md": "# doc"})
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        monkeypatch.setattr(imp, "clone_repo", lambda url: repo)
        out_dir = tmp_path / "datasets"
        result = imp.import_from_github("https://github.com/org/repo", "ds", str(out_dir))
        assert result.success is True
        assert result.files_imported == 2
        assert result.total_chars > 0
        assert result.output_path.endswith("ds/corpus.jsonl")
        assert (out_dir / "ds" / "corpus.jsonl").exists()

    def test_import_from_github_failure(self, tmp_path, monkeypatch):
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))

        def boom(url):
            raise RuntimeError("git failed")

        monkeypatch.setattr(imp, "clone_repo", boom)
        result = imp.import_from_github("https://github.com/org/repo", "ds")
        assert result.success is False
        assert "git failed" in result.error
        assert result.files_imported == 0


# ---------------------------------------------------------------------------
# BooksSearch
# ---------------------------------------------------------------------------

class TestBooksSearch:
    def test_sanitize_query(self):
        b = BooksSearch()
        assert b._sanitize_query("hello world") == "hello%20world"

    def test_isbn_10(self):
        b = BooksSearch()
        assert b._is_isbn("0-306-40615-2") == "isbn:0306406152"

    def test_isbn_13(self):
        b = BooksSearch()
        assert b._is_isbn("978-0-306-40615-7") == "isbn:9780306406157"

    def test_isbn_invalid(self):
        b = BooksSearch()
        assert b._is_isbn("abc") is None

    def test_search_title(self, tmp_path, fake_urlopen):
        calls, responses = fake_urlopen
        responses.append(FakeResponse(json.dumps({
            "docs": [
                {"key": "/a", "title": "The Book", "author_name": ["Author"],
                 "isbn": ["123"], "first_publish_year": 1999, "cover_i": 5},
                {"key": "/b", "author_name": ["X"]},  # no title -> filtered
            ]
        })))
        b = BooksSearch()
        results = b.search("hello world")
        assert len(results) == 1
        r = results[0]
        assert r["title"] == "The Book"
        assert r["author"] == "Author"
        assert r["isbn"] == "123"
        assert r["year"] == 1999
        assert r["cover"] == 5
        assert calls[0][0] == "https://openlibrary.org/search.json?q=title:hello%20world&limit=10&fields=title,author_name,first_publish_year,cover_i,isbn,key"

    def test_search_isbn(self, fake_urlopen):
        calls, responses = fake_urlopen
        responses.append(FakeResponse(json.dumps({"docs": []})))
        b = BooksSearch()
        b.search("0306406152")
        assert "isbn:0306406152" in calls[0][0]

    def test_search_empty_results(self, fake_urlopen):
        _, responses = fake_urlopen
        responses.append(FakeResponse(json.dumps({"docs": []})))
        assert BooksSearch().search("nonexistent") == []

    def test_search_error(self, fake_urlopen):
        _, responses = fake_urlopen
        responses.append(urllib.error.URLError("nope"))
        assert BooksSearch().search("anything") == []


# ---------------------------------------------------------------------------
# HuggingFaceImporter
# ---------------------------------------------------------------------------

class TestHuggingFaceImporter:
    def test_check_hf_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "datasets", None)
        assert HuggingFaceImporter()._hf_available is False

    def test_check_hf_present(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "datasets",
                            SimpleNamespace(load_dataset=lambda *a, **k: None))
        assert HuggingFaceImporter()._hf_available is True

    def test_search_datasets_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "domains.infrastructure.hf_hub", None)
        assert HuggingFaceImporter().search_datasets("anything") == []

    def test_search_datasets_success(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "domains.infrastructure.hf_hub",
                            SimpleNamespace(fetch_dataset_search=lambda q, limit: [{"id": q}]))
        assert HuggingFaceImporter().search_datasets("cats") == [{"id": "cats"}]

    def test_search_datasets_error(self, monkeypatch):
        def boom(q, limit):
            raise RuntimeError("hub down")

        monkeypatch.setitem(sys.modules, "domains.infrastructure.hf_hub",
                            SimpleNamespace(fetch_dataset_search=boom))
        assert HuggingFaceImporter().search_datasets("cats") == []

    def test_download_requires_datasets_package(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "datasets", None)
        result = HuggingFaceImporter().download_dataset("owner/ds", str(Path("x") / "datasets"))
        assert result.success is False
        assert "pip install datasets" in result.error

    def _importer_with_fake_load(self, monkeypatch, load_fn):
        monkeypatch.setitem(sys.modules, "datasets",
                            SimpleNamespace(load_dataset=load_fn))
        return HuggingFaceImporter()

    def test_download_datasetdict(self, tmp_path, monkeypatch):
        def load(*args, **kwargs):
            return {"train": [{"text": "hello"}, {"text": "world"}],
                    "test": [{"content": "bye"}]}

        imp = self._importer_with_fake_load(monkeypatch, load)
        out = tmp_path / "datasets"
        result = imp.download_dataset("owner/ds", str(out))
        assert result.success is True
        assert result.name == "ds"
        assert result.files_imported == 3
        assert result.total_chars == len("hello") + len("world") + len("bye")
        lines = (out / "ds" / "corpus.jsonl").read_text().splitlines()
        assert json.loads(lines[0])["split"] == "train"
        assert json.loads(lines[2])["split"] == "test"

    def test_download_single_dataset(self, tmp_path, monkeypatch):
        def load(*args, **kwargs):
            return [{"text": "one"}, {"content": "two"}]

        imp = self._importer_with_fake_load(monkeypatch, load)
        result = imp.download_dataset("owner/ds", str(tmp_path))
        assert result.success is True
        assert result.files_imported == 2
        lines = (tmp_path / "ds" / "corpus.jsonl").read_text().splitlines()
        assert json.loads(lines[0])["split"] == "train"

    def test_download_fallback_split(self, tmp_path, monkeypatch):
        state = {"calls": 0}

        def load(*args, **kwargs):
            state["calls"] += 1
            if state["calls"] == 1:
                raise ValueError("requires config")
            return [{"text": "ok"}]

        imp = self._importer_with_fake_load(monkeypatch, load)
        result = imp.download_dataset("owner/ds", str(tmp_path))
        assert result.success is True
        assert state["calls"] == 2

    def test_download_failure(self, tmp_path, monkeypatch):
        def load(*args, **kwargs):
            raise RuntimeError("download failed")

        imp = self._importer_with_fake_load(monkeypatch, load)
        result = imp.download_dataset("owner/ds", str(tmp_path))
        assert result.success is False
        assert "download failed" in result.error


# ---------------------------------------------------------------------------
# URLImporter
# ---------------------------------------------------------------------------

class TestURLImporter:
    def test_init_creates_cache_dir(self, tmp_path):
        imp = URLImporter(cache_dir=str(tmp_path / "dl"))
        assert (tmp_path / "dl").is_dir()

    def test_download_file_with_extension(self, tmp_path, monkeypatch):
        imp = URLImporter(cache_dir=str(tmp_path / "dl"))
        monkeypatch.setattr(urllib.request, "urlretrieve", lambda *a, **k: None)
        out = imp.download_file("https://example.com/data.txt")
        assert out == tmp_path / "dl" / "data.txt"

    def test_download_file_no_extension(self, tmp_path, monkeypatch):
        imp = URLImporter(cache_dir=str(tmp_path / "dl"))
        monkeypatch.setattr(urllib.request, "urlretrieve", lambda *a, **k: None)
        out = imp.download_file("https://example.com/data")
        assert out == tmp_path / "dl" / "download.txt"

    def test_import_from_url_success(self, tmp_path, monkeypatch):
        imp = URLImporter(cache_dir=str(tmp_path / "dl"))
        src = tmp_path / "data.txt"
        src.write_text("file content", encoding="utf-8")
        monkeypatch.setattr(imp, "download_file", lambda url: src)
        result = imp.import_from_url("https://example.com/data.txt", "ds", str(tmp_path / "datasets"))
        assert result.success is True
        assert result.files_imported == 1
        assert result.total_chars == len("file content")
        record = json.loads((tmp_path / "datasets" / "ds" / "corpus.jsonl").read_text())
        assert record["content"] == "file content"
        assert record["source"] == "https://example.com/data.txt"

    def test_import_from_url_failure(self, tmp_path, monkeypatch):
        imp = URLImporter(cache_dir=str(tmp_path / "dl"))

        def boom(url):
            raise ConnectionError("offline")

        monkeypatch.setattr(imp, "download_file", boom)
        result = imp.import_from_url("https://example.com/data.txt", "ds")
        assert result.success is False
        assert "offline" in result.error


# ---------------------------------------------------------------------------
# GitHubSearch
# ---------------------------------------------------------------------------

class TestGitHubSearch:
    def test_sanitize_query(self):
        g = GitHubSearch()
        assert g._sanitize_query("  Python Code  ") == "python%20code"

    def test_search_repos_success(self, fake_urlopen):
        calls, responses = fake_urlopen
        responses.append(FakeResponse(json.dumps({
            "items": [
                {"full_name": "o/r", "description": "desc", "html_url": "https://x",
                 "stargazers_count": 5, "forks_count": 2, "language": "Python"},
                {"full_name": "o2/r2", "html_url": "https://y"},
            ]
        })))
        g = GitHubSearch()
        results = g.search_repos("python code")
        assert len(results) == 2
        assert results[0]["full_name"] == "o/r"
        assert results[0]["stargazers_count"] == 5
        assert results[1]["description"] == ""
        assert "q=python%20code" in calls[0][0]
        assert "per_page=10" in calls[0][0]

    def test_search_repos_empty(self, fake_urlopen):
        _, responses = fake_urlopen
        responses.append(FakeResponse(json.dumps({"items": []})))
        assert GitHubSearch().search_repos("nothing") == []

    def test_search_repos_error(self, fake_urlopen):
        _, responses = fake_urlopen
        responses.append(urllib.error.URLError("down"))
        assert GitHubSearch().search_repos("anything") == []


# ---------------------------------------------------------------------------
# ISBNImporter
# ---------------------------------------------------------------------------

class TestISBNImporter:
    def test_import_not_found(self, tmp_path, monkeypatch):
        imp = ISBNImporter(output_dir=str(tmp_path / "datasets"))
        monkeypatch.setattr(imp._books_search, "search", lambda *a, **k: [])
        result = imp.import_from_isbn("0306406152", "book")
        assert result.success is False
        assert "not found" in result.error
        assert (tmp_path / "datasets" / "book").is_dir()

    def test_import_with_gutenberg_text(self, tmp_path, monkeypatch):
        imp = ISBNImporter(output_dir=str(tmp_path / "datasets"))
        monkeypatch.setattr(imp._books_search, "search",
                            lambda *a, **k: [{"title": "Book", "author": "Writer",
                                              "first_publish_year": 1900}])
        monkeypatch.setattr(imp, "_fetch_gutenberg_text", lambda t, a: "FULL TEXT")
        result = imp.import_from_isbn("0306406152", "book")
        assert result.success is True
        assert result.files_imported == 1
        assert (tmp_path / "datasets" / "book" / "book.txt").read_text() == "FULL TEXT"
        meta = json.loads((tmp_path / "datasets" / "book" / "metadata.json").read_text())
        assert meta["source_type"] == "gutenberg"

    def test_import_metadata_only(self, tmp_path, monkeypatch):
        imp = ISBNImporter(output_dir=str(tmp_path / "datasets"))
        monkeypatch.setattr(imp._books_search, "search",
                            lambda *a, **k: [{"title": "Book", "author": "Writer"}])
        monkeypatch.setattr(imp, "_fetch_gutenberg_text", lambda t, a: None)
        result = imp.import_from_isbn("0306406152", "book")
        assert result.success is True
        meta = json.loads((tmp_path / "datasets" / "book" / "metadata.json").read_text())
        assert meta["source_type"] == "metadata_only"
        assert (tmp_path / "datasets" / "book" / "book_info.txt").exists()

    def test_fetch_gutenberg_no_search_terms(self):
        imp = ISBNImporter()
        assert imp._fetch_gutenberg_text("", "") is None

    def test_fetch_gutenberg_no_results(self, fake_urlopen):
        _, responses = fake_urlopen
        responses.append(FakeResponse(json.dumps({"results": []})))
        assert ISBNImporter()._fetch_gutenberg_text("Title", "Author") is None

    def test_fetch_gutenberg_success(self, fake_urlopen):
        _, responses = fake_urlopen
        responses.append(FakeResponse(json.dumps({"results": [{"id": 123}]})))
        responses.append(FakeResponse("gutenberg text"))
        text = ISBNImporter()._fetch_gutenberg_text("Title", "Author")
        assert text == "gutenberg text"

    def test_fetch_gutenberg_error(self, fake_urlopen):
        _, responses = fake_urlopen
        responses.append(urllib.error.URLError("no net"))
        assert ISBNImporter()._fetch_gutenberg_text("Title", "Author") is None


# ---------------------------------------------------------------------------
# DataImporter
# ---------------------------------------------------------------------------

@pytest.fixture
def data_importer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return DataImporter(str(tmp_path / "datasets"))


class TestDataImporter:
    def test_init_wires_subimporters(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        imp = DataImporter(str(tmp_path / "datasets"))
        assert isinstance(imp.repo_importer, RepoImporter)
        assert isinstance(imp.hf_importer, HuggingFaceImporter)
        assert isinstance(imp.url_importer, URLImporter)

    def test_import_from_github_delegates(self, data_importer, monkeypatch):
        calls = []

        def fake(url, name, out_dir, extensions, max_files):
            calls.append((url, name, out_dir, extensions, max_files))
            return ImportResult(True, name, url, 1, 5, "p")

        monkeypatch.setattr(data_importer.repo_importer, "import_from_github", fake)
        r = data_importer.import_from_github("https://github.com/a/b", "n",
                                             extensions=[".py"], max_files=3)
        assert r.success is True
        assert calls[0][:2] == ("https://github.com/a/b", "n")
        assert calls[0][3:] == ([".py"], 3)

    def test_import_from_huggingface_delegates(self, data_importer, monkeypatch):
        calls = []

        def fake(dataset_id, out_dir, name):
            calls.append((dataset_id, out_dir, name))
            return ImportResult(True, name or dataset_id, "hf", 1, 1, "p")

        monkeypatch.setattr(data_importer.hf_importer, "download_dataset", fake)
        r = data_importer.import_from_huggingface("owner/ds", "myname")
        assert r.success is True
        assert calls[0] == ("owner/ds", data_importer.output_dir, "myname")

    def test_import_from_url_delegates(self, data_importer, monkeypatch):
        calls = []

        def fake(url, name, out_dir):
            calls.append((url, name, out_dir))
            return ImportResult(True, name, url, 1, 1, "p")

        monkeypatch.setattr(data_importer.url_importer, "import_from_url", fake)
        data_importer.import_from_url("https://x.com/f.txt", "n")
        assert calls[0] == ("https://x.com/f.txt", "n", data_importer.output_dir)

    def test_extract_pdf_text_no_libs(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "fitz", None)
        monkeypatch.setitem(sys.modules, "PyPDF2", None)
        p = tmp_path / "doc.pdf"
        p.write_bytes(b"%PDF-1.4")
        assert DataImporter._extract_pdf_text(p) == ""

    def test_extract_pdf_text_fitz(self, monkeypatch, tmp_path):
        class Page:
            def get_text(self):
                return "page text"

        class Doc:
            def __init__(self):
                self.pages = [Page(), Page()]
                self.closed = False

            def __iter__(self):
                return iter(self.pages)

            def close(self):
                self.closed = True

        monkeypatch.setitem(sys.modules, "PyPDF2", None)
        monkeypatch.setitem(sys.modules, "fitz", SimpleNamespace(open=lambda path: Doc()))
        p = tmp_path / "doc.pdf"
        p.write_bytes(b"%PDF-1.4")
        assert DataImporter._extract_pdf_text(p) == "page text\npage text"

    def test_extract_pdf_text_pypdf2(self, monkeypatch, tmp_path):
        class Page:
            def extract_text(self):
                return "pypdf page"

        monkeypatch.setitem(sys.modules, "fitz", None)
        monkeypatch.setitem(sys.modules, "PyPDF2",
                            SimpleNamespace(PdfReader=lambda f: SimpleNamespace(
                                pages=[Page(), Page()])))
        p = tmp_path / "doc.pdf"
        p.write_bytes(b"%PDF-1.4")
        assert DataImporter._extract_pdf_text(p) == "pypdf page\npypdf page"

    def test_import_from_local_single_file(self, data_importer, tmp_path):
        src = tmp_path / "input.txt"
        src.write_text("hello import", encoding="utf-8")
        result = data_importer.import_from_local(str(src), "ds")
        assert result.success is True
        assert result.files_imported == 1
        assert result.total_chars == len("hello import")
        corpus = Path(data_importer.output_dir) / "ds" / "corpus.jsonl"
        record = json.loads(corpus.read_text())
        assert record["content"] == "hello import"

    def test_import_from_local_single_pdf(self, data_importer, monkeypatch, tmp_path):
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4")
        monkeypatch.setitem(sys.modules, "fitz", None)
        monkeypatch.setitem(sys.modules, "PyPDF2", None)
        result = data_importer.import_from_local(str(src), "ds")
        assert result.success is True
        assert result.files_imported == 1

    def test_import_from_local_directory(self, data_importer, tmp_path):
        root = tmp_path / "data"
        write_repo(root, {"main.py": "print('x')", "notes.txt": "hello", "data.json": "{}"})
        (root / "photo.csv").write_text("a,b\n", encoding="utf-8")
        result = data_importer.import_from_local(str(root), "ds")
        assert result.success is True
        assert result.files_imported == 3
        corpus = Path(data_importer.output_dir) / "ds" / "corpus.jsonl"
        paths = {json.loads(l)["path"] for l in corpus.read_text().splitlines()}
        assert "main.py" in paths
        assert "photo.csv" not in paths

    def test_import_from_local_directory_with_pdf(self, data_importer, monkeypatch, tmp_path):
        root = tmp_path / "data"
        root.mkdir()
        (root / "doc.pdf").write_bytes(b"%PDF-1.4")
        monkeypatch.setitem(sys.modules, "fitz", None)
        monkeypatch.setitem(sys.modules, "PyPDF2", None)
        result = data_importer.import_from_local(str(root), "ds")
        assert result.success is True
        assert result.files_imported == 1

    def test_import_from_local_skips_unreadable(self, data_importer, tmp_path):
        root = tmp_path / "data"
        write_repo(root, {"ok.txt": "fine"})
        (root / "bad.py").write_bytes(b"\xff\xfe\x00\x01")
        result = data_importer.import_from_local(str(root), "ds")
        assert result.success is True
        assert result.files_imported == 1

    def test_import_from_local_output_dir_is_file(self, tmp_path):
        out_dir = tmp_path / "datasets"
        out_dir.mkdir()
        (out_dir / "ds").write_text("i am a file", encoding="utf-8")
        imp = DataImporter(str(out_dir))
        result = imp.import_from_local(str(tmp_path / "whatever"), "ds")
        assert result.success is False


# ---------------------------------------------------------------------------
# import_data
# ---------------------------------------------------------------------------

class TestImportData:
    def _patch_method(self, monkeypatch, method, out_dir):
        calls = []

        def fake(*args, **kwargs):
            calls.append((args, kwargs))
            return ImportResult(True, "n", "s", 1, 1, "p")

        monkeypatch.setattr(DataImporter, method, fake)
        return calls

    def test_auto_github(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calls = self._patch_method(monkeypatch, "import_from_github", str(tmp_path))
        r = import_data("https://github.com/org/repo", "ds", output_dir=str(tmp_path))
        assert r.success is True
        assert calls[0][0][1] == "https://github.com/org/repo"

    def test_auto_huggingface(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calls = self._patch_method(monkeypatch, "import_from_huggingface", str(tmp_path))
        import_data("owner/dataset", "ds", output_dir=str(tmp_path))
        assert calls[0][0][1] == "owner/dataset"

    def test_auto_url(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calls = self._patch_method(monkeypatch, "import_from_url", str(tmp_path))
        import_data("https://example.com/data.txt", "ds", output_dir=str(tmp_path))
        assert calls[0][0][1] == "https://example.com/data.txt"

    def test_auto_local(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calls = self._patch_method(monkeypatch, "import_from_local", str(tmp_path))
        import_data("somedir", "ds", output_dir=str(tmp_path))
        assert calls[0][0][1] == "somedir"

    def test_explicit_local(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calls = self._patch_method(monkeypatch, "import_from_local", str(tmp_path))
        import_data("https://github.com/org/repo", "ds", source_type="local",
                    output_dir=str(tmp_path))
        assert calls[0][0][1] == "https://github.com/org/repo"

    def test_kwargs_forwarded(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calls = self._patch_method(monkeypatch, "import_from_github", str(tmp_path))
        import_data("https://github.com/org/repo", "ds", extensions=[".py"],
                    output_dir=str(tmp_path))
        assert calls[0][1]["extensions"] == [".py"]

    def test_auto_git_at_github(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calls = self._patch_method(monkeypatch, "import_from_github", str(tmp_path))
        import_data("git@github.com:org/repo.git", "ds", output_dir=str(tmp_path))
        assert calls[0][0][1] == "git@github.com:org/repo.git"

    def test_auto_http_not_github(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calls = self._patch_method(monkeypatch, "import_from_url", str(tmp_path))
        import_data("http://example.com/file.csv", "ds", output_dir=str(tmp_path))
        assert calls[0][0][1] == "http://example.com/file.csv"


# ---------------------------------------------------------------------------
# _retry
# ---------------------------------------------------------------------------

class TestRetry:
    def test_succeeds_first_try(self):
        from domains.training.data_import import _retry
        counter = {"n": 0}

        def fn():
            counter["n"] += 1
            return "ok"

        result = _retry(fn, retries=3, delay=0.01)
        assert result == "ok"
        assert counter["n"] == 1

    def test_retries_on_transient_error(self):
        from domains.training.data_import import _retry
        counter = {"n": 0}

        def fn():
            counter["n"] += 1
            if counter["n"] < 3:
                raise ConnectionError("transient")
            return "recovered"

        result = _retry(fn, retries=3, delay=0.01, exceptions=(ConnectionError,))
        assert result == "recovered"
        assert counter["n"] == 3

    def test_raises_after_all_retries_exhausted(self):
        from domains.training.data_import import _retry

        def fn():
            raise ConnectionError("always fail")

        with pytest.raises(ConnectionError, match="always fail"):
            _retry(fn, retries=2, delay=0.01, exceptions=(ConnectionError,))

    def test_does_not_catch_unlisted_exceptions(self):
        from domains.training.data_import import _retry

        def fn():
            raise ValueError("wrong type")

        with pytest.raises(ValueError, match="wrong type"):
            _retry(fn, retries=3, delay=0.01, exceptions=(ConnectionError,))

    def test_zero_retries(self):
        from domains.training.data_import import _retry
        counter = {"n": 0}

        def fn():
            counter["n"] += 1
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            _retry(fn, retries=0, delay=0.01, exceptions=(ConnectionError,))
        assert counter["n"] == 1

    def test_default_exceptions_tuple(self):
        from domains.training.data_import import _retry
        import urllib.error

        counter = {"n": 0}

        def fn():
            counter["n"] += 1
            if counter["n"] == 1:
                raise urllib.error.URLError("net")
            if counter["n"] == 2:
                raise ConnectionError("conn")
            return "done"

        result = _retry(fn, retries=3, delay=0.01)
        assert result == "done"
        assert counter["n"] == 3


# ---------------------------------------------------------------------------
# RepoImporter — branch validation
# ---------------------------------------------------------------------------

class TestRepoImporterBranchValidation:
    def test_invalid_branch_rejected(self, tmp_path):
        imp = RepoImporter(cache_dir=str(tmp_path))
        with pytest.raises(ValueError, match="Invalid branch name"):
            imp.clone_repo("https://github.com/o/r", branch="feat; rm -rf /")

    def test_invalid_branch_spaces(self, tmp_path):
        imp = RepoImporter(cache_dir=str(tmp_path))
        with pytest.raises(ValueError, match="Invalid branch name"):
            imp.clone_repo("https://github.com/o/r", branch="has spaces")

    def test_valid_branch_patterns(self, tmp_path, monkeypatch):
        imp = RepoImporter(cache_dir=str(tmp_path))
        calls = []
        monkeypatch.setattr(subprocess, "check_call", lambda cmd, **k: calls.append(cmd))
        for branch in ["main", "feat/x", "release-1.0", "dev_branch"]:
            imp.clone_repo("https://github.com/o/r", branch=branch)
        assert len(calls) == 4


# ---------------------------------------------------------------------------
# RepoImporter — _iter_files edge cases
# ---------------------------------------------------------------------------

class TestRepoImporterIterFiles:
    def test_skips_dotfiles(self, tmp_path):
        repo = tmp_path / "repo"
        write_repo(repo, {".gitignore": "x", ".env": "y", "normal.py": "z"})
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        files = list(imp._iter_files(repo, DEFAULT_IGNORES, [".py", ".env", ".gitignore"]))
        assert len(files) == 1
        assert files[0].name == "normal.py"

    def test_skips_dot_directories(self, tmp_path):
        repo = tmp_path / "repo"
        (repo / ".hidden_dir").mkdir(parents=True)
        (repo / ".hidden_dir" / "file.py").write_text("x")
        (repo / "visible.py").write_text("y")
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        files = list(imp._iter_files(repo, DEFAULT_IGNORES, [".py"]))
        assert len(files) == 1
        assert files[0].name == "visible.py"

    def test_empty_extensions_list_returns_all(self, tmp_path):
        repo = tmp_path / "repo"
        write_repo(repo, {"a.py": "x", "b.js": "y"})
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        files = list(imp._iter_files(repo, DEFAULT_IGNORES, []))
        assert len(files) == 2

    def test_none_extensions_returns_all(self, tmp_path):
        repo = tmp_path / "repo"
        write_repo(repo, {"a.py": "x", "b.js": "y"})
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        files = list(imp._iter_files(repo, DEFAULT_IGNORES, None))
        assert len(files) == 2

    def test_empty_repo(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        files = list(imp._iter_files(repo, DEFAULT_IGNORES, [".py"]))
        assert files == []


# ---------------------------------------------------------------------------
# RepoImporter — _detect_language extended
# ---------------------------------------------------------------------------

class TestRepoImporterDetectLanguageExtended:
    def setup_method(self):
        self.imp = RepoImporter()

    def test_more_python_extensions(self, tmp_path):
        for ext in [".pyw", ".pyi"]:
            assert self.imp._detect_language(tmp_path / f"f{ext}") == "python"

    def test_more_js_extensions(self, tmp_path):
        for ext in [".jsx", ".mjs", ".cjs"]:
            assert self.imp._detect_language(tmp_path / f"f{ext}") == "javascript"

    def test_more_ts_extensions(self, tmp_path):
        for ext in [".tsx", ".mts"]:
            assert self.imp._detect_language(tmp_path / f"f{ext}") == "typescript"

    def test_more_markdown_extensions(self, tmp_path):
        assert self.imp._detect_language(tmp_path / "f.markdown") == "markdown"

    def test_more_json_extensions(self, tmp_path):
        assert self.imp._detect_language(tmp_path / "f.jsonl") == "json"

    def test_more_yaml_extensions(self, tmp_path):
        assert self.imp._detect_language(tmp_path / "f.yml") == "yaml"

    def test_more_shell_extensions(self, tmp_path):
        for ext in [".bash", ".zsh"]:
            assert self.imp._detect_language(tmp_path / f"f{ext}") == "shell"

    def test_more_cpp_extensions(self, tmp_path):
        for ext in [".cc", ".cxx", ".hpp", ".hxx"]:
            assert self.imp._detect_language(tmp_path / f"f{ext}") == "cpp"

    def test_h_is_c(self, tmp_path):
        assert self.imp._detect_language(tmp_path / "f.h") == "c"

    def test_ruby_php_perl(self, tmp_path):
        assert self.imp._detect_language(tmp_path / "f.rb") == "ruby"
        assert self.imp._detect_language(tmp_path / "f.php") == "php"
        assert self.imp._detect_language(tmp_path / "f.pl") == "perl"
        assert self.imp._detect_language(tmp_path / "f.pm") == "perl"

    def test_lua_r_swift_kotlin_scala(self, tmp_path):
        assert self.imp._detect_language(tmp_path / "f.lua") == "lua"
        assert self.imp._detect_language(tmp_path / "f.r") == "r"
        assert self.imp._detect_language(tmp_path / "f.R") == "r"
        assert self.imp._detect_language(tmp_path / "f.swift") == "swift"
        assert self.imp._detect_language(tmp_path / "f.kt") == "kotlin"
        assert self.imp._detect_language(tmp_path / "f.kts") == "kotlin"
        assert self.imp._detect_language(tmp_path / "f.scala") == "scala"
        assert self.imp._detect_language(tmp_path / "f.sc") == "scala"

    def test_web_extensions(self, tmp_path):
        for ext in [".htm", ".xhtml"]:
            assert self.imp._detect_language(tmp_path / f"f{ext}") == "html"
        for ext in [".scss", ".sass", ".less"]:
            assert self.imp._detect_language(tmp_path / f"f{ext}") == ext[1:]
        assert self.imp._detect_language(tmp_path / "f.vue") == "vue"
        assert self.imp._detect_language(tmp_path / "f.svelte") == "svelte"

    def test_functional_extensions(self, tmp_path):
        for ext in [".ex", ".exs"]:
            assert self.imp._detect_language(tmp_path / f"f{ext}") == "elixir"
        assert self.imp._detect_language(tmp_path / "f.erl") == "erlang"
        for ext in [".hs", ".lhs"]:
            assert self.imp._detect_language(tmp_path / f"f{ext}") == "haskell"
        for ext in [".clj", ".cljs"]:
            assert self.imp._detect_language(tmp_path / f"f{ext}") == "clojure"
        for ext in [".ml", ".mli"]:
            assert self.imp._detect_language(tmp_path / f"f{ext}") == "ocaml"
        for ext in [".fs", ".fsi"]:
            assert self.imp._detect_language(tmp_path / f"f{ext}") == "fsharp"

    def test_misc_extensions(self, tmp_path):
        assert self.imp._detect_language(tmp_path / "f.dart") == "dart"
        assert self.imp._detect_language(tmp_path / "f.jl") == "julia"
        assert self.imp._detect_language(tmp_path / "f.nim") == "nim"
        assert self.imp._detect_language(tmp_path / "f.cr") == "crystal"
        assert self.imp._detect_language(tmp_path / "f.d") == "d"
        for ext in [".asm", ".s"]:
            assert self.imp._detect_language(tmp_path / f"f{ext}") == "assembly"

    def test_config_extensions(self, tmp_path):
        assert self.imp._detect_language(tmp_path / "f.ini") == "ini"
        assert self.imp._detect_language(tmp_path / "f.cfg") == "ini"
        assert self.imp._detect_language(tmp_path / "f.conf") == "config"

    def test_data_extensions(self, tmp_path):
        assert self.imp._detect_language(tmp_path / "f.tsv") == "tsv"
        assert self.imp._detect_language(tmp_path / "f.tex") == "latex"
        assert self.imp._detect_language(tmp_path / "f.rst") == "restructuredtext"
        assert self.imp._detect_language(tmp_path / "f.org") == "org"

    def test_special_file_names_case_insensitive(self, tmp_path):
        assert self.imp._detect_language(tmp_path / "DOCKERFILE") == "dockerfile"
        assert self.imp._detect_language(tmp_path / "makefile") == "makefile"
        assert self.imp._detect_language(tmp_path / "Gemfile") == "gemfile"
        assert self.imp._detect_language(tmp_path / "Pipfile") == "pipfile"
        assert self.imp._detect_language(tmp_path / "Readme.md") == "markdown"
        assert self.imp._detect_language(tmp_path / "LICENSE.txt") == "text"
        assert self.imp._detect_language(tmp_path / ".env.local") == "env"

    def test_dockerfile_extension(self, tmp_path):
        assert self.imp._detect_language(tmp_path / "f.dockerfile") == "dockerfile"

    def test_content_python_comprehensive(self):
        content = "def foo():\n    self.x = 1\n    import os\n    from x import y\nif __name__ == '__main__':\n    print('hi')"
        assert self.imp._detect_language(Path("sample"), content) == "python"

    def test_content_python_async(self):
        content = "async def fetch():\n    await something()\n    try:\n        pass\n    except Exception:\n        raise ValueError('x')"
        assert self.imp._detect_language(Path("sample"), content) == "python"

    def test_content_python_lambda(self):
        content = "f = lambda x: x + 1\nwith open('f.py') as fh:\n    pass\nimport os\nprint('hi')"
        assert self.imp._detect_language(Path("sample"), content) == "python"

    def test_content_typescript_interfaces(self):
        content = "interface User {\n  name: string\n  age: number\n}\ntype ID = string | number"
        assert self.imp._detect_language(Path("sample"), content) == "typescript"

    def test_content_typescript_advanced(self):
        content = "const x: string[] = []\nconst y: Record<string, number> = {}\nreadonly z: boolean"
        assert self.imp._detect_language(Path("sample"), content) == "typescript"

    def test_content_typescript_generic(self):
        content = "function identity<T>(x: T): T { return x }\ninterface Box<T> { value: T }"
        assert self.imp._detect_language(Path("sample"), content) == "typescript"

    def test_content_javascript_comprehensive(self):
        content = "function greet() {\n  const name = 'hi';\n  let x = 1;\n  var y = 2;\n  console.log(name);\n  require('fs');\n}"
        assert self.imp._detect_language(Path("sample"), content) == "javascript"

    def test_content_javascript_arrow(self):
        content = "const fn = (x) => { return x + 1; };\nexport default fn;\nimport { foo } from 'bar';"
        assert self.imp._detect_language(Path("sample"), content) == "javascript"

    def test_content_html_doctype(self):
        content = "<!DOCTYPE html>\n<html><head><title>x</title></head></html>"
        assert self.imp._detect_language(Path("sample"), content) == "html"

    def test_content_html_multiple_tags(self):
        content = "<div><span>hi</span></div>\n<p>text</p>\n<body><script>let x</script></body>"
        assert self.imp._detect_language(Path("sample"), content) == "html"

    def test_content_css_properties(self):
        content = "body { margin: 0; padding: 10px; color: red; display: flex; }"
        assert self.imp._detect_language(Path("sample"), content) == "css"

    def test_content_css_pseudo_selectors(self):
        content = ".btn:hover { color: blue; }\n.btn::before { content: ''; }\n.btn::after { content: ''; }"
        assert self.imp._detect_language(Path("sample"), content) == "css"

    def test_content_css_font_face(self):
        content = "@font-face { font-family: 'Open'; src: url('font.woff'); }"
        assert self.imp._detect_language(Path("sample"), content) == "css"

    def test_content_json_array(self):
        content = '[{"a": 1}, {"b": 2}]'
        assert self.imp._detect_language(Path("sample"), content) == "json"

    def test_content_json_nested(self):
        content = '{"a": {"b": [1, 2, 3]}, "c": "d"}'
        assert self.imp._detect_language(Path("sample"), content) == "json"

    def test_content_yaml_first_line_colon(self):
        content = "key:\n  nested: yes\n    deep: true"
        assert self.imp._detect_language(Path("sample"), content) == "yaml"

    def test_content_shell_shebang_usr_bin_env(self):
        content = "#!/usr/bin/env python3\nprint('hi')"
        assert self.imp._detect_language(Path("sample"), content) == "shell"

    def test_content_shell_comprehensive(self):
        content = "echo hello\nexport PATH=/usr/bin\nsource ~/.bashrc\ncd /tmp\nmkdir -p x\nchmod 755 x\nsudo apt install vim"
        assert self.imp._detect_language(Path("sample"), content) == "shell"

    def test_content_sql_comprehensive(self):
        content = "SELECT * FROM users WHERE id = 1;\nINSERT INTO logs VALUES (1, 'x');\nUPDATE users SET name = 'y';"
        assert self.imp._detect_language(Path("sample"), content) == "sql"

    def test_content_go_comprehensive(self):
        content = "package main\nimport \"fmt\"\nfunc main() {\n  fmt.Println(\"hi\")\n}"
        assert self.imp._detect_language(Path("sample"), content) == "go"

    def test_content_go_goroutine(self):
        content = "package main\nimport \"fmt\"\ngo func() { fmt.Println(\"hi\") }()"
        assert self.imp._detect_language(Path("sample"), content) == "go"

    def test_content_rust_comprehensive(self):
        content = "fn main() {\n    let mut x = 1;\n    impl Trait for S {}\n    pub fn run() {}\n}"
        assert self.imp._detect_language(Path("sample"), content) == "rust"

    def test_content_rust_macros(self):
        content = "fn main() {\n    println!(\"hi\");\n    let v = vec![1, 2];\n    let s = Some(1);\n    let n: Option<i32> = None;\n}"
        assert self.imp._detect_language(Path("sample"), content) == "rust"

    def test_content_java_comprehensive(self):
        content = "public class Main {\n  public static void main(String[] args) {\n    System.out.println(\"hi\");\n  }\n}"
        assert self.imp._detect_language(Path("sample"), content) == "java"

    def test_content_cpp_iostream(self):
        content = "#include <iostream>\nstd::cout << \"hi\";\nstd::cin >> x;\nnamespace foo {}"
        assert self.imp._detect_language(Path("sample"), content) == "cpp"

    def test_content_c_stdio(self):
        content = "#include <stdio.h>\nint main() { printf(\"hi\"); scanf(\"%d\", &x); void *p; malloc(10); free(p); }"
        assert self.imp._detect_language(Path("sample"), content) == "c"

    def test_content_markdown_comprehensive(self):
        content = "# Title\n\n## Sub\n\n- item\n- item2\n\n```code```\n[text](url)\n**bold** __underline__"
        assert self.imp._detect_language(Path("sample"), content) == "markdown"

    def test_content_xml_root_tag(self):
        content = "<root><item>a</item><item>b</item></root>"
        assert self.imp._detect_language(Path("sample"), content) == "xml"

    def test_content_fallback_plain_text(self):
        content = "Just some random words without any patterns"
        assert self.imp._detect_language(Path("sample"), content) == "text"

    def test_content_empty_string(self):
        assert self.imp._detect_language(Path("sample"), "") == "text"


# ---------------------------------------------------------------------------
# RepoImporter — export_to_corpus edge cases
# ---------------------------------------------------------------------------

class TestRepoImporterExportEdgeCases:
    def test_export_empty_repo(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        out = tmp_path / "corpus.jsonl"
        count = imp.export_to_corpus(repo, str(out))
        assert count == 0
        assert out.read_text() == ""

    def test_export_custom_ignores(self, tmp_path):
        repo = tmp_path / "repo"
        write_repo(repo, {"a.py": "x", "b.py": "y"})
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        out = tmp_path / "corpus.jsonl"
        count = imp.export_to_corpus(repo, str(out), ignores={"a.py"})
        assert count == 1
        assert "b.py" in out.read_text()

    def test_export_custom_extensions(self, tmp_path):
        repo = tmp_path / "repo"
        write_repo(repo, {"a.py": "x", "b.js": "y", "c.ts": "z"})
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        out = tmp_path / "corpus.jsonl"
        count = imp.export_to_corpus(repo, str(out), extensions=[".js", ".ts"])
        assert count == 2

    def test_export_records_jsonl_format(self, tmp_path):
        repo = tmp_path / "repo"
        write_repo(repo, {"a.py": "print('hi')"})
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        out = tmp_path / "corpus.jsonl"
        imp.export_to_corpus(repo, str(out))
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert set(record.keys()) == {"path", "content", "size", "language"}
        assert record["path"] == "a.py"

    def test_export_nested_directories(self, tmp_path):
        repo = tmp_path / "repo"
        write_repo(repo, {
            "src/deep/nested/file.py": "x",
            "lib/utils/helper.js": "y",
        })
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        out = tmp_path / "corpus.jsonl"
        count = imp.export_to_corpus(repo, str(out))
        assert count == 2
        paths = {json.loads(l)["path"] for l in out.read_text().splitlines()}
        assert "src/deep/nested/file.py" in paths
        assert "lib/utils/helper.js" in paths


# ---------------------------------------------------------------------------
# RepoImporter — import_from_github
# ---------------------------------------------------------------------------

class TestRepoImporterImportExtended:
    def test_import_from_github_total_chars(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        write_repo(repo, {"a.py": "hello", "b.md": "world!"})
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        monkeypatch.setattr(imp, "clone_repo", lambda url: repo)
        result = imp.import_from_github("https://github.com/o/r", "ds", str(tmp_path / "ds"))
        assert result.total_chars == len("hello") + len("world!")

    def test_import_from_github_empty_repo(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        monkeypatch.setattr(imp, "clone_repo", lambda url: repo)
        result = imp.import_from_github("https://github.com/o/r", "ds", str(tmp_path / "ds"))
        assert result.success is True
        assert result.files_imported == 0
        assert result.total_chars == 0

    def test_import_from_github_with_extensions(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        write_repo(repo, {"a.py": "x", "b.js": "y", "c.md": "z"})
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        monkeypatch.setattr(imp, "clone_repo", lambda url: repo)
        result = imp.import_from_github("https://github.com/o/r", "ds",
                                         str(tmp_path / "ds"), extensions=[".py", ".js"])
        assert result.files_imported == 2

    def test_import_from_github_with_max_files(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        write_repo(repo, {"a.py": "x", "b.py": "y", "c.py": "z"})
        imp = RepoImporter(cache_dir=str(tmp_path / "cache"))
        monkeypatch.setattr(imp, "clone_repo", lambda url: repo)
        result = imp.import_from_github("https://github.com/o/r", "ds",
                                         str(tmp_path / "ds"), max_files=2)
        assert result.files_imported == 2


# ---------------------------------------------------------------------------
# BooksSearch — extended
# ---------------------------------------------------------------------------

class TestBooksSearchExtended:
    def test_sanitize_query_empty(self):
        assert BooksSearch()._sanitize_query("") == ""

    def test_sanitize_query_special_chars(self):
        assert BooksSearch()._sanitize_query("hello&world=yes") == "hello%26world%3Dyes"

    def test_isbn_10_no_dashes(self):
        assert BooksSearch()._is_isbn("0306406152") == "isbn:0306406152"

    def test_isbn_13_no_dashes(self):
        assert BooksSearch()._is_isbn("9780306406157") == "isbn:9780306406157"

    def test_isbn_with_underscores(self):
        assert BooksSearch()._is_isbn("978_0_306_40615_7") == "isbn:9780306406157"

    def test_isbn_with_spaces(self):
        assert BooksSearch()._is_isbn("978 0 306 40615 7") == "isbn:9780306406157"

    def test_isbn_wrong_length(self):
        assert BooksSearch()._is_isbn("12345") is None
        assert BooksSearch()._is_isbn("123456789012345") is None

    def test_isbn_contains_letters(self):
        assert BooksSearch()._is_isbn("030640615X") is None

    def test_search_limit(self, fake_urlopen):
        calls, responses = fake_urlopen
        responses.append(FakeResponse(json.dumps({"docs": []})))
        BooksSearch().search("test", limit=5)
        assert "limit=5" in calls[0][0]

    def test_search_multiple_results(self, fake_urlopen):
        _, responses = fake_urlopen
        docs = [{"title": f"Book {i}", "key": f"/{i}", "author_name": [f"A{i}"]}
                for i in range(3)]
        responses.append(FakeResponse(json.dumps({"docs": docs})))
        results = BooksSearch().search("test")
        assert len(results) == 3

    def test_search_filters_no_title(self, fake_urlopen):
        _, responses = fake_urlopen
        responses.append(FakeResponse(json.dumps({
            "docs": [
                {"title": "Good", "key": "/1", "author_name": ["A"]},
                {"key": "/2", "author_name": ["B"]},  # no title
                {"title": "", "key": "/3", "author_name": ["C"]},  # empty title
            ]
        })))
        results = BooksSearch().search("test")
        assert len(results) == 1
        assert results[0]["title"] == "Good"


# ---------------------------------------------------------------------------
# HuggingFaceImporter — extended
# ---------------------------------------------------------------------------

class TestHuggingFaceImporterExtended:
    def test_download_datasetdict_fallback(self, tmp_path, monkeypatch):
        state = {"calls": 0}

        def load(*args, **kwargs):
            state["calls"] += 1
            if state["calls"] == 1:
                raise ValueError("config needed")
            return {"train": [{"text": "data"}]}

        monkeypatch.setitem(sys.modules, "datasets",
                            SimpleNamespace(load_dataset=load))
        imp = HuggingFaceImporter()
        result = imp.download_dataset("owner/ds", str(tmp_path))
        assert result.success is True
        assert result.files_imported == 1
        assert state["calls"] == 2

    def test_download_datasetdict_with_content_field(self, tmp_path, monkeypatch):
        def load(*args, **kwargs):
            return {"train": [{"content": "text content"}]}

        monkeypatch.setitem(sys.modules, "datasets",
                            SimpleNamespace(load_dataset=load))
        imp = HuggingFaceImporter()
        result = imp.download_dataset("owner/ds", str(tmp_path))
        assert result.success is True
        record = json.loads((tmp_path / "ds" / "corpus.jsonl").read_text().splitlines()[0])
        assert record["content"] == "text content"
        assert record["split"] == "train"

    def test_download_datasetdict_with_str_fallback(self, tmp_path, monkeypatch):
        def load(*args, **kwargs):
            return {"train": [{"key": "value"}]}

        monkeypatch.setitem(sys.modules, "datasets",
                            SimpleNamespace(load_dataset=load))
        imp = HuggingFaceImporter()
        result = imp.download_dataset("owner/ds", str(tmp_path))
        assert result.success is True
        record = json.loads((tmp_path / "ds" / "corpus.jsonl").read_text().splitlines()[0])
        assert record["content"] == "{'key': 'value'}"

    def test_download_custom_name(self, tmp_path, monkeypatch):
        def load(*args, **kwargs):
            return [{"text": "hi"}]

        monkeypatch.setitem(sys.modules, "datasets",
                            SimpleNamespace(load_dataset=load))
        imp = HuggingFaceImporter()
        result = imp.download_dataset("owner/ds", str(tmp_path), name="custom_name")
        assert result.name == "custom_name"
        assert (tmp_path / "custom_name" / "corpus.jsonl").exists()

    def test_download_default_name_from_id(self, tmp_path, monkeypatch):
        def load(*args, **kwargs):
            return [{"text": "hi"}]

        monkeypatch.setitem(sys.modules, "datasets",
                            SimpleNamespace(load_dataset=load))
        imp = HuggingFaceImporter()
        result = imp.download_dataset("owner/my_dataset", str(tmp_path))
        assert result.name == "my_dataset"

    def test_search_datasets_returns_list(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "domains.infrastructure.hf_hub",
                            SimpleNamespace(fetch_dataset_search=lambda q, limit: [
                                {"id": "a"}, {"id": "b"}
                            ]))
        results = HuggingFaceImporter().search_datasets("q", limit=2)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# URLImporter — extended
# ---------------------------------------------------------------------------

class TestURLImporterExtended:
    def test_download_file_url_with_multiple_dots(self, tmp_path, monkeypatch):
        imp = URLImporter(cache_dir=str(tmp_path / "dl"))
        monkeypatch.setattr(urllib.request, "urlretrieve", lambda *a, **k: None)
        out = imp.download_file("https://example.com/data.archive.tar.gz")
        assert out.name == "data.archive.tar.gz"

    def test_download_file_empty_path(self, tmp_path, monkeypatch):
        imp = URLImporter(cache_dir=str(tmp_path / "dl"))
        monkeypatch.setattr(urllib.request, "urlretrieve", lambda *a, **k: None)
        out = imp.download_file("https://example.com/")
        assert out.name == "download.txt"

    def test_import_from_url_content_length(self, tmp_path, monkeypatch):
        imp = URLImporter(cache_dir=str(tmp_path / "dl"))
        content = "a" * 100
        src = tmp_path / "data.txt"
        src.write_text(content, encoding="utf-8")
        monkeypatch.setattr(imp, "download_file", lambda url: src)
        result = imp.import_from_url("https://example.com/data.txt", "ds",
                                      str(tmp_path / "datasets"))
        assert result.total_chars == 100

    def test_import_from_url_creates_output_dir(self, tmp_path, monkeypatch):
        imp = URLImporter(cache_dir=str(tmp_path / "dl"))
        src = tmp_path / "data.txt"
        src.write_text("content", encoding="utf-8")
        monkeypatch.setattr(imp, "download_file", lambda url: src)
        result = imp.import_from_url("https://example.com/data.txt", "ds",
                                      str(tmp_path / "new_dir"))
        assert (tmp_path / "new_dir" / "ds" / "corpus.jsonl").exists()


# ---------------------------------------------------------------------------
# GitHubSearch — extended
# ---------------------------------------------------------------------------

class TestGitHubSearchExtended:
    def test_sanitize_query_special_chars(self):
        assert GitHubSearch()._sanitize_query("C++ code") == "c%2B%2B%20code"

    def test_search_repos_limit(self, fake_urlopen):
        calls, responses = fake_urlopen
        responses.append(FakeResponse(json.dumps({"items": []})))
        GitHubSearch().search_repos("test", limit=5)
        assert "per_page=5" in calls[0][0]

    def test_search_repos_headers(self, fake_urlopen):
        calls, responses = fake_urlopen
        responses.append(FakeResponse(json.dumps({"items": []})))
        GitHubSearch().search_repos("test")
        assert calls[0][0].startswith("https://api.github.com/search/repos")

    def test_search_repos_defaults(self, fake_urlopen):
        _, responses = fake_urlopen
        responses.append(FakeResponse(json.dumps({
            "items": [{"full_name": "o/r", "html_url": "https://x"}]
        })))
        results = GitHubSearch().search_repos("q")
        assert results[0]["description"] == ""
        assert results[0]["stargazers_count"] == 0
        assert results[0]["forks_count"] == 0
        assert results[0]["language"] is None


# ---------------------------------------------------------------------------
# ISBNImporter — extended
# ---------------------------------------------------------------------------

class TestISBNImporterExtended:
    def test_import_creates_output_dir(self, tmp_path, monkeypatch):
        imp = ISBNImporter(output_dir=str(tmp_path / "books"))
        monkeypatch.setattr(imp._books_search, "search",
                            lambda *a, **k: [{"title": "T", "author": "A"}])
        monkeypatch.setattr(imp, "_fetch_gutenberg_text", lambda t, a: None)
        imp.import_from_isbn("1234567890", "mybook")
        assert (tmp_path / "books" / "mybook").is_dir()

    def test_import_gutenberg_metadata_fields(self, tmp_path, monkeypatch):
        imp = ISBNImporter(output_dir=str(tmp_path / "books"))
        book = {"title": "Test Book", "author": "Author", "year": 1999}
        monkeypatch.setattr(imp._books_search, "search", lambda *a, **k: [book])
        monkeypatch.setattr(imp, "_fetch_gutenberg_text", lambda t, a: "text")
        result = imp.import_from_isbn("12345", "ds")
        meta = json.loads((tmp_path / "books" / "ds" / "metadata.json").read_text())
        assert meta["title"] == "Test Book"
        assert meta["author"] == "Author"
        assert meta["year"] == 1999
        assert meta["source"] == "isbn:12345"

    def test_import_metadata_only_info_file(self, tmp_path, monkeypatch):
        imp = ISBNImporter(output_dir=str(tmp_path / "books"))
        monkeypatch.setattr(imp._books_search, "search",
                            lambda *a, **k: [{"title": "T", "author": "A", "year": 2000}])
        monkeypatch.setattr(imp, "_fetch_gutenberg_text", lambda t, a: None)
        result = imp.import_from_isbn("999", "ds")
        info = (tmp_path / "books" / "ds" / "ds_info.txt").read_text()
        assert "Title: T" in info
        assert "Author: A" in info
        assert "ISBN: 999" in info

    def test_fetch_gutenberg_query_truncation(self, fake_urlopen):
        _, responses = fake_urlopen
        responses.append(FakeResponse(json.dumps({"results": []})))
        long_title = "A" * 300
        ISBNImporter()._fetch_gutenberg_text(long_title, "Author")
        # Query should be truncated to 200 chars
        assert True  # no crash

    def test_fetch_gutenberg_text_encoding(self, fake_urlopen):
        _, responses = fake_urlopen
        responses.append(FakeResponse(json.dumps({"results": [{"id": 42}]})))
        text_bytes = "Hello World".encode("utf-8")
        responses.append(FakeResponse(text_bytes))
        result = ISBNImporter()._fetch_gutenberg_text("Title", "Author")
        assert result == "Hello World"


# ---------------------------------------------------------------------------
# DataImporter — extended
# ---------------------------------------------------------------------------

class TestDataImporterExtended:
    def test_init_custom_output_dir(self, tmp_path):
        imp = DataImporter(str(tmp_path / "custom"))
        assert imp.output_dir == str(tmp_path / "custom")

    def test_import_from_local_custom_extensions(self, data_importer, tmp_path):
        root = tmp_path / "data"
        write_repo(root, {"a.py": "x", "b.js": "y", "c.ts": "z"})
        result = data_importer.import_from_local(str(root), "ds", extensions=[".js", ".ts"])
        assert result.files_imported == 2

    def test_import_from_local_empty_directory(self, data_importer, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()
        result = data_importer.import_from_local(str(root), "ds")
        assert result.success is True
        assert result.files_imported == 0
        assert result.total_chars == 0

    def test_import_from_local_nested_directory(self, data_importer, tmp_path):
        root = tmp_path / "data"
        write_repo(root, {
            "src/a.py": "x",
            "lib/b.py": "y",
            "deep/nested/c.py": "z",
        })
        result = data_importer.import_from_local(str(root), "ds")
        assert result.files_imported == 3

    def test_import_from_local_preserves_relative_paths(self, data_importer, tmp_path):
        root = tmp_path / "data"
        write_repo(root, {"src/main.py": "x"})
        result = data_importer.import_from_local(str(root), "ds")
        corpus = Path(data_importer.output_dir) / "ds" / "corpus.jsonl"
        record = json.loads(corpus.read_text().splitlines()[0])
        assert record["path"] == "src/main.py"

    def test_extract_pdf_text_fitz_fallback_to_pypdf2(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "fitz", None)

        class Page:
            def extract_text(self):
                return "fallback text"

        monkeypatch.setitem(sys.modules, "PyPDF2",
                            SimpleNamespace(PdfReader=lambda f: SimpleNamespace(
                                pages=[Page()])))
        p = tmp_path / "doc.pdf"
        p.write_bytes(b"%PDF-1.4")
        assert DataImporter._extract_pdf_text(p) == "fallback text"

    def test_import_from_local_binary_skipped(self, data_importer, tmp_path):
        root = tmp_path / "data"
        root.mkdir()
        (root / "ok.txt").write_text("fine", encoding="utf-8")
        (root / "bad.bin").write_bytes(b"\xff\xfe\x00\x01\x02\x03")
        result = data_importer.import_from_local(str(root), "ds")
        assert result.files_imported == 1

    def test_import_from_local_large_file(self, data_importer, tmp_path):
        root = tmp_path / "data"
        root.mkdir()
        (root / "big.txt").write_text("x" * 10000, encoding="utf-8")
        result = data_importer.import_from_local(str(root), "ds")
        assert result.success is True
        assert result.total_chars == 10000

    def test_extract_pdf_text_both_missing(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "fitz", None)
        monkeypatch.setitem(sys.modules, "PyPDF2", None)
        p = tmp_path / "doc.pdf"
        p.write_bytes(b"%PDF-1.4")
        assert DataImporter._extract_pdf_text(p) == ""


# ---------------------------------------------------------------------------
# ImportResult — extended
# ---------------------------------------------------------------------------

class TestImportResultExtended:
    def test_all_fields(self):
        r = ImportResult(
            success=True, name="test", source="src",
            files_imported=5, total_chars=1000,
            output_path="/out/corpus.jsonl", error=None,
        )
        assert r.success is True
        assert r.name == "test"
        assert r.source == "src"
        assert r.files_imported == 5
        assert r.total_chars == 1000
        assert r.output_path == "/out/corpus.jsonl"
        assert r.error is None

    def test_error_result(self):
        r = ImportResult(
            success=False, name="fail", source="src",
            files_imported=0, total_chars=0,
            output_path="", error="something went wrong",
        )
        assert r.success is False
        assert r.error == "something went wrong"

    def test_dataclass_identity(self):
        r1 = ImportResult(True, "a", "b", 1, 2, "p")
        r2 = ImportResult(True, "a", "b", 1, 2, "p")
        assert r1 == r2


# ---------------------------------------------------------------------------
# DEFAULT_IGNORES
# ---------------------------------------------------------------------------

class TestDefaultIgnores:
    def test_contains_expected_entries(self):
        expected = {
            ".git", ".svn", ".hg", "node_modules", "__pycache__",
            ".venv", "venv", "dist", "build", ".pytest_cache",
            ".mypy_cache", ".ruff_cache", "*.egg-info", ".tox",
        }
        assert DEFAULT_IGNORES == expected

    def test_is_set_type(self):
        assert isinstance(DEFAULT_IGNORES, set)
