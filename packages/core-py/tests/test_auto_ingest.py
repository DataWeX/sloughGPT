"""Tests for the auto-ingestion pipeline: RepoScanner, CodeChunker, AutoIngester."""

import asyncio
import os
import sys
from pathlib import Path

import pytest

from domains.infrastructure.auto_ingest import (
    AutoIngester,
    CodeChunker,
    FileChunk,
    RepoScanner,
    DEFAULT_IGNORE_DIRS,
    main,
)


@pytest.fixture
def sample_tree(tmp_path):
    """Build a small repo-like tree for scanning tests."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main():\n    return 1\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("hello world\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("module.exports = 1;")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    (tmp_path / "package-lock.json").write_text("{}")
    return tmp_path


class TestRepoScanner:
    def test_scans_relevant_files_only(self, sample_tree):
        scanner = RepoScanner(root_path=str(sample_tree))
        found = {str(p.relative_to(sample_tree)) for p, _ in scanner.iter_files()}
        assert "src/main.py" in found
        assert "docs/guide.md" in found
        assert "node_modules/dep.js" not in found
        assert "__pycache__/x.pyc" not in found
        assert "package-lock.json" not in found

    def test_should_ignore_ignored_file(self, sample_tree):
        scanner = RepoScanner(root_path=str(sample_tree))
        assert scanner.should_ignore(sample_tree / "package-lock.json") is True

    def test_should_ignore_extension(self, tmp_path):
        scanner = RepoScanner(root_path=str(tmp_path))
        f = tmp_path / "lib.pyc"
        f.write_bytes(b"x")
        assert scanner.should_ignore(f) is True

    def test_should_ignore_dir_in_path(self, sample_tree):
        scanner = RepoScanner(root_path=str(sample_tree))
        assert scanner.should_ignore(sample_tree / "node_modules" / "dep.js") is True

    def test_custom_ignore_sets_override_defaults(self, tmp_path):
        scanner = RepoScanner(
            root_path=str(tmp_path),
            ignore_dirs={"custom_ignore"},
            ignore_exts={".zzz"},
            ignore_files={"custom.txt"},
        )
        assert DEFAULT_IGNORE_DIRS.isdisjoint(scanner.ignore_dirs) or "custom_ignore" in scanner.ignore_dirs
        f = tmp_path / "keep.py"
        f.write_text("x")
        assert scanner.should_ignore(f) is False

    def test_get_file_type(self, tmp_path):
        scanner = RepoScanner(root_path=str(tmp_path))
        cases = {
            "a.py": "python",
            "b.ts": "javascript",
            "c.rs": "rust",
            "d.md": "markdown",
            "e.json": "json",
            "f.yaml": "yaml",
            "g.sql": "sql",
            "h.css": "stylesheet",
            "i.html": "html",
            "j.unknownext": "unknown",
        }
        for name, expected in cases.items():
            assert scanner.get_file_type(tmp_path / name) == expected

    def test_guess_language(self, tmp_path):
        scanner = RepoScanner(root_path=str(tmp_path))
        assert scanner.guess_language(tmp_path / "a.py") == "python"
        assert scanner.guess_language(tmp_path / "a.ts") == "typescript"
        assert scanner.guess_language(tmp_path / "a.tsx") == "tsx"
        assert scanner.guess_language(tmp_path / "a.nope") == "text"

    def test_oversized_file_skipped_with_placeholder(self, tmp_path):
        big = tmp_path / "big.txt"
        big.write_text("x" * 100)
        scanner = RepoScanner(root_path=str(tmp_path), max_file_size=10)
        results = list(scanner.iter_files())
        assert len(results) == 1
        path, content = results[0]
        assert path == big
        assert content.startswith("[File too large")

    def test_unreadable_file_falls_back(self, tmp_path, monkeypatch):
        target = tmp_path / "weird.txt"
        target.write_bytes(b"\xff\xfe\x00")
        from pathlib import Path
        original = Path.read_text

        def fake_read_text(self, *args, **kwargs):
            if self.name == "weird.txt":
                raise UnicodeDecodeError("utf-8", b"", 0, 1, "bad")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", fake_read_text)
        scanner = RepoScanner(root_path=str(tmp_path))
        results = list(scanner.iter_files())
        assert len(results) == 1
        assert results[0][1] == "[Binary or unreadable file]"

    def test_iter_files_skips_stat_errors(self, tmp_path, monkeypatch):
        from pathlib import Path
        target = tmp_path / "locked.txt"
        target.write_text("x")
        original_stat = Path.stat
        original_is_file = Path.is_file

        def fake_stat(self, *args, **kwargs):
            if str(self) == str(target):
                raise OSError("permission denied")
            return original_stat(self, *args, **kwargs)

        def fake_is_file(self):
            if str(self) == str(target):
                return True
            return original_is_file(self)

        monkeypatch.setattr(Path, "stat", fake_stat)
        monkeypatch.setattr(Path, "is_file", fake_is_file)
        scanner = RepoScanner(root_path=str(tmp_path))
        assert list(scanner.iter_files()) == []


class TestCodeChunker:
    def test_chunk_text_has_deterministic_id(self):
        chunker = CodeChunker()
        c1 = chunker.chunk_text("abc", "/x.py", 0)
        c2 = chunker.chunk_text("abc", "/x.py", 0)
        assert c1.id == c2.id
        assert isinstance(c1, FileChunk)
        assert c1.chunk_index == 0
        assert c1.file_path == "/x.py"

    def test_chunk_text_different_index_different_id(self):
        chunker = CodeChunker()
        assert chunker.chunk_text("abc", "/x.py", 0).id != chunker.chunk_text("abc", "/x.py", 1).id

    def test_chunk_file_empty_content(self):
        chunker = CodeChunker()
        chunks = chunker.chunk_file("/f.txt", "")
        assert len(chunks) == 1

    def test_chunk_file_too_large_placeholder_single_chunk(self):
        chunker = CodeChunker()
        chunks = chunker.chunk_file("/f.txt", "[File too large: 9999 bytes — skipped]")
        assert len(chunks) == 1

    def test_chunk_code_splits_on_definitions(self):
        content = "def one():\n    pass\n\nclass Two:\n    def m(self):\n        pass\n"
        chunker = CodeChunker(chunk_size=10)
        chunks = chunker.chunk_file("/m.py", content)
        assert len(chunks) >= 2
        combined = "".join(c.content for c in chunks)
        assert "def one()" in combined
        assert "class Two" in combined

    def test_chunk_prose_splits_by_paragraphs(self):
        content = "Para one has enough text to fill the buffer boundary.\n\nSecond paragraph.\n\nThird paragraph."
        chunker = CodeChunker(chunk_size=40)
        chunks = chunker.chunk_file("/notes.txt", content)
        assert len(chunks) >= 2
        for c in chunks:
            assert "Para" in c.content or "Second" in c.content or "Third" in c.content

    def test_chunk_prose_single_chunk_when_small(self):
        chunker = CodeChunker()
        chunks = chunker.chunk_file("/notes.txt", "tiny")
        assert len(chunks) == 1
        assert chunks[0].content == "tiny"

    def test_chunk_prose_overlap_preserved(self):
        content = "A" * 600 + "\n\n" + "B" * 600
        chunker = CodeChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_file("/big.txt", content)
        assert len(chunks) >= 2

    def test_is_code_file(self):
        chunker = CodeChunker()
        assert chunker._is_code_file("/a.py") is True
        assert chunker._is_code_file("/a.md") is False


class TestAutoIngester:
    def test_stats_initial_state(self, tmp_path):
        ingester = AutoIngester(root_path=str(tmp_path))
        assert ingester.stats == {
            "files_scanned": 0,
            "files_ingested": 0,
            "chunks_created": 0,
            "errors": 0,
        }

    def test_build_metadata(self, tmp_path):
        ingester = AutoIngester(root_path=str(tmp_path))
        (tmp_path / "mod.py").write_text("x")
        chunk = ingester.chunker.chunk_text("code", str(tmp_path / "mod.py"), 3)
        meta = ingester.build_metadata(chunk)
        assert meta["file"] == "mod.py"
        assert meta["file_name"] == "mod.py"
        assert meta["extension"] == "py"
        assert meta["file_type"] == "python"
        assert meta["language"] == "python"
        assert meta["chunk_index"] == 3
        assert meta["repo"] == tmp_path.name

    def test_build_metadata_nested(self, tmp_path):
        (tmp_path / "src").mkdir()
        ingester = AutoIngester(root_path=str(tmp_path))
        chunk = ingester.chunker.chunk_text("code", str(tmp_path / "src" / "a.ts"), 0)
        meta = ingester.build_metadata(chunk)
        assert meta["file"] == os.path.join("src", "a.ts")

    @pytest.mark.asyncio
    async def test_ingest_dry_run_returns_stats(self, tmp_path):
        (tmp_path / "a.py").write_text("def f():\n    return 1\n")
        ingester = AutoIngester(root_path=str(tmp_path))
        stats = await ingester.ingest(dry_run=True)
        assert stats["files_scanned"] == 1
        assert stats["chunks_created"] >= 1
        assert stats["errors"] == 0

    @pytest.mark.asyncio
    async def test_ingest_with_vector_store(self, tmp_path, monkeypatch):
        (tmp_path / "a.py").write_text("def f():\n    return 1\n")
        ingester = AutoIngester(root_path=str(tmp_path))

        class FakeStore:
            def __init__(self):
                self.upserted = []
                self.connected = False

            async def connect(self):
                self.connected = True

            async def upsert(self, entries):
                self.upserted.extend(entries)
                return len(entries)

        store = FakeStore()

        async def fake_get_vector_store():
            return store

        monkeypatch.setattr(ingester, "get_vector_store", fake_get_vector_store)
        stats = await ingester.ingest(dry_run=False)
        assert stats["files_scanned"] == 1
        assert stats["files_ingested"] >= 1
        assert store.upserted and store.upserted[0].text.startswith("def f()")

    @pytest.mark.asyncio
    async def test_ingest_without_store_is_dry_run(self, tmp_path, monkeypatch):
        (tmp_path / "a.py").write_text("x = 1\n")
        ingester = AutoIngester(root_path=str(tmp_path))

        async def no_store():
            return None

        monkeypatch.setattr(ingester, "get_vector_store", no_store)
        stats = await ingester.ingest(dry_run=False)
        assert stats["chunks_created"] >= 1
        assert stats["files_ingested"] == 0

    @pytest.mark.asyncio
    async def test_ingest_chunk_error_increments_stats(self, tmp_path, monkeypatch):
        (tmp_path / "a.py").write_text("def f():\n    return 1\n")
        ingester = AutoIngester(root_path=str(tmp_path))

        def boom(path, content):
            raise RuntimeError("chunk fail")

        monkeypatch.setattr(ingester.chunker, "chunk_file", boom)

        async def no_store():
            return None

        monkeypatch.setattr(ingester, "get_vector_store", no_store)
        stats = await ingester.ingest(dry_run=False)
        assert stats["files_scanned"] == 1
        assert stats["errors"] == 1

    @pytest.mark.asyncio
    async def test_get_vector_store_in_memory(self, tmp_path):
        ingester = AutoIngester(root_path=str(tmp_path), provider="in_memory")
        store = await ingester.get_vector_store()
        assert store is not None
        assert await store.connect() is True

    @pytest.mark.asyncio
    async def test_ingest_single_file_with_store(self, tmp_path, monkeypatch):
        (tmp_path / "one.py").write_text("def f():\n    return 1\n")
        ingester = AutoIngester(root_path=str(tmp_path), provider="in_memory")
        count = await ingester.ingest_single_file(str(tmp_path / "one.py"))
        assert count >= 1
        assert ingester.stats["files_scanned"] == 1
        assert ingester.stats["chunks_created"] >= 1

    def test_query_relevant_with_store(self, tmp_path, monkeypatch):
        (tmp_path / "doc.txt").write_text("knowledge about cats\n")
        ingester = AutoIngester(root_path=str(tmp_path), provider="in_memory")
        shared = {}

        async def fake_get_store():
            from domains.inference.vector_store import create_vector_store
            if shared.get("store") is None:
                shared["store"] = await create_vector_store(provider="in_memory", dimension=384)
            return shared["store"]

        monkeypatch.setattr(ingester, "get_vector_store", fake_get_store)

        async def seed():
            await ingester.ingest_single_file(str(tmp_path / "doc.txt"))

        asyncio.run(seed())
        results = ingester.query_relevant("cats", top_k=5)
        assert isinstance(results, list)
        assert len(results) >= 1
        assert results[0]["file"] == "doc.txt"

    def test_main_dry_run(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(sys, "argv",
                            ["auto_ingest", "--path", str(tmp_path), "--dry-run"])
        with caplog.at_level("INFO"):
            asyncio.run(main())
        assert "Done" in caplog.text

    def test_main_file(self, tmp_path, monkeypatch, caplog):
        (tmp_path / "one.py").write_text("def f():\n    return 1\n")
        monkeypatch.setattr(sys, "argv",
                            ["auto_ingest", "--path", str(tmp_path), "--file", str(tmp_path / "one.py")])
        with caplog.at_level("INFO"):
            asyncio.run(main())
        assert "Ingested" in caplog.text

    def test_module_main_block(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(sys, "argv", ["auto_ingest", "--path", str(tmp_path), "--dry-run"])
        module_path = os.path.abspath(os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "domains",
                         "infrastructure", "auto_ingest.py")))
        with open(module_path) as fh:
            source = fh.read()
        with caplog.at_level("INFO"):
            exec(compile(source, module_path, "exec"), {"__name__": "__main__", "__file__": module_path})
        assert "Done" in caplog.text

    @pytest.mark.asyncio
    async def test_ingest_single_file_missing_returns_zero(self, tmp_path):
        ingester = AutoIngester(root_path=str(tmp_path))
        assert await ingester.ingest_single_file(str(tmp_path / "ghost.py")) == 0

    @pytest.mark.asyncio
    async def test_ingest_single_file_no_store_returns_chunk_count(self, tmp_path, monkeypatch):
        (tmp_path / "one.py").write_text("def f():\n    return 1\n")
        ingester = AutoIngester(root_path=str(tmp_path))

        async def no_store():
            return None

        monkeypatch.setattr(ingester, "get_vector_store", no_store)
        count = await ingester.ingest_single_file(str(tmp_path / "one.py"))
        assert count >= 1
        assert ingester.stats["files_scanned"] == 1

    def test_query_relevant_no_store_returns_empty(self, tmp_path, monkeypatch):
        ingester = AutoIngester(root_path=str(tmp_path))

        async def no_store():
            return None

        monkeypatch.setattr(ingester, "get_vector_store", no_store)
        assert ingester.query_relevant("anything") == []
