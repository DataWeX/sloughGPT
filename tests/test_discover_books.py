"""Tests for scripts/discover_books.py — algorithmized book discovery
(card 20261116_095846). All HTTP is mocked; no network in this suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import discover_books as db  # noqa: E402


def _book(
    book_id: int,
    title: str,
    *,
    plain: bool = True,
    lang: str = "en",
    subjects: list[str] | None = None,
    downloads: int = 100,
) -> dict:
    formats: dict[str, str] = {"text/html": f"https://x/{book_id}.html"}
    if plain:
        formats["text/plain; charset=utf-8"] = f"https://x/{book_id}.txt"
    return {
        "id": book_id,
        "title": title,
        "authors": [{"name": "A. Author"}],
        "subjects": subjects or [],
        "languages": [lang],
        "formats": formats,
        "download_count": downloads,
    }


class TestSlugify:
    def test_basic(self):
        assert db.slugify("Alice's Adventures in Wonderland") == "alice-s-adventures-in-wonderland"

    def test_strips_non_ascii(self):
        assert db.slugify("Les Misérables") == "les-miserables"

    def test_collapses_separators(self):
        assert db.slugify("  a -- b  ") == "a-b"

    def test_empty(self):
        assert db.slugify("!!!") == ""


class TestDiscover:
    def test_single_page(self, monkeypatch):
        calls: list[str] = []

        def fake_get(url: str) -> dict:
            calls.append(url)
            return {"results": [_book(1, "One")], "next": None}

        monkeypatch.setattr(db, "_http_get_json", fake_get)
        out, src = db.discover("philosophy")
        assert len(out) == 1
        assert src == "gutendex"
        assert len(calls) == 1
        assert "search=philosophy" in calls[0]
        assert "page=1" in calls[0]

    def test_follows_pagination(self, monkeypatch):
        def fake_get(url: str) -> dict:
            if "page=1" in url:
                return {"results": [_book(1, "One")], "next": "https://gutendex.com/books?page=2"}
            return {"results": [_book(2, "Two")], "next": None}

        monkeypatch.setattr(db, "_http_get_json", fake_get)
        out, _src = db.discover("science", max_pages=5)
        assert [b["id"] for b in out] == [1, 2]

    def test_max_pages_caps_requests(self, monkeypatch):
        calls = 0

        def fake_get(url: str) -> dict:
            nonlocal calls
            calls += 1
            return {"results": [_book(calls, f"B{calls}")], "next": f"https://x?page={calls + 1}"}

        monkeypatch.setattr(db, "_http_get_json", fake_get)
        db.discover("q", max_pages=2)
        assert calls == 2


class TestHtmlFallback:
    def _html_page(self, *entries: tuple[str, str, str]) -> str:
        blocks = []
        for book_id, title, extra in entries:
            blocks.append(
                f'<li class="booklink">'
                f'<a class="link" href="/ebooks/{book_id}" accesskey="3">'
                f'<span class="cell content">'
                f'<span class="title">{title}</span>'
                f'<span class="subtitle">{extra}</span>'
                f'<span class="extra">1,234 downloads</span>'
                f"</span></a></li>"
            )
        return "<html>" + "".join(blocks) + "</html>"

    def test_parse_booklinks(self, monkeypatch):
        html = self._html_page((174, "The Picture of Dorian Gray", "Oscar Wilde"))
        monkeypatch.setattr(db, "_http_get_text", lambda url: html)
        out, src = db.discover("dorian", source="gutenberg")
        assert src == "gutenberg"
        assert len(out) == 1
        book = out[0]
        assert book["id"] == 174
        assert book["title"] == "The Picture of Dorian Gray"
        assert book["authors"][0]["name"] == "Oscar Wilde"
        assert book["download_count"] == 1234
        assert book["languages"] is None and book["formats"] is None

    def test_falls_back_when_gutendex_down(self, monkeypatch, capsys):
        def boom(url):
            raise TimeoutError("hang")

        html = self._html_page((205, "Walden", "Henry David Thoreau"))
        monkeypatch.setattr(db, "_http_get_json", boom)
        monkeypatch.setattr(db, "_http_get_text", lambda url: html)
        out, src = db.discover("walden", source="auto")
        assert src == "gutenberg"
        assert [b["id"] for b in out] == [205]
        assert "falling back" in capsys.readouterr().err

    def test_gutendex_source_raises_instead_of_falling_back(self, monkeypatch):
        def boom(url):
            raise TimeoutError("hang")

        monkeypatch.setattr(db, "_http_get_json", boom)
        with pytest.raises(TimeoutError):
            db.discover("x", source="gutendex")

    def test_html_pagination_follows_start_index(self, monkeypatch):
        calls: list[str] = []

        def fake_text(url: str) -> str:
            calls.append(url)
            return self._html_page(*[(i, f"B{i}", "") for i in range(25)])

        monkeypatch.setattr(db, "_http_get_text", fake_text)
        out, _src = db.discover("q", max_pages=3, source="gutenberg")
        assert len(calls) == 3, "every page full -> run to max_pages"
        assert "start_index=26" in calls[1]
        assert "start_index=51" in calls[2]
        assert len(out) == 75

    def test_html_pagination_stops_on_partial_page(self, monkeypatch):
        calls: list[str] = []

        def fake_text(url: str) -> str:
            calls.append(url)
            n = 25 if len(calls) == 1 else 7  # second page short -> stop
            return self._html_page(*[(i, f"B{i}", "") for i in range(n)])

        monkeypatch.setattr(db, "_http_get_text", fake_text)
        out, _src = db.discover("q", max_pages=5, source="gutenberg")
        assert len(calls) == 2, "a partial page means no further results"
        assert len(out) == 32


class TestEnrichFromBookPage:
    def test_resolves_language_and_plain_text(self, monkeypatch):
        page = (
            '<tr property="dcterms:language" content="en"><th>Language</th>'
            "<td>English</td></tr>"
            '<a href="/ebooks/174.txt.utf-8">Plain Text</a>'
        )
        monkeypatch.setattr(db, "_http_get_text", lambda url: page)
        book = {"id": 174, "languages": None, "formats": None}
        db.enrich_from_book_page(book)
        assert book["languages"] == ["en"]
        assert db._plain_text_url(book["formats"]).endswith("pg174.txt")

    def test_no_plain_text_yields_empty_formats(self, monkeypatch):
        page = '<tr property="dcterms:language" content="fr"></tr>'
        monkeypatch.setattr(db, "_http_get_text", lambda url: page)
        book = {"id": 9, "languages": None, "formats": None}
        db.enrich_from_book_page(book)
        assert book["languages"] == ["fr"]
        assert book["formats"] == {}

    def test_fetch_failure_keeps_unknown(self, monkeypatch):
        def boom(url):
            raise TimeoutError("hang")

        monkeypatch.setattr(db, "_http_get_text", boom)
        book = {"id": 9, "languages": None, "formats": None}
        db.enrich_from_book_page(book)  # must not raise
        assert book["languages"] is None and book["formats"] is None

    def test_never_overwrites_known_metadata(self, monkeypatch):
        monkeypatch.setattr(db, "_http_get_text", lambda url: "<html></html>")
        book = {"id": 9, "languages": ["de"], "formats": {"text/html": "x"}}
        db.enrich_from_book_page(book)
        assert book["languages"] == ["de"]
        assert book["formats"] == {"text/html": "x"}

    def test_known_metadata_short_circuits_fetch(self, monkeypatch):
        def boom(url):
            raise AssertionError("must not fetch when metadata is known")

        monkeypatch.setattr(db, "_http_get_text", boom)
        book = {"id": 9, "languages": ["en"], "formats": {"text/plain": "x"}}
        db.enrich_from_book_page(book)


class TestMetadataRejectReason:
    def test_known_good_passes(self):
        assert db._metadata_reject_reason(_book(1, "A"), "en") is None

    def test_unknown_metadata_defers(self):
        book = {"id": 1, "title": "A", "languages": None, "formats": None}
        assert db._metadata_reject_reason(book, "en") is None

    def test_known_bad_language_rejects(self):
        assert db._metadata_reject_reason(_book(1, "A", lang="fr"), "en") == "language not en"

    def test_missing_plain_text_rejects(self):
        assert (
            db._metadata_reject_reason(_book(1, "A", plain=False), "en") == "no plain-text format"
        )

    def test_no_lang_filter_skips_language(self):
        assert db._metadata_reject_reason(_book(1, "A", lang="fr"), "") is None


class TestHttpRetry:
    def test_transient_timeout_retries_then_succeeds(self, monkeypatch):
        attempts = {"n": 0}

        class _Resp:
            def __init__(self, payload: bytes):
                self._payload = payload

            def read(self) -> bytes:
                return self._payload

        def fake_urlopen(req, timeout=None):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise TimeoutError("read timed out")
            return _Resp(b'{"results": [], "next": null}')

        monkeypatch.setattr(db.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(db.time, "sleep", lambda s: None)
        out = db._http_get_json("https://x/")
        assert out == {"results": [], "next": None}
        assert attempts["n"] == 3

    def test_persistent_failure_raises_after_retries(self, monkeypatch):
        attempts = {"n": 0}

        def fake_urlopen(req, timeout=None):
            attempts["n"] += 1
            raise TimeoutError("read timed out")

        monkeypatch.setattr(db.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(db.time, "sleep", lambda s: None)
        with pytest.raises(TimeoutError):
            db._http_get_json("https://x/")
        assert attempts["n"] == db.HTTP_RETRIES


class TestRank:
    def test_filters_missing_plain_text(self):
        ranked, rejected = db.rank(
            [_book(1, "Plain"), _book(2, "Html only", plain=False)],
            query="",
            lang="en",
            existing_slugs=set(),
        )
        assert [b["id"] for _, b, _ in ranked] == [1]
        assert rejected == [("Html only", "no plain-text format")]

    def test_filters_language(self):
        ranked, rejected = db.rank(
            [_book(1, "French", lang="fr")],
            query="",
            lang="en",
            existing_slugs=set(),
        )
        assert ranked == []
        assert rejected[0][1] == "language not en"

    def test_any_language_when_lang_empty(self):
        ranked, _ = db.rank(
            [_book(1, "French", lang="fr")],
            query="",
            lang="",
            existing_slugs=set(),
        )
        assert len(ranked) == 1

    def test_dedupes_existing_slug(self):
        ranked, rejected = db.rank(
            [_book(1, "Alice in Wonderland")],
            query="",
            lang="en",
            existing_slugs={"alice-in-wonderland"},
        )
        assert ranked == []
        assert rejected[0][1].startswith("already in data/")

    def test_dedupes_duplicate_ids(self):
        ranked, rejected = db.rank(
            [_book(1, "Same"), _book(1, "Same")],
            query="",
            lang="en",
            existing_slugs=set(),
        )
        assert len(ranked) == 1
        assert rejected == [("Same", "duplicate id")]

    def test_query_overlap_scores_higher(self):
        ranked, _ = db.rank(
            [
                _book(1, "Cooking for Everyone", subjects=["cooking"]),
                _book(2, "Meditations", subjects=["philosophy", "stoicism"]),
            ],
            query="philosophy",
            lang="en",
            existing_slugs=set(),
        )
        assert ranked[0][1]["id"] == 2, "subject overlap must beat download_count"

    def test_download_count_breaks_ties(self):
        ranked, _ = db.rank(
            [_book(1, "A", downloads=10), _book(2, "B", downloads=999)],
            query="",
            lang="en",
            existing_slugs=set(),
        )
        assert ranked[0][1]["id"] == 2


class TestValidateText:
    def test_accepts_normal_text(self):
        text = "Hello world.\n" * 1000
        assert db.validate_text(text, min_chars=5000, min_printable=0.9) is None

    def test_rejects_too_short(self):
        reason = db.validate_text("short", min_chars=5000, min_printable=0.9)
        assert reason is not None and "too short" in reason

    def test_rejects_binary_garbage(self):
        text = "".join(chr(i % 32) for i in range(6000))  # control chars, long enough
        reason = db.validate_text(text, min_chars=5000, min_printable=0.9)
        assert reason is not None and "printable ratio" in reason

    def test_empty_rejected(self):
        assert db.validate_text("", min_chars=1, min_printable=0.9) is not None


class TestWriteDataset:
    def test_writes_canonical_v1_layout(self, tmp_path):
        target = db.write_dataset(tmp_path, "my-book", "body text", {"title": "My Book"})
        assert (target / "input.txt").read_text() == "body text"
        meta = json.loads((target / "metadata.json").read_text())
        assert meta["title"] == "My Book"
        manifest = json.loads((target / "manifest.json").read_text())
        assert manifest["schema_version"] == "1.0"
        assert manifest["dataset_id"] == "my-book"
        assert manifest["splits"] == {"train": "input.txt"}

    def test_manifest_satisfies_v1_contract(self, tmp_path):
        from domain.training._internal.dataset_manifest import (
            load_manifest,
            resolve_training_data_path,
        )

        target = db.write_dataset(
            tmp_path,
            "my-book",
            "body text",
            {"source": "gutenberg:1", "source_type": "gutenberg", "text_url": "https://x/1.txt"},
        )
        resolved, manifest = resolve_training_data_path(target / "manifest.json")
        assert resolved == target / "input.txt"
        assert load_manifest(target / "manifest.json")["sources"] == ["https://x/1.txt"]

    def test_leaves_no_tmp_dir(self, tmp_path):
        db.write_dataset(tmp_path, "my-book", "x", {})
        assert not any(p.name.startswith(".") for p in tmp_path.iterdir())

    def test_refuses_overwrite(self, tmp_path):
        db.write_dataset(tmp_path, "my-book", "x", {})
        with pytest.raises(FileExistsError):
            db.write_dataset(tmp_path, "my-book", "y", {})

    def test_failure_cleans_tmp(self, tmp_path, monkeypatch):
        # simulate a crash mid-write: metadata write blows up
        real_write_text = Path.write_text

        def boom(self, *args, **kwargs):
            if self.name == "metadata.json":
                raise OSError("disk full")
            return real_write_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", boom)
        with pytest.raises(OSError):
            db.write_dataset(tmp_path, "my-book", "x", {})
        assert not any(tmp_path.iterdir()), "no partial dataset may survive a failed write"


class TestStandardize:
    def _legacy_dir(self, tmp_path, slug="old-book", text="legacy body"):
        d = tmp_path / slug
        d.mkdir()
        (d / f"{slug}.txt").write_text(text, encoding="utf-8")
        (d / "metadata.json").write_text(
            json.dumps({"source": "gutenberg:9", "source_type": "gutenberg", "title": "Old"}),
            encoding="utf-8",
        )
        return d

    def test_renames_and_writes_manifest(self, tmp_path):
        d = self._legacy_dir(tmp_path)
        assert db.standardize_dir(d) == "renamed + manifest written"
        assert (d / "input.txt").read_text() == "legacy body"
        assert not (d / "old-book.txt").exists()
        manifest = json.loads((d / "manifest.json").read_text())
        assert manifest["dataset_id"] == "old-book"
        assert manifest["sources"] == ["gutenberg:9"]
        assert manifest["splits"] == {"train": "input.txt"}
        assert (d / "metadata.json").exists(), "metadata sidecar must never be deleted"

    def test_idempotent_second_run_preserves_bytes(self, tmp_path):
        d = self._legacy_dir(tmp_path)
        db.standardize_dir(d)
        before = (d / "input.txt").read_bytes()
        assert db.standardize_dir(d) == "already conformant"
        assert (d / "input.txt").read_bytes() == before

    def test_skips_foreign_dirs(self, tmp_path):
        foreign = tmp_path / "someone-elses"
        foreign.mkdir()
        (foreign / "data.txt").write_text("keep", encoding="utf-8")
        (foreign / "metadata.json").write_text(json.dumps({"title": "x"}), encoding="utf-8")
        assert db.standardize_dir(foreign) == "skipped (not acquired by this tool)"
        assert (foreign / "data.txt").exists()
        assert not (foreign / "manifest.json").exists()

    def test_skips_dirs_without_our_sidecar(self, tmp_path):
        d = tmp_path / "no-meta"
        d.mkdir()
        (d / "no-meta.txt").write_text("x", encoding="utf-8")
        assert db.standardize_dir(d) == "skipped (not acquired by this tool)"

    def test_skips_when_no_corpus_file(self, tmp_path):
        d = tmp_path / "empty-book"
        d.mkdir()
        (d / "metadata.json").write_text(
            json.dumps({"source": "gutenberg:1", "source_type": "gutenberg"}),
            encoding="utf-8",
        )
        assert db.standardize_dir(d) == "skipped (no corpus file)"

    def test_rewrites_invalid_manifest(self, tmp_path):
        from domain.training._internal.dataset_manifest import load_manifest

        d = self._legacy_dir(tmp_path)
        (d / "manifest.json").write_text("{not json", encoding="utf-8")
        assert db.standardize_dir(d) == "renamed + manifest written"
        assert load_manifest(d / "manifest.json")["dataset_id"] == "old-book"

    def test_valid_manifest_left_alone(self, tmp_path):
        target = db.write_dataset(
            tmp_path, "new-book", "x", {"source": "gutenberg:1", "source_type": "gutenberg"}
        )
        before = (target / "manifest.json").read_bytes()
        assert db.standardize_dir(target) == "already conformant"
        assert (target / "manifest.json").read_bytes() == before

    def test_run_standardize_report(self, tmp_path, capsys):
        self._legacy_dir(tmp_path, "old-book")
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        (foreign / "x.txt").write_text("y", encoding="utf-8")
        rc = db.run_standardize(argparse_ns({"data_dir": str(tmp_path)}))
        assert rc == 0
        out = capsys.readouterr().out
        assert "old-book" in out and "renamed + manifest written" in out
        assert "1 foreign dir(s) untouched" in out

    def test_run_standardize_missing_dir_errors(self, tmp_path, capsys):
        rc = db.run_standardize(argparse_ns({"data_dir": str(tmp_path / "nope")}))
        assert rc == 2
        assert "not a directory" in capsys.readouterr().err


class TestCliDispatch:
    def test_standardize_flag_routes_to_run_standardize(self, tmp_path, monkeypatch):
        seen = {}

        def fake_run_standardize(args):
            seen["data_dir"] = args.data_dir
            return 0

        monkeypatch.setattr(db, "run_standardize", fake_run_standardize)
        assert db.main(["--standardize", "--data-dir", str(tmp_path)]) == 0
        assert seen["data_dir"] == str(tmp_path)

    def test_missing_query_is_a_parser_error(self):
        with pytest.raises(SystemExit):
            db.main([])

    def test_standardize_rejects_query(self, tmp_path):
        with pytest.raises(SystemExit):
            db.main(["--standardize", "-q", "philosophy", "--data-dir", str(tmp_path)])


class TestRun:
    def _args(self, tmp_path, **overrides):
        base = dict(
            query=["philosophy"],
            max_pages=1,
            max_books=5,
            lang="en",
            min_chars=10,
            min_printable=0.9,
            data_dir=str(tmp_path),
            dry_run=False,
            source="auto",
            max_rejects=15,
        )
        base.update(overrides)
        return argparse_ns(base)

    def test_end_to_end_acquisition(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            db, "_http_get_json", lambda url: {"results": [_book(7, "Deep Thinking")], "next": None}
        )
        monkeypatch.setattr(db, "_download_text", lambda gid: "real text body " * 10)
        rc = db.run(self._args(tmp_path))
        assert rc == 0
        out = tmp_path / "deep-thinking" / "input.txt"
        assert out.exists()
        assert out.read_text() == "real text body " * 10
        meta = json.loads((out.parent / "metadata.json").read_text())
        assert meta["gutenberg_id"] == 7
        assert meta["source_type"] == "gutenberg"
        manifest = json.loads((out.parent / "manifest.json").read_text())
        assert manifest["dataset_id"] == "deep-thinking"
        report = capsys.readouterr().out
        assert "1 dataset(s)" in report

    def test_dry_run_writes_nothing(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            db, "_http_get_json", lambda url: {"results": [_book(7, "Deep Thinking")], "next": None}
        )
        monkeypatch.setattr(db, "_download_text", lambda gid: "real text body " * 10)
        rc = db.run(self._args(tmp_path, dry_run=True))
        assert rc == 0
        assert not any(tmp_path.iterdir()), "dry-run must not create files"
        assert "dry-run" in capsys.readouterr().out

    def test_rejected_download_not_written(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            db, "_http_get_json", lambda url: {"results": [_book(7, "Deep Thinking")], "next": None}
        )
        monkeypatch.setattr(db, "_download_text", lambda gid: None)
        rc = db.run(self._args(tmp_path))
        assert rc == 0
        assert not any(p.is_dir() for p in tmp_path.iterdir())
        assert "download failed" in capsys.readouterr().out

    def test_short_download_not_written(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            db, "_http_get_json", lambda url: {"results": [_book(7, "Tiny")], "next": None}
        )
        monkeypatch.setattr(db, "_download_text", lambda gid: "tiny")
        db.run(self._args(tmp_path))
        assert not any(p.is_dir() for p in tmp_path.iterdir())
        assert "too short" in capsys.readouterr().out

    def test_gutendex_failure_pinned_for_later_queries(self, tmp_path, monkeypatch, capsys):
        # query 1: gutendex times out -> fallback; query 2 must NOT re-probe
        gutendex_calls = {"n": 0}

        def fake_json(url: str) -> dict:
            gutendex_calls["n"] += 1
            raise TimeoutError("hang")

        def fake_text(url: str) -> str:
            if "/ebooks/search/" in url:
                blocks = "".join(
                    f'<li class="booklink"><a class="link" href="/ebooks/{i}">'
                    f'<span class="cell content"><span class="title">Book {i}</span>'
                    f"</span></a></li>"
                    for i in (1, 2)
                )
                return f"<html>{blocks}</html>"
            book_id = url.rsplit("/", 1)[-1]
            return (
                '<tr property="dcterms:language" content="en"></tr>'
                f'<a href="/ebooks/{book_id}.txt">Plain Text</a>'
            )

        monkeypatch.setattr(db, "_http_get_json", fake_json)
        monkeypatch.setattr(db, "_http_get_text", fake_text)
        monkeypatch.setattr(db, "_download_text", lambda gid: "body " * 10)
        rc = db.run(self._args(tmp_path, query=["philosophy", "science"], max_books=5))
        assert rc == 0
        assert gutendex_calls["n"] == 1, "second query must use the pinned fallback source"
        assert (tmp_path / "book-1").is_dir()
        assert (tmp_path / "book-2").is_dir()

    def test_existing_slug_skipped_without_download(self, tmp_path, monkeypatch):
        (tmp_path / "deep-thinking").mkdir()
        monkeypatch.setattr(
            db, "_http_get_json", lambda url: {"results": [_book(7, "Deep Thinking")], "next": None}
        )
        downloaded: list[int] = []
        monkeypatch.setattr(db, "_download_text", lambda gid: downloaded.append(gid) or "text")
        db.run(self._args(tmp_path))
        assert downloaded == [], "dedupe must happen before any download"

    def test_query_merge_scores_subjects(self, tmp_path, monkeypatch, capsys):
        # two queries, one book matching only the second — still acquired
        def fake_get(url: str) -> dict:
            if "search=science" in url:
                return {"results": [_book(9, "The Science of Getting Rich")], "next": None}
            return {"results": [], "next": None}

        monkeypatch.setattr(db, "_http_get_json", fake_get)
        monkeypatch.setattr(db, "_download_text", lambda gid: "body " * 10)
        db.run(self._args(tmp_path, query=["philosophy", "science"]))
        assert (tmp_path / "the-science-of-getting-rich").is_dir()

    def test_gutenberg_source_enriches_lazily(self, tmp_path, monkeypatch, capsys):
        # HTML source: search page yields 3 books, only the FIRST passes
        # enrichment (others lack plain text) — exactly one book page may
        # be fetched per acquisition attempt, not one per candidate up front
        pages_fetched: list[str] = []

        def fake_text(url: str) -> str:
            if "/ebooks/search/" in url:
                blocks = "".join(
                    f'<li class="booklink"><a class="link" href="/ebooks/{i}">'
                    f'<span class="cell content"><span class="title">Book {i}</span>'
                    f'<span class="subtitle">A. Author</span>'
                    f'<span class="extra">10 downloads</span></span></a></li>'
                    for i in (1, 2, 3)
                )
                return f"<html>{blocks}</html>"
            pages_fetched.append(url)
            book_id = url.rsplit("/", 1)[-1]
            if book_id == "1":
                return '<tr property="dcterms:language" content="en"></tr><a href="/ebooks/1.txt">'
            return '<tr property="dcterms:language" content="en"></tr>'  # no plain text

        monkeypatch.setattr(db, "_http_get_text", fake_text)
        monkeypatch.setattr(db, "_download_text", lambda gid: "body " * 10)
        rc = db.run(self._args(tmp_path, source="gutenberg", max_books=3))
        assert rc == 0
        assert len(pages_fetched) == 3, "stop fetching after fills or rejections exhaust"
        dirs = [p.name for p in tmp_path.iterdir() if p.is_dir()]
        assert dirs == ["book-1"], "books 2/3 lack plain text -> rejected after enrichment"
        report = capsys.readouterr().out
        assert "no plain-text format" in report


def argparse_ns(d: dict):
    import argparse

    return argparse.Namespace(**d)
