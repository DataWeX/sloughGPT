"""Tests for first-code-corpus additions to build_corpus: license provenance."""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_corpus.py"
_spec = importlib.util.spec_from_file_location("build_corpus", _SCRIPT)
assert _spec and _spec.loader
build_corpus_mod = importlib.util.module_from_spec(_spec)
sys.modules["build_corpus"] = build_corpus_mod
_spec.loader.exec_module(build_corpus_mod)

TEXT_A = (
    "The mitochondria is the powerhouse of the cell. It converts nutrients "
    "into ATP through oxidative phosphorylation, a process studied extensively "
    "in molecular biology since the mid twentieth century."
)


class TestNormalizeLicense:
    def test_known_alias(self):
        assert build_corpus_mod.normalize_license("mit") == "MIT"
        assert build_corpus_mod.normalize_license("Apache 2.0") == "Apache-2.0"
        assert build_corpus_mod.normalize_license("gpl-3.0-or-later") == "GPL-3.0-or-later"

    def test_already_spdx(self):
        assert build_corpus_mod.normalize_license("MIT") == "MIT"
        assert build_corpus_mod.normalize_license("Apache-2.0") == "Apache-2.0"

    def test_unknown_returns_none(self):
        assert build_corpus_mod.normalize_license("Weird Custom License v2") is None
        assert build_corpus_mod.normalize_license("") is None
        assert build_corpus_mod.normalize_license(None) is None


def _http_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestFetchHelpers:
    def test_github_license_found(self):
        with patch.object(
            urllib.request,
            "urlopen",
            return_value=_http_response({"license": {"spdx_id": "MIT"}}),
        ):
            assert build_corpus_mod.fetch_github_license("https://github.com/org/repo") == "MIT"

    def test_github_noassertion_is_none(self):
        with patch.object(
            urllib.request,
            "urlopen",
            return_value=_http_response({"license": {"spdx_id": "NOASSERTION"}}),
        ):
            assert build_corpus_mod.fetch_github_license("https://github.com/org/repo") is None

    def test_github_network_failure_is_none(self):
        with patch.object(
            urllib.request,
            "urlopen",
            side_effect=OSError("no network"),
        ):
            assert build_corpus_mod.fetch_github_license("https://github.com/org/repo") is None

    def test_github_bad_url_is_none(self):
        assert build_corpus_mod.fetch_github_license("not-a-url") is None

    def test_hf_license_from_card_data(self):
        payload = {"cardData": {"license": "apache-2.0"}}
        with patch.object(
            urllib.request,
            "urlopen",
            return_value=_http_response(payload),
        ):
            assert build_corpus_mod.fetch_hf_license("org/ds") == "Apache-2.0"

    def test_hf_license_list_form(self):
        payload = {"cardData": {"license": ["mit"]}}
        with patch.object(
            urllib.request,
            "urlopen",
            return_value=_http_response(payload),
        ):
            assert build_corpus_mod.fetch_hf_license("org/ds") == "MIT"

    def test_hf_missing_license_is_none(self):
        with patch.object(
            urllib.request,
            "urlopen",
            return_value=_http_response({"id": "org/ds"}),
        ):
            assert build_corpus_mod.fetch_hf_license("org/ds") is None


class TestBuildCorpusAutoLicense:
    def test_github_auto_license_lands_in_manifest(self, tmp_path: Path):
        src = tmp_path / "corpus.jsonl"
        src.write_text(json.dumps({"content": TEXT_A}) + "\n", encoding="utf-8")
        # Simulate acquire via the local path, but drive build_corpus with a
        # github source by pre-seeding the work dir and stubbing clone/export.
        work = tmp_path / "work"
        work.mkdir()

        def fake_acquire(source, src, *, work_dir):
            yield TEXT_A

        with (
            patch.object(build_corpus_mod, "acquire_records", fake_acquire),
            patch.object(
                build_corpus_mod,
                "fetch_github_license",
                return_value="MIT",
            ),
        ):
            manifest_path = build_corpus_mod.build_corpus(
                "code_demo",
                "github",
                "https://github.com/org/repo",
                store_root=tmp_path / "crunched",
                work_dir=work,
            )
        m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        assert m["sources"][0]["license_spdx"] == "MIT"
        assert m["sources"][0]["type"] == "github"

    def test_unknown_license_omitted_from_manifest(self, tmp_path: Path):
        work = tmp_path / "work"
        work.mkdir()

        def fake_acquire(source, src, *, work_dir):
            yield TEXT_A

        with (
            patch.object(build_corpus_mod, "acquire_records", fake_acquire),
            patch.object(build_corpus_mod, "fetch_github_license", return_value=None),
        ):
            manifest_path = build_corpus_mod.build_corpus(
                "code_demo2",
                "github",
                "https://github.com/org/unknown-repo",
                store_root=tmp_path / "crunched",
                work_dir=work,
            )
        m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        assert "license_spdx" not in m["sources"][0]

    def test_explicit_license_wins_over_fetch(self, tmp_path: Path):
        work = tmp_path / "work"
        work.mkdir()

        def fake_acquire(source, src, *, work_dir):
            yield TEXT_A

        with (
            patch.object(build_corpus_mod, "acquire_records", fake_acquire),
            patch.object(
                build_corpus_mod,
                "fetch_github_license",
                return_value="MIT",
            ),
        ):
            manifest_path = build_corpus_mod.build_corpus(
                "code_demo3",
                "github",
                "https://github.com/org/repo",
                license_spdx="Apache-2.0",
                store_root=tmp_path / "crunched",
                work_dir=work,
            )
        m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        assert m["sources"][0]["license_spdx"] == "Apache-2.0"
