"""Tests for scripts/build_corpus.py — acquire → crunch → seal wiring (offline)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from domain.training._internal.cruncher import SnapshotStore, trainer_data_path
from domain.training._internal.cruncher.snapshot import normalize_source
from domain.training._internal.dataset_manifest import load_manifest, resolve_training_data_path

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
TEXT_B = (
    "Photosynthesis converts sunlight into chemical energy stored in glucose. "
    "Chloroplasts in plant cells contain chlorophyll, the pigment that absorbs "
    "light most efficiently in the blue and red portions of the spectrum."
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


# ── normalize_source (producer fix) ───────────────────────────────────


class TestNormalizeSource:
    def test_plain_string_becomes_local_path(self):
        assert normalize_source("unit-test") == {"type": "local_path", "uri": "unit-test"}

    def test_github_prefix(self):
        assert normalize_source("github:org/repo") == {"type": "github", "uri": "org/repo"}

    def test_hf_prefixes(self):
        assert normalize_source("hf:org/ds") == {"type": "huggingface", "uri": "org/ds"}
        assert normalize_source("huggingface:org/ds") == {
            "type": "huggingface",
            "uri": "org/ds",
        }

    def test_full_https_url(self):
        url = "https://example.com/page"
        assert normalize_source(url) == {"type": "https", "uri": url}

    def test_url_prefix(self):
        assert normalize_source("url:https://x.com") == {
            "type": "https",
            "uri": "https://x.com",
        }

    def test_dict_passthrough_with_required_check(self):
        src = {"type": "github", "uri": "org/repo", "license_spdx": "MIT"}
        assert normalize_source(src) == src
        with pytest.raises(Exception, match="missing"):
            normalize_source({"type": "github"})


# ── source_entry / record readers ─────────────────────────────────────


class TestSourceEntry:
    def test_types_and_license(self):
        assert build_corpus_mod.source_entry("github", "https://github.com/o/r", "MIT") == {
            "type": "github",
            "uri": "https://github.com/o/r",
            "license_spdx": "MIT",
        }
        assert build_corpus_mod.source_entry("bawl", "x.jsonl") == {
            "type": "https",
            "uri": "x.jsonl",
        }
        assert build_corpus_mod.source_entry("hf", "org/ds")["type"] == "huggingface"
        assert build_corpus_mod.source_entry("local", "f.txt")["type"] == "local_path"


class TestRecordReaders:
    def test_jsonl_prefers_text_then_content(self, tmp_path: Path):
        p = _write_jsonl(
            tmp_path / "pages.jsonl",
            [
                {"text": "from text", "content": "ignored"},
                {"content": "from content"},
                {"url": "x"},
            ],
        )
        assert list(build_corpus_mod._records_from_jsonl(p)) == ["from text", "from content"]

    def test_jsonl_bad_line_names_location(self, tmp_path: Path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"text": "ok"}\n{broken\n', encoding="utf-8")
        with pytest.raises(SystemExit, match="bad.jsonl:2"):
            list(build_corpus_mod._records_from_jsonl(p))

    def test_plain_text_single_record(self, tmp_path: Path):
        p = tmp_path / "doc.txt"
        p.write_text(TEXT_A, encoding="utf-8")
        assert list(build_corpus_mod._records_from_text_file(p)) == [TEXT_A]


# ── build_corpus end-to-end (offline) ─────────────────────────────────


class TestBuildCorpus:
    def test_local_txt_seals_valid_snapshot(self, tmp_path: Path):
        src = tmp_path / "doc.txt"
        src.write_text(TEXT_A + "\n\n" + TEXT_B, encoding="utf-8")
        store_root = tmp_path / "crunched"
        manifest_path = build_corpus_mod.build_corpus(
            "demo_txt", "local", str(src), store_root=store_root
        )
        m = load_manifest(manifest_path)
        assert m["dataset_id"] == "demo_txt"
        assert m["version"] == 1
        assert m["sources"] == [{"type": "local_path", "uri": str(src)}]
        text_path, _ = resolve_training_data_path(manifest_path)
        # A plain .txt file is one record (see _records_from_text_file) — it
        # lands in train.txt as a single line-run, not one line per paragraph.
        assert text_path.read_text(encoding="utf-8") == TEXT_A + "\n\n" + TEXT_B + "\n"
        # Sealed manifest passes the shipped validator CLI.
        proc = subprocess.run(
            [sys.executable, str(_SCRIPT.parent / "validate_dataset_manifest.py"), manifest_path],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr

    def test_bawl_jsonl_source(self, tmp_path: Path):
        crawl = _write_jsonl(
            tmp_path / "crawl.jsonl",
            [
                {"url": "https://a.com", "text": TEXT_A},
                {"url": "https://b.com", "text": TEXT_B},
            ],
        )
        manifest_path = build_corpus_mod.build_corpus(
            "demo_bawl", "bawl", str(crawl), store_root=tmp_path / "crunched"
        )
        m = load_manifest(manifest_path)
        assert m["sources"][0]["type"] == "https"
        text_path, _ = resolve_training_data_path(manifest_path)
        assert len(text_path.read_text(encoding="utf-8").splitlines()) == 2

    def test_license_recorded_when_given(self, tmp_path: Path):
        src = tmp_path / "doc.txt"
        src.write_text(TEXT_A, encoding="utf-8")
        manifest_path = build_corpus_mod.build_corpus(
            "demo_lic",
            "local",
            str(src),
            license_spdx="CC-BY-4.0",
            store_root=tmp_path / "crunched",
        )
        assert load_manifest(manifest_path)["sources"][0]["license_spdx"] == "CC-BY-4.0"

    def test_second_seal_dedups_against_history(self, tmp_path: Path):
        store_root = tmp_path / "crunched"
        # jsonl so each record is one text — a whole .txt file is a single
        # record and can never match another file's fingerprint for dedup.
        src1 = _write_jsonl(tmp_path / "a.jsonl", [{"text": TEXT_A}, {"text": TEXT_B}])
        build_corpus_mod.build_corpus("demo_dd", "local", str(src1), store_root=store_root)

        # v2 re-ingests TEXT_A (dropped as cross-snapshot duplicate) + new TEXT_C.
        text_c = (
            "Ribosomes translate messenger RNA into polypeptide chains by "
            "reading codons in sets of three nucleotides, a mechanism conserved "
            "across bacteria and eukaryotes alike."
        )
        src2 = _write_jsonl(tmp_path / "b.jsonl", [{"text": TEXT_A}, {"text": text_c}])
        manifest_path = build_corpus_mod.build_corpus(
            "demo_dd", "local", str(src2), store_root=store_root
        )
        m = load_manifest(manifest_path)
        assert m["version"] == 2
        text_path, _ = resolve_training_data_path(manifest_path)
        lines = text_path.read_text(encoding="utf-8").splitlines()
        assert lines == [text_c]  # TEXT_A deduped against v1 history

    def test_trainer_data_path_contract(self, tmp_path: Path):
        store = SnapshotStore(tmp_path / "crunched")
        from domain.training._internal.cruncher import Cruncher

        ref = Cruncher("adapter_demo", store=store).crunch([TEXT_A], sources=["unit"])
        path = trainer_data_path(ref)
        assert Path(path) == ref.manifest_path
