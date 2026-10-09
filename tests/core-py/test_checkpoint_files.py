"""Tests for domain.training._internal.checkpoint_files (ownership predicate).

The regression these pin: three operations must agree about a checkpoint's
identity, and the delete side is the one that used to disagree — a shared
``checkpoint_dir`` was swept directory-wide, so a finishing run deleted other
jobs' checkpoints and their Resume then failed on a file that had been valid
when it was recorded.
"""

from __future__ import annotations

import os

import pytest

from domain.training._internal.checkpoint_files import (
    STEM_MAX_LEN,
    is_owned,
    is_temp_artifact,
    list_owned,
    sanitize_stem,
    stem_for,
)

# ── sanitize_stem ─────────────────────────────────────────────────────────


class TestSanitizeStem:
    def test_replaces_path_separators_and_spaces(self):
        assert sanitize_stem("my dataset/v2") == "my_dataset_v2"

    def test_keeps_alnum_hyphen_underscore(self):
        assert sanitize_stem("api_conversations-1790_v2") == "api_conversations-1790_v2"

    def test_empty_and_none_are_empty_string(self):
        assert sanitize_stem("") == ""
        assert sanitize_stem(None) == ""

    def test_truncates_to_the_single_documented_cap(self):
        assert len(sanitize_stem("x" * 500)) == STEM_MAX_LEN == 120

    def test_cap_argument_overrides_the_default(self):
        assert sanitize_stem("x" * 500, cap=10) == "x" * 10

    def test_shorter_than_cap_is_untouched(self):
        # The length matters only because two writers used to disagree (80 vs
        # 120): a reader re-sanitising must return what the writer stored.
        stem = "a" * 80
        assert sanitize_stem(stem) == stem

    def test_unicode_alnum_survives(self):
        # str.isalnum() is Unicode-aware — non-ASCII never degrades to "_".
        assert sanitize_stem("café") == "café"

    def test_is_idempotent(self):
        once = sanitize_stem("my dataset v2")
        assert sanitize_stem(once) == once


# ── stem_for ──────────────────────────────────────────────────────────────


class TestStemFor:
    def test_uses_the_dataset_paths_parent_directory(self):
        assert stem_for("/repo/datasets/python_flask/corpus.jsonl") == "python_flask"

    def test_sanitises_a_parent_directory_containing_a_space(self):
        # The write path used a RAW Path(...).parent.name, so "my dataset"
        # produced a filename no sanitising reader could ever own.
        assert stem_for("/repo/data/my dataset/corpus.jsonl") == "my_dataset"

    def test_empty_data_path_falls_back_to_the_soul_name(self):
        assert stem_for("", "sloughgpt") == "sloughgpt"

    def test_bare_filename_has_no_usable_parent_and_falls_back(self):
        assert stem_for("corpus.txt", "sloughgpt") == "sloughgpt"

    def test_never_returns_an_empty_stem(self):
        # An empty stem makes is_owned match nothing, which would silently
        # disable retention entirely.
        for data_path in ("", "corpus.txt", "///", " "):
            assert stem_for(data_path, "") != ""
            assert stem_for(data_path, "assistant") != ""

    def test_truncates_to_the_shared_cap(self):
        assert len(stem_for("/repo/" + "d" * 400 + "/corpus.txt")) == STEM_MAX_LEN


# ── is_owned ──────────────────────────────────────────────────────────────


class TestIsOwned:
    def test_exact_stem_matches(self):
        assert is_owned("run_1700000000.soul", "run")

    def test_stem_underscore_suffix_matches(self):
        assert is_owned("run_1700000000.soul", "run") is True
        assert is_owned("run_2026_01.soul", "run") is True

    def test_a_different_job_sharing_a_prefix_does_not_match(self):
        # "api_conversations" must not claim "api_conversations_other_1.soul",
        # and "apiconversations" is a different job entirely. The underscore is
        # the boundary.
        assert is_owned("api_conversations_other_1.soul", "api_conversations")
        assert not is_owned("apiconversations_1790.soul", "api_conversations")
        assert not is_owned("api_conversationsX_1790.soul", "api_conversations")

    def test_none_or_empty_stem_owns_nothing(self):
        assert not is_owned("run_1700000000.soul", None)
        assert not is_owned("run_1700000000.soul", "")


# ── is_temp_artifact ──────────────────────────────────────────────────────


class TestIsTempArtifact:
    @pytest.mark.parametrize("name", ["x.tmp", "run_1790.tmp.npz", "abcDEF123.tmp"])
    def test_both_temp_spellings_are_recognised(self, name):
        assert is_temp_artifact(name)

    def test_finished_checkpoints_are_not_temp(self):
        assert not is_temp_artifact("run_1790.soul")
        assert not is_temp_artifact("run_1790.npz")

    def test_a_temp_that_looks_owned_is_still_excluded(self):
        # "run_1790.tmp.npz" has stem "run_1790.tmp", which startswith "run_"
        # — so ownership alone would admit a half-written file to both the
        # resume scan and the retention window.
        assert is_owned("run_1790.tmp.npz", "run")
        assert is_temp_artifact("run_1790.tmp.npz")


# ── list_owned ────────────────────────────────────────────────────────────


class TestListOwned:
    @pytest.fixture
    def directory(self, tmp_path):
        d = tmp_path / "shared"
        d.mkdir()
        return d

    @staticmethod
    def _touch(directory, name, mtime):
        p = directory / name
        p.write_bytes(b"x")
        os.utime(p, (mtime, mtime))
        return p

    def test_missing_directory_yields_empty_list(self, tmp_path):
        # The caller decides whether an absent directory is an error.
        assert list_owned(tmp_path / "nope", "jobA") == []

    def test_filters_to_one_jobs_files(self, directory):
        self._touch(directory, "jobA_1.soul", 100)
        self._touch(directory, "jobB_1.soul", 200)
        assert [p.name for p in list_owned(directory, "jobA")] == ["jobA_1.soul"]

    def test_excludes_temp_artifacts(self, directory):
        self._touch(directory, "jobA_1.soul", 100)
        self._touch(directory, "jobA_2.tmp.npz", 999)
        assert [p.name for p in list_owned(directory, "jobA")] == ["jobA_1.soul"]

    def test_newest_first_with_name_tiebreak(self, directory):
        self._touch(directory, "jobA_1.soul", 100)
        self._touch(directory, "jobA_2.soul", 300)
        self._touch(directory, "jobA_3.soul", 200)
        assert [p.name for p in list_owned(directory, "jobA")] == [
            "jobA_2.soul",
            "jobA_3.soul",
            "jobA_1.soul",
        ]

    def test_stem_none_applies_no_ownership_filter(self, directory):
        self._touch(directory, "jobA_1.soul", 100)
        self._touch(directory, "jobB_1.soul", 200)
        assert len(list_owned(directory, None)) == 2

    def test_suffixes_default_to_soul_and_npz(self, directory):
        self._touch(directory, "jobA_1.soul", 100)
        self._touch(directory, "jobA_1.npz", 200)
        assert len(list_owned(directory, "jobA")) == 2

    def test_suffixes_can_be_narrowed_to_what_retention_writes(self, directory):
        # Retention keeps its historical .soul-only scope: the .npz fallback is
        # not a file this run wrote, and sweeping it up would change what
        # survives a prune.
        self._touch(directory, "jobA_1.soul", 100)
        self._touch(directory, "jobA_1.npz", 200)
        only_soul = list_owned(directory, "jobA", suffixes=(".soul",))
        assert [p.name for p in only_soul] == ["jobA_1.soul"]


# ── writer / reader agreement (the live bug) ──────────────────────────────


class TestWriterReaderAgreement:
    def test_a_parent_directory_with_a_space_yields_an_ownable_file(self, tmp_path):
        dataset = tmp_path / "my dataset" / "corpus.jsonl"
        dataset.parent.mkdir(parents=True)
        dataset.write_text('{"text": "x"}\n', encoding="utf-8")

        stem = stem_for(str(dataset), "sloughgpt")
        written = tmp_path / f"{stem}_1790000000.soul"
        written.write_bytes(b"x")

        assert stem == "my_dataset"
        assert is_owned(written, stem)

    def test_the_pre_fix_raw_name_is_not_ownable(self, tmp_path):
        # Documents why the writer had to change: naming the file after a raw
        # parent.name put a space in it, and the sanitising reader then could
        # not match it — the resume scan came back empty with nothing to report.
        dataset = tmp_path / "my dataset" / "corpus.jsonl"
        dataset.parent.mkdir(parents=True)
        dataset.write_text("x", encoding="utf-8")

        stem = stem_for(str(dataset), "sloughgpt")
        raw_named = tmp_path / "my dataset_1790000000.soul"
        raw_named.write_bytes(b"x")

        assert not is_owned(raw_named, stem)

    def test_scan_and_prune_go_through_the_same_glob(self, tmp_path):
        # One function, so read and delete cannot drift apart again.
        d = tmp_path / "shared"
        d.mkdir()
        (d / "jobA_1.soul").write_bytes(b"x")
        (d / "jobB_1.soul").write_bytes(b"x")

        scanned = list_owned(d, "jobA")
        assert [p.name for p in scanned] == ["jobA_1.soul"]
        # Retention narrows only the suffix set, never the ownership set.
        assert [p.name for p in list_owned(d, "jobA", suffixes=(".soul",))] == [
            p.name for p in scanned
        ]
