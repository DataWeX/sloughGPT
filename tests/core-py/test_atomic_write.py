"""Tests for domain.infrastructure._internal.atomic_write (single save owner)."""

from __future__ import annotations

import pytest

from domain.infrastructure._internal.atomic_write import (
    TEMP_SUFFIX,
    atomic_write,
    atomic_writer,
    is_temp_name,
)


class TestIsTempName:
    @pytest.mark.parametrize("name", ["x.tmp", "run_1790.tmp.npz", "abc.tmp"])
    def test_both_spellings(self, name):
        assert is_temp_name(name)

    @pytest.mark.parametrize("name", ["x.soul", "x.npz", "x.soul.meta.json", "tmp"])
    def test_finished_files_are_not_temp(self, name):
        assert not is_temp_name(name)

    def test_the_documented_suffix_constant_is_what_the_predicate_uses(self):
        assert TEMP_SUFFIX == ".tmp"


class TestAtomicWrite:
    def test_creates_missing_parent_directories(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "out.txt"
        atomic_write(target, "hello")
        assert target.read_text(encoding="utf-8") == "hello"

    def test_accepts_bytes(self, tmp_path):
        target = tmp_path / "out.bin"
        atomic_write(target, b"\x00\x01")
        assert target.read_bytes() == b"\x00\x01"

    def test_str_payload_is_encoded(self, tmp_path):
        target = tmp_path / "out.txt"
        atomic_write(target, "café", encoding="utf-8")
        assert target.read_bytes() == "café".encode()

    def test_replaces_existing_content(self, tmp_path):
        target = tmp_path / "out.txt"
        target.write_text("old", encoding="utf-8")
        atomic_write(target, "new")
        assert target.read_text(encoding="utf-8") == "new"

    def test_leaves_no_temp_behind(self, tmp_path):
        target = tmp_path / "out.txt"
        atomic_write(target, "x")
        leftovers = [p.name for p in tmp_path.iterdir() if is_temp_name(p.name)]
        assert leftovers == []

    def test_returns_the_destination_never_the_temp(self, tmp_path):
        target = tmp_path / "out.txt"
        assert atomic_write(target, "x") == str(target)


class TestAtomicWriter:
    def test_body_writes_are_committed_on_clean_exit(self, tmp_path):
        target = tmp_path / "out.txt"
        with atomic_writer(target, binary=False, encoding="utf-8") as f:
            f.write("line")
        assert target.read_text(encoding="utf-8") == "line"

    def test_temp_is_created_beside_the_target(self, tmp_path):
        # A cross-device temp would make the rename non-atomic (EXDEV, or a
        # silent copy-then-delete) — the whole point of the primitive.
        # Note the handle's .name is the raw fd (os.fdopen), not a path, so
        # observe the directory rather than the handle.
        target = tmp_path / "out.txt"
        seen_during_write = []
        with atomic_writer(target) as f:
            f.write(b"x")
            seen_during_write = [p.name for p in tmp_path.iterdir()]
        assert any(is_temp_name(n) for n in seen_during_write), seen_during_write
        assert not any(is_temp_name(n) for n in (p.name for p in tmp_path.iterdir()))

    def test_exception_in_body_leaves_original_untouched(self, tmp_path):
        target = tmp_path / "out.txt"
        target.write_text("original", encoding="utf-8")
        with pytest.raises(RuntimeError, match="boom"):
            with atomic_writer(target, binary=False, encoding="utf-8") as f:
                f.write("partial")
                raise RuntimeError("boom")
        assert target.read_text(encoding="utf-8") == "original"

    def test_exception_in_body_creates_no_file_at_all(self, tmp_path):
        target = tmp_path / "out.txt"
        with pytest.raises(RuntimeError):
            with atomic_writer(target) as f:
                f.write(b"partial")
                raise RuntimeError("boom")
        assert not target.exists()
        assert [p.name for p in tmp_path.iterdir() if is_temp_name(p.name)] == []

    def test_text_mode_rejects_bytes_and_binary_mode_rejects_str(self, tmp_path):
        with atomic_writer(tmp_path / "a.txt", binary=False, encoding="utf-8") as f:
            with pytest.raises(TypeError):
                f.write(b"bytes")  # type: ignore[arg-type]
        with atomic_writer(tmp_path / "b.bin", binary=True) as f:
            with pytest.raises(TypeError):
                f.write("str")  # type: ignore[arg-type]

    def test_multiple_writes_to_one_handle_all_land(self, tmp_path):
        target = tmp_path / "out.txt"
        with atomic_writer(target, binary=False, encoding="utf-8") as f:
            f.write("a")
            f.write("b")
        assert target.read_text(encoding="utf-8") == "ab"
