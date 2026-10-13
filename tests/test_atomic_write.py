"""Tests for domain.infrastructure._internal.atomic_write (single save owner).

The base classes are ported from ``tests/core-py/test_atomic_write.py`` on
``feat/atomic-save-domain-sweep`` (source sha ``b7ec9a7e6``); ``TestDurable``
is the extension added on ``feat/file-handler-fast-save``.
"""

from __future__ import annotations

import os
import stat

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


class TestDurable:
    """The durable tier: fsync(temp) -> os.replace -> fsync(parent dir).

    These assert the *shape* of the guarantee, not just that bytes landed:
    that the cheap default really is free, and that the expensive path flushes
    both the file and the directory that names it, in that order.
    """

    @staticmethod
    def _trace_fs(monkeypatch):
        """Record fsyncs (labelled by what kind of thing got flushed) and replaces."""
        events: list[tuple] = []
        real_fsync = os.fsync
        real_replace = os.replace

        def fsync(fd):
            mode = os.fstat(fd).st_mode
            kind = "dir" if stat.S_ISDIR(mode) else "file"
            events.append(("fsync", kind))
            return real_fsync(fd)

        def replace(src, dst):
            events.append(("replace", os.fspath(dst)))
            return real_replace(src, dst)

        monkeypatch.setattr(os, "fsync", fsync)
        monkeypatch.setattr(os, "replace", replace)
        return events

    def test_default_path_never_calls_fsync(self, tmp_path, monkeypatch):
        """The fast tier must stay free — this is the whole point of the split."""
        calls: list = []
        monkeypatch.setattr(os, "fsync", lambda fd: calls.append(fd))

        atomic_write(tmp_path / "out.txt", "x")
        with atomic_writer(tmp_path / "out2.txt") as f:
            f.write(b"y")

        assert calls == [], "the atomic (default) path must not flush"

    def test_durable_flushes_file_then_rename_then_directory(
        self, tmp_path, monkeypatch
    ):
        events = self._trace_fs(monkeypatch)
        target = tmp_path / "out.txt"

        atomic_write(target, "durable", durable=True)

        assert events == [
            ("fsync", "file"),
            ("replace", str(target)),
            ("fsync", "dir"),
        ], events

    def test_durable_writer_flushes_in_the_same_order(self, tmp_path, monkeypatch):
        events = self._trace_fs(monkeypatch)
        target = tmp_path / "out.bin"

        with atomic_writer(target, durable=True) as f:
            f.write(b"payload")

        # Uniform labels: fsync events carry what was flushed, replace carries
        # its own kind (its second slot holds the destination path).
        labels = [e[1] if e[0] == "fsync" else e[0] for e in events]
        assert labels == ["file", "replace", "dir"], labels

    def test_durable_file_fsync_happens_while_the_handle_is_still_open(
        self, tmp_path, monkeypatch
    ):
        """fsync after `with` closes the handle would flush nothing at all."""
        seen: list[bool] = []
        real_fsync = os.fsync

        def fsync(fd):
            # A closed fd is exactly what a post-`with` fsync would try to use.
            try:
                os.fstat(fd)
                seen.append(True)
            except OSError:
                seen.append(False)
            return real_fsync(fd)

        monkeypatch.setattr(os, "fsync", fsync)
        with atomic_writer(tmp_path / "out.bin", durable=True) as f:
            f.write(b"x")

        assert seen and all(seen), "an fsync ran against a closed fd"

    def test_durable_directory_is_the_targets_parent(self, tmp_path, monkeypatch):
        """The dir flush must name the target's own directory, not CWD."""
        flushed: list[str] = []
        real_open, real_fsync, real_close = os.open, os.fsync, os.close

        def open_(path, flags, *a, **kw):
            if isinstance(path, str):
                try:
                    if stat.S_ISDIR(os.stat(path).st_mode):
                        flushed.append(path)
                except OSError:
                    pass  # e.g. mkstemp's O_CREAT open — nothing to stat yet
            return real_open(path, flags, *a, **kw)

        def close(fd):
            return real_close(fd)

        monkeypatch.setattr(os, "open", open_)
        monkeypatch.setattr(os, "fsync", real_fsync)
        monkeypatch.setattr(os, "close", close)

        target = tmp_path / "sub" / "out.txt"
        atomic_write(target, "x", durable=True)

        assert flushed == [str(tmp_path / "sub")], flushed

    def test_durable_writes_real_content_and_leaves_no_temp(self, tmp_path):
        target = tmp_path / "out.txt"
        atomic_write(target, "survives", durable=True)
        assert target.read_text(encoding="utf-8") == "survives"
        assert [p.name for p in tmp_path.iterdir() if is_temp_name(p.name)] == []

    def test_durable_creates_missing_parents_then_flushes_them(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "out.txt"
        atomic_write(target, "x", durable=True)
        assert target.exists()

    def test_durable_failure_in_body_leaves_original_and_no_temp(self, tmp_path):
        target = tmp_path / "out.txt"
        target.write_text("original", encoding="utf-8")
        with pytest.raises(RuntimeError, match="boom"):
            with atomic_writer(target, binary=False, encoding="utf-8", durable=True) as f:
                f.write("partial")
                raise RuntimeError("boom")
        assert target.read_text(encoding="utf-8") == "original"
        assert [p.name for p in tmp_path.iterdir() if is_temp_name(p.name)] == []

    def test_dir_fsync_failure_propagates_rather_than_pretending(
        self, tmp_path, monkeypatch
    ):
        """A swallowed dir-fsync error is a false guarantee — worse than none."""

        def boom(fd):
            raise OSError(28, "Invalid argument")

        monkeypatch.setattr(os, "fsync", boom)

        with pytest.raises(OSError, match="Invalid argument"):
            atomic_write(tmp_path / "out.txt", "x", durable=True)
