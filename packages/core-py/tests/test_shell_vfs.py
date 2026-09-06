"""Tests for shell.vfs — VFS singleton accessors."""

from __future__ import annotations

import pytest

from domains.shell.vfs import get_vfs, reset_vfs, VFS


# ── Singleton accessors ───────────────────────────────────────────────────


class TestVFSSingleton:

    def setup_method(self):
        reset_vfs()

    def teardown_method(self):
        reset_vfs()

    def test_get_vfs_returns_vfs_instance(self):
        vfs = get_vfs()
        assert isinstance(vfs, VFS)

    def test_get_vfs_returns_same_instance(self):
        vfs1 = get_vfs()
        vfs2 = get_vfs()
        assert vfs1 is vfs2

    def test_reset_vfs_clears_instance(self):
        vfs1 = get_vfs()
        reset_vfs()
        vfs2 = get_vfs()
        assert vfs1 is not vfs2

    def test_reset_vfs_allows_fresh_start(self):
        get_vfs()
        reset_vfs()
        # After reset, get_vfs should create a new instance
        vfs = get_vfs()
        assert isinstance(vfs, VFS)


# ── Re-exports ────────────────────────────────────────────────────────────


class TestVFSReExports:

    def test_vfs_class_importable(self):
        from domains.shell.vfs import VFS
        assert VFS is not None

    def test_vfs_entry_importable(self):
        from domains.shell.vfs import VFSEntry
        assert VFSEntry is not None

    def test_vfs_directory_importable(self):
        from domains.shell.vfs import VFSDirectory
        assert VFSDirectory is not None

    def test_vfs_generated_file_importable(self):
        from domains.shell.vfs import VFSGeneratedFile
        assert VFSGeneratedFile is not None

    def test_vfs_write_only_file_importable(self):
        from domains.shell.vfs import VFSWriteOnlyFile
        assert VFSWriteOnlyFile is not None
