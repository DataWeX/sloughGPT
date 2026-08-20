"""Tests for domains.shell.addons.filesystem — VFSEntry, VFSGeneratedFile, VFSWriteOnlyFile, VFSDirectory."""

from domains.shell.addons.filesystem import (
    VFSEntry, VFSGeneratedFile, VFSWriteOnlyFile, VFSDirectory,
)


class TestVFSEntry:
    def test_init(self):
        e = VFSEntry("test.txt")
        assert e.name == "test.txt"
        assert e.is_dir is False
        assert e.size == 0

    def test_read_returns_empty(self):
        e = VFSEntry("test.txt")
        assert e.read() == ""

    def test_write_noop(self):
        e = VFSEntry("test.txt")
        e.write("hello")

    def test_directory(self):
        e = VFSEntry("dir", is_dir=True)
        assert e.is_dir is True


class TestVFSGeneratedFile:
    def test_read_calls_fn(self):
        gf = VFSGeneratedFile("gen.txt", read_fn=lambda: "generated content")
        assert gf.read() == "generated content"

    def test_init(self):
        gf = VFSGeneratedFile("gen.txt", read_fn=lambda: "x")
        assert gf.name == "gen.txt"


class TestVFSWriteOnlyFile:
    def test_write_calls_fn(self):
        captured = []
        wf = VFSWriteOnlyFile("out.txt", write_fn=lambda d: captured.append(d) or None)
        wf.write("data")
        assert captured == ["data"]

    def test_read_returns_empty(self):
        wf = VFSWriteOnlyFile("out.txt", write_fn=lambda d: None)
        assert wf.read() == ""


class TestVFSDirectory:
    def test_is_dir(self):
        d = VFSDirectory("root")
        assert d.is_dir is True

    def test_list_empty(self):
        d = VFSDirectory("root")
        assert d.list() == []

    def test_list_entries(self):
        entries = {"a.txt": VFSEntry("a.txt"), "b.txt": VFSEntry("b.txt")}
        d = VFSDirectory("root", entries)
        names = d.list()
        assert names == ["a.txt", "b.txt"]

    def test_add_entry(self):
        d = VFSDirectory("root")
        e = VFSEntry("file.txt")
        d.add(e)
        assert "file.txt" in d.list()

    def test_get_entry(self):
        e = VFSEntry("x.txt")
        d = VFSDirectory("root", {"x.txt": e})
        assert d.get("x.txt") is e

    def test_get_missing(self):
        d = VFSDirectory("root")
        assert d.get("nope") is None
