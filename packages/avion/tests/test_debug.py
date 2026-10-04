"""Debug tests — capture, step, diff."""

import pytest
from avion.debug import PageSnapshot, TimeTravel


class TestSnapshots:
    def test_capture_sequences(self):
        tt = TimeTravel()
        a = tt.capture(url="/a")
        b = tt.capture(url="/b")
        assert (a.seq, b.seq) == (1, 2)
        assert tt.current.url == "/b"

    def test_excerpt_truncated(self):
        tt = TimeTravel()
        assert len(tt.capture(text_excerpt="x" * 900).text_excerpt) == 500

    def test_screenshot_hashed_not_stored(self):
        tt = TimeTravel()
        snap = tt.capture(url="/a", screenshot=b"raw-bytes")
        assert snap.screenshot_hash != ""
        assert "raw-bytes" not in str(snap.to_dict())


class TestTravel:
    def _tt(self):
        tt = TimeTravel()
        tt.capture(url="/a", title="A")
        tt.capture(url="/b", title="B")
        return tt

    def test_back_and_forward(self):
        tt = self._tt()
        assert tt.back().url == "/a"
        assert tt.back() is None
        assert tt.forward().url == "/b"
        assert tt.forward() is None

    def test_diff(self):
        tt = self._tt()
        d = tt.diff_between(0, 1)
        assert d["url"] == {"before": "/a", "after": "/b"}
        assert "title" in d

    def test_diff_same_is_empty(self):
        s = PageSnapshot(seq=1, url="/a")
        assert s.diff(PageSnapshot(seq=2, url="/a")) == {}

    def test_diff_bad_index_raises(self):
        with pytest.raises(IndexError):
            self._tt().diff_between(0, 9)

    def test_clear(self):
        tt = self._tt()
        tt.clear()
        assert len(tt.timeline) == 0 and tt.current is None
