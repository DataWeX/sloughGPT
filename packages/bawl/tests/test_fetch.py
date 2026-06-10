"""Tests for bawl.fetch"""

from bawl.fetch import fetch, _throttle
import time


def test_fetch_real():
    html = fetch("https://example.com")
    assert html is not None
    assert "Example Domain" in html


def test_fetch_bad_url():
    assert fetch("https://nosuchdomain99999.xyz") is None


def test_throttle():
    t0 = time.time()
    _throttle("test.local", 0.1)
    t1 = time.time()
    _throttle("test.local", 0.1)
    t2 = time.time()
    assert (t2 - t1) >= 0.08  # slight tolerance
