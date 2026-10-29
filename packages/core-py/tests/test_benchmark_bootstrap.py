"""Tests for scripts/benchmark_bootstrap.py envelope metrics + journey gates."""

import json
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import benchmark_bootstrap as bb  # noqa: E402

# ── pure helpers ───────────────────────────────────────────────────────────


def test_percentile_nearest_rank():
    """Nearest-rank p95 of 1..20 is the 19th value (ceil(0.95*n)-1)."""
    samples = [float(i) for i in range(1, 21)]
    assert bb.percentile(samples, 95) == 19.0
    assert bb.percentile(samples, 50) == 10.0
    assert bb.percentile([7.0], 95) == 7.0


def test_stats_mean_p95_max():
    s = bb.stats([1.0, 2.0, 3.0, 4.0])
    assert s["mean"] == pytest.approx(2.5)
    assert s["max"] == 4.0
    assert s["p95"] == 4.0


def test_summarize_envelope_success_shape():
    """The success_response envelope (status/data) unwraps to members+errors."""
    payload = {
        "status": "success",
        "data": {"members": {k: {} for k in range(12)}, "errors": {}},
    }
    members, errors = bb.summarize_envelope(payload)
    assert len(members) == 12
    assert errors == {}


def test_summarize_envelope_bare_shape_and_null_errors():
    """A bare body without the status wrapper also parses; null errors → {}."""
    members, errors = bb.summarize_envelope({"members": {"a": 1}, "errors": None})
    assert members == {"a": 1}
    assert errors == {}


def test_summarize_envelope_rejects_shape_drift():
    """Missing members map or non-dict body fails loudly, never zeroes."""
    with pytest.raises(ValueError, match="no members map"):
        bb.summarize_envelope({"status": "success", "data": {"errors": {}}})
    with pytest.raises(ValueError):
        bb.summarize_envelope(["not", "a", "dict"])
    with pytest.raises(ValueError):
        bb.summarize_envelope({"members": [], "errors": {}})


# ── main() gates (urlopen mocked; no server needed) ────────────────────────


class _FakeResponse:
    def __init__(self, payload: dict, delay: float = 0.0):
        self._body = json.dumps(payload).encode()
        self._delay = delay

    def read(self) -> bytes:
        import time

        if self._delay:
            time.sleep(self._delay)
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _envelope(member_errors: dict | None = None) -> dict:
    return {
        "status": "success",
        "data": {
            "members": {f"m{i}": {"ok": True} for i in range(12)},
            "errors": member_errors or {},
        },
    }


def _run_main(monkeypatch, tmp_path, response, extra=()):
    """Run bb.main() with urlopen mocked to return ``response`` (or raise)."""

    def fake_urlopen(req, timeout=None):
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(bb.urllib.request, "urlopen", fake_urlopen)
    out = tmp_path / "bench.json"
    monkeypatch.setattr(
        sys, "argv", ["benchmark_bootstrap.py", "--runs", "3", "--json-out", str(out), *extra]
    )
    return bb.main(), out


def test_main_passes_on_healthy_envelope(monkeypatch, tmp_path, capsys):
    rc, out = _run_main(monkeypatch, tmp_path, _FakeResponse(_envelope()))
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["members"] == 12
    assert data["member_errors"] == 0
    assert data["runs"] == 3
    assert data["first_s"] >= 0
    assert data["warm_p95_s"] is not None


def test_main_fails_on_member_errors(monkeypatch, tmp_path, capsys):
    rc, _ = _run_main(
        monkeypatch, tmp_path, _FakeResponse(_envelope({"models": "TimeoutError: slow"}))
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "member error" in err and "models" in err


def test_main_fails_when_watchdog_budget_exceeded(monkeypatch, tmp_path, capsys):
    """A request slower than --budget-s fails like the 8s overlay watchdog."""
    slow = _FakeResponse(_envelope(), delay=0.05)
    rc, _ = _run_main(monkeypatch, tmp_path, slow, extra=["--budget-s", "0.01"])
    assert rc == 1
    assert "overlay_stuck" in capsys.readouterr().err


def test_main_fails_cleanly_on_404_without_router(monkeypatch, tmp_path, capsys):
    err = urllib.error.HTTPError("http://localhost:8000/bootstrap", 404, "Not Found", None, None)
    rc, _ = _run_main(monkeypatch, tmp_path, err)
    assert rc == 1
    assert "HTTP 404" in capsys.readouterr().err


def test_main_fails_on_unreachable_server(monkeypatch, tmp_path, capsys):
    rc, _ = _run_main(monkeypatch, tmp_path, urllib.error.URLError("connection refused"))
    assert rc == 1
    assert "unreachable" in capsys.readouterr().err
