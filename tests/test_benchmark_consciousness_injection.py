"""The injection probe matrix must survive a dead server (card 20261115_081635).

Defect found during the first live baseline run (2026-11-17): the API server
died at cell 7/15 mid-SSE-stream, ``chat_turn`` raised ``ChunkedEncodingError``,
and the whole run aborted with no JSON and no record — an always-measure gate
that loses its own evidence the moment the thing it measures flaks is not a
gate. Three behaviours are pinned here:

1. ``run_matrix`` records a transport failure as an errored cell and keeps
   going — one dead cell never costs the other 14;
2. ``aggregate`` excludes errored cells from the rate denominators — a cell
   that produced no response is NOT evidence that the baseline latched (or
   that it stayed clean);
3. ``verdict`` refuses to pass a run carrying errors — incomplete
   measurement is not a green gate, even when the baseline looks clean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_consciousness_injection as inj  # noqa: E402


def _ok(text: str = "The answer is 4.") -> dict:
    return {
        "text": text,
        "latency_ms": 12.5,
        "response_tokens": 8,
        "error": None,
    }


def test_run_matrix_survives_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """One raising cell becomes an errored cell; the other 14 still run."""
    calls = {"n": 0}

    def flaky_chat_turn(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise requests.ConnectionError("server went away")
        return _ok()

    monkeypatch.setattr(inj, "chat_turn", flaky_chat_turn)

    cells = inj.run_matrix("http://localhost:8000", "narrative", "engine", timeout=1.0)

    assert len(cells) == 15, "a transport failure must not abort the matrix"
    errored = [c for c in cells if c.get("error")]
    assert len(errored) == 1
    assert "server went away" in errored[0]["error"]
    assert errored[0]["error"].startswith("transport:")
    # every cell is fully shaped — the errored one included (print_matrix
    # indexes these keys unconditionally)
    for c in cells:
        for key in ("config", "probe", "latch", "on_topic", "parrot", "error"):
            assert key in c


def test_aggregate_excludes_errored_cells_from_rates() -> None:
    """An errored cell is absence of evidence, not a latched baseline."""
    clean = {
        "config": "off",
        "probe": "2+2",
        "latch": False,
        "on_topic": True,
        "keyword_hit": True,
        "parrot": False,
        "injected_tokens": 0,
        "response_tokens": 8,
        "token_delta_vs_off": None,
        "chars": 16,
        "excerpt": "4",
    }
    # the trap: an errored cell that WOULD read as latched if counted —
    # score_cell on empty text never latches, but a transport failure carries
    # no answer at all and must not enter the denominator either
    errored = dict(clean, error="transport: reset", latch=True, on_topic=False)

    summary = inj.aggregate([clean, clean, errored])

    assert summary["off"]["cells"] == 3  # total rows kept for reporting
    assert summary["off"]["latch_rate"] == 0.0, "errored cell must not latch the baseline"
    assert summary["off"]["on_topic_rate"] == 1.0


def test_aggregate_skips_config_with_only_errors() -> None:
    """A config whose cells all errored contributes no rates at all."""
    errored = {
        "config": "off",
        "probe": "2+2",
        "latch": False,
        "on_topic": False,
        "keyword_hit": False,
        "parrot": False,
        "injected_tokens": 0,
        "response_tokens": None,
        "token_delta_vs_off": None,
        "chars": 0,
        "excerpt": "",
        "error": "transport: refused",
    }

    summary = inj.aggregate([errored])

    assert "off" not in summary, "an all-errored baseline must not produce clean-looking rates"


def test_verdict_fails_run_with_errors() -> None:
    """Clean baseline + one dead cell = incomplete, not green."""
    off_cell = {
        "config": "off",
        "probe": "2+2",
        "latch": False,
        "on_topic": True,
        "keyword_hit": True,
        "parrot": False,
        "injected_tokens": 0,
        "response_tokens": 8,
        "token_delta_vs_off": None,
        "chars": 16,
        "excerpt": "4",
        "error": None,
    }
    dead_cell = dict(off_cell, config="tail+strip", error="transport: reset")

    passed, score, baseline_clean = inj.verdict([off_cell, dead_cell])

    assert baseline_clean is True, "the off baseline itself is clean"
    assert passed is False, "a run carrying transport errors must not pass"
    assert score > 0  # score still reports the baseline's on-topic quality


def test_verdict_fails_when_baseline_all_errored() -> None:
    """No working baseline → nothing to compare against → fail closed."""
    dead = {
        "config": "off",
        "probe": "2+2",
        "latch": False,
        "on_topic": False,
        "keyword_hit": False,
        "parrot": False,
        "injected_tokens": 0,
        "response_tokens": None,
        "token_delta_vs_off": None,
        "chars": 0,
        "excerpt": "",
        "error": "transport: refused",
    }

    passed, score, baseline_clean = inj.verdict([dead])

    assert baseline_clean is False, "a missing baseline defaults to unclean (latch_rate 1.0)"
    assert passed is False
    assert score == 0.0


def test_verdict_passes_clean_complete_run() -> None:
    """The normal case still passes — no behaviour regression."""
    off_cell = {
        "config": "off",
        "probe": "2+2",
        "latch": False,
        "on_topic": True,
        "keyword_hit": True,
        "parrot": False,
        "injected_tokens": 0,
        "response_tokens": 8,
        "token_delta_vs_off": None,
        "chars": 16,
        "excerpt": "4",
        "error": None,
    }

    passed, score, baseline_clean = inj.verdict([off_cell])

    assert baseline_clean is True
    assert passed is True
    assert score == 100.0
