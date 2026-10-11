"""The site doctor must separate its own artefacts from real defects.

A probe that reports "108 broken routes" when most of them are query-required
calls, SSE streams that never finish, or auth gates behaving correctly, is
worse than no probe: the number is large, true-ish, and points nowhere. The
classification is the product here, so it is what gets pinned.

Real run 2026-10-11, before the classification: 108 of 266 "failures", none of
which were defects. After: 4, of which two are hard 500s with named error codes
(`/phoneme/languages` -> `'list' object is not callable`, `/providers` ->
`E_SERIALIZATION`), one is an 11.8s route mistaken for a dead one by the
client's own timeout, and one genuinely hangs.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import site_doctor_probe as doctor  # noqa: E402


def _ops(*parameters: dict, responses: dict | None = None) -> dict:
    return {"get": {"parameters": list(parameters), "responses": responses or {}}}


def test_a_required_query_param_makes_a_route_uncallable_bare() -> None:
    """Such a route returns 422 for want of its query, not for being unhealthy."""
    ops = _ops({"name": "q", "in": "query", "required": True})

    assert doctor._required_query_params(ops) == ["q"]


def test_optional_parameters_do_not_disqualify_a_route() -> None:
    """OpenAPI defaults `required` to false — absent must not mean skipped."""
    ops = _ops({"name": "limit", "in": "query"})

    assert doctor._required_query_params(ops) == [], (
        "an optional param must not get a route skipped, or the probe would cover almost nothing"
    )


def test_streaming_is_detected_by_path_when_openapi_stays_silent() -> None:
    """Our SSE routes under-declare their content type — the path must catch them."""
    ops = _ops(responses={"200": {"content": {"application/json": {}}}})

    for path in ("/consciousness/stream", "/dashboard/stream", "/registry/events"):
        assert doctor._is_streaming(ops, path), f"{path} was not treated as streaming"


def test_streaming_is_detected_by_content_type_when_the_path_is_innocuous() -> None:
    ops = _ops(responses={"200": {"content": {"text/event-stream": {}}}})

    assert doctor._is_streaming(ops, "/models/watch")


def test_an_ordinary_route_is_not_skipped() -> None:
    ops = _ops(responses={"200": {"content": {"application/json": {}}}})

    assert not doctor._is_streaming(ops, "/providers")
    assert doctor._required_query_params(ops) == []


def test_an_unreachable_host_reads_as_none_instead_of_raising() -> None:
    """The probe must survive every endpoint it touches.

    `IncompleteRead` is an `http.client.HTTPException`, not an `OSError`, so
    the first version of the probe crashed on a truncated chunked response and
    reported nothing at all — the one outcome a diagnostic must never produce.
    An unreachable host is the milder cousin of that case and must likewise
    come back as a status, not an exception.
    """
    status, body, _elapsed = doctor.get("http://127.0.0.1:1/nope", timeout=0.3)

    assert status is None, f"an unreachable host must read as None, got {status!r}"
    assert body is None
