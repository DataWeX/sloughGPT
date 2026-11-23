"""CORS expose_headers contract — the 429 backoff family must be JS-readable.

Cross-origin response headers (web :5173 → api :8000) are opaque to
browser JS unless listed in ``Access-Control-Expose-Headers``.
``apps/lib/request.ts:192`` already reads ``Retry-After`` to honor a 429
bucket reset; without exposure the read returns null and the client falls
back to 500ms/1000ms exponential backoff into the same exhausted bucket —
the retry amplification captured in card 20261119_091258 Phase 1.

Card 20261119_124725 (residual 1 of 20261119_091258, user-approved).
"""

from test_support import get_test_client

_ORIGIN = "http://localhost:5173"


def _exposed_names(resp) -> set[str]:
    exposed = resp.headers.get("access-control-expose-headers", "")
    return {n.strip().lower() for n in exposed.split(",") if n.strip()}


class TestExposeHeadersOnCrossOriginResponses:
    """The header family the client contract consumes must be JS-readable."""

    def test_retry_after_is_js_readable(self):
        client = get_test_client()
        resp = client.get("/health", headers={"Origin": _ORIGIN})
        assert "retry-after" in _exposed_names(resp)

    def test_rate_limit_headers_are_js_readable(self):
        client = get_test_client()
        resp = client.get("/health", headers={"Origin": _ORIGIN})
        names = _exposed_names(resp)
        assert "x-ratelimit-remaining" in names
        assert "x-ratelimit-limit" in names

    def test_allow_origin_still_echoes_for_local_dev(self):
        """Exposure must not disturb the existing origin contract."""
        client = get_test_client()
        resp = client.get("/health", headers={"Origin": _ORIGIN})
        assert resp.headers.get("access-control-allow-origin") == _ORIGIN
