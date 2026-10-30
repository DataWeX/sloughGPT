"""Fast-lane pins for the live-stack skip gates — card 20261029_041801.

The gates themselves run only against a live stack (slow-marked, deselected
by default), so these tests pin the PURE parts in the fast lane:

- ``readiness_payload_ok`` — the API identity check, including the exact
  nested envelope shape our server emits (a wrong shape here would make the
  gate skip our own server, or accept a foreign one);
- ``web_tree_id`` — the tree-identity recipe shared with
  apps/web/vite.config.ts, including recipe parity (header name + hash
  steps present on BOTH sides — a rename on one side alone must fail);
- the probe budget invariant — the old budget (158s) exceeded pytest.ini's
  per-test timeout (120s), which is exactly how a foreign listener became
  27 setup-timeout ERRORS instead of skips.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.skip_gate_identity import (
    PROBE_ATTEMPTS,
    PROBE_SLEEP,
    PROBE_TIMEOUT,
    WEB_TREE_HEADER,
    probe_budget_seconds,
    readiness_payload_ok,
    web_tree_id,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VITE_CONFIG = REPO_ROOT / "apps" / "web" / "vite.config.ts"
PYTEST_INI = REPO_ROOT / "pytest.ini"

# The real body of GET /health/ready — success_response(get_readiness())
# (routers/health.py wrapping controllers/health.py, envelope from
# schemas/common.py). Shape pinned from source, not from a live run.
OUR_READY = {
    "status": "success",
    "data": {
        "status": "ready",
        "checks": {"app_lifecycle": "ready", "model": "loaded", "mogdb": "ok"},
    },
}


class TestReadinessIdentity:
    """readiness_payload_ok: OUR ready envelope only."""

    def test_accepts_our_ready_envelope(self):
        assert readiness_payload_ok(OUR_READY) is True

    def test_rejects_our_own_not_ready(self):
        # Same server, still loading: must NOT pass (that's the
        # foreign/mid-load case the old port-open gate could not tell).
        payload = {
            "status": "success",
            "data": {"status": "not_ready", "checks": {"model": "loading"}},
        }
        assert readiness_payload_ok(payload) is False

    def test_rejects_foreign_shapes(self):
        rejects = [
            None,
            "ready",
            200,
            [],
            {},  # empty dict
            {"status": "success"},  # no data
            {"status": "success", "data": None},  # data null
            {"status": "success", "data": "ready"},  # data scalar
            {"status": "error", "data": {"status": "ready", "checks": {}}},
            {"status": "ready"},  # flat (liveness-ish) shape
            {"status": "success", "data": {"status": "ready"}},  # no checks
            {"status": "success", "data": {"status": "ok", "checks": {}}},
            # plausible foreign service (K8s-style readiness probe)
            {"status": "success", "data": {"ready": True, "checks": {}}},
        ]
        for payload in rejects:
            assert readiness_payload_ok(payload) is False, payload

    def test_checks_must_be_a_dict(self):
        payload = {"status": "success", "data": {"status": "ready", "checks": None}}
        assert readiness_payload_ok(payload) is False


class TestWebTreeIdentity:
    """web_tree_id: stable, tree-specific, and in lockstep with vite.config.ts."""

    def test_default_id_is_16_hex_chars(self):
        ident = web_tree_id()
        assert re.fullmatch(r"[0-9a-f]{16}", ident), ident

    def test_deterministic(self):
        assert web_tree_id() == web_tree_id()

    def test_different_roots_different_ids(self):
        # Two checkouts (worktree vs main, or a foreign clone) must never
        # collide — that's the whole point of the header comparison.
        a = web_tree_id(Path("/tmp/fake-checkout-a"))
        b = web_tree_id(Path("/tmp/fake-checkout-b"))
        assert a != b
        assert re.fullmatch(r"[0-9a-f]{16}", a)
        assert re.fullmatch(r"[0-9a-f]{16}", b)

    def test_vite_config_emits_the_same_header(self):
        # Recipe parity: rename the header on ONE side only and this fails.
        source = VITE_CONFIG.read_text(encoding="utf-8")
        assert WEB_TREE_HEADER in source, (
            f"vite.config.ts no longer emits {WEB_TREE_HEADER} — "
            "tests/test_header_screenshots.py would skip forever"
        )

    def test_vite_config_hashes_realpath_the_same_way(self):
        source = VITE_CONFIG.read_text(encoding="utf-8")
        assert "createHash('sha256')" in source, "vite hash recipe changed"
        assert "realpathSync(__dirname)" in source, "vite no longer realpath-resolves"
        assert ".slice(0, 16)" in source, "vite truncation changed"


class TestProbeBudget:
    """The budget MUST stay under pytest.ini's per-test timeout.

    Regression guard for the original defect: old budget 5x30s + 4x2s =
    158s > timeout=120, so the fixture itself timed out per test and a
    foreign :8000 listener produced errors instead of a skip.
    """

    def test_budget_under_pytest_timeout(self):
        ini = PYTEST_INI.read_text(encoding="utf-8")
        match = re.search(r"^timeout\s*=\s*(\d+)", ini, re.MULTILINE)
        assert match, "pytest.ini no longer declares a per-test timeout"
        ceiling = int(match.group(1))
        budget = probe_budget_seconds()
        assert budget < ceiling, (
            f"probe budget {budget}s >= pytest timeout {ceiling}s — a dead "
            "or foreign listener would ERROR every test instead of skipping"
        )

    def test_budget_arithmetic(self):
        assert probe_budget_seconds() == (
            PROBE_ATTEMPTS * PROBE_TIMEOUT + (PROBE_ATTEMPTS - 1) * PROBE_SLEEP
        )
        # Sanity: the intended scale (tens of seconds, not minutes).
        assert probe_budget_seconds() <= 60
