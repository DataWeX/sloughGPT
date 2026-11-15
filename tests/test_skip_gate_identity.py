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
  27 setup-timeout ERRORS instead of skips;
- ``served_tree_status`` / ``environment_gate`` — the journey serve check
  (foreign tree or dead server → skip, never content-assert reds) and the
  one shared environment gate both journey fixtures call (card
  20261104_085623);
- ``browser_build_status`` — playwright's default chromium builds present
  in the ms-playwright registry (missing build → skip, not a launch ERROR).
"""

from __future__ import annotations

import json
import re
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.skip_gate_identity import (
    PROBE_ATTEMPTS,
    PROBE_SLEEP,
    PROBE_TIMEOUT,
    WEB_TREE_HEADER,
    _default_browsers_json,
    _registry_dir,
    _required_chromium_builds,
    browser_build_status,
    environment_gate,
    probe_budget_seconds,
    readiness_payload_ok,
    served_tree_status,
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


# Real browsers.json shape (playwright 1.62): tip-of-tree builds are
# installByDefault=false and must NEVER be required; firefox is default but
# the journeys only launch chromium.
CHROMIUM_BUILDS = [
    {"name": "chromium", "revision": "1234", "installByDefault": True},
    {"name": "chromium-headless-shell", "revision": "1234", "installByDefault": True},
    {"name": "chromium-tip-of-tree", "revision": "1433", "installByDefault": False},
    {"name": "firefox", "revision": "1538", "installByDefault": True},
]


@contextmanager
def _serve(*, status: int = 200, headers: dict[str, str] | None = None):
    """A local HTTP server answering with fixed status/headers (ephemeral port)."""
    headers = headers or {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):  # silence request logging
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


class TestServedTreeStatus:
    """served_tree_status: whose tree answers, not merely whether a port is open."""

    def test_ok_when_header_is_this_tree(self):
        with _serve(headers={WEB_TREE_HEADER: web_tree_id()}) as url:
            status, detail = served_tree_status(url)
        assert status == "ok"
        assert web_tree_id() in detail

    def test_foreign_when_header_is_another_tree(self):
        with _serve(headers={WEB_TREE_HEADER: "deadbeefdeadbeef"}) as url:
            status, detail = served_tree_status(url)
        assert status == "foreign"
        assert "foreign/stale tree deadbeefdeadbeef" in detail

    def test_foreign_when_header_absent(self):
        with _serve() as url:
            status, detail = served_tree_status(url)
        assert status == "foreign"
        assert WEB_TREE_HEADER in detail  # names what is missing

    def test_down_when_nothing_listens(self):
        status, detail = served_tree_status("http://127.0.0.1:1")
        assert status == "down"
        assert "no web server" in detail

    def test_down_on_http_error_status(self):
        with _serve(status=503) as url:
            status, detail = served_tree_status(url)
        assert status == "down"
        assert "HTTP 503" in detail


class TestBrowserBuildStatus:
    """browser_build_status: chromium builds present in the ms-playwright registry."""

    @staticmethod
    def _env(
        tmp_path,
        *,
        browsers=CHROMIUM_BUILDS,
        dirs=("chromium-1234", "chromium_headless_shell-1234"),
    ):
        bj = tmp_path / "browsers.json"
        bj.write_text(json.dumps({"browsers": browsers}), encoding="utf-8")
        registry = tmp_path / "ms-playwright"
        for name in dirs:
            (registry / name).mkdir(parents=True, exist_ok=True)
        return bj, registry

    def test_ok_when_default_chromium_builds_present(self, tmp_path):
        bj, registry = self._env(tmp_path)
        status, detail = browser_build_status(browsers_json=bj, registry_dir=registry)
        assert status == "ok"
        # firefox is installByDefault but journeys never launch it → not required.
        assert str(registry) in detail

    def test_missing_when_chromium_dir_absent(self, tmp_path):
        bj, registry = self._env(tmp_path, dirs=("chromium_headless_shell-1234",))
        status, detail = browser_build_status(browsers_json=bj, registry_dir=registry)
        assert status == "missing"
        assert "chromium-1234" in detail

    def test_missing_when_headless_shell_dir_absent(self, tmp_path):
        # Journeys launch headless=True → chromium-headless-shell is required;
        # its dir name underscores the dashes (driver coreBundle formula).
        bj, registry = self._env(tmp_path, dirs=("chromium-1234",))
        status, detail = browser_build_status(browsers_json=bj, registry_dir=registry)
        assert status == "missing"
        assert "chromium_headless_shell-1234" in detail

    def test_missing_when_registry_empty(self, tmp_path):
        bj, registry = self._env(tmp_path, dirs=())
        status, detail = browser_build_status(browsers_json=bj, registry_dir=registry)
        assert status == "missing"
        assert "chromium-1234" in detail
        assert "chromium_headless_shell-1234" in detail

    def test_tip_of_tree_dir_never_satisfies_the_guard(self, tmp_path):
        # chromium-tip-of-tree is installByDefault=false → never required and
        # never sufficient: its presence must not mask a missing default build.
        bj, registry = self._env(tmp_path, dirs=("chromium_tip_of_tree-1433",))
        status, detail = browser_build_status(browsers_json=bj, registry_dir=registry)
        assert status == "missing"
        assert "chromium-1234" in detail
        assert "chromium_tip_of_tree" not in detail

    def test_missing_when_browsers_json_unreadable(self, tmp_path):
        bj = tmp_path / "nope" / "browsers.json"
        status, detail = browser_build_status(browsers_json=bj, registry_dir=tmp_path)
        assert status == "missing"
        assert str(bj) in detail

    def test_missing_when_no_default_chromium_declared(self, tmp_path):
        bj, registry = self._env(
            tmp_path,
            browsers=[{"name": "firefox", "revision": "1538", "installByDefault": True}],
            dirs=(),
        )
        status, detail = browser_build_status(browsers_json=bj, registry_dir=registry)
        assert status == "missing"
        assert "no default chromium" in detail

    def test_required_builds_filter(self):
        assert _required_chromium_builds(CHROMIUM_BUILDS) == [
            ("chromium", "1234"),
            ("chromium-headless-shell", "1234"),
        ]

    def test_registry_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path / "custom"))
        assert _registry_dir(Path("/any/browsers.json")) == tmp_path / "custom"

    def test_registry_env_zero_uses_package_local_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", "0")
        bj = tmp_path / "driver" / "package" / "browsers.json"
        assert _registry_dir(bj) == bj.parent / ".local-browsers"

    def test_registry_env_relative_resolves_against_cwd(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", "rel-browsers")
        monkeypatch.chdir(tmp_path)
        assert _registry_dir(Path("/any/browsers.json")) == tmp_path / "rel-browsers"

    def test_registry_xdg_cache_home(self, monkeypatch, tmp_path):
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert _registry_dir(Path("/any/browsers.json")) == tmp_path / "ms-playwright"

    def test_registry_default_cache(self, monkeypatch):
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        assert _registry_dir(Path("/any/browsers.json")) == (
            Path.home() / ".cache" / "ms-playwright"
        )

    def test_real_playwright_browsers_json_declares_chromium(self):
        pytest.importorskip("playwright")
        path = _default_browsers_json()
        assert path is not None and path.is_file()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert _required_chromium_builds(data["browsers"])  # non-empty, chromium-family

    def test_real_cache_status_is_a_known_state(self):
        pytest.importorskip("playwright")
        status, detail = browser_build_status()
        assert status in {"ok", "missing"}
        assert detail
        if status == "missing":
            assert "chromium" in detail  # names what to install


class TestEnvironmentGate:
    """environment_gate: one shared skip for foreign tree / missing browser."""

    def test_skips_on_foreign_tree(self):
        with _serve(headers={WEB_TREE_HEADER: "0000000000000000"}) as url:
            with pytest.raises(pytest.skip.Exception, match="foreign/stale tree"):
                environment_gate(url)

    def test_skips_without_a_server(self):
        with pytest.raises(pytest.skip.Exception, match="no web server"):
            environment_gate("http://127.0.0.1:1")

    def test_skips_when_browser_build_missing(self, monkeypatch):
        monkeypatch.setattr(
            "tests.skip_gate_identity.browser_build_status",
            lambda: ("missing", "chromium-1234 absent"),
        )
        with _serve(headers={WEB_TREE_HEADER: web_tree_id()}) as url:
            with pytest.raises(pytest.skip.Exception, match="chromium-1234"):
                environment_gate(url)

    def test_proceeds_when_tree_and_browser_ok(self, monkeypatch):
        monkeypatch.setattr(
            "tests.skip_gate_identity.browser_build_status",
            lambda: ("ok", "chromium builds present"),
        )
        with _serve(headers={WEB_TREE_HEADER: web_tree_id()}) as url:
            assert environment_gate(url) is None
