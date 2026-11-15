"""Identity checks and the environment gate for the live-stack skips — cards
20261029_041801, 20261104_085623.

The gates in ``tests/test_integration.py`` (API) and
``tests/test_header_screenshots.py`` (web) historically checked only
PORT-OPEN, so a foreign listener (another checkout's API mid-load, another
checkout's vite on :5173) turned live tests into 120s-per-test
pytest-timeout ERRORS instead of module skips — 27 setup-timeouts in one
run, because the old probe budget (5 x 30s + 4 x 2s = 158s) itself
EXCEEDED pytest.ini's ``timeout = 120``.

Both helpers here answer one question: "is the thing on that port OURS?"
They live in this non-test module (not in the slow test files) so the
fast lane can pin them (``tests/test_skip_gate_identity.py``): the slow
modules are deselected by default (``-m "not slow"``), and importing
``test_header_screenshots`` would drag its ``pytest.importorskip``
contagion along.

Card 20261104_085623 adds the journey half of the same idea:
``served_tree_status`` (whose tree answers — extracted from the inline
check test_header_screenshots.py used to carry), ``browser_build_status``
(are playwright's chromium builds installed), and ``environment_gate``
(the one shared ``pytest.skip`` wiring both journey fixtures call). The
first two stay side-effect-free (probe + read only) so the fast lane can
pin them the same way; only ``environment_gate`` raises.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# Dev-server header apps/web/vite.config.ts emits to identify its checkout.
WEB_TREE_HEADER = "X-Slo-Web-Tree"

# Probe budget for the API skip gate (card 20261029_041801): identity, not
# port-open. Must stay well under pytest.ini's per-test timeout=120 — the
# old budget (5 attempts x 30s + 4 x 2s = 158s) EXCEEDED it, so a foreign
# listener holding :8000 turned every test into a 120s setup-timeout ERROR
# instead of a module skip. New budget: 5 x 6s + 4 x 2s = 38s max, pinned
# by tests/test_skip_gate_identity.py against pytest.ini's timeout.
PROBE_ATTEMPTS = 5
PROBE_TIMEOUT = 6
PROBE_SLEEP = 2

_REPO_ROOT = Path(__file__).resolve().parents[1]


def probe_budget_seconds() -> int:
    """Worst-case seconds the API fixture may spend probing before skipping."""
    return PROBE_ATTEMPTS * PROBE_TIMEOUT + (PROBE_ATTEMPTS - 1) * PROBE_SLEEP


def readiness_payload_ok(payload: object) -> bool:
    """True only for OUR ready API: our nested envelope says ``status: "ready"``.

    The real body from ``GET /health/ready`` is
    ``success_response(get_readiness())`` (apps/api/server/routers/health.py
    wrapping apps/api/server/controllers/health.py, envelope from
    apps/api/server/schemas/common.py)::

        {"status": "success",
         "data": {"status": "ready" | "not_ready",
                  "checks": {"app_lifecycle": ..., "model": ..., ...}}}

    Any listener can hold a port; only this exact nested envelope with
    ``data.status == "ready"`` counts as our server. A foreign service, a
    different JSON shape, or our own instance still loading (model
    ``"loading"``/``"not_loaded"`` → ``"not_ready"``) answers with anything
    else → the module SKIPS instead of failing every test against a
    listener that is not ours.
    """
    if not isinstance(payload, dict):
        return False
    if payload.get("status") != "success":
        return False
    data = payload.get("data")
    if not isinstance(data, dict):
        return False
    if data.get("status") != "ready":
        return False
    return isinstance(data.get("checks"), dict)


def web_tree_id(root: Path | None = None) -> str:
    """Identity token of a web checkout's app — mirrors vite.config.ts.

    ``apps/web/vite.config.ts`` serves this value as the
    ``X-Slo-Web-Tree`` dev-server header: ``sha256(realpath(apps/web))``
    truncated to 16 hex chars. Node's ``fs.realpathSync`` and Python's
    ``os.path.realpath`` both resolve symlinks, so the two sides agree
    byte-for-byte. A server WITHOUT the header (foreign checkout's vite,
    a pre-fix server, a static preview) or with a different value is not
    this tree → the screenshot tests skip instead of capturing a foreign
    checkout's UI.

    ``root`` defaults to this repo (``tests/..``); tests pass other roots
    to prove different checkouts produce different ids.
    """
    base = root if root is not None else _REPO_ROOT
    web_dir = os.path.realpath(str(base / "apps" / "web"))
    return hashlib.sha256(web_dir.encode("utf-8")).hexdigest()[:16]


def served_tree_status(base_url: str, *, timeout: float = 5.0) -> tuple[str, str]:
    """Whose web tree answers at ``base_url`` — identity, not port-open.

    Returns ``(status, detail)`` with status one of:

    - ``"ok"`` — the server answered 2xx/3xx with OUR ``X-Slo-Web-Tree``
      header (``web_tree_id()``, same recipe apps/web/vite.config.ts stamps);
    - ``"foreign"`` — a server answered but the header is missing or a
      different value: another checkout's vite, a pre-fix server, a static
      preview. Its sha (or the missing header name) is in ``detail``;
    - ``"down"`` — nothing usable answered: connection refused/timeout, or
      an HTTP error status.

    Extraction of the inline check tests/test_header_screenshots.py used to
    carry, shared with the journey env gate (card 20261104_085623): a
    live-but-foreign server used to pass readiness and then fail CONTENT
    asserts — the failure pointed at the journeys instead of the
    environment. Never raises; callers decide skip vs fail.
    """
    expected = web_tree_id()
    try:
        req = urllib.request.Request(base_url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if not (200 <= resp.status < 400):
                return "down", f"{base_url} answered HTTP {resp.status}"
            served = resp.headers.get(WEB_TREE_HEADER)
    except urllib.error.HTTPError as exc:
        return "down", f"{base_url} answered HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 — any transport failure = not up
        return "down", f"no web server on {base_url} ({type(exc).__name__})"
    if served == expected:
        return "ok", f"{base_url} serves this tree ({expected})"
    if served is None:
        return (
            "foreign",
            f"{base_url} serves no {WEB_TREE_HEADER} header — not this tree (expected {expected})",
        )
    return (
        "foreign",
        f"{base_url} serves foreign/stale tree {served} (expected {expected})",
    )


def _default_browsers_json() -> Path | None:
    """playwright's driver/package/browsers.json, or None if not installed."""
    try:
        import playwright
    except ImportError:
        return None
    if not playwright.__file__:
        return None
    return Path(playwright.__file__).resolve().parent / "driver" / "package" / "browsers.json"


def _registry_dir(browsers_json: Path) -> Path:
    """playwright's browser registry dir — mirrors its driver resolution.

    The driver (coreBundle.js) resolves, in order: ``PLAYWRIGHT_BROWSERS_PATH``
    (``"0"`` → package-local ``.local-browsers``, else that path, relative
    paths resolved against cwd), then ``XDG_CACHE_HOME || ~/.cache`` +
    ``/ms-playwright``.
    """
    env = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if env == "0":
        # package-local browsers: <packageRoot>/.local-browsers, and
        # browsers.json lives at <packageRoot>/browsers.json.
        return browsers_json.parent / ".local-browsers"
    if env:
        path = Path(env)
        return path if path.is_absolute() else Path.cwd() / path
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "ms-playwright"


def _required_chromium_builds(browsers: list) -> list[tuple[str, str]]:
    """``[(name, revision), ...]`` of the chromium builds a journey needs.

    Only ``installByDefault`` chromium-family entries: the journeys launch
    ``chromium.launch(headless=True)`` / avion's headless chromium, which the
    driver resolves to BOTH ``chromium`` and ``chromium-headless-shell``
    (coreBundle.js: ``options.headless ? "chromium-headless-shell" :
    "chromium"``). Firefox/webkit are never launched here; tip-of-tree builds
    are opt-in (installByDefault=false) and are excluded explicitly as well —
    requiring one would skip every normally-installed machine.
    """
    required: list[tuple[str, str]] = []
    for entry in browsers:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", ""))
        if not name.startswith("chromium") or "tip-of-tree" in name:
            continue
        if not entry.get("installByDefault"):
            continue
        revision = str(entry.get("revision", ""))
        if revision:
            required.append((name, revision))
    return required


def browser_build_status(
    *,
    browsers_json: Path | None = None,
    registry_dir: Path | None = None,
) -> tuple[str, str]:
    """Are playwright's default chromium builds installed? Identity, not assumed.

    Returns ``(status, detail)``: ``"ok"`` when every required build directory
    (``<name with dashes→underscores>-<revision>``, the driver's formula —
    e.g. ``chromium_headless_shell-1234``) exists in the registry, else
    ``"missing"`` naming what to ``playwright install``. Absent/unreadable
    browsers.json counts as missing: an environment we cannot verify must
    skip, not launch into an ERROR.

    Overrides exist for the fast-lane pins (tests/test_skip_gate_identity.py);
    production callers pass nothing and get playwright's own resolution.
    """
    if browsers_json is None:
        browsers_json = _default_browsers_json()
        if browsers_json is None:
            return "missing", (
                "playwright not installed — cannot verify chromium build "
                "(python -m playwright install chromium)"
            )
    try:
        data = json.loads(browsers_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return "missing", f"cannot read {browsers_json} ({type(exc).__name__})"
    browsers = data.get("browsers") if isinstance(data, dict) else None
    if not isinstance(browsers, list):
        return "missing", f"no browsers list in {browsers_json}"
    builds = _required_chromium_builds(browsers)
    if not builds:
        return "missing", f"{browsers_json} declares no default chromium builds"
    if registry_dir is None:
        registry_dir = _registry_dir(browsers_json)
    missing = [
        f"{name.replace('-', '_')}-{revision}"
        for name, revision in builds
        if not (registry_dir / f"{name.replace('-', '_')}-{revision}").is_dir()
    ]
    if missing:
        return "missing", (
            f"playwright chromium build(s) not installed: {', '.join(missing)} "
            f"in {registry_dir} — run: python -m playwright install chromium"
        )
    return "ok", f"chromium builds present in {registry_dir}"


def environment_gate(base_url: str, *, timeout: float = 5.0) -> None:
    """Skip (never fail) unless the journey environment is OUR stack.

    One shared gate for route-level journey files (card 20261104_085623),
    called from their session fixtures AFTER the readiness waits (the waits
    let a booting stack come up; this then verifies identity):

    1. serve check — ``base_url`` must serve THIS checkout's web tree
       (``served_tree_status``), else skip "served by foreign/stale tree …";
    2. browser guard — playwright's chromium builds must be present
       (``browser_build_status``), else skip naming the missing build.

    Genuine content failures still fail: once this gate passes, every later
    assertion is the journey's own.
    """
    status, detail = served_tree_status(base_url, timeout=timeout)
    if status != "ok":
        pytest.skip(detail)
    status, detail = browser_build_status()
    if status != "ok":
        pytest.skip(detail)
