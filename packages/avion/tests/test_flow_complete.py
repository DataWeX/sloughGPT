"""Flow-completion tests — a real browser completes a recorded flow on a
real UI, and the login survives the browser's death via the profile.

The fixture is a stdlib ``ThreadingHTTPServer`` (the ``test_cdp.py``
pattern) serving a tiny site: landing -> Sign in -> cookie-gated
dashboard -> search -> results.  The tests are the user story in full:

1. perform a flow once (recorded automatically), then ``complete_flow``
   re-drives it end to end on the same browser;
2. log in once, record the search-only part, KILL the browser, relaunch
   the same profile, and ``complete_flow`` still completes — no login
   step in the flow, because the profile carried the session.

Degrades explicitly (stated skip) when no Chromium is available.
"""

from __future__ import annotations

import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from avion.core.element import ElementLocator
from avion.core.session import Arken
from avion.events.models import EventType
from avion.flow.complete import complete_flow
from avion.launch import Profile, find_chromium, launch_chromium

_RESULTS_TEXT = "Found: avion flows - 1 match"


class _SiteHandler(BaseHTTPRequestHandler):
    """The mini-app: landing, login (cookie), gated dashboard, search."""

    def log_message(self, *args):  # silence the server thread
        pass

    def _html(self, body: str, *, cookie: str | None = None, status: int = 200):
        payload = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        if cookie:
            # Persistent cookie — what a real login sets, and what the
            # profile's cookie store keeps across browser restarts.
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(payload)

    def _redirect(self, location: str):
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        logged_in = "session=ok" in self.headers.get("Cookie", "")

        if parsed.path == "/":
            self._html("<h1>Welcome</h1><a href='/login'>Sign in</a>")
        elif parsed.path == "/login":
            self._html(
                "<h1>Sign in</h1>"
                "<form action='/do_login' method='get'>"
                "<input name='user'><input name='pass' type='password'>"
                "<button id='go' type='submit'>Log in</button></form>"
            )
        elif parsed.path == "/do_login":
            if not query.get("user"):
                self._redirect("/login")
            else:
                self._redirect_set_cookie()
        elif parsed.path == "/app":
            if not logged_in:
                self._redirect("/login")
            elif query.get("q"):
                self._html(f"<h1>Results</h1><p>Found: {query['q'][0]} - 1 match</p>")
            else:
                self._html(
                    "<h1>Dashboard</h1>"
                    "<form action='/app' method='get'>"
                    "<input name='q' id='q'>"
                    "<button id='search' type='submit'>Search</button></form>"
                )
        else:
            self._html("<h1>404</h1>", status=404)

    def _redirect_set_cookie(self):
        self.send_response(302)
        self.send_header("Location", "/app")
        self.send_header("Set-Cookie", "session=ok; Max-Age=3600; Path=/")
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture
def site():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SiteHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _chromium_or_skip() -> str:
    try:
        return find_chromium()
    except RuntimeError:
        pytest.skip("no Chromium available — set AVION_CHROMIUM or install chromium")


_LAUNCH_ARGS = ("--no-sandbox", "--disable-gpu")


async def _record(base: str, launched, *, login: bool) -> list:
    """Perform the flow once; the session's recorder captures the events."""
    session = Arken(base_url=base)
    await session.start(backend=launched.backend)
    try:
        if login:
            await session.goto("/")
            await session.click(await session.find(ElementLocator.text("Sign in")))
            await session.fill(ElementLocator.css("input[name=user]"), "ada")
            await session.fill(ElementLocator.css("input[name=pass]"), "lovelace")
            await session.click(await session.find(ElementLocator.css("#go")))
            # The submit navigates (/do_login -> cookie -> /app). An
            # immediate next action would CANCEL the in-flight login
            # request — wait for the dashboard to actually land first.
            await session.find(ElementLocator.text("Dashboard"))
        # The search half: dashboard -> results.
        await session.goto("/app")
        await session.fill(ElementLocator.css("input[name=q]"), "avion flows")
        await session.click(await session.find(ElementLocator.css("#search")))
        return list(session.recorder.events)
    finally:
        await session.stop()


async def _assert_results_show(base: str, launched):
    """Reconnect and prove the UI actually shows the result, not just steps."""
    check = Arken(base_url=base)
    await check.start(backend=launched.backend)
    try:
        el = await check.find(ElementLocator.text(_RESULTS_TEXT))
        assert el is not None, "results page did not render the hit"
    finally:
        await check.stop()


class TestCompleteFlow:
    async def test_record_once_then_complete(self, site, tmp_path):
        """The user story: do the flow once; avion completes it again."""
        binary = _chromium_or_skip()
        launched = launch_chromium(
            Profile("flowdemo", root=tmp_path / "profiles"),
            headless=True,
            binary=binary,
            extra_args=_LAUNCH_ARGS,
            timeout=30.0,
        )
        try:
            events = await _record(site, launched, login=True)
            kinds = {e.type for e in events}
            assert EventType.NAVIGATE in kinds
            assert EventType.CLICK in kinds
            assert EventType.FILL in kinds

            outcome = await complete_flow(events, launched, base_url=site)
            assert outcome.completed, f"flow did not complete: {outcome.failed_steps}"
            assert outcome.failed_steps == []
            assert outcome.steps_done == outcome.steps_total
            # The real final URL — reached by form submission, not a goto.
            assert "/app?q=avion" in outcome.final_url
            # And the page truly shows the result.
            await _assert_results_show(site, launched)
        finally:
            launched.stop()

    async def test_login_survives_browser_death(self, site, tmp_path):
        """Log in once, kill the browser, relaunch — the flow still completes.

        The recorded flow contains NO login step: it starts at the gated
        dashboard.  Only the persistent profile carries the session.
        """
        binary = _chromium_or_skip()
        profile = Profile("persist", root=tmp_path / "profiles")

        first = launch_chromium(
            profile, headless=True, binary=binary, extra_args=_LAUNCH_ARGS, timeout=30.0
        )
        try:
            # Log in "by hand" once — the human-login equivalent.
            session = Arken(base_url=site)
            await session.start(backend=first.backend)
            try:
                await session.goto("/do_login?user=ada&pass=lovelace")
                # Record the SEARCH-ONLY flow while authenticated.
                await session.goto("/app")
                await session.fill(ElementLocator.css("input[name=q]"), "avion flows")
                await session.click(await session.find(ElementLocator.css("#search")))
                search_events = list(session.recorder.events)
            finally:
                await session.stop()
        finally:
            first.stop()  # the browser dies; the profile persists
        assert not first.alive

        second = launch_chromium(
            profile, headless=True, binary=binary, extra_args=_LAUNCH_ARGS, timeout=30.0
        )
        try:
            outcome = await complete_flow(search_events, second, base_url=site)
            assert outcome.completed, f"flow did not complete: {outcome.failed_steps}"
            assert "/app?q=avion" in outcome.final_url
            await _assert_results_show(site, second)
        finally:
            second.stop()

    async def test_dry_run_resolves_without_touching(self, site, tmp_path):
        """A dress rehearsal: every locator resolves, no action taken.

        Dry-run skips NAVIGATE too, so the browser is landed on the
        dashboard by hand first — then the recorded search half must
        resolve cleanly and change nothing.
        """
        from avion.events.replay import ReplayConfig

        binary = _chromium_or_skip()
        launched = launch_chromium(
            Profile("dryrun", root=tmp_path / "profiles"),
            headless=True,
            binary=binary,
            extra_args=_LAUNCH_ARGS,
            timeout=30.0,
        )
        try:
            # Log in by hand first (goto awaits the full redirect chain,
            # so the cookie is set and we land on the dashboard).
            setup = Arken(base_url=site)
            await setup.start(backend=launched.backend)
            try:
                await setup.goto("/do_login?user=ada&pass=lovelace")
            finally:
                await setup.stop()

            events = await _record(site, launched, login=False)
            search_only = [e for e in events if e.type != EventType.NAVIGATE]

            session = Arken(base_url=site)
            await session.start(backend=launched.backend)
            await session.goto("/app")
            await session.stop()

            outcome = await complete_flow(
                search_only, launched, base_url=site, config=ReplayConfig(dry_run=True)
            )
            assert outcome.completed, f"dry run failed to resolve: {outcome.failed_steps}"
            # Nothing was submitted: still on the dashboard, no results URL.
            assert "q=" not in outcome.final_url
        finally:
            launched.stop()
