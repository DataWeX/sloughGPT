#!/usr/bin/env python3
"""
Run the UX_FLOWS user journeys (docs/UX_FLOWS.md) against a live stack.

Engine: avion (packages/avion, formerly voyager/arken) on Firefox — chromium
never commits navigations on this hardware.

Usage:
    .venv/bin/python scripts/run_ux_flows.py                # all journeys
    .venv/bin/python scripts/run_ux_flows.py --flow 2-write
    .venv/bin/python scripts/run_ux_flows.py --list
    .venv/bin/python scripts/run_ux_flows.py --strict-errors

Env:
    SLO_WEB_URL         default http://localhost:5175
    SLO_API_URL         default http://localhost:8000
    SLO_JOURNEY_BROWSER firefox (default) | chromium
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "packages", "avion", "src"))

WEB = os.environ.get("SLO_WEB_URL", "http://localhost:5175")
API = os.environ.get("SLO_API_URL", "http://localhost:8000")
BROWSER = os.environ.get("SLO_JOURNEY_BROWSER", "firefox")
SHOTS = os.environ.get("SLO_JOURNEY_SHOTS", "/home/mana/.cache/slog-journeys/shots/ux")
REPORT = os.environ.get("SLO_JOURNEY_REPORT", "/home/mana/.cache/slog-journeys/ux-flows-report.json")

ASSISTANT_BUBBLE = '[aria-label="Message from Assistant"]'

# ── Step helpers (avion TaskStep builders) ──────────────────────────────────


def _step(name: str, fn: Callable[[dict[str, Any]], Any], optional: bool = False):
    from avion.core.task import TaskStep

    async def action(ctx: dict[str, Any]) -> None:
        res = fn(ctx)
        if hasattr(res, "__await__"):
            await res

    return TaskStep(name=name, action=action, optional=optional)


def s_goto(url: str, verify: str | None = None):
    async def action(ctx):
        ok = await ctx["arken"].goto(url)
        assert ok, f"goto failed: {url}"
        # Full reloads re-show the StartupOverlay (aria-label="Startup progress")
        # for every stage — wait until it fully unmounts, sustained, before asserting.
        page = ctx["backend"].page
        deadline = time.monotonic() + 60
        gone_since: float | None = None
        while time.monotonic() < deadline:
            n = await page.locator('[aria-label="Startup progress"]').count()
            if n == 0:
                if gone_since is None:
                    gone_since = time.monotonic()
                elif time.monotonic() - gone_since >= 1.0:
                    break
            else:
                gone_since = None
            await asyncio.sleep(0.25)
        else:
            raise AssertionError("boot overlay never dismissed")
        if verify:
            found = await ctx["arken"].wait_for_text(verify, timeout=25)
            assert found, f"after goto: missing text {verify!r}"

    return _step(f"goto {url}", action)


def s_click(text: str, optional: bool = False, timeout: float = 10):
    async def action(ctx):
        from avion.core.element import ElementLocator

        # exact=True: substring text matching is case-insensitive and would
        # also hit accumulated session titles/messages ("...friendly email..."),
        # whose hidden copies are never visible.
        el = await ctx["arken"].find(ElementLocator.text(text, exact=True), timeout=timeout)
        await ctx["arken"].click(el)

    return _step(f"click '{text}'", action, optional=optional)


def s_wait(text: str, timeout: float = 20):
    async def action(ctx):
        ok = await ctx["arken"].wait_for_text(text, timeout=timeout)
        assert ok, f"text not found: {text!r}"

    return _step(f"wait '{text}'", action)


def s_send(prompt: str, reply_timeout: float = 90):
    """Fill the composer, press Enter, wait for the assistant bubble."""

    async def action(ctx):
        page = ctx["backend"].page  # async Playwright page — every call awaits
        ta = page.locator("textarea").last
        await ta.wait_for(timeout=20000)
        await ta.fill(prompt)
        await ta.press("Enter")
        await page.locator(ASSISTANT_BUBBLE).first.wait_for(timeout=reply_timeout * 1000)

    return _step("send + assistant reply", action)


def s_exists(selector: str, description: str, timeout: float = 15):
    async def action(ctx):
        page = ctx["backend"].page
        deadline = time.monotonic() + timeout
        while True:
            if await page.locator(selector).count() > 0:
                return
            if time.monotonic() >= deadline:
                raise AssertionError(f"missing: {description}")
            await asyncio.sleep(0.25)

    return _step(f"exists: {description}", action)


def s_upload(path: str):
    async def action(ctx):
        page = ctx["backend"].page
        await page.locator("input[type=file]").first.set_input_files(path)
        await ctx["arken"].wait_for_text(os.path.basename(path), timeout=15)

    return _step(f"upload {os.path.basename(path)}", action)


def s_note(description: str, fn: Callable[[dict[str, Any]], Any]):
    """Record an observation (never fails the flow unless fn raises)."""
    return _step(description, fn)


# ── The 12 journeys (+ home baseline) ───────────────────────────────────────


@dataclass
class Flow:
    id: str
    label: str
    url: str
    build: Callable[[], list]


def _chat_flow(flow_id: str, label: str, url: str, prompt: str, pill: str | None = None,
               marker: str | None = None) -> Flow:
    def build():
        steps = [s_goto(url, verify="Chat")]
        if marker:
            steps.append(s_wait(marker))
        if pill:
            # Gate on the pill itself: the marker label can substring-match
            # decoys (e.g. 'Tone' in 'Tone_flexibility') before the ModeBar
            # mounts, leaving the click racing session hydration.
            steps.append(s_wait(pill, timeout=30))
            steps.append(s_click(pill, timeout=20))
        steps.append(s_send(prompt))
        return steps

    return Flow(flow_id, label, url, build)


def _lease_file() -> str:
    path = "/home/mana/.cache/slog-journeys/lease.txt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(
            "LEASE AGREEMENT\n"
            "Section 4.2: The move-out notice period is 30 days in writing.\n"
            "Rent is due on the 1st of each month.\n"
        )
    return path


def flow_home() -> Flow:
    def build():
        return [s_goto("/"), s_wait("Teach me"), s_wait("Chat")]

    return Flow("0-home", "Home screen + sidebar", "/", build)


def flow_read() -> Flow:
    def build():
        return [
            s_goto("/chat?mode=read", verify="Chat"),
            s_upload(_lease_file()),
            s_send("What's the move-out notice period?"),
        ]

    return Flow("3-read", "Read My Files", "/chat?mode=read", build)


def flow_talk() -> Flow:
    def build():
        # Firefox has no Web Speech API — the app must open voice mode and
        # degrade with a plain-language notice (never a crash/jargon).
        return [
            s_goto("/chat?mode=talk", verify="Chat"),
            s_exists('button[aria-label="Exit voice mode"]', "voice mode open"),
            s_wait("Speech recognition not supported", timeout=15),
        ]

    return Flow("7-talk", "Talk Out Loud", "/chat?mode=talk", build)


def flow_training() -> Flow:
    def build():
        return [
            s_goto("/training", verify="Teach"),
            s_wait("Pick your data"),
            s_wait("Next: Configure"),
            s_click("Next: Configure", optional=True),
        ]

    return Flow("12-training", "Train My AI (3-click)", "/training", build)


FLOWS: list[Flow] = [
    flow_home(),
    _chat_flow(
        "1-chat",
        "Chat That Remembers Me",
        "/chat",
        "What was that recipe we talked about yesterday?",
    ),
    _chat_flow(
        "2-write",
        "Writing Assistant",
        "/chat?mode=write",
        "Tell my landlord the sink is broken and ask when he can fix it",
        marker="Tone",
        pill="Friendly",
    ),
    flow_read(),
    _chat_flow(
        "4-brainstorm",
        "Brainstorm With Me",
        "/chat?mode=brainstorm",
        "Gift ideas for my dad's 60th birthday, he loves fishing and cooking",
        marker="Topic",
        pill="Gift Ideas",
    ),
    _chat_flow(
        "5-rewrite",
        "Rewrite & Polish",
        "/chat?mode=rewrite",
        "Rewrite: teh quick bown fox dont jump over teh lazy dogg",
        marker="Action",
        pill="Fix Grammar",
    ),
    _chat_flow(
        "6-create",
        "Create Images",
        "/chat?mode=create",
        "Create an image: a cozy cabin in the mountains at sunset",
        marker="Style",
        pill="Realistic",
    ),
    flow_talk(),
    _chat_flow(
        "8-translate",
        "Translate",
        "/chat?mode=translate",
        "How much does this cost?",
        marker="To",
        pill="EN→ES",
    ),
    _chat_flow(
        "9-decide",
        "Help Me Decide",
        "/chat?mode=decide",
        "Should I take the job in New York or stay in my current role?",
        marker="Output",
        pill="Pros & Cons",
    ),
    _chat_flow(
        "10-explain",
        "Explain Things Simply",
        "/chat?mode=explain",
        "How does the internet work?",
        marker="Level",
        pill="Simple",
    ),
    _chat_flow(
        "11-wellness",
        "Make Me Well (Wellness)",
        "/chat?mode=wellness",
        "I want a short sleep story about the ocean",
        marker="Type",
        pill="Sleep Story",
    ),
    flow_training(),
]


# ── Preflight ───────────────────────────────────────────────────────────────


def _http_ok(url: str, timeout: float = 5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


# ── Runner ──────────────────────────────────────────────────────────────────


async def run(flows: list[Flow], headed: bool, strict_errors: bool) -> int:
    from avion import Avion
    from avion.backends.playwright import PlaywrightBackend
    from avion.core.task import Task

    os.makedirs(SHOTS, exist_ok=True)
    backend = PlaywrightBackend(headless=not headed, browser_type=BROWSER)
    console_errors: list[str] = []
    network_errors: list[str] = []

    results: list[dict[str, Any]] = []

    def persist() -> dict[str, Any]:
        """Write the report after every flow — a crash mid-run must not lose results."""
        report = {
            "started_web": WEB,
            "browser": BROWSER,
            "flows": results,
            "passed": sum(1 for r in results if r["status"] == "passed"),
            "failed": sum(1 for r in results if r["status"] != "passed"),
            "console_error_count": len(console_errors),
            "network_error_count": len(network_errors),
            "console_errors_sample": console_errors[:20],
            "network_errors_sample": network_errors[:20],
        }
        with open(REPORT, "w") as f:
            json.dump(report, f, indent=2)
        return report

    async with Avion(base_url=WEB) as a:
        await a.start(backend)
        page = backend.page
        page.on(
            "console",
            lambda m: console_errors.append(m.text[:300]) if m.type == "error" else None,
        )
        page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"[:300]))
        page.on(
            "response",
            lambda r: network_errors.append(f"{r.status} {r.url[:160]}")
            if r.status >= 400
            else None,
        )

        for flow in flows:
            t0 = time.perf_counter()
            task = Task(name=flow.id, description=flow.label, steps=flow.build())
            result = await a.run_task(task)
            dur = time.perf_counter() - t0
            rec = result.to_dict()
            rec["label"] = flow.label
            rec["url"] = flow.url
            rec["duration_s"] = round(dur, 2)
            if result.status.value != "passed":
                try:
                    shot = os.path.join(SHOTS, f"{flow.id}-FAIL.png")
                    await page.screenshot(path=shot)
                    rec["screenshot"] = shot
                except Exception:
                    pass
            results.append(rec)
            mark = "PASS" if result.status.value == "passed" else "FAIL"
            print(
                f"[{mark}] {flow.id:14s} {flow.label} ({dur:.1f}s) {result.error}".rstrip(),
                flush=True,
            )
            persist()

        try:
            print(a._reporter.report("terminal"))
        except Exception:
            pass

        await a.stop()

    report = persist()
    print(
        f"\n{report['passed']}/{len(results)} journeys passed | "
        f"console_errors={report['console_error_count']} "
        f"network_errors={report['network_error_count']}",
        flush=True,
    )
    print(f"report: {REPORT}", flush=True)

    failed = report["failed"]
    if strict_errors and (console_errors or network_errors):
        print("strict-errors: console/network errors present → failing")
        failed = max(failed, 1)
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow", action="append", help="run only these flow ids (repeatable)")
    ap.add_argument("--list", action="store_true", help="list flow ids and exit")
    ap.add_argument("--headed", action="store_true", help="show the browser window")
    ap.add_argument("--strict-errors", action="store_true", help="fail on console/network errors")
    args = ap.parse_args()

    if args.list:
        for f in FLOWS:
            print(f"{f.id:14s} {f.label}")
        return 0

    flows = FLOWS
    if args.flow:
        wanted = set(args.flow)
        flows = [f for f in FLOWS if f.id in wanted or f.id.split("-", 1)[-1] in wanted]
        if not flows:
            print(f"no flows match {sorted(wanted)}")
            return 2

    if not _http_ok(f"{API}/health"):
        print(f"API not reachable at {API}/health — start uvicorn first")
        return 2
    if not _http_ok(f"{WEB}/"):
        print(f"web not reachable at {WEB} — start vite first")
        return 2

    return asyncio.run(run(flows, headed=args.headed, strict_errors=args.strict_errors))


if __name__ == "__main__":
    sys.exit(main())
