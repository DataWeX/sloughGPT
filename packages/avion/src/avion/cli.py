"""avion CLI — drive a debuggable Chromium from the terminal.

Python-side parity with the chrome-devtools MCP verbs, over raw CDP.
One invocation = one CDP session; the browser keeps its state between
invocations, so ``open`` then ``click`` operate on the same page.

Usage::

    avion open http://localhost:3000/chat
    avion snapshot
    avion click "#send"            # --by css|text|testid|label|role|xpath
    avion fill "input[name=q]" "hello"
    avion wait "Start Training" --timeout 15
    avion shot page.png
    avion console --errors
    avion network --json
    avion launch --profile x           # own a browser; --no-wait detaches it
    avion run flow.json --profile x    # complete a recorded flow (or --cdp attach)
    avion repl --record flow.json      # author a flow by driving; avion run completes it
    avion repl

One invocation = one CDP session, but ``console``/``network`` keep history:
each session appends what it captured to a per-endpoint store, and ``open``
starts a fresh capture — so ``open`` then ``click`` then ``console`` reports
the errors the page produced, not an empty list.

Environment: AVION_CDP_URL overrides the debug endpoint
(default ws://localhost:9222). Requires a Chromium started with
``--remote-debugging-port=9222`` — on hosts without unprivileged
user namespaces add ``--no-sandbox --password-store=basic`` so
navigation cannot stall. No dependencies: the websocket transport
is avion's own stdlib client.
AVION_STATE_DIR relocates the capture store (default: $TMPDIR/avion).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from avion.events.models import Event, EventType

_DEFAULT_CDP = os.environ.get("AVION_CDP_URL", "ws://localhost:9222")
_HINT = (
    "Start one with:\n"
    "  chromium --headless --no-sandbox --password-store=basic"
    " --remote-debugging-port=9222\n"
    "(or set AVION_CDP_URL to an existing debug endpoint)"
)


def _state_dir() -> Path:
    """Where captured console/network events live between invocations."""
    override = os.environ.get("AVION_STATE_DIR")
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "avion"


def _events_path(cdp_url: str) -> Path:
    key = hashlib.sha1(cdp_url.encode()).hexdigest()[:16]
    return _state_dir() / f"events-{key}.json"


def _load_events(cdp_url: str) -> dict[str, list]:
    """Events captured by earlier invocations against this endpoint."""
    try:
        data = json.loads(_events_path(cdp_url).read_text())
    except (OSError, ValueError):
        return {"console": [], "network": []}
    return {
        "console": list(data.get("console", [])),
        "network": list(data.get("network", [])),
    }


def _reset_events(cdp_url: str) -> None:
    try:
        _events_path(cdp_url).unlink()
    except OSError:
        pass


def _merge(history: list[dict[str, Any]], live: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """History plus this session's replay, never repeating a known entry."""
    out = list(history)
    for entry in live:
        if entry not in history:
            out.append(entry)
    return out


def _persist_events(cdp_url: str, backend: Any) -> None:
    """Append this session's capture to the endpoint's store (atomic write)."""
    existing = _load_events(cdp_url)
    data = {
        "console": existing["console"] + backend.console_entries(),
        "network": existing["network"] + backend.network_entries(),
    }
    path = _events_path(cdp_url)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)
    except OSError:
        pass  # capture is best-effort; never fail the command over it


def _fail(msg: str, code: int = 1) -> int:
    print(f"avion: {msg}", file=sys.stderr)
    return code


async def _session(cdp_url: str, fn: Callable[[Any], Any], cmd: str | None = None) -> int:
    """Run one command against a live CDP endpoint; map errors to exits."""
    from avion.backends.cdp import CdpBackend

    backend = CdpBackend(cdp_url=cdp_url)
    try:
        await backend.start()
    except Exception as e:
        print(f"avion: cannot reach Chromium at {cdp_url}: {e}", file=sys.stderr)
        print(_HINT, file=sys.stderr)
        return 2
    if cmd == "open":
        _reset_events(cdp_url)  # a new journey starts a fresh capture
    # Chrome re-sends its buffered console to every new client. A command that
    # drives the page must drop it (or each run would duplicate the previous
    # run's output); console/network keep it so attaching late still reports
    # what the page said.
    if cmd not in ("console", "network"):
        backend.clear_events()
    try:
        result = await fn(backend)
        return int(result) if isinstance(result, int) else 0
    except Exception as e:
        return _fail(str(e))
    finally:
        await backend.stop()
        if cmd not in ("console", "network"):
            _persist_events(cdp_url, backend)


def _locator(value: str, by: str):
    from avion.core.element import ElementLocator, Selector, SelectorStrategy

    if by == "css":
        return ElementLocator.css(value)
    if by == "text":
        return ElementLocator.text(value)
    if by == "testid":
        return ElementLocator.test_id(value)
    if by == "label":
        return ElementLocator.label(value)
    if by == "role":
        return ElementLocator.role(value)
    if by == "xpath":
        return ElementLocator(selectors=(Selector(SelectorStrategy.XPATH, value),))
    raise SystemExit(_fail(f"unknown --by {by!r} (css|text|testid|label|role|xpath)"))


# ── commands ────────────────────────────────────────────────────────────


async def _cmd_open(b, url: str) -> int:
    await b.navigate(url)
    await b.wait_for_timeout(400)
    title = await b.get_title()
    print(f"{await b.get_url()}  ({title})")
    return 0


async def _cmd_snapshot(b) -> int:
    url = await b.get_url()
    title = await b.get_title()
    print(f"URL:   {url}")
    print(f"Title: {title}")
    tree = await b.get_accessibility_tree()
    nodes = tree.get("nodes", [])
    lines = []
    for node in nodes:
        role = (node.get("role") or {}).get("value", "")
        name = (node.get("name") or {}).get("value", "")
        if not name or role in ("none", "presentation", "InlineTextBox"):
            continue
        lines.append(f"  {role:<14} {name[:80]}")
    print(f"A11y:  {len(nodes)} nodes, {len(lines)} named")
    for line in lines[:40]:
        print(line)
    if len(lines) > 40:
        print(f"  ... {len(lines) - 40} more")
    return 0


async def _cmd_click(b, value: str, by: str) -> int:
    element = await b.find_element(_locator(value, by))
    if element is None:
        return _fail(f"element not found ({by}): {value}")
    await b.click(element)
    print(f"clicked {by}: {value}")
    return 0


async def _cmd_fill(b, value: str, text: str, by: str) -> int:
    element = await b.find_element(_locator(value, by))
    if element is None:
        return _fail(f"element not found ({by}): {value}")
    await b.fill(element, text)
    print(f"filled {by}: {value}")
    return 0


async def _cmd_wait(b, text: str, timeout: float) -> int:
    if await b.wait_for_text(text, timeout=timeout):
        print(f"found: {text}")
        return 0
    return _fail(f"timeout after {timeout}s: {text!r} not on page")


async def _cmd_shot(b, path: str) -> int:
    data = await b.screenshot()
    with open(path, "wb") as fh:
        fh.write(data)
    print(f"wrote {path} ({len(data)} bytes)")
    return 0


async def _cmd_console(
    b, errors_only: bool, as_json: bool, history: list[dict[str, Any]] | None = None
) -> int:
    entries = _merge(history or [], b.console_entries())
    if errors_only:
        entries = [e for e in entries if e["level"] in ("error", "assert")]
    if as_json:
        print(json.dumps(entries, indent=2))
        return 0
    if not entries:
        print("(no console output)")
        return 0
    for e in entries:
        print(f"[{e['level']}] {e['text']}")
    return 0


async def _cmd_network(b, as_json: bool, history: list[dict[str, Any]] | None = None) -> int:
    entries = _merge(history or [], b.network_entries())
    if as_json:
        print(json.dumps(entries, indent=2))
        return 0
    if not entries:
        print("(no network activity)")
        return 0
    for e in entries:
        status = e.get("error") or e.get("status", "?")
        print(f"{status}  {e.get('method', '')}  {e.get('url', '')}")
    return 0


async def _cmd_repl(cdp_url: str, record_path: str | None = None) -> int:
    """Persistent session: read verb-args lines until quit."""
    from avion.backends.cdp import CdpBackend

    backend = CdpBackend(cdp_url=cdp_url)
    try:
        await backend.start()
    except Exception as e:
        print(f"avion: cannot reach Chromium at {cdp_url}: {e}", file=sys.stderr)
        print(_HINT, file=sys.stderr)
        return 2
    recorder = None
    if record_path:
        from avion.events.recorder import EventRecorder

        recorder = EventRecorder(session_name="repl")
        print(f"avion repl — connected to {cdp_url} (quit to exit) — RECORDING -> {record_path}")
    else:
        print(f"avion repl — connected to {cdp_url} (quit to exit)")
    try:
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: sys.stdin.readline()
                )
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            if line in ("quit", "exit", "q"):
                break
            try:
                code = await _dispatch(
                    backend, shlex.split(line), in_session=True, recorder=recorder
                )
            except SystemExit as e:
                code = int(e.code or 1)
            if code:
                print(f"(exit {code})")
    finally:
        await backend.stop()
        _persist_events(cdp_url, backend)
        if recorder is not None:
            recorder.save(record_path)
            print(f"recorded {len(recorder)} events -> {record_path}")
    return 0


async def _dispatch(backend, argv: list[str], in_session: bool = False, recorder=None) -> int:
    """Run one command against an existing backend (repl path)."""
    verb, rest = argv[0], argv[1:]
    if verb == "open" and len(rest) >= 1:
        code = await _cmd_open(backend, rest[0])
        rec = _replayable(recorder, code)
        if rec is not None:
            rec.record(EventType.NAVIGATE, name=rest[0])
        return code
    if verb == "snapshot":
        return await _cmd_snapshot(backend)
    if verb == "click" and len(rest) >= 1:
        by, rest = _pop_by(rest)
        code = await _cmd_click(backend, rest[0], by)
        rec = _replayable(recorder, code)
        if rec is not None:
            loc = _locator(rest[0], by)
            # The click resolved the element (a find) — record that as the
            # evidence barrier first, exactly like session-driven
            # recordings, so replay settles after submits instead of
            # racing them. Unparseable (xpath) names are skipped as
            # noise by replay itself.
            rec.record(EventType.ELEMENT_FOUND, name=loc.describe())
            rec.record(EventType.CLICK, name=loc.describe())
        return code
    if verb == "fill" and len(rest) >= 2:
        by, rest = _pop_by(rest)
        code = await _cmd_fill(backend, rest[0], rest[1], by)
        rec = _replayable(recorder, code)
        if rec is not None:
            loc = _locator(rest[0], by)
            rec.record(EventType.ELEMENT_FOUND, name=loc.describe())
            rec.record(EventType.FILL, name=loc.describe(), data={"value": rest[1]})
        return code
    if verb == "wait" and len(rest) >= 1:
        timeout = 10.0
        if "--timeout" in rest:
            i = rest.index("--timeout")
            timeout = float(rest[i + 1])
            rest = rest[:i] + rest[i + 2 :]
        code = await _cmd_wait(backend, rest[0], timeout)
        rec = _replayable(recorder, code)
        if rec is not None:
            from avion.core.element import ElementLocator

            rec.record(EventType.ELEMENT_FOUND, name=ElementLocator.text(rest[0]).describe())
        return code
    if verb == "shot" and len(rest) >= 1:
        return await _cmd_shot(backend, rest[0])
    if verb == "console":
        return await _cmd_console(backend, "--errors" in rest, "--json" in rest)
    if verb == "network":
        return await _cmd_network(backend, "--json" in rest)
    if verb == "help":
        print("verbs: open snapshot click fill wait shot console network quit")
        print("quote arguments with spaces: fill 'input[name=q]' \"hello world\"")
        return 0
    return _fail(f"unknown or incomplete command: {' '.join(argv)}")


def _replayable(recorder, code: int):
    """Record only what replay can re-drive: successes, when recording.

    A failed verb is a typo the human retries — putting it in the flow
    would break every future run of it.
    """
    return recorder if recorder is not None and code == 0 else None


def _pop_by(rest: list[str]) -> tuple[str, list[str]]:
    """Strip optional ``--by <strategy>``; return strategy + remaining args."""
    by = "css"
    if "--by" in rest:
        i = rest.index("--by")
        by = rest[i + 1]
        rest = rest[:i] + rest[i + 2 :]
    return by, rest


def _load_flow_events(path: str) -> list[Event]:
    """Events from a flow file saved by ``EventRecorder.save()``.

    A malformed file surfaces as ``ValueError`` — the CLI turns that into a
    loud message and exit 1, never a traceback.
    """
    payload = json.loads(Path(path).read_text())
    raw = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(raw, list):
        raise ValueError(f"no events in flow file: {path}")
    try:
        return [Event.from_dict(d) for d in raw]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed event in {path}: {exc}") from exc


def _cmd_launch(args: argparse.Namespace) -> int:
    """Own a browser start: visible until Enter, or detached with --no-wait."""
    from avion.launch import Profile, launch_chromium

    launched = launch_chromium(Profile(args.profile), headless=args.headless, extra_args=args.arg)
    print(f"launched profile={args.profile} pid={launched.process.pid}")
    print(f"endpoint={launched.endpoint.base_ws_url}")
    if args.no_wait:
        # Detach: the browser outlives this invocation — that is the point
        # (persistent profile, other verbs attach to it).
        print(f"detach: export AVION_CDP_URL={launched.endpoint.base_ws_url}")
        return 0
    print("browser running — press Enter to stop it...", flush=True)
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
    launched.stop()
    print("stopped.")
    return 0


async def _cmd_run(args: argparse.Namespace) -> int:
    """Complete a recorded flow: own a profile browser or attach via --cdp."""
    from avion.backends.cdp import CdpBackend
    from avion.events.replay import ReplayConfig
    from avion.flow import complete_flow
    from avion.launch import Profile, launch_chromium

    try:
        events = _load_flow_events(args.flow)
    except (OSError, ValueError) as exc:
        return _fail(str(exc))

    launched = None
    if args.profile:
        launched = launch_chromium(
            Profile(args.profile), headless=not args.headed, extra_args=args.arg
        )
        target: Any = launched
        print(f"running flow on profile={args.profile} ({len(events)} events)", file=sys.stderr)
    else:
        target = CdpBackend(cdp_url=args.cdp)
        print(f"running flow on {args.cdp} ({len(events)} events)", file=sys.stderr)
    try:
        outcome = await complete_flow(
            events,
            target,
            base_url=args.base_url,
            config=ReplayConfig(dry_run=args.dry_run),
        )
    except Exception as exc:
        return _fail(f"flow run failed: {exc}")
    finally:
        if launched is not None:
            launched.stop()
    if args.json:
        print(json.dumps(outcome.to_dict(), indent=2))
    else:
        # Summarize, don't dump: one status line + the failures, nothing else.
        mark = "completed" if outcome.completed else "FAILED"
        print(
            f"{mark}: {outcome.steps_done}/{outcome.steps_total} steps — final: {outcome.final_url}"
        )
        for step in outcome.failed_steps:
            print(
                f"  step {step.get('event_seq', '?')} {step.get('action', '')}:"
                f" {step.get('error') or 'failed'}"
            )
    return 0 if outcome.completed else 1


# ── parser / entry ──────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="avion", description="Drive a debuggable Chromium over CDP.")
    p.add_argument("--cdp", default=_DEFAULT_CDP, help="CDP websocket endpoint")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("open", help="navigate to a URL")
    sp.add_argument("url")

    sub.add_parser("snapshot", help="url, title and named a11y nodes")

    sp = sub.add_parser("click", help="click an element")
    sp.add_argument("selector")
    sp.add_argument(
        "--by", default="css", choices=["css", "text", "testid", "label", "role", "xpath"]
    )

    sp = sub.add_parser("fill", help="fill an input")
    sp.add_argument("selector")
    sp.add_argument("text")
    sp.add_argument(
        "--by", default="css", choices=["css", "text", "testid", "label", "role", "xpath"]
    )

    sp = sub.add_parser("wait", help="wait until page text appears")
    sp.add_argument("text")
    sp.add_argument("--timeout", type=float, default=10.0)

    sp = sub.add_parser("shot", help="save a PNG screenshot")
    sp.add_argument("path")

    sp = sub.add_parser("console", help="show captured console output")
    sp.add_argument("--errors", action="store_true", help="errors only")
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("network", help="show captured network requests")
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("repl", help="interactive session (same verbs)")
    sp.add_argument(
        "--record",
        metavar="FLOW_JSON",
        default=None,
        help="record successful verbs into a flow file avion run can complete",
    )

    sp = sub.add_parser("launch", help="start a persistent-profile browser")
    sp.add_argument("--profile", default="default", help="named profile (logins persist in it)")
    sp.add_argument("--headless", action="store_true", help="run without a visible window")
    sp.add_argument(
        "--no-wait",
        action="store_true",
        help="detach: leave it running and print its endpoint",
    )
    sp.add_argument(
        "--arg",
        action="append",
        default=[],
        metavar="CHROMIUM_ARG",
        help="extra chromium arg, repeatable (use --arg=--no-sandbox for dashed values)",
    )

    sp = sub.add_parser(
        "run",
        help="complete a recorded flow (an EventRecorder save file)",
        description=(
            "Owns a browser with --profile; otherwise attaches to the global "
            "--cdp flag (before the subcommand) or AVION_CDP_URL."
        ),
    )
    sp.add_argument("flow", help="flow JSON saved by EventRecorder.save()")
    sp.add_argument("--profile", default=None, help="launch and own this profile's browser")
    sp.add_argument("--base-url", default="", help="base URL for relative navigations")
    sp.add_argument(
        "--headed",
        action="store_true",
        help="show the window when launching (default: headless)",
    )
    sp.add_argument("--dry-run", action="store_true", help="resolve every locator, touch nothing")
    sp.add_argument("--json", action="store_true", help="print the full report as JSON")
    sp.add_argument(
        "--arg",
        action="append",
        default=[],
        metavar="CHROMIUM_ARG",
        help="extra chromium arg when launching, repeatable (use --arg=--no-sandbox)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.cmd:
        build_parser().print_help()
        return 0
    if args.cmd == "repl":
        return asyncio.run(_cmd_repl(args.cdp, record_path=args.record))
    if args.cmd == "launch":
        return _cmd_launch(args)
    if args.cmd == "run":
        return asyncio.run(_cmd_run(args))

    async def run(b) -> int:
        if args.cmd == "open":
            return await _cmd_open(b, args.url)
        if args.cmd == "snapshot":
            return await _cmd_snapshot(b)
        if args.cmd == "click":
            return await _cmd_click(b, args.selector, args.by)
        if args.cmd == "fill":
            return await _cmd_fill(b, args.selector, args.text, args.by)
        if args.cmd == "wait":
            return await _cmd_wait(b, args.text, args.timeout)
        if args.cmd == "shot":
            return await _cmd_shot(b, args.path)
        if args.cmd == "console":
            history = _load_events(args.cdp)["console"]
            return await _cmd_console(b, args.errors, args.json, history=history)
        if args.cmd == "network":
            history = _load_events(args.cdp)["network"]
            return await _cmd_network(b, args.json, history=history)
        return _fail(f"unknown command {args.cmd}")

    return asyncio.run(_session(args.cdp, run, cmd=args.cmd))


if __name__ == "__main__":
    raise SystemExit(main())
