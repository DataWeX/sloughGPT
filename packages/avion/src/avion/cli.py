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
    avion repl

One invocation = one CDP session, but ``console``/``network`` keep history:
each session appends what it captured to a per-endpoint store, and ``open``
starts a fresh capture — so ``open`` then ``click`` then ``console`` reports
the errors the page produced, not an empty list.

Environment: AVION_CDP_URL overrides the debug endpoint
(default ws://localhost:9222). Requires a Chromium started with
``--remote-debugging-port=9222`` — on hosts without unprivileged
user namespaces add ``--no-sandbox --password-store=basic`` so
navigation cannot stall — and ``pip install websockets``.
AVION_STATE_DIR relocates the capture store (default: $TMPDIR/avion).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

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
        return ElementLocator(
            selectors=(Selector(SelectorStrategy.XPATH, value),)
        )
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


async def _cmd_network(
    b, as_json: bool, history: list[dict[str, Any]] | None = None
) -> int:
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


async def _cmd_repl(cdp_url: str) -> int:
    """Persistent session: read verb-args lines until quit."""
    from avion.backends.cdp import CdpBackend

    backend = CdpBackend(cdp_url=cdp_url)
    try:
        await backend.start()
    except Exception as e:
        print(f"avion: cannot reach Chromium at {cdp_url}: {e}", file=sys.stderr)
        print(_HINT, file=sys.stderr)
        return 2
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
                code = await _dispatch(backend, line.split(), in_session=True)
            except SystemExit as e:
                code = int(e.code or 1)
            if code:
                print(f"(exit {code})")
    finally:
        await backend.stop()
        _persist_events(cdp_url, backend)
    return 0


async def _dispatch(backend, argv: list[str], in_session: bool = False) -> int:
    """Run one command against an existing backend (repl path)."""
    verb, rest = argv[0], argv[1:]
    if verb == "open" and len(rest) >= 1:
        return await _cmd_open(backend, rest[0])
    if verb == "snapshot":
        return await _cmd_snapshot(backend)
    if verb == "click" and len(rest) >= 1:
        by, rest = _pop_by(rest)
        return await _cmd_click(backend, rest[0], by)
    if verb == "fill" and len(rest) >= 2:
        by, rest = _pop_by(rest)
        return await _cmd_fill(backend, rest[0], rest[1], by)
    if verb == "wait" and len(rest) >= 1:
        timeout = 10.0
        if "--timeout" in rest:
            i = rest.index("--timeout")
            timeout = float(rest[i + 1])
            rest = rest[:i] + rest[i + 2 :]
        return await _cmd_wait(backend, rest[0], timeout)
    if verb == "shot" and len(rest) >= 1:
        return await _cmd_shot(backend, rest[0])
    if verb == "console":
        return await _cmd_console(backend, "--errors" in rest, "--json" in rest)
    if verb == "network":
        return await _cmd_network(backend, "--json" in rest)
    if verb == "help":
        print("verbs: open snapshot click fill wait shot console network quit")
        return 0
    return _fail(f"unknown or incomplete command: {' '.join(argv)}")


def _pop_by(rest: list[str]) -> tuple[str, list[str]]:
    """Strip optional ``--by <strategy>``; return strategy + remaining args."""
    by = "css"
    if "--by" in rest:
        i = rest.index("--by")
        by = rest[i + 1]
        rest = rest[:i] + rest[i + 2 :]
    return by, rest


# ── parser / entry ──────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="avion", description="Drive a debuggable Chromium over CDP."
    )
    p.add_argument("--cdp", default=_DEFAULT_CDP, help="CDP websocket endpoint")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("open", help="navigate to a URL")
    sp.add_argument("url")

    sub.add_parser("snapshot", help="url, title and named a11y nodes")

    sp = sub.add_parser("click", help="click an element")
    sp.add_argument("selector")
    sp.add_argument("--by", default="css",
                    choices=["css", "text", "testid", "label", "role", "xpath"])

    sp = sub.add_parser("fill", help="fill an input")
    sp.add_argument("selector")
    sp.add_argument("text")
    sp.add_argument("--by", default="css",
                    choices=["css", "text", "testid", "label", "role", "xpath"])

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

    sub.add_parser("repl", help="interactive session (same verbs)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.cmd:
        build_parser().print_help()
        return 0
    if args.cmd == "repl":
        return asyncio.run(_cmd_repl(args.cdp))

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
