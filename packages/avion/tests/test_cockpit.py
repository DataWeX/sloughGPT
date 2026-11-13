"""Cockpit tests — the live agent console: four panes over a local server.

The cockpit is the *outside* half of the control plane: a stdlib HTTP
server serving a self-contained page (log / work / message / terminal),
an incremental event log (``?since=`` long-poll), and ``POST /command``
straight into the agent's ``CommandInbox``. Nothing here touches a real
browser or model — the seam being tested is ours (page, routes,
validation, wiring), not Chromium's.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from avion.ai import AgentConfig
from avion.ai.control import CommandInbox, ControlState
from avion.cli import build_parser
from avion.cockpit import Cockpit, EventStream


def run(coro):
    return asyncio.run(coro)


async def _request(port: int, method: str, path: str, body: str | None = None) -> tuple[int, bytes]:
    """Raw HTTP over asyncio streams — no client library involved."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    payload = body.encode() if body else b""
    head = f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n"
    if body is not None:
        head += f"Content-Length: {len(payload)}\r\nContent-Type: application/json\r\n"
    head += "\r\n"
    writer.write(head.encode() + payload)
    await writer.drain()
    raw = await reader.read()
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass
    status_line, _, rest = raw.partition(b"\r\n")
    status = int(status_line.split()[1])
    _, _, response_body = rest.partition(b"\r\n\r\n")
    return status, response_body


def _make_cockpit(**overrides) -> Cockpit:
    defaults = dict(
        inbox=CommandInbox(),
        events=EventStream(),
        state=lambda: {"running": True, "paused": False, "steps": 0},
        records=lambda: [],
    )
    defaults.update(overrides)
    return Cockpit(host="127.0.0.1", port=0, **defaults)


class TestRoutes:
    def test_serves_the_four_pane_page(self):
        async def scenario():
            ck = _make_cockpit()
            url = await ck.start()
            try:
                assert url.startswith("http://127.0.0.1:")
                return await _request(ck.port, "GET", "/")
            finally:
                await ck.stop()

        status, body = run(scenario())
        assert status == 200
        html = body.decode()
        for pane in ("log", "work", "message", "terminal"):
            assert f'id="{pane}"' in html, f"missing {pane} pane"

    def test_post_command_lands_in_the_inbox(self):
        async def scenario():
            inbox = CommandInbox()
            ck = _make_cockpit(inbox=inbox)
            await ck.start()
            try:
                status, body = await _request(
                    ck.port,
                    "POST",
                    "/command",
                    json.dumps({"kind": "set_params", "payload": {"max_steps": 3}}),
                )
                assert status == 200, body
                commands = inbox.drain(ControlState(), AgentConfig())
                assert [c["kind"] for c in commands] == ["set_params"]
                assert inbox.pending == 0
                return status
            finally:
                await ck.stop()

        assert run(scenario()) == 200

    @pytest.mark.parametrize(
        "bad_body",
        [
            "{not json",
            "[]",
            json.dumps({"kind": "explode"}),
            json.dumps({"kind": "pause", "payload": "x"}),
        ],
    )
    def test_post_command_rejects_malformed(self, bad_body):
        async def scenario():
            inbox = CommandInbox()
            ck = _make_cockpit(inbox=inbox)
            await ck.start()
            try:
                status, _ = await _request(ck.port, "POST", "/command", bad_body)
                assert inbox.pending == 0, "a malformed command must never be queued"
                return status
            finally:
                await ck.stop()

        assert run(scenario()) == 400

    def test_state_endpoint_returns_the_state_provider_json(self):
        async def scenario():
            ck = _make_cockpit(state=lambda: {"paused": True, "steps": 7})
            await ck.start()
            try:
                return await _request(ck.port, "GET", "/state")
            finally:
                await ck.stop()

        status, body = run(scenario())
        assert status == 200
        assert json.loads(body) == {"paused": True, "steps": 7}

    def test_unknown_path_is_404(self):
        async def scenario():
            ck = _make_cockpit()
            await ck.start()
            try:
                return await _request(ck.port, "GET", "/nope")
            finally:
                await ck.stop()

        status, _ = run(scenario())
        assert status == 404


class TestEventStream:
    def test_incremental_since_semantics(self):
        events = EventStream()
        events.emit("step", {"n": 1})
        events.emit("step", {"n": 2})
        first = events.since(0)
        assert [e["seq"] for e in first] == [1, 2]
        assert events.since(first[-1]["seq"]) == []
        events.emit("control", {"kind": "pause"})
        tail = events.since(first[-1]["seq"])
        assert len(tail) == 1 and tail[0]["kind"] == "control"

    def test_events_endpoint_serves_batches(self):
        async def scenario():
            events = EventStream()
            events.emit("step", {"n": 1})
            ck = _make_cockpit(events=events)
            await ck.start()
            try:
                status, body = await _request(ck.port, "GET", "/events?since=0&wait=0")
                assert status == 200
                batch = json.loads(body)
                assert batch["next"] == 1
                assert batch["events"][0]["kind"] == "step"
                status2, body2 = await _request(
                    ck.port, "GET", f"/events?since={batch['next']}&wait=0"
                )
                assert json.loads(body2)["events"] == []
                return True
            finally:
                await ck.stop()

        assert run(scenario())

    def test_events_endpoint_pulls_control_records(self):
        async def scenario():
            records = [{"kind": "set_params", "payload": {"applied": {"max_steps": 2}}}]
            ck = _make_cockpit(records=lambda: records)
            await ck.start()
            try:
                status, body = await _request(ck.port, "GET", "/events?since=0&wait=0")
                assert status == 200
                batch = json.loads(body)
                kinds = [e["kind"] for e in batch["events"]]
                assert "control" in kinds, "control records never reached the log pane"
                # A second pull must not duplicate what was already served.
                _, body2 = await _request(ck.port, "GET", "/events?since=0&wait=0")
                again = json.loads(body2)
                assert len([e for e in again["events"] if e["kind"] == "control"]) == 1
                return True
            finally:
                await ck.stop()

        assert run(scenario())


class TestCockpitCli:
    def test_parser_declares_the_verb_and_window_flags(self):
        args = build_parser().parse_args(
            ["cockpit", "watch the page", "--window", "--pin", "--port", "8899"]
        )
        assert args.cmd == "cockpit"
        assert args.task == "watch the page"
        assert args.window is True
        assert args.pin is True
        assert args.port == 8899

    def test_defaults_are_safe(self):
        args = build_parser().parse_args(["cockpit"])
        assert args.port == 0, "port 0 (ephemeral) must be the default — no port conflicts"
        assert args.host == "127.0.0.1"
        assert args.window is False and args.pin is False

    def test_task_is_optional(self):
        args = build_parser().parse_args(["cockpit", "--soul", "x.soul"])
        assert args.task, "a default task must exist so the loop has a prompt"
