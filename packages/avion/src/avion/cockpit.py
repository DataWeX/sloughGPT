"""Cockpit — the live console window for a running agent.

Four panes (log / work / message / terminal) served as one self-contained
HTML page by a stdlib-only HTTP server (precedent: ``avion view``). The
page is the outside half of the control plane: it tails an incremental
:event:`EventStream` and posts typed commands straight into the agent's
``CommandInbox`` — it never mutates agent state itself.

Routes::

    GET  /                 the console page
    GET  /state            current control state (JSON)
    GET  /events?since=N   events after seq N (JSON long-poll, ?wait=S)
    POST /command          {"kind": ..., "payload": {...}} -> inbox

``open_window`` opens the page as its own browser window (``--app``),
not a tab, so it sits in the window system beside the agent's own
browser windows; ``pin`` asks the window manager (wmctrl) to keep it on
top. Both are best-effort: a missing Chromium/wmctrl degrades to "open
the printed URL by hand", never to a crash.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections import deque
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

from avion.ai.control import CommandInbox

#: window title the page sets and wmctrl matches for --pin
WINDOW_TITLE = "Avion Agent Console"

_STATUS_REASONS = {200: "OK", 400: "Bad Request", 404: "Not Found", 500: "Internal Server Error"}


class EventStream:
    """Incremental, bounded event log with monotonic sequence numbers.

    ``since(seq)`` returns everything after ``seq`` — the console's tail
    cursor. The buffer may drop old events under pressure; the sequence
    number never goes backwards, so a client still advances correctly.
    """

    def __init__(self, maxlen: int = 1000) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._seq = 0

    @property
    def last_seq(self) -> int:
        return self._seq

    def emit(self, kind: str, data: Any) -> dict[str, Any]:
        self._seq += 1
        event = {"seq": self._seq, "kind": kind, "data": data}
        self._events.append(event)
        return event

    def since(self, seq: int) -> list[dict[str, Any]]:
        return [event for event in self._events if event["seq"] > seq]


class Cockpit:
    """HTTP console for one agent: page, state, events, command inbox."""

    def __init__(
        self,
        *,
        inbox: CommandInbox,
        events: EventStream,
        state: Callable[[], dict[str, Any]],
        records: Callable[[], list[dict[str, Any]]] | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._inbox = inbox
        self._events = events
        self._state = state
        self._records = records
        self._host = host
        self._port = port
        self._server: asyncio.AbstractServer | None = None
        self._records_emitted = 0
        self._window: subprocess.Popen | None = None

    # ── lifecycle ──────────────────────────────────────────────────────

    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}/"

    async def start(self) -> str:
        """Bind the server; returns the console URL (port 0 -> ephemeral)."""
        self._server = await asyncio.start_server(self._handle, self._host, self._port)
        self._port = self._server.sockets[0].getsockname()[1]
        return self.url

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    # ── own window (best-effort, never fatal) ─────────────────────────

    def open_window(self, url: str, *, pin: bool = False, extra_args: tuple | list = ()) -> bool:
        """Open the console as its own browser window. False = degrade."""
        from avion.launch import find_chromium

        try:
            binary = find_chromium()
            self._window = subprocess.Popen(
                [binary, f"--app={url}", *extra_args],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            return False
        if pin:
            self.pin_window()
        return True

    def pin_window(self) -> bool:
        """Ask the window manager to keep the console on top (X11)."""
        try:
            done = subprocess.run(
                ["wmctrl", "-r", WINDOW_TITLE, "-b", "add,always_on_top"],
                capture_output=True,
                timeout=5,
                check=False,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return False
        return done.returncode == 0

    def close_window(self) -> None:
        if self._window is not None and self._window.poll() is None:
            self._window.terminate()
        self._window = None

    # ── HTTP ───────────────────────────────────────────────────────────

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10.0)
        except Exception:
            writer.close()
            return
        try:
            head_text = raw.decode("latin-1")
            lines = head_text.split("\r\n")
            method, target, *_ = lines[0].split(" ")
            headers = {}
            for line in lines[1:]:
                if ":" in line:
                    key, _, value = line.partition(":")
                    headers[key.strip().lower()] = value.strip()
            body = b""
            length = int(headers.get("content-length", "0") or 0)
            if length:
                body = await asyncio.wait_for(reader.readexactly(length), timeout=10.0)
            await self._route(writer, method.upper(), target, body)
        except Exception:
            try:
                await self._respond(writer, 500, {"error": "internal error"})
            except Exception:
                pass
            writer.close()

    async def _route(self, writer, method: str, target: str, body: bytes) -> None:
        parts = urlsplit(target)
        path = parts.path or "/"

        if method == "GET" and path == "/":
            await self._respond_raw(writer, 200, CONSOLE_HTML.encode(), "text/html; charset=utf-8")
            return
        if method == "GET" and path == "/state":
            try:
                await self._respond(writer, 200, self._state())
            except Exception as exc:
                await self._respond(writer, 500, {"error": str(exc)})
            return
        if method == "GET" and path == "/events":
            query = parse_qs(parts.query)
            try:
                since = int(query.get("since", ["0"])[0])
                wait = float(query.get("wait", ["0"])[0])
            except ValueError:
                await self._respond(writer, 400, {"error": "since/wait must be numeric"})
                return
            wait = max(0.0, min(wait, 10.0))
            self._pull_records()
            batch = self._events.since(since)
            remaining = wait
            while not batch and remaining > 0:
                slice_s = min(0.2, remaining)
                await asyncio.sleep(slice_s)
                remaining -= slice_s
                self._pull_records()
                batch = self._events.since(since)
            await self._respond(
                writer, 200, {"events": batch, "next": max(since, self._events.last_seq)}
            )
            return
        if method == "POST" and path == "/command":
            try:
                payload = json.loads(body.decode("utf-8")) if body else None
                self._inbox.submit_dict(payload)
            except (ValueError, json.JSONDecodeError) as exc:
                await self._respond(writer, 400, {"ok": False, "error": str(exc)})
                return
            except Exception as exc:  # never let a bad command kill the server
                await self._respond(writer, 400, {"ok": False, "error": str(exc)})
                return
            await self._respond(writer, 200, {"ok": True})
            return
        await self._respond(writer, 404, {"error": f"no route: {method} {path}"})

    def _pull_records(self) -> None:
        """Fold newly applied control records into the event log once."""
        if self._records is None:
            return
        try:
            records = self._records()
        except Exception:
            return
        while self._records_emitted < len(records):
            self._events.emit("control", records[self._records_emitted])
            self._records_emitted += 1

    async def _respond(
        self, writer: asyncio.StreamWriter, status: int, payload: dict[str, Any]
    ) -> None:
        body = json.dumps(payload).encode()
        await self._respond_raw(writer, status, body, "application/json; charset=utf-8")

    async def _respond_raw(self, writer, status: int, body: bytes, ctype: str) -> None:
        reason = _STATUS_REASONS.get(status, "OK")
        head = (
            f"HTTP/1.1 {status} {reason}\r\n"
            f"Content-Type: {ctype}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        writer.write(head.encode("latin-1") + body)
        try:
            await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass


CONSOLE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Avion Agent Console</title>
<style>
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; background: #0d0f14; color: #d7dae0;
  font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
body { display: grid; grid-template-rows: auto 1fr; }
header { display: flex; gap: 10px; align-items: center; padding: 8px 12px;
  background: #12151c; border-bottom: 1px solid #232833; }
#status { color: #4fd1a5; }
#status.paused { color: #e0b64f; }
#status.stopped, #status.finished { color: #d06060; }
#status-text { flex: 1; color: #8b93a7; }
header button { background: #1b2130; color: #d7dae0; border: 1px solid #2a3244;
  padding: 4px 12px; border-radius: 4px; cursor: pointer; font: inherit; }
header button:hover { background: #232a3d; }
main { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr;
  gap: 6px; padding: 6px; min-height: 0; }
section.pane { background: #12151c; border: 1px solid #232833; border-radius: 6px;
  display: grid; grid-template-rows: auto 1fr auto; min-height: 0; overflow: hidden; }
section.pane h2 { margin: 0; padding: 6px 10px; font-size: 11px; letter-spacing: .12em;
  text-transform: uppercase; color: #8b93a7; border-bottom: 1px solid #232833; }
#log, #terminal, #message { overflow-y: auto; padding: 8px 10px; min-height: 0; }
#work { overflow: auto; padding: 8px 10px; margin: 0; font: inherit; white-space: pre-wrap; }
#log div { padding: 1px 0; }
#log div.err { color: #d06060; }
#log div.cmd { color: #8fb7e8; }
#message div { padding: 2px 0; }
#message div.you { color: #4fd1a5; }
form { display: flex; gap: 6px; padding: 6px 8px; border-top: 1px solid #232833; }
input { flex: 1; background: #0d0f14; border: 1px solid #232833; color: inherit;
  padding: 6px 8px; border-radius: 4px; font: inherit; min-width: 0; }
input:focus-visible, button:focus-visible { outline: 2px solid #8fb7e8; outline-offset: 1px; }
button { background: #1b2130; color: #d7dae0; border: 1px solid #2a3244;
  padding: 4px 10px; border-radius: 4px; cursor: pointer; font: inherit; }
button:hover { background: #232a3d; }
</style>
</head>
<body>
<header>
  <span id="status" aria-hidden="true">&#9679;</span>
  <span id="status-text" role="status">connecting&hellip;</span>
  <button type="button" id="btn-pause">Pause</button>
  <button type="button" id="btn-resume">Resume</button>
  <button type="button" id="btn-stop">Stop</button>
</header>
<main>
  <section class="pane"><h2>log</h2><div id="log"></div></section>
  <section class="pane"><h2>work</h2><pre id="work"></pre></section>
  <section class="pane"><h2>message</h2><div id="message"></div>
    <form id="msg-form">
      <input id="msg-input" aria-label="message to the agent"
             placeholder="message the agent&hellip;" autocomplete="off">
      <button type="submit">send</button>
    </form></section>
  <section class="pane"><h2>terminal</h2><div id="terminal"></div>
    <form id="term-form">
      <input id="term-input" aria-label="terminal command"
             placeholder="pause | resume | stop | params max_steps=10 | instructions &hellip;"
             autocomplete="off">
      <button type="submit">run</button>
    </form></section>
</main>
<script>
(function () {
  var $ = function (id) { return document.getElementById(id); };
  var seq = 0, busy = false;
  var MAX = 400;

  function line(boxId, text, cls) {
    var box = $(boxId);
    var div = document.createElement('div');
    div.textContent = text;
    if (cls) div.className = cls;
    box.appendChild(div);
    while (box.childElementCount > MAX) box.removeChild(box.firstChild);
    box.scrollTop = box.scrollHeight;
  }

  function post(kind, payload) {
    return fetch('/command', {
      method: 'POST',
      body: JSON.stringify({ kind: kind, payload: payload || {} })
    }).then(function (r) {
      if (!r.ok) {
        return r.json().catch(function () { return {}; }).then(function (e) {
          line('log', 'rejected: ' + (e.error || r.status), 'err');
          return false;
        });
      }
      return true;
    }).catch(function (e) {
      line('log', 'console link down: ' + e, 'err');
      return false;
    });
  }

  function render(ev) {
    var d = ev.data || {};
    if (ev.kind === 'step') {
      line('log', 'step ' + d.step + ' &middot; ' + (d.action || '') +
        (d.observation ? ' \u2014 ' + d.observation : ''));
    } else if (ev.kind === 'control') {
      line('log', 'cmd &middot; ' + (d.note || d.kind), 'cmd');
    } else {
      line('log', ev.kind + ' &middot; ' + JSON.stringify(d));
    }
  }

  function pollEvents() {
    if (busy) return;
    busy = true;
    fetch('/events?since=' + seq + '&wait=1')
      .then(function (r) { return r.json(); })
      .then(function (batch) {
        (batch.events || []).forEach(function (ev) {
          if (ev.seq > seq) seq = ev.seq;
          render(ev);
        });
      })
      .catch(function () {})
      .then(function () { busy = false; });
  }

  function pollState() {
    fetch('/state')
      .then(function (r) { return r.json(); })
      .then(function (s) {
        $('work').textContent = JSON.stringify(s, null, 2);
        var st = s.stopped ? 'stopped' : (s.paused ? 'paused' :
          (s.done ? 'finished: ' + s.stop_reason : (s.running ? 'running' : 'idle')));
        $('status-text').textContent = st + ' \u00b7 step ' + (s.steps || 0) +
          ' \u00b7 ' + (s.model || '');
        var dot = $('status');
        dot.className = s.stopped ? 'stopped' : (s.paused ? 'paused' :
          (s.done ? 'finished' : ''));
      })
      .catch(function () {});
  }

  function handleTerm(raw) {
    var s = raw.trim();
    if (!s) return;
    line('terminal', '> ' + s, 'cmd');
    var sp = s.split(/\\s+/);
    var head = sp[0].toLowerCase();
    var arg = sp.slice(1).join(' ');
    if (head === 'pause') { post('pause'); return; }
    if (head === 'resume') { post('resume'); return; }
    if (head === 'stop') { post('stop'); return; }
    if (head === 'instructions') {
      if (!arg) { line('terminal', 'usage: instructions <text>', 'err'); return; }
      post('set_instructions', { text: arg });
      return;
    }
    if (head === 'message') {
      if (!arg) { line('terminal', 'usage: message <text>', 'err'); return; }
      post('inject_message', { text: arg });
      line('message', 'you: ' + arg, 'you');
      return;
    }
    if (head === 'params') {
      var obj = {};
      arg.split(',').forEach(function (pair) {
        var i = pair.indexOf('=');
        if (i < 0) return;
        var k = pair.slice(0, i).trim();
        var v = pair.slice(i + 1).trim();
        if (v === 'true') v = true;
        else if (v === 'false') v = false;
        else if (v !== '' && !isNaN(Number(v))) v = Number(v);
        if (k) obj[k] = v;
      });
      if (!Object.keys(obj).length) {
        line('terminal', 'usage: params key=value,key=value', 'err');
        return;
      }
      post('set_params', obj);
      return;
    }
    line('terminal', 'unknown: ' + head +
      ' (pause|resume|stop|params|instructions|message)', 'err');
  }

  $('term-form').addEventListener('submit', function (e) {
    e.preventDefault();
    handleTerm($('term-input').value);
    $('term-input').value = '';
  });
  $('msg-form').addEventListener('submit', function (e) {
    e.preventDefault();
    var text = $('msg-input').value.trim();
    if (!text) return;
    post('inject_message', { text: text });
    line('message', 'you: ' + text, 'you');
    $('msg-input').value = '';
  });
  $('btn-pause').addEventListener('click', function () { post('pause'); });
  $('btn-resume').addEventListener('click', function () { post('resume'); });
  $('btn-stop').addEventListener('click', function () { post('stop'); });

  pollEvents();
  pollState();
  setInterval(pollEvents, 500);
  setInterval(pollState, 1000);
})();
</script>
</body>
</html>
"""
