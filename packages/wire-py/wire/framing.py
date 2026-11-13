"""HTTP/1.1 framing — request parser and response serializer.

stdlib-only, synchronous, byte-oriented: this module is the wire layer's
contract with RFC 9112 §2-§6. It never imports asyncio (execution
philosophy: the wire core stays sync; the ASGI host adapter from card
ecea49c8 phase 2 sits ABOVE this seam).

Security-relevant behaviour, all pinned by tests:
- ``Transfer-Encoding`` + ``Content-Length`` in one message is rejected
  (request-smuggling guard — the wire never guesses which framing wins);
- conflicting duplicate ``Content-Length`` values are rejected, identical
  duplicates accepted (RFC 9112 §6.3);
- header line and header count caps raise 431, malformed anything else 400;
- obsolete line folding (obs-fold) is rejected rather than reinterpreted;
- bodies are capped (``max_body``, default 64 MiB → 413) because the wire
  buffers them whole — chunked streams count toward the same cap;
- ``Expect`` is honoured: ``100-continue`` fires ``on_expect`` (the
  interim response) only after the framing validated, anything else → 417.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from email.utils import formatdate
from http import HTTPStatus
from typing import BinaryIO

MAX_REQUEST_LINE = 8192
MAX_HEADER_LINE = 8192
MAX_HEADERS = 100
# the wire buffers whole bodies (uvicorn streams) — bound them per request
DEFAULT_MAX_BODY = 64 * 1024 * 1024
# bodies up to this are collected inline at parse time (the hot path —
# small JSON bodies stay one buffered read); anything larger streams
# through Request.body_reader for the host to deliver incrementally
DEFAULT_BODY_BUFFER_THRESHOLD = 64 * 1024

_METHOD_RE = re.compile(rb"^[A-Za-z0-9!#$%&'*+\-.^_`|~]+$")
_TOKEN_RE = re.compile(rb"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_HEX_RE = re.compile(rb"^[0-9A-Fa-f]+$")

_HEADER = ("Server", "wire")  # Date/Server are wire-owned framing metadata


class WireError(Exception):
    """Base class for framing failures carryable as an HTTP status."""

    status = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class MalformedRequest(WireError):
    """Anything the wire refuses to interpret — answered with 400."""

    status = 400


class HeaderTooLarge(WireError):
    """Header line or header count over the configured cap — answered with 431."""

    status = 431


class BodyTooLarge(WireError):
    """Declared or streamed body over ``max_body`` — answered with 413."""

    status = 413


class ExpectationFailed(WireError):
    """Unrecognized ``Expect`` header value — answered with 417."""

    status = 417


@dataclass(frozen=True)
class Request:
    """One parsed HTTP request."""

    method: str
    target: str
    version: str  # "1.1" | "1.0"
    headers: tuple[tuple[str, str], ...]
    body: bytes
    keep_alive: bool
    # transport-internal: set when the body exceeded the buffer threshold
    # — the host streams it (ASGI receive loop) or drains it (handler mode)
    body_reader: BodyReader | None = field(default=None, compare=False, repr=False)

    def header(self, name: str) -> str | None:
        """First header value matching ``name`` case-insensitively."""
        want = name.lower()
        for key, value in self.headers:
            if key.lower() == want:
                return value
        return None


def should_keep_alive(version: str, headers: Sequence[tuple[str, str]]) -> bool:
    """Connection persistence per RFC 9112 §9.3 (version defaults + Connection)."""
    tokens: set[str] = set()
    for key, value in headers:
        if key.lower() == "connection":
            tokens.update(part.strip().lower() for part in value.split(","))
    if version == "1.1":
        return "close" not in tokens
    # HTTP/1.0: persistent only when explicitly requested
    return "keep-alive" in tokens


def _read_line(reader: BinaryIO, cap: int) -> bytes:
    """Read one CRLF-terminated line under ``cap`` (used by chunk framing)."""
    line = reader.readline()
    if not line:
        raise MalformedRequest("unexpected end of stream")
    if len(line) > cap:
        raise HeaderTooLarge("line over cap")
    if not line.endswith(b"\r\n"):
        raise MalformedRequest("line not CRLF-terminated")
    return line[:-2]


def _parse_request_line(line: bytes) -> tuple[str, str, str]:
    parts = line.split(b" ")
    if len(parts) != 3:
        raise MalformedRequest("request line must be METHOD SP TARGET SP VERSION")
    method_raw, target_raw, version_raw = parts
    if not _METHOD_RE.match(method_raw):
        raise MalformedRequest("invalid method token")
    if not target_raw:
        raise MalformedRequest("empty request target")
    if not all(0x21 <= byte <= 0x7E for byte in target_raw):
        # origin-form is printable ASCII (pct-encoded); a raw high byte here
        # would otherwise reach decode() and crash the connection thread
        raise MalformedRequest("non-printable byte in request target")
    if version_raw == b"HTTP/1.1":
        version = "1.1"
    elif version_raw == b"HTTP/1.0":
        version = "1.0"
    else:
        raise MalformedRequest(f"unsupported HTTP version {version_raw!r}")
    return (
        method_raw.decode("ascii"),
        target_raw.decode("ascii"),
        version,
    )


def _parse_headers(reader: BinaryIO) -> list[tuple[str, str]]:
    headers: list[tuple[str, str]] = []
    while True:
        raw = reader.readline()
        if not raw:
            raise MalformedRequest("unexpected end of stream in headers")
        if len(raw) > MAX_HEADER_LINE:
            raise HeaderTooLarge("header line over cap")
        if not raw.endswith(b"\r\n"):
            raise MalformedRequest("header line not CRLF-terminated")
        line = raw[:-2]
        if line == b"":
            return headers
        if line[:1] in (b" ", b"\t"):
            raise MalformedRequest("obsolete line folding is not supported")
        if b":" not in line:
            raise MalformedRequest("header line without colon")
        name_raw, _, value_raw = line.partition(b":")
        if not _TOKEN_RE.match(name_raw):
            raise MalformedRequest("invalid header field name")
        headers.append((name_raw.decode("ascii"), value_raw.strip(b" \t").decode("latin-1")))
        if len(headers) > MAX_HEADERS:
            raise HeaderTooLarge("header count over cap")


class BodyReader:
    """Incremental request-body reader — the streaming seam (sync by design).

    Reads through the connection's buffered reader, so pipelined bytes are
    never stolen, and enforces ``max_body`` as a RUNNING total (→ 413 via
    :class:`BodyTooLarge` — chunked bodies have no declared size to
    pre-check). ``read_some`` loops one underlying read at a time, so
    bytes reach the app as they arrive; any read-ahead taken by the
    buffer-threshold decision is queued and handed out first.
    """

    def __init__(
        self,
        reader: BinaryIO,
        *,
        length: int | None,
        chunked: bool,
        max_body: int = DEFAULT_MAX_BODY,
    ) -> None:
        self._reader = reader
        self._chunked = chunked
        self._remaining = length or 0
        self._max = max_body
        self._consumed = 0
        self._pending = b""
        self._chunk_left = 0
        self._done = not chunked and length is None

    @property
    def done(self) -> bool:
        return self._done

    def queue(self, data: bytes) -> None:
        """Re-queue read-ahead bytes so a streaming host emits them first."""
        self._pending = data + self._pending

    def read_some(self, size: int = 65536) -> bytes:
        """Next chunk; ``b""`` only once the body is complete."""
        if self._pending:
            out, self._pending = self._pending, b""
            return out
        if self._done:
            return b""
        if self._chunked:
            out = self._read_chunked_some(size)
        else:
            take = min(size, self._remaining)
            data = _read_at_least(self._reader, take)
            self._remaining -= len(data)
            if self._remaining == 0:
                self._done = True
            out = data
        self._consumed += len(out)
        if self._consumed > self._max:
            raise BodyTooLarge(f"body over cap {self._max}")
        return out

    def read_all(self) -> bytes:
        """Drain the body (handler mode); honours the same running cap."""
        out = bytearray(self._pending)
        self._pending = b""
        while not self._done:
            out += self.read_some()
        return bytes(out)

    def _read_chunked_some(self, size: int) -> bytes:
        if self._chunk_left == 0:
            size_line = _read_line(self._reader, MAX_HEADER_LINE)
            size_raw = size_line.split(b";", 1)[0].strip()
            if not _HEX_RE.match(size_raw):
                raise MalformedRequest("invalid chunk size")
            self._chunk_left = int(size_raw, 16)
            if self._chunk_left == 0:
                self._read_trailers()
                self._done = True
                return b""
        data = _read_at_least(self._reader, min(size, self._chunk_left), "chunk data")
        self._chunk_left -= len(data)
        if self._chunk_left == 0:
            if self._reader.read(2) != b"\r\n":
                raise MalformedRequest("chunk data not CRLF-terminated")
        return data

    def _read_trailers(self) -> None:
        # trailer lines up to the cap, then an empty line
        trailers = 0
        while True:
            trailer = self._reader.readline()
            if not trailer or not trailer.endswith(b"\r\n"):
                raise MalformedRequest("truncated chunked trailers")
            if len(trailer) > MAX_HEADER_LINE:
                raise HeaderTooLarge("trailer line over cap")
            if trailer == b"\r\n":
                return
            if b":" not in trailer:
                raise MalformedRequest("malformed trailer line")
            trailers += 1
            if trailers > MAX_HEADERS:
                raise HeaderTooLarge("trailer count over cap")


def _read_at_least(reader: BinaryIO, n: int, what: str = "request body") -> bytes:
    """Exactly ``n`` bytes, one underlying read at a time (EOF truncates)."""
    out = bytearray()
    while len(out) < n:
        piece = (
            reader.read1(n - len(out)) if hasattr(reader, "read1") else reader.read(n - len(out))
        )
        if not piece:
            break
        out += piece
    if len(out) != n:
        raise MalformedRequest(f"truncated {what}")
    return bytes(out)


def _body_plan(headers: list[tuple[str, str]], max_body: int) -> tuple[int | None, bool]:
    """Validate body framing; return ``(declared length | None, chunked)``.

    Rejects smuggling shapes (TE+CL together, unsupported encodings,
    conflicting or non-numeric Content-Length) and over-declared bodies
    (413 before a single body byte is read).
    """
    te = [v for k, v in headers if k.lower() == "transfer-encoding"]
    cl = [v for k, v in headers if k.lower() == "content-length"]

    if te and cl:
        raise MalformedRequest("Transfer-Encoding together with Content-Length")
    if te:
        if te[0].strip().lower() != "chunked":
            raise MalformedRequest(f"unsupported transfer encoding {te[0]!r}")
        return None, True
    if cl:
        if len({v.strip() for v in cl}) > 1:
            raise MalformedRequest("conflicting Content-Length values")
        raw = cl[0].strip()
        if not raw.isdigit():
            raise MalformedRequest("invalid Content-Length")
        size = int(raw)
        if size > max_body:
            raise BodyTooLarge(f"body of {size} bytes over cap {max_body}")
        return size, False
    return None, False  # no body


def read_request(
    reader: BinaryIO,
    *,
    max_body: int = DEFAULT_MAX_BODY,
    on_expect: Callable[[], None] | None = None,
    body_buffer_threshold: int = DEFAULT_BODY_BUFFER_THRESHOLD,
) -> Request | None:
    """Read one request from ``reader``.

    Returns ``None`` on a clean EOF (keep-alive peer closed between
    requests); raises :class:`MalformedRequest` / :class:`HeaderTooLarge`
    / :class:`BodyTooLarge` / :class:`ExpectationFailed` with the status
    the server should answer. Reads exactly one message — pipelined bytes
    stay buffered in ``reader`` for the next call.

    Bodies are read through a :class:`BodyReader`: up to
    ``body_buffer_threshold`` bytes are collected inline (the hot path —
    small bodies stay one buffered read, ``Request.body`` is complete);
    anything larger leaves ``Request.body_reader`` set with the read-ahead
    queued, so the host streams it (ASGI receive loop) or drains it
    (handler mode). ``max_body`` bounds the declared AND running totals
    alike.

    ``Expect: 100-continue`` is evaluated here: an unrecognized
    expectation raises 417 before the body is touched; a recognized one
    invokes ``on_expect`` (the server sends the interim response) only
    after the framing and the body cap have validated — never for a
    request the wire is about to refuse.
    """
    line = reader.readline()
    if not line:
        return None
    if len(line) > MAX_REQUEST_LINE:
        raise MalformedRequest("request line over cap")
    if not line.endswith(b"\r\n"):
        raise MalformedRequest("truncated request line")
    method, target, version = _parse_request_line(line[:-2])

    headers = _parse_headers(reader)
    expect = next((v for k, v in headers if k.lower() == "expect"), None)
    if expect is not None and expect.strip().lower() != "100-continue":
        raise ExpectationFailed(f"unsupported expectation {expect.strip()!r}")
    length, chunked = _body_plan(headers, max_body)
    # only after the framing validated: no interim response for a
    # request the wire is about to refuse
    if expect is not None and on_expect is not None and (length is not None or chunked):
        on_expect()
    body_reader = BodyReader(reader, length=length, chunked=chunked, max_body=max_body)
    collected = bytearray()
    while not body_reader.done and len(collected) <= body_buffer_threshold:
        collected += body_reader.read_some(body_buffer_threshold + 1)
    keep = should_keep_alive(version, headers)
    if body_reader.done:
        return Request(
            method=method,
            target=target,
            version=version,
            headers=tuple(headers),
            body=bytes(collected),
            keep_alive=keep,
        )
    body_reader.queue(bytes(collected))
    return Request(
        method=method,
        target=target,
        version=version,
        headers=tuple(headers),
        body=b"",
        keep_alive=keep,
        body_reader=body_reader,
    )


def _reason_phrase(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return ""


def serialize_head(
    status: int,
    headers: Mapping[str, str] | Sequence[tuple[str, str]],
    *,
    keep_alive: bool,
) -> bytes:
    """Serialize status line + headers, wire-owned metadata included.

    The wire is the one authority for ``Connection``/``Date``/``Server``:
    caller-supplied values of those three are discarded, everything else
    passes through in order. Ends with the header/body separator.
    """
    if not 100 <= status <= 599:
        raise ValueError(f"status {status} out of range")
    items = list(headers.items()) if isinstance(headers, Mapping) else list(headers)
    items = [(k, v) for k, v in items if k.lower() not in ("connection", "date", "server")]
    items.append(("Connection", "keep-alive" if keep_alive else "close"))
    items.append(("Date", formatdate(usegmt=True)))
    items.append(_HEADER)

    head = f"HTTP/1.1 {status} {_reason_phrase(status)}".encode("ascii")
    for key, value in items:
        head += b"\r\n" + key.encode("ascii") + b": " + str(value).encode("latin-1")
    return head + b"\r\n\r\n"


def build_response(
    status: int,
    headers: Mapping[str, str] | Sequence[tuple[str, str]],
    body: bytes,
    *,
    keep_alive: bool,
    chunked: bool = False,
) -> bytes:
    """Serialize a full HTTP/1.1 response message.

    The wire owns framing metadata: ``Content-Length`` (or the chunked
    framing), ``Connection``, ``Date`` and ``Server`` are written here and
    caller-supplied duplicates of them are ignored — one authority for how
    the message ends.
    """
    items = list(headers.items()) if isinstance(headers, Mapping) else list(headers)
    lower = {k.lower() for k, _ in items}

    if chunked:
        framed = (
            b"0\r\n\r\n" if not body else (f"{len(body):X}\r\n".encode() + body + b"\r\n0\r\n\r\n")
        )
        if "content-length" in lower:
            items = [(k, v) for k, v in items if k.lower() != "content-length"]
            lower.discard("content-length")
        if "transfer-encoding" not in lower:
            items.append(("Transfer-Encoding", "chunked"))
    else:
        framed = body
        if "content-length" not in lower:
            items.append(("Content-Length", str(len(body))))

    return serialize_head(status, items, keep_alive=keep_alive) + framed
