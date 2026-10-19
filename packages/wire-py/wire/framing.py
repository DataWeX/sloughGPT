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
- obsolete line folding (obs-fold) is rejected rather than reinterpreted.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from email.utils import formatdate
from http import HTTPStatus
from typing import BinaryIO

MAX_REQUEST_LINE = 8192
MAX_HEADER_LINE = 8192
MAX_HEADERS = 100

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


@dataclass(frozen=True)
class Request:
    """One parsed HTTP request."""

    method: str
    target: str
    version: str  # "1.1" | "1.0"
    headers: tuple[tuple[str, str], ...]
    body: bytes
    keep_alive: bool

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


def _read_chunked(reader: BinaryIO) -> bytes:
    out = bytearray()
    while True:
        size_line = _read_line(reader, MAX_HEADER_LINE)
        size_raw = size_line.split(b";", 1)[0].strip()
        if not _HEX_RE.match(size_raw):
            raise MalformedRequest("invalid chunk size")
        size = int(size_raw, 16)
        if size == 0:
            # trailers: any number of header lines, terminated by an empty line
            while True:
                trailer = reader.readline()
                if not trailer or not trailer.endswith(b"\r\n"):
                    raise MalformedRequest("truncated chunked trailers")
                if len(trailer) > MAX_HEADER_LINE:
                    raise HeaderTooLarge("trailer line over cap")
                if trailer == b"\r\n":
                    return bytes(out)
                if b":" not in trailer:
                    raise MalformedRequest("malformed trailer line")
            # unreachable
        data = reader.read(size)
        if len(data) != size:
            raise MalformedRequest("truncated chunk data")
        out += data
        if reader.read(2) != b"\r\n":
            raise MalformedRequest("chunk data not CRLF-terminated")


def _read_body(reader: BinaryIO, headers: list[tuple[str, str]]) -> bytes:
    te = [v for k, v in headers if k.lower() == "transfer-encoding"]
    cl = [v for k, v in headers if k.lower() == "content-length"]

    if te and cl:
        raise MalformedRequest("Transfer-Encoding together with Content-Length")

    if te:
        if te[0].strip().lower() != "chunked":
            raise MalformedRequest(f"unsupported transfer encoding {te[0]!r}")
        return _read_chunked(reader)

    if cl:
        if len({v.strip() for v in cl}) > 1:
            raise MalformedRequest("conflicting Content-Length values")
        raw = cl[0].strip()
        if not raw.isdigit():
            raise MalformedRequest("invalid Content-Length")
        size = int(raw)
        data = reader.read(size)
        if len(data) != size:
            raise MalformedRequest("truncated request body")
        return data

    return b""


def read_request(reader: BinaryIO) -> Request | None:
    """Read one request from ``reader``.

    Returns ``None`` on a clean EOF (keep-alive peer closed between
    requests); raises :class:`MalformedRequest` / :class:`HeaderTooLarge`
    with the status the server should answer. Reads exactly one message —
    pipelined bytes stay buffered in ``reader`` for the next call.
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
    body = _read_body(reader, headers)
    keep = should_keep_alive(version, headers)
    return Request(
        method=method,
        target=target,
        version=version,
        headers=tuple(headers),
        body=body,
        keep_alive=keep,
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
