"""HTTP/1.1 framing — parser and serializer (card ecea49c8, phase 1).

The wire owns bytes on a socket; everything here is stdlib-only and
synchronous by design (execution philosophy: the wire core stays sync,
asyncio is confined to a future ASGI-host adapter above it).

Each test pins one contract of RFC 9112 §2-§6 that a naive ``split("\\r\\n")``
parser gets wrong: keep-alive defaults per version, chunked request bodies,
the TE+CL smuggling guard, and header caps.
"""

from __future__ import annotations

import io

import pytest
from wire.framing import (
    MAX_HEADERS,
    HeaderTooLarge,
    MalformedRequest,
    build_response,
    read_request,
)


def _reader(raw: bytes) -> io.BytesIO:
    return io.BytesIO(raw)


# ── request parsing ───────────────────────────────────────────────────


def test_parse_simple_get():
    req = read_request(_reader(b"GET /v1/chat HTTP/1.1\r\nHost: localhost\r\nAccept: */*\r\n\r\n"))
    assert req is not None
    assert req.method == "GET"
    assert req.target == "/v1/chat"
    assert req.version == "1.1"
    assert req.header("Host") == "localhost"
    assert req.header("accept") == "*/*"  # case-insensitive lookup
    assert req.body == b""
    assert req.keep_alive is True  # HTTP/1.1 defaults to persistent


def test_parse_post_with_content_length_body():
    body = b'{"prompt": "hello"}'
    raw = (
        b"POST /generate HTTP/1.1\r\nHost: x\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    )
    req = read_request(_reader(raw))
    assert req is not None
    assert req.method == "POST"
    assert req.body == body
    assert req.header("Content-Length") == str(len(body))


def test_http10_defaults_to_close_unless_keep_alive_asked():
    req = read_request(_reader(b"GET / HTTP/1.0\r\n\r\n"))
    assert req is not None
    assert req.keep_alive is False

    req = read_request(_reader(b"GET / HTTP/1.0\r\nConnection: Keep-Alive\r\n\r\n"))
    assert req is not None
    assert req.keep_alive is True


def test_http11_connection_close_disables_keep_alive():
    req = read_request(_reader(b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"))
    assert req is not None
    assert req.keep_alive is False


def test_chunked_request_body_is_decoded():
    raw = (
        b"POST /upload HTTP/1.1\r\nHost: x\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n"
        b"4\r\nWiki\r\n5\r\npedia\r\n0\r\n\r\n"
    )
    req = read_request(_reader(raw))
    assert req is not None
    assert req.body == b"Wikipedia"


def test_chunked_size_line_extensions_are_ignored():
    raw = (
        b"POST / HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n"
        b"5;name=value\r\nhello\r\n0\r\n\r\n"
    )
    req = read_request(_reader(raw))
    assert req is not None
    assert req.body == b"hello"


def test_te_and_content_length_together_is_rejected():
    """Request-smuggling guard: TE+CL in one message is never ambiguous here."""
    raw = (
        b"POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n"
    )
    with pytest.raises(MalformedRequest) as exc:
        read_request(_reader(raw))
    assert exc.value.status == 400


def test_conflicting_duplicate_content_length_is_rejected():
    raw = b"POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\nContent-Length: 7\r\n\r\nabcde"
    with pytest.raises(MalformedRequest) as exc:
        read_request(_reader(raw))
    assert exc.value.status == 400


def test_identical_duplicate_content_length_is_accepted():
    raw = b"POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 3\r\nContent-Length: 3\r\n\r\nabc"
    req = read_request(_reader(raw))
    assert req is not None
    assert req.body == b"abc"


def test_malformed_request_line_is_400():
    for bad in (
        b"GET /\r\n\r\n",  # missing version
        b"GET / HTTP/9.9\r\n\r\n",  # unknown version
        b"GET\t/ HTTP/1.1\r\n\r\n",  # tab separator
        b"\x00\x01 / HTTP/1.1\r\n\r\n",  # binary junk
    ):
        with pytest.raises(MalformedRequest) as exc:
            read_request(_reader(bad))
        assert exc.value.status == 400


def test_header_without_colon_is_400():
    with pytest.raises(MalformedRequest) as exc:
        read_request(_reader(b"GET / HTTP/1.1\r\nbad-header-line\r\n\r\n"))
    assert exc.value.status == 400


def test_header_line_over_cap_is_431():
    long_value = b"X-Big: " + b"a" * 9000 + b"\r\n"
    with pytest.raises(HeaderTooLarge) as exc:
        read_request(_reader(b"GET / HTTP/1.1\r\nHost: x\r\n" + long_value + b"\r\n"))
    assert exc.value.status == 431


def test_header_count_over_cap_is_431():
    headers = b"".join(b"X-H%d: v\r\n" % i for i in range(MAX_HEADERS + 5))
    with pytest.raises(HeaderTooLarge) as exc:
        read_request(_reader(b"GET / HTTP/1.1\r\n" + headers + b"\r\n"))
    assert exc.value.status == 431


def test_clean_eof_returns_none():
    """Keep-alive peer closed between requests — not an error."""
    assert read_request(_reader(b"")) is None


def test_truncated_request_is_malformed():
    with pytest.raises(MalformedRequest) as exc:
        read_request(_reader(b"GET / HTTP/1.1\r\nHost: x\r\n"))
    assert exc.value.status == 400


def test_truncated_body_is_malformed():
    raw = b"POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 10\r\n\r\nabc"
    with pytest.raises(MalformedRequest) as exc:
        read_request(_reader(raw))
    assert exc.value.status == 400


def test_pipelined_requests_read_in_order():
    raw = b"GET /a HTTP/1.1\r\nHost: x\r\n\r\nGET /b HTTP/1.1\r\nHost: x\r\n\r\n"
    reader = _reader(raw)
    first = read_request(reader)
    second = read_request(reader)
    assert first is not None and second is not None
    assert (first.target, second.target) == ("/a", "/b")
    assert read_request(reader) is None


# ── response serialization ────────────────────────────────────────────


def test_build_response_sets_framing_headers():
    out = build_response(200, {"Content-Type": "text/plain"}, b"ok", keep_alive=True)
    head, _, body = out.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.1 200 OK\r\n")
    assert body == b"ok"
    assert b"Content-Length: 2\r\n" in head
    assert b"Connection: keep-alive\r\n" in head
    assert b"Date: " in head
    assert head.endswith(b"Server: wire")  # last header sits right before CRLFCRLF


def test_build_response_close_says_close():
    out = build_response(404, {}, b"nope", keep_alive=False)
    head = out.partition(b"\r\n\r\n")[0]
    assert b"Connection: close\r\n" in head
    assert b"Content-Length: 4\r\n" in head


def test_build_response_chunked_body():
    out = build_response(
        200, {"Content-Type": "text/plain"}, b"hello", keep_alive=True, chunked=True
    )
    head, _, body = out.partition(b"\r\n\r\n")
    assert b"Transfer-Encoding: chunked\r\n" in head
    assert b"Content-Length" not in head
    assert body == b"5\r\nhello\r\n0\r\n\r\n"


def test_build_response_error_status_reason_phrases():
    for code, phrase in ((400, b"Bad Request"), (431, b"Request Header Fields Too Large")):
        out = build_response(code, {}, b"", keep_alive=False)
        assert out.startswith(b"HTTP/1.1 " + str(code).encode() + b" " + phrase)
