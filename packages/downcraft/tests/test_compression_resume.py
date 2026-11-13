"""Correctness tests: compression in the download path (kanban card 19cd41dc).

These tests PROVE the byte-space invariant behind resume:

    The ``Range: bytes=N-`` offset sent by the client, the bytes on disk in
    ``file.sgpart``, the ``Content-Range`` served by the gateway, and the
    ``bytes_downloaded`` field in ``~/.downcraft/state.json`` must all be
    measured in the SAME byte-space — decoded (identity) bytes.

Why this is not trivial: a full ``application/x-lz4`` response carries a
44-byte SLZ4 header + LZ4 frame (wire bytes); wire bytes ≠ decoded bytes,
so a naive ``Range: bytes=len(sgpart)-`` against a server that slices the
compressed representation corrupts the file *silently* (the numbers match,
the spaces do not).

Covered invariants:
- Full x-lz4 responses are stream-decoded to identity before writing.
- Range responses are identity slices (downcraft protocol, see
  ``CompressedFileServer.serve_range``); wire-space 206 responses are
  detected and rejected (restart from 0), never appended.
- After a mid-stream crash + resume: SHA-256 via ``download/verify.py``
  AND the SLZ4 header checksum both verify; atomic rename to the final
  filename happens; state.json agrees with disk.
- multipart groups pass compression through and resume per-part.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from pathlib import Path

import pytest
from downcraft.download.compress import (
    CompressedFileServer,
    compress_file,
    peek_compressed_header,
)
from downcraft.download.http import (
    DownloadError,
    _identity_total,
    _part_path,
    download_file,
    download_range,
)
from downcraft.download.multipart import download_parts
from downcraft.download.verify import verify_file
from helpers import RangeHandler, _range_url, slz4_wire

# Keep per-chunk granularity small so crash points are deterministic
# (default CHUNK_SIZE is 8 MB — a 200 KB payload would arrive in one chunk).
SMALL_CHUNK = 32 * 1024


class SimulatedCrash(Exception):
    """Deterministic client-side mid-stream crash (like SIGKILL between writes)."""


def crash_after(n: int):
    """Return an ``on_chunk`` callback that raises ``SimulatedCrash`` on call *n*."""

    calls: list[int] = []

    def _cb(done: int, total: int) -> None:
        calls.append(done)
        if len(calls) >= n:
            raise SimulatedCrash(f"crash after {n} chunks (done={done})")

    return _cb


def _lz4_gets(path: str) -> list[dict]:
    return [e for e in RangeHandler.requests_log if e["path"] == path and e["status"] in (200, 206)]


# ---------------------------------------------------------------------------
# Fresh (non-resumed) compressed downloads
# ---------------------------------------------------------------------------


class TestLz4FreshDownload:
    def test_fresh_lz4_response_decodes_to_identity(self, range_server, tmp_path, monkeypatch):
        """A full x-lz4 response lands as decoded identity bytes on disk."""
        from downcraft.download import http as http_mod

        monkeypatch.setattr(http_mod, "CHUNK_SIZE", SMALL_CHUNK)
        payload = b"fresh slz4 payload " * 5000  # 95 KB
        RangeHandler.payloads["/f.bin"] = payload
        RangeHandler.lz4_paths["/f.bin"] = True

        dest = tmp_path / "f.bin"
        download_file(_range_url(range_server, "/f.bin"), dest, compressed=True)

        assert dest.read_bytes() == payload
        assert not _part_path(dest).exists()
        assert verify_file(dest, hashlib.sha256(payload).hexdigest())
        # Request-side integration: compressed=True advertises lz4 willingness.
        assert "lz4" in _lz4_gets("/f.bin")[0]["accept_encoding"]

    def test_fresh_lz4_progress_totals_are_identity_size(self, range_server, tmp_path):
        """Progress totals never leak the wire (compressed) size."""
        payload = os.urandom(120_000)
        RangeHandler.payloads["/p.bin"] = payload
        RangeHandler.lz4_paths["/p.bin"] = True
        seen: list[tuple[int, int]] = []

        dest = tmp_path / "p.bin"
        download_file(
            _range_url(range_server, "/p.bin"),
            dest,
            compressed=True,
            on_chunk=lambda done, total: seen.append((done, total)),
        )
        assert dest.read_bytes() == payload
        assert seen, "no progress reported"
        assert seen[-1] == (len(payload), len(payload))

    def test_corrupt_slz4_header_checksum_fails_download(self, range_server, tmp_path):
        """A bad SLZ4 header SHA-256 must fail loudly and leave no files."""
        payload = b"header integrity matters " * 1000
        RangeHandler.payloads["/bad.bin"] = payload
        RangeHandler.lz4_paths["/bad.bin"] = True
        RangeHandler.lz4_bad_header["/bad.bin"] = True

        dest = tmp_path / "bad.bin"
        with pytest.raises(DownloadError, match="(?i)slz4|integrity"):
            download_file(_range_url(range_server, "/bad.bin"), dest, compressed=True)
        assert not dest.exists()
        assert not _part_path(dest).exists()

    def test_identity_total_ignores_lz4_wire_length(self):
        """Wire Content-Length is not an identity total for x-lz4 responses."""
        assert (
            _identity_total({"Content-Type": "application/x-lz4", "Content-Length": "999"}) == 0
        )
        assert (
            _identity_total(
                {
                    "Content-Encoding": "lz4",
                    "Content-Length": "999",
                    "X-Uncompressed-Content-Length": "500",
                }
            )
            == 500
        )


# ---------------------------------------------------------------------------
# THE core: mid-stream crash + resume + decompress in one byte-space
# ---------------------------------------------------------------------------


class TestLz4ResumeByteSpace:
    def test_crash_resume_decompress_end_to_end(
        self, range_server, tmp_path, monkeypatch
    ):
        """Crash mid-decode, resume with Range=decoded-size, verify both checksums.

        This is the provable form of http.py's claim: the N in
        ``Range: bytes=N-`` equals the DECODED bytes on disk, the gateway
        slices identity at N, and the final file verifies against both
        ``download/verify.py`` (caller SHA-256) and the SLZ4 header
        checksum relayed by the gateway on the 206 response.
        """
        from downcraft.download import compress as compress_mod
        from downcraft.download import http as http_mod

        monkeypatch.setattr(http_mod, "CHUNK_SIZE", SMALL_CHUNK)
        monkeypatch.setattr(compress_mod, "DEFAULT_CHUNK_SIZE", SMALL_CHUNK)

        payload = os.urandom(200_000)
        RangeHandler.payloads["/m.bin"] = payload
        RangeHandler.lz4_paths["/m.bin"] = True

        dest = tmp_path / "m.bin"
        part = _part_path(dest)

        # -- phase 1: fresh download crashes after 3 decoded chunks -------
        with pytest.raises(SimulatedCrash):
            download_file(
                _range_url(range_server, "/m.bin"),
                dest,
                compressed=True,
                on_chunk=crash_after(3),
            )

        prefix = part.read_bytes()
        assert 0 < len(prefix) < len(payload), "expected a mid-stream decoded prefix"
        assert payload.startswith(prefix), ".sgpart must hold DECODED bytes"

        # -- phase 2: process restarts, resumes from the decoded offset ---
        download_file(_range_url(range_server, "/m.bin"), dest, compressed=True)

        gets = _lz4_gets("/m.bin")
        assert len(gets) == 2
        # PROOF of same byte-space: the client asked for exactly its
        # decoded-prefix size, and the server answered in identity space.
        assert gets[1]["range"] == f"bytes={len(prefix)}-"
        assert gets[1]["status"] == 206

        # -- end-to-end integrity ----------------------------------------
        assert dest.read_bytes() == payload
        sha = hashlib.sha256(payload).hexdigest()
        assert verify_file(dest, sha), "SHA-256 via download/verify.py failed"

        # The relayed value IS the SLZ4 44-byte header checksum:
        wire = slz4_wire(payload)
        assert wire[0:4] == b"SLZ4"
        assert struct.unpack(">Q", wire[4:12])[0] == len(payload)
        header_sha = wire[12:44].hex()
        assert header_sha == sha, "SLZ4 header SHA-256 covers the payload"
        assert gets[1]["slz4_sha"] == header_sha, "gateway relayed the header checksum"

    def test_wire_space_206_restarts_instead_of_corrupting(
        self, range_server, tmp_path, monkeypatch
    ):
        """A server slicing the WIRE for Range must never be appended to.

        The trap: a decoded 80 000-byte prefix and a wire offset of 80 000
        are numerically equal but different byte-spaces — a naive client
        silently corrupts the file. The client must detect the lz4 markers
        on the 206 and restart from 0 instead.
        """
        payload = os.urandom(200_000)
        RangeHandler.payloads["/naive.bin"] = payload
        RangeHandler.lz4_paths["/naive.bin"] = True
        RangeHandler.lz4_naive["/naive.bin"] = True

        dest = tmp_path / "naive.bin"
        part = _part_path(dest)
        part.write_bytes(payload[:80_000])  # decoded prefix from an earlier crash

        download_file(_range_url(range_server, "/naive.bin"), dest, compressed=True)

        # Single call must recover: wire-space 206 rejected, full restart.
        assert dest.read_bytes() == payload
        assert verify_file(dest, hashlib.sha256(payload).hexdigest())
        gets = _lz4_gets("/naive.bin")
        assert gets[0]["range"] == "bytes=80000-"
        assert any(e["range"] is None for e in gets), "restarted with a full GET"

    def test_stale_range_retry_never_duplicates_bytes(self, range_server, tmp_path, monkeypatch):
        """In-call retry must re-derive Range from disk, not reuse the stale offset.

        Attempt 1 dies mid-body after appending more bytes to .sgpart;
        attempt 2 must resume from the NEW file size. Reusing attempt 1's
        Range would append an overlapping slice and corrupt the file.

        CHUNK_SIZE is lowered so the stack yields full chunks before the
        truncation error fires (with 8 MB chunks a 300 KB body dies before
        the first yield — nothing written, offset unchanged).
        """
        from downcraft.download import http as http_mod

        monkeypatch.setattr(http_mod, "MAX_RETRIES", 3)
        monkeypatch.setattr(http_mod, "RETRY_DELAY", 0)
        monkeypatch.setattr(http_mod, "CHUNK_SIZE", 8192)

        payload = os.urandom(300_000)
        RangeHandler.payloads["/stale.bin"] = payload

        dest = tmp_path / "stale.bin"
        part = _part_path(dest)
        part.write_bytes(payload[:50_000])
        # First resume response promises everything, dies after 60 KB.
        RangeHandler.truncate_once["/stale.bin"] = 60_000

        download_file(_range_url(range_server, "/stale.bin"), dest)

        assert dest.read_bytes() == payload, "retry must not duplicate or skip bytes"
        ranges = [e["range"] for e in _lz4_gets("/stale.bin")]
        assert ranges[0] == "bytes=50000-"
        # Attempt 2's offset must come from the grown .sgpart — a stale
        # Range would repeat "bytes=50000-" and duplicate the written chunk.
        assert len(ranges) >= 2 and ranges[1] != ranges[0]
        retry_start = int(ranges[1].removeprefix("bytes=").rstrip("-"))
        assert retry_start > 50_000, "retry Range must be re-derived from disk"

    def test_wire_truncated_mid_frame_recovers(self, range_server, tmp_path, monkeypatch):
        """Network crash mid-SLZ4-frame: clean DownloadError, then resume works."""
        from downcraft.download import http as http_mod

        monkeypatch.setattr(http_mod, "MAX_RETRIES", 1)
        payload = os.urandom(300_000)
        RangeHandler.payloads["/cut.bin"] = payload
        RangeHandler.lz4_paths["/cut.bin"] = True
        # Header (44) + ~150 KB of frame: client decodes ≥1 block, then the
        # connection dies with the promised Content-Length unfulfilled.
        RangeHandler.truncate_once["/cut.bin"] = 44 + 150_000

        dest = tmp_path / "cut.bin"
        part = _part_path(dest)
        with pytest.raises(DownloadError):
            download_file(_range_url(range_server, "/cut.bin"), dest, compressed=True)

        prefix_len = part.stat().st_size if part.exists() else 0
        if prefix_len:
            assert payload.startswith(part.read_bytes())

        # -- resume / retry after the crash --------------------------------
        download_file(_range_url(range_server, "/cut.bin"), dest, compressed=True)
        assert dest.read_bytes() == payload
        assert verify_file(dest, hashlib.sha256(payload).hexdigest())
        if prefix_len:
            resumed = [e for e in _lz4_gets("/cut.bin") if e["range"]]
            assert resumed and resumed[0]["range"] == f"bytes={prefix_len}-"

    def test_resume_rejects_mismatched_slz4_header_checksum(
        self, range_server, tmp_path
    ):
        """If the relayed SLZ4 header checksum disagrees → discard, never accept.

        A mismatch means the server's file changed (version skew) or the
        partial is corrupt; completing the download would mix versions.
        """
        payload = b"version one " * 8000
        RangeHandler.payloads["/skew.bin"] = payload
        RangeHandler.lz4_paths["/skew.bin"] = True
        RangeHandler.lz4_bad_resume_sha["/skew.bin"] = True

        dest = tmp_path / "skew.bin"
        part = _part_path(dest)
        part.write_bytes(payload[:40_000])

        with pytest.raises(DownloadError, match="(?i)slz4|integrity|checksum"):
            download_file(_range_url(range_server, "/skew.bin"), dest, compressed=True)
        assert not dest.exists()
        assert not part.exists(), "unverifiable partial must be discarded"

        # A fresh attempt (full 200 carries the real header) still succeeds.
        download_file(_range_url(range_server, "/skew.bin"), dest, compressed=True)
        assert dest.read_bytes() == payload


# ---------------------------------------------------------------------------
# Edge (gateway) compression: the existing docstring claim, proven
# ---------------------------------------------------------------------------


class TestEdgeCrashResume:
    """``Content-Encoding: gzip|zstd`` → decoded to identity before writing,
    so Range resume offsets stay consistent (http.py module docstring)."""

    @pytest.mark.parametrize("enc", ["zstd", "gzip"])
    def test_edge_crash_then_identity_resume(self, range_server, tmp_path, monkeypatch, enc):
        from downcraft.download import http as http_mod

        monkeypatch.setattr(http_mod, "CHUNK_SIZE", SMALL_CHUNK)
        payload = os.urandom(200_000)
        path = f"/edge-{enc}.bin"
        RangeHandler.payloads[path] = payload
        RangeHandler.encodings[path] = enc

        dest = tmp_path / f"edge-{enc}.bin"
        part = _part_path(dest)

        with pytest.raises(SimulatedCrash):
            download_file(_range_url(range_server, path), dest, on_chunk=crash_after(3))

        prefix = part.read_bytes()
        assert 0 < len(prefix) < len(payload)
        assert payload.startswith(prefix), "crash must leave DECODED bytes"

        download_file(_range_url(range_server, path), dest)
        gets = _lz4_gets(path)
        assert gets[-1]["range"] == f"bytes={len(prefix)}-"
        assert dest.read_bytes() == payload
        assert verify_file(dest, hashlib.sha256(payload).hexdigest())


# ---------------------------------------------------------------------------
# multipart: multi-part groups + compressed responses
# ---------------------------------------------------------------------------


class TestMultipartCompressed:
    def test_download_parts_compressed_decodes_checksums_state(
        self, range_server, tmp_path
    ):
        """download_parts(compressed=True) decodes every part and tracks
        identity bytes in ~/.downcraft/state.json."""
        payloads = {
            "part1.bin": b"model weights alpha " * 3000,
            "part2.bin": b"model weights bravo " * 3000,
            "part3.bin": os.urandom(60_000),
        }
        urls = []
        checksums = []
        for name, data in payloads.items():
            RangeHandler.payloads[f"/{name}"] = data
            RangeHandler.lz4_paths[f"/{name}"] = True
            urls.append(_range_url(range_server, f"/{name}"))
            checksums.append(hashlib.sha256(data).hexdigest())

        dest_dir = tmp_path / "parts"
        result = download_parts(
            urls,
            dest_dir,
            filenames=list(payloads),
            checksums=checksums,
            compressed=True,
        )
        assert result.status == "complete"
        for name, data in payloads.items():
            f = dest_dir / name
            assert f.read_bytes() == data
            assert verify_file(f, hashlib.sha256(data).hexdigest())
            assert not _part_path(f).exists(), "atomic rename must consume .sgpart"

        state = json.loads((tmp_path / "state" / "state.json").read_text())
        group = state["models"][f"parts:{dest_dir}"]
        assert group["status"] == "complete"
        for fp in group["files"]:
            assert fp["complete"] is True
            assert fp["bytes_downloaded"] == fp["total_bytes"] == len(
                payloads[Path(fp["path"]).name]
            )

    def test_download_parts_crash_state_matches_sgpart_then_resume(
        self, range_server, tmp_path, monkeypatch
    ):
        """At crash time, state.json bytes == .sgpart size (same byte-space);
        after a second run the group is complete and consistent on disk."""
        from downcraft.download import compress as compress_mod
        from downcraft.download import http as http_mod

        monkeypatch.setattr(http_mod, "CHUNK_SIZE", SMALL_CHUNK)
        monkeypatch.setattr(compress_mod, "DEFAULT_CHUNK_SIZE", SMALL_CHUNK)

        payloads = {
            "one.bin": b"first part payload " * 4000,
            "two.bin": os.urandom(200_000),
            "three.bin": b"third part payload " * 1000,
        }
        urls = []
        checksums = []
        for name, data in payloads.items():
            RangeHandler.payloads[f"/{name}"] = data
            RangeHandler.lz4_paths[f"/{name}"] = True
            urls.append(_range_url(range_server, f"/{name}"))
            checksums.append(hashlib.sha256(data).hexdigest())

        dest_dir = tmp_path / "grp"
        state_file = tmp_path / "state" / "state.json"

        # Crash part 2 (index 1) after 3 decoded chunks. Sleep on the
        # previous chunk so PersistentState's 2 s flush interval writes the
        # crashing chunk's progress to disk before the exception lands.
        calls = {"n": 0}

        def _progress(part_idx, done, total, speed):
            if part_idx == 1:
                calls["n"] += 1
                if calls["n"] == 2:
                    time.sleep(2.1)
                elif calls["n"] >= 3:
                    raise SimulatedCrash(f"part2 crash at {done} bytes")

        with pytest.raises(SimulatedCrash):
            download_parts(
                urls, dest_dir, filenames=list(payloads), checksums=checksums,
                compressed=True, on_progress=_progress,
            )

        # -- crash-time consistency: state == disk, both in decoded bytes --
        part2 = _part_path(dest_dir / "two.bin")
        assert part2.exists()
        part2_size = part2.stat().st_size
        assert 0 < part2_size < len(payloads["two.bin"])
        assert payloads["two.bin"].startswith(part2.read_bytes())

        state = json.loads(state_file.read_text())
        group = state["models"][f"parts:{dest_dir}"]
        fp2 = next(f for f in group["files"] if f["path"].endswith("two.bin"))
        assert fp2["bytes_downloaded"] == part2_size, (
            "state bytes_downloaded must equal the .sgpart size (decoded byte-space)"
        )

        # -- resume: second run finishes the group --------------------------
        result = download_parts(
            urls, dest_dir, filenames=list(payloads), checksums=checksums, compressed=True
        )
        assert result.status == "complete"
        for name, data in payloads.items():
            f = dest_dir / name
            assert f.read_bytes() == data
            assert verify_file(f, checksums[list(payloads).index(name)])
            assert not _part_path(f).exists()

        state = json.loads(state_file.read_text())
        group = state["models"][f"parts:{dest_dir}"]
        assert group["status"] == "complete"
        for fp in group["files"]:
            assert fp["complete"] is True
            assert fp["bytes_downloaded"] == fp["total_bytes"]


# ---------------------------------------------------------------------------
# Byte-range slices are identity space; serve_range carries the header hash
# ---------------------------------------------------------------------------


class TestRangeIsIdentitySpace:
    def test_download_range_returns_identity_slice_from_lz4_url(
        self, range_server, tmp_path
    ):
        """Slices of an SLZ4-served URL are identity bytes (protocol rule:
        Range is answered in decoded byte-space), verified by checksum."""
        payload = os.urandom(80_000)
        RangeHandler.payloads["/big.lz4"] = payload
        RangeHandler.lz4_paths["/big.lz4"] = True

        start, end = 10_000, 29_999
        dest = tmp_path / "slice.bin"
        download_range(
            _range_url(range_server, "/big.lz4"),
            dest,
            start,
            end,
            checksum=hashlib.sha256(payload[start : end + 1]).hexdigest(),
        )
        assert dest.read_bytes() == payload[start : end + 1]


class TestServeRangeMetadata:
    def test_serve_range_relays_slz4_header_checksum(self, tmp_path):
        """CompressedFileServer.serve_range (protocol source) must expose the
        SLZ4 header SHA-256 so resumed clients can verify end-to-end."""
        payload = b"serve_range metadata proof " * 500
        src = tmp_path / "src.bin"
        src.write_bytes(payload)
        slz4_file = tmp_path / "src.slz4"
        compress_file(src, slz4_file)

        with open(slz4_file, "rb") as f:
            header = peek_compressed_header(f)
        assert header is not None
        assert header["sha256"] == hashlib.sha256(payload).hexdigest()

        resp = CompressedFileServer().serve_range(slz4_file, 10, 19)
        assert resp["data"] == payload[10:20]
        assert resp["headers"]["Content-Range"] == f"bytes 10-19/{len(payload)}"
        assert resp["headers"]["X-SLZ4-SHA256"] == header["sha256"]
