"""Tests for WireStats bandwidth accounting + the Range identity pin.

Covers the wire-vs-decoded savings report:
- identity downloads: wire == decoded, zero savings
- edge-compressed downloads (gzip/zstd): wire < decoded, savings measured
- no Content-Length: identity resolves exactly, encoded reports unknown
- resume: only THIS call's bytes are attributed (prior partial excluded)
- ``download_range`` pins ``Accept-Encoding: identity`` and rejects
  content-coded 206 slices (byte-space guard)
- the ``download()`` result dict and CLI savings line
- multipart group aggregation
"""

import gzip
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure local conftest is importable
sys.path.insert(0, str(Path(__file__).parent))

from downcraft import __main__ as cli  # noqa: E402
from downcraft import download as download_api  # noqa: E402
from downcraft.download.http import (  # noqa: E402
    DownloadError,
    WireStats,
    _part_path,
    download_file,
    download_range,
)
from downcraft.download.multipart import download_parts  # noqa: E402
from helpers import RangeHandler, _range_url  # noqa: E402


class TestIdentityWireAccounting:
    def test_identity_wire_equals_decoded(self, range_server):
        content = b"x" * 10_000
        RangeHandler.payloads["/plain.bin"] = content

        stats = WireStats()
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "plain.bin"
            download_file(_range_url(range_server, "/plain.bin"), dest, stats=stats)
            assert dest.read_bytes() == content

        assert stats.wire_known
        assert stats.decoded_bytes == len(content)
        assert stats.wire_bytes == len(content)
        assert stats.saved_bytes == 0
        assert stats.saved_pct == 0.0
        assert stats.content_encoding == "identity"
        assert stats.header_bytes > 0  # response headers are counted separately

    def test_identity_without_content_length_resolves_wire(self, range_server):
        content = b"n" * 5_000
        RangeHandler.payloads["/noclen.bin"] = content
        RangeHandler.no_cl_paths["/noclen.bin"] = True

        stats = WireStats()
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "noclen.bin"
            download_file(_range_url(range_server, "/noclen.bin"), dest, stats=stats)
            assert dest.read_bytes() == content

        assert stats.wire_known  # identity: wire == decoded by definition
        assert stats.wire_bytes == len(content)
        assert stats.saved_bytes == 0

    def test_resume_excludes_prior_session_partial(self, range_server):
        """Wire/decoded cover this call only — not the partial already on disk."""
        content = b"resume" * 5_000  # 30_000
        RangeHandler.payloads["/resume.bin"] = content
        prior = 12_000

        stats = WireStats()
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "resume.bin"
            _part_path(dest).write_bytes(content[:prior])
            download_file(_range_url(range_server, "/resume.bin"), dest, stats=stats)
            assert dest.read_bytes() == content

        assert stats.wire_known
        assert stats.decoded_bytes == len(content) - prior
        assert stats.wire_bytes == len(content) - prior  # 206 Content-Length
        assert stats.saved_bytes == 0


class TestCompressedWireSavings:
    def test_gzip_edge_reports_savings(self, range_server):
        content = b"compressible padding text " * 500  # gzip shrinks hard
        RangeHandler.payloads["/gz.bin"] = content
        RangeHandler.encodings["/gz.bin"] = "gzip"

        stats = WireStats()
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "gz.bin"
            download_file(_range_url(range_server, "/gz.bin"), dest, stats=stats)
            assert dest.read_bytes() == content  # identity on disk

        wire_gz = len(gzip.compress(content))
        assert stats.wire_known
        assert stats.decoded_bytes == len(content)
        assert stats.wire_bytes == wire_gz
        assert stats.saved_bytes == len(content) - wire_gz
        assert stats.saved_pct is not None and stats.saved_pct > 50
        assert stats.content_encoding == "gzip"

    def test_zstd_edge_reports_savings(self, range_server):
        content = b"zstd padding record " * 400
        RangeHandler.payloads["/zs.bin"] = content
        RangeHandler.encodings["/zs.bin"] = "zstd"

        stats = WireStats()
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "zs.bin"
            download_file(_range_url(range_server, "/zs.bin"), dest, stats=stats)
            assert dest.read_bytes() == content

        assert stats.wire_known
        assert stats.wire_bytes < len(content)
        assert stats.saved_bytes and stats.saved_bytes > 0
        assert stats.content_encoding == "zstd"

    def test_encoded_without_content_length_reports_unknown(self, range_server):
        content = b"unknown wire length padding " * 300
        RangeHandler.payloads["/noclen.bin"] = content
        RangeHandler.encodings["/noclen.bin"] = "gzip"
        RangeHandler.no_cl_paths["/noclen.bin"] = True

        stats = WireStats()
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "noclen.bin"
            download_file(_range_url(range_server, "/noclen.bin"), dest, stats=stats)
            assert dest.read_bytes() == content  # decode still works

        assert not stats.wire_known  # pre-decode length is unrecoverable
        assert stats.saved_bytes is None  # no invented numbers
        assert stats.saved_pct is None
        assert stats.content_encoding == "gzip"

    def test_failed_attempt_bandwidth_still_counted(self, range_server):
        """A crashed attempt still counts its wire spend; no fake savings.

        urllib3 raises IncompleteRead without yielding socket-buffered
        bytes to the writer, so the decoded prefix here is NOT guaranteed
        (resume-prefix retention on decode paths is covered by
        ``test_compression_resume``).  What must hold: the promised
        Content-Length was spent on the wire, and saved_* never reports
        savings that didn't happen.
        """
        content = b"crash me " * 1_200
        RangeHandler.payloads["/crash.bin"] = content
        RangeHandler.truncate_once["/crash.bin"] = 4_000  # then connection dies

        stats = WireStats()
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "crash.bin"
            with pytest.raises(DownloadError):  # MAX_RETRIES=1 in tests
                download_file(_range_url(range_server, "/crash.bin"), dest, stats=stats)

        # The crashed response promised a full Content-Length — spent is spent.
        assert stats.wire_bytes == len(content)
        assert stats.wire_known
        assert stats.decoded_bytes <= 4_000  # prefix (if any) reached disk
        assert stats.saved_bytes == 0  # never invent savings from a failure
        assert not stats.saved_pct  # None (nothing delivered) or 0.0


class TestRangeIdentityPin:
    def test_download_range_pins_accept_encoding_identity(self, range_server):
        content = b"slice-payload-" * 400  # 5_600
        RangeHandler.payloads["/slice.bin"] = content

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "slice.bin"
            download_range(_range_url(range_server, "/slice.bin"), dest, 100, 199)
            assert dest.read_bytes() == content[100:200]

        range_reqs = [r for r in RangeHandler.requests_log if r["range"]]
        assert range_reqs, "expected a Range request"
        assert all(r["accept_encoding"] == "identity" for r in range_reqs)

    def test_download_range_rejects_coded_206(self, range_server):
        """A 206 with content-coding markers slices WIRE bytes — refuse it."""
        content = b"trap" * 500
        RangeHandler.payloads["/naive.bin"] = content
        RangeHandler.lz4_paths["/naive.bin"] = True
        RangeHandler.lz4_naive["/naive.bin"] = True  # Range slices the WIRE

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "naive.bin"
            with pytest.raises(DownloadError, match="content coding"):
                download_range(_range_url(range_server, "/naive.bin"), dest, 100, 199)
            assert not dest.exists()


class TestDownloadResultDict:
    def test_download_dict_carries_wire_keys(self, range_server):
        content = b"dict savings payload " * 300
        RangeHandler.payloads["/dict.bin"] = content
        RangeHandler.encodings["/dict.bin"] = "gzip"

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "dict.bin"
            result = download_api(_range_url(range_server, "/dict.bin"), str(dest))
            assert dest.read_bytes() == content

        assert result["status"] == "complete"
        assert result["wire_bytes"] < result["decoded_bytes"]
        assert result["saved_bytes"] > 0
        assert result["saved_pct"] > 0
        assert result["content_encoding"] == "gzip"

    def test_skip_path_reports_zero_wire(self, range_server, tmp_path):
        content = b"already here"
        RangeHandler.payloads["/skip.bin"] = content
        import hashlib

        checksum = hashlib.sha256(content).hexdigest()
        dest = tmp_path / "skip.bin"
        dest.write_bytes(content)

        result = download_api(
            _range_url(range_server, "/skip.bin"),
            str(dest),
            checksum=checksum,
        )
        assert result["status"] == "already_downloaded"
        assert result["wire_bytes"] == 0
        assert result["saved_bytes"] == 0


class TestMultipartAggregation:
    def test_group_sums_wire_and_savings(self, range_server, tmp_path):
        plain = b"p" * 4_000
        squeezed = b"compress me " * 400
        RangeHandler.payloads["/p1.bin"] = plain
        RangeHandler.payloads["/p2.bin"] = squeezed
        RangeHandler.encodings["/p2.bin"] = "gzip"

        result = download_parts(
            [_range_url(range_server, "/p1.bin"), _range_url(range_server, "/p2.bin")],
            dest_dir=tmp_path,
            filenames=["p1.bin", "p2.bin"],
        )

        assert result.status == "complete"
        assert result.wire_known
        assert result.wire_bytes < result.decoded_bytes
        assert result.saved_bytes > 0
        assert result.saved_pct is not None and result.saved_pct > 0

        p_plain, p_gz = result.parts
        assert p_plain.wire_bytes == p_plain.decoded_bytes == len(plain)
        assert p_gz.wire_bytes < p_gz.decoded_bytes
        assert (tmp_path / "p2.bin").read_bytes() == squeezed

    def test_group_saved_pct_none_when_wire_unknown(self, range_server, tmp_path):
        content = b"unknown group wire " * 200
        RangeHandler.payloads["/u.bin"] = content
        RangeHandler.encodings["/u.bin"] = "gzip"
        RangeHandler.no_cl_paths["/u.bin"] = True

        result = download_parts(
            [_range_url(range_server, "/u.bin")],
            dest_dir=tmp_path,
            filenames=["u.bin"],
        )
        assert not result.wire_known
        assert result.saved_pct is None
        assert result.parts[0].saved_bytes is None


class TestCliSavingsLine:
    def test_url_prints_wire_savings(self, range_server, capsys, tmp_path):
        content = b"cli wire report " * 400
        RangeHandler.payloads["/cli-wire.bin"] = content
        RangeHandler.encodings["/cli-wire.bin"] = "gzip"
        dest = tmp_path / "cli-wire.bin"

        cli.cmd_url(
            type(
                "A",
                (),
                {
                    "url": _range_url(range_server, "/cli-wire.bin"),
                    "dest": str(dest),
                    "compressed": False,
                },
            )
        )
        out = capsys.readouterr().out
        assert "Done" in out
        assert "wire" in out
        assert "saved" in out
        assert "gzip" in out
        assert dest.read_bytes() == content

    def test_url_prints_no_compression_line(self, range_server, capsys, tmp_path):
        content = b"plain cli payload"
        RangeHandler.payloads["/cli-plain.bin"] = content
        dest = tmp_path / "cli-plain.bin"

        cli.cmd_url(
            type(
                "A",
                (),
                {
                    "url": _range_url(range_server, "/cli-plain.bin"),
                    "dest": str(dest),
                    "compressed": False,
                },
            )
        )
        out = capsys.readouterr().out
        assert "no compression" in out
