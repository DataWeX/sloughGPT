"""ExternalDownloadBackend — progress accounting (card f6bb491c)."""

from __future__ import annotations

import io
import json
import urllib.request

from domain.infrastructure._internal.external_download import (
    ExternalDownloadBackend,
)


class _FakeResponse:
    """urlopen stand-in that serves deterministic chunk sizes."""

    def __init__(self, payload: bytes):
        self._buf = io.BytesIO(payload)

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serve(monkeypatch, routes: dict[str, bytes]) -> None:
    """Route URL suffix -> canned bytes for urllib.request.urlopen."""

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        for suffix, payload in routes.items():
            if url.endswith(suffix):
                return _FakeResponse(payload)
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_plain_progress_is_cumulative_within_and_across_files(monkeypatch, tmp_path):
    """Regression (card f6bb491c): the plain (compressed=False) path passed the
    CURRENT CHUNK size to _progress — the registry saw 65536/65536/... (chunk
    sizes, resetting per file) instead of a running total, so DownloadsCard
    showed '14.0 KB / 78.0 KB' for a completed 79872 B download and mid-transfer
    percentages were wrong. Progress must accumulate within a file and carry
    across files (bytes_done += file_size after each completion)."""
    monkeypatch.setenv("SLO_CACHE_DIR", str(tmp_path))
    # Keep the artifact registry out of the test: download() calls it
    # best-effort at the end and its state is not this test's subject.
    monkeypatch.setattr(
        "domain.infrastructure._internal.artifact_registry.try_register",
        lambda *a, **k: None,
    )

    payload_a = b"a" * 131072  # 2 full 64 KiB chunks
    payload_b = b"b" * 1000  # short final chunk
    manifest = {
        "files": [
            {"path": "a.bin", "size": len(payload_a)},
            {"path": "b.bin", "size": len(payload_b)},
        ]
    }
    _serve(
        monkeypatch,
        {
            "/models/prog-demo/manifest.json": json.dumps(manifest).encode(),
            "/models/prog-demo/file/a.bin": payload_a,
            "/models/prog-demo/file/b.bin": payload_b,
        },
    )

    backend = ExternalDownloadBackend("http://peer.example", compressed=False)
    reports: list[tuple[int, int]] = []
    completed_files: list[str] = []

    result = backend.download(
        "prog-demo",
        on_progress=lambda rid, done, total, speed: reports.append((done, total)),
        on_file_complete=lambda rid, path: completed_files.append(path),
    )

    assert result["status"] == "completed", result
    assert len(completed_files) == 2

    totals = [done for done, _ in reports]
    # Chunk schedule: a.bin -> 65536, 65536; b.bin -> 1000.
    # Cumulative: 65536, 131072 (end of a.bin), 132072 (end of b.bin).
    # With the bug this was [65536, 65536, 132072] — the per-chunk size
    # repeated for every chunk of the same file.
    assert totals == [65536, 131072, 132072]
    # The denominator stays the grand total for every report.
    assert all(total == 132072 for _, total in reports)
    # Running total never regresses.
    assert totals == sorted(totals)
