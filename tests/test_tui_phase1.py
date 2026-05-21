"""
TDD for Phase 1 TUI shell: session (repo + API base URL) and HTTP read-only adapters.

See: docs/plans/tui-cli-port.md (Phase 1 — Shell).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("httpx")


def test_discover_repo_root_via_pyproject_name(tmp_path: Path) -> None:
    from apps.tui.session import discover_repo_root

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "sloughgpt"\nversion = "0.0.1"\n',
        encoding="utf-8",
    )
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    assert discover_repo_root(sub) == tmp_path.resolve()


def test_discover_repo_root_via_config_and_core_py_layout(tmp_path: Path) -> None:
    from apps.tui.session import discover_repo_root

    (tmp_path / "config.yaml").write_text("training: {}\n", encoding="utf-8")
    (tmp_path / "packages" / "core-py").mkdir(parents=True)

    assert discover_repo_root(tmp_path) == tmp_path.resolve()


def test_discover_repo_root_returns_none_without_markers(tmp_path: Path) -> None:
    from apps.tui.session import discover_repo_root

    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    assert discover_repo_root(tmp_path) is None


def test_tui_session_api_base_url() -> None:
    from apps.tui.session import TuiSession

    s = TuiSession(repo_root=Path("/tmp/repo"), api_host="127.0.0.1", api_port=9000)
    assert s.api_base_url == "http://127.0.0.1:9000"


def test_tui_session_default_api_port() -> None:
    from apps.tui.session import TuiSession

    s = TuiSession(repo_root=Path("."))
    assert s.api_port == 8000
    assert "8000" in s.api_base_url


def test_fetch_health_success_parses_json() -> None:
    from apps.tui.adapters.http_api import fetch_health

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "healthy", "model_loaded": False}

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = mock_resp

    with patch("apps.tui.adapters.http_api.httpx.Client", return_value=mock_client):
        r = fetch_health("http://localhost:8000")

    assert r.status_code == 200
    assert r.payload is not None
    assert r.payload["status"] == "healthy"
    assert r.error is None


def test_fetch_health_non_json_body() -> None:
    from apps.tui.adapters.http_api import fetch_health

    mock_resp = MagicMock()
    mock_resp.status_code = 502
    mock_resp.json.side_effect = ValueError("not json")

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = mock_resp

    with patch("apps.tui.adapters.http_api.httpx.Client", return_value=mock_client):
        r = fetch_health("http://127.0.0.1:1")

    assert r.status_code == 502
    assert r.payload is None


def test_fetch_health_transport_error() -> None:
    import httpx

    from apps.tui.adapters.http_api import fetch_health

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.side_effect = httpx.ConnectError("refused")

    with patch("apps.tui.adapters.http_api.httpx.Client", return_value=mock_client):
        r = fetch_health("http://127.0.0.1:1")

    assert r.status_code == 0
    assert r.payload is None
    assert r.error is not None


def test_tui_main_exits_zero() -> None:
    from apps.tui import app

    with pytest.raises(SystemExit) as exc:
        app.main(["--help"])
    assert exc.value.code == 0


def test_scan_local_repo_counts_models_and_datasets(tmp_path: Path) -> None:
    from apps.tui.adapters.local_status import scan_local_repo

    (tmp_path / "models" / "nested").mkdir(parents=True)
    (tmp_path / "models" / "nested" / "a.pt").write_text("x", encoding="utf-8")
    (tmp_path / "models" / "b.pth").write_text("y", encoding="utf-8")
    (tmp_path / "datasets" / "shakespeare").mkdir(parents=True)
    (tmp_path / "datasets" / "other.txt").write_text("z", encoding="utf-8")

    snap = scan_local_repo(tmp_path)
    assert snap.repo_root == tmp_path.resolve()
    assert snap.models_dir_found is True
    assert snap.model_file_count == 2
    assert snap.datasets_dir_found is True
    assert snap.dataset_entry_count == 2
    assert "shakespeare" in snap.dataset_sample_names or "other.txt" in snap.dataset_sample_names


def test_scan_local_repo_without_dirs(tmp_path: Path) -> None:
    from apps.tui.adapters.local_status import scan_local_repo

    snap = scan_local_repo(tmp_path)
    assert snap.model_file_count == 0
    assert snap.dataset_entry_count == 0
    assert snap.models_dir_found is False
    assert snap.datasets_dir_found is False


def test_fetch_metrics_hits_metrics_url() -> None:
    from apps.tui.adapters.http_api import fetch_metrics

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"uptime": 1.0, "requests_per_minute": 60}

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = mock_resp

    with patch("apps.tui.adapters.http_api.httpx.Client", return_value=mock_client):
        r = fetch_metrics("http://127.0.0.1:8000")

    assert r.status_code == 200
    assert r.payload is not None
    assert r.payload["requests_per_minute"] == 60
    mock_client.get.assert_called_once()
    assert mock_client.get.call_args[0][0].endswith("/metrics")


def test_fetch_health_detailed_hits_detailed_url() -> None:
    from apps.tui.adapters.http_api import fetch_health_detailed

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "healthy", "model_loaded": True}

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = mock_resp

    with patch("apps.tui.adapters.http_api.httpx.Client", return_value=mock_client):
        r = fetch_health_detailed("http://localhost:9999/")

    assert r.status_code == 200
    assert r.payload is not None
    assert r.payload["model_loaded"] is True
    called = mock_client.get.call_args[0][0]
    assert called.endswith("/health/detailed")


def test_app_main_local_status_prints(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from apps.tui import app

    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "x.pt").write_text("1", encoding="utf-8")
    app.main(["--local-status", "--repo-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "Local Repository" in out
    assert "Total files" in out or "x.pt" in out
    assert "x.pt" in out or "1" in out


def test_app_main_api_health_mocked(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from apps.tui import app
    from apps.tui.adapters import http_api

    def _fake_fetch(base: str):
        return http_api.ApiJsonResult(200, {"status": "healthy"}, None)

    monkeypatch.setattr(http_api, "fetch_health", _fake_fetch)
    app.main(["--api-health"])
    out = capsys.readouterr().out
    assert "/health" in out
    assert "healthy" in out


def test_app_main_api_metrics_mocked(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from apps.tui import app
    from apps.tui.adapters import http_api

    def _fake_fetch(base: str):
        return http_api.ApiJsonResult(200, {"requests_per_minute": 60}, None)

    monkeypatch.setattr(http_api, "fetch_metrics", _fake_fetch)
    app.main(["--api-metrics"])
    out = capsys.readouterr().out
    assert "/metrics" in out
    assert "60" in out


def test_app_main_api_health_detailed_mocked(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from apps.tui import app
    from apps.tui.adapters import http_api

    def _fake_fetch(base: str):
        return http_api.ApiJsonResult(200, {"model_loaded": True}, None)

    monkeypatch.setattr(http_api, "fetch_health_detailed", _fake_fetch)
    app.main(["--api-health-detailed"])
    out = capsys.readouterr().out
    assert "/health/detailed" in out
    assert "model_loaded" in out


def test_app_main_no_action_message(capsys: pytest.CaptureFixture[str]) -> None:
    from apps.tui import app

    app.main([])
    out = capsys.readouterr().out
    assert "--local-status" in out
    assert "run with" in out
