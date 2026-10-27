"""Honest model-readiness verdict (fix: infinite "Model still loading" 503).

Root cause being locked down here: when autoload fails permanently (e.g. fresh
worktree with empty models/hf-cache and SLO_BOOTSTRAP_HF off), startup used to
log "Model loaded + providers registered" with model=None, flip the phase to
ready, and every readiness surface answered "Model still loading" forever.

Contract:
  - The load thread records the permanent failure as STARTUP_PHASE["model_error"].
  - _get_model_status surfaces it as E_NO_MODEL with the real reason.
  - "still loading" (E_MODEL_LOADING) is reserved for a load genuinely in flight.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from routers.inference import _get_model_status
from startup_progress import STARTUP_PHASE


@pytest.fixture(autouse=True)
def _reset_startup_phase():
    """STARTUP_PHASE is a process-wide dict — snapshot/restore around each test."""
    saved = dict(STARTUP_PHASE)
    yield
    STARTUP_PHASE.clear()
    STARTUP_PHASE.update(saved)


def _force_not_ready():
    """Patch every source _model_ready consults so the verdict matrix is isolated."""
    return (
        patch("state.model", None),
        patch("state.provider", None),
        patch("domain.infrastructure.server_state.get_server_state"),
    )


class TestGetModelStatusVerdict:
    """_get_model_status must tell loading apart from failed."""

    def test_ready_short_circuits(self):
        with patch("routers.inference._model_ready", return_value=True):
            assert _get_model_status() == {"ready": True}

    def test_autoload_disabled_is_no_model(self):
        with (
            patch("routers.inference._model_ready", return_value=False),
            patch("config.ServerConfig.from_env") as mock_cfg,
        ):
            mock_cfg.return_value = SimpleNamespace(autoload_model="none", default_model="qwen")
            STARTUP_PHASE.update(phase="ready", model_error=None)
            status = _get_model_status()
        assert status["ready"] is False
        assert status["code"] == "E_NO_MODEL"

    def test_phase_not_ready_is_starting(self):
        STARTUP_PHASE.update(phase="config", step=3, total=9, message="Validating config...")
        with (
            patch("routers.inference._model_ready", return_value=False),
            patch("config.ServerConfig.from_env") as mock_cfg,
        ):
            mock_cfg.return_value = SimpleNamespace(autoload_model="Qwen/Qwen2.5-0.5B-Instruct")
            status = _get_model_status()
        assert status["ready"] is False
        assert status["code"] == "E_STARTING"

    def test_permanent_failure_is_no_model_with_real_reason(self):
        """THE bug: phase ready + autoload on + load permanently failed."""
        STARTUP_PHASE.update(
            phase="ready",
            model_error="No local weights for Qwen/Qwen2.5-0.5B-Instruct and SLO_BOOTSTRAP_HF is off",
        )
        with (
            patch("routers.inference._model_ready", return_value=False),
            patch("config.ServerConfig.from_env") as mock_cfg,
        ):
            mock_cfg.return_value = SimpleNamespace(autoload_model="Qwen/Qwen2.5-0.5B-Instruct")
            status = _get_model_status()
        assert status["ready"] is False
        assert status["code"] == "E_NO_MODEL"
        assert "SLO_BOOTSTRAP_HF" in status["reason"]

    def test_genuinely_in_flight_is_model_loading(self):
        """E_MODEL_LOADING stays for a real in-flight load (no failure recorded)."""
        STARTUP_PHASE.update(phase="ready", model_error=None)
        with (
            patch("routers.inference._model_ready", return_value=False),
            patch("config.ServerConfig.from_env") as mock_cfg,
        ):
            mock_cfg.return_value = SimpleNamespace(autoload_model="Qwen/Qwen2.5-0.5B-Instruct")
            status = _get_model_status()
        assert status["ready"] is False
        assert status["code"] == "E_MODEL_LOADING"


class TestAutoloadRecordsModelError:
    """Every permanent failure exit of _autoload_model records the verdict."""

    def _cfg(self, autoload_model="Qwen/Qwen2.5-0.5B-Instruct"):
        return SimpleNamespace(
            native_soul_path="",
            autoload_model=autoload_model,
            autoload_device="auto",
            quantize_slonet=False,
            quant_bits=8,
            quant_mode="symmetric",
        )

    def _failed_result(
        self, model_id="Qwen/Qwen2.5-0.5B-Instruct", error="No .slnc or .soul file found"
    ):
        from domain.infrastructure._internal.model_loader import LoadResult

        return LoadResult(success=False, model_id=model_id, model_type="unknown", error=error)

    @patch("state.model", None)
    @patch("state.provider", None)
    @patch("time.sleep")
    @patch("domain.infrastructure.model_loader.ModelLoader.load")
    def test_bootstrap_off_records_error(self, mock_load, _sleep):
        """No local weights + HF bootstrap off → model_error names the model."""
        mock_load.return_value = self._failed_result()
        from infrastructure.startup import _autoload_model

        with patch.dict("os.environ", {"SLO_BOOTSTRAP_HF": "0"}):
            result = _autoload_model(self._cfg())

        assert result is None
        assert STARTUP_PHASE.get("model_error")
        assert "Qwen/Qwen2.5-0.5B-Instruct" in STARTUP_PHASE["model_error"]

    @patch("state.model", None)
    @patch("state.provider", None)
    @patch("time.sleep")
    @patch("domain.infrastructure.hf_hub.download_hf_model", side_effect=RuntimeError("no network"))
    @patch("domain.infrastructure.model_loader.ModelLoader.load")
    def test_hf_download_failure_records_error(self, mock_load, _dl, _sleep):
        """Bootstrap on but the download itself failed → error recorded."""
        mock_load.return_value = self._failed_result()
        from infrastructure.startup import _autoload_model

        with patch.dict("os.environ", {"SLO_BOOTSTRAP_HF": "1"}):
            result = _autoload_model(self._cfg())

        assert result is None
        assert "no network" in (STARTUP_PHASE.get("model_error") or "")

    @patch("state.model", None)
    @patch("state.provider", None)
    @patch("time.sleep")
    @patch("domain.infrastructure.model_loader.ModelLoader.load")
    def test_post_download_retries_exhausted_records_error(self, mock_load, _sleep):
        """Download succeeded but every load attempt failed → error recorded."""
        mock_load.return_value = self._failed_result(error="corrupt safetensors")
        from infrastructure.startup import _autoload_model

        with (
            patch.dict("os.environ", {"SLO_BOOTSTRAP_HF": "1"}),
            patch("domain.infrastructure.hf_hub.download_hf_model"),
        ):
            result = _autoload_model(self._cfg())

        assert result is None
        assert "corrupt safetensors" in (STARTUP_PHASE.get("model_error") or "")

    @patch("state.model", None)
    @patch("state.provider", None)
    @patch("time.sleep")
    @patch("domain.infrastructure.model_loader.ModelLoader.load")
    def test_successful_load_clears_error(self, mock_load, _sleep):
        """A successful autoload must not carry a stale failure verdict."""
        from domain.infrastructure._internal.model_loader import LoadResult

        STARTUP_PHASE["model_error"] = "stale failure from a previous attempt"
        provider = MagicMock()
        mock_load.return_value = LoadResult(
            success=True,
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            model_type="slonet",
            provider=provider,
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        from infrastructure.startup import _autoload_model

        result = _autoload_model(self._cfg())

        assert result is provider
        assert not STARTUP_PHASE.get("model_error")


class TestPgqLoadDoneRecordsFailure:
    """The PGQ completion callback's failure branch records the verdict too."""

    def _orchestrator(self):
        from infrastructure.startup import StartupOrchestrator

        return StartupOrchestrator.__new__(StartupOrchestrator)

    @patch("infrastructure.staged_loader.get_staged_loader")
    def test_failed_proc_records_error(self, mock_loader):
        from domain.infrastructure._internal.pugqeep.engine import ProcessStatus

        proc = MagicMock()
        proc.status = ProcessStatus.FAILED
        proc.error = "worker crashed"

        self._orchestrator()._on_pgq_model_load_done(proc)

        assert "worker crashed" in (STARTUP_PHASE.get("model_error") or "")
        mock_loader.return_value.set_model_progress.assert_called_once()
