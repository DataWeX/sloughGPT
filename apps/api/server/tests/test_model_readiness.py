"""Honest model-readiness verdict (fix: infinite "Model still loading" 503).

Root cause being locked down here: when autoload fails permanently (e.g. fresh
worktree with empty models/hf-cache and SLO_BOOTSTRAP_HF off), startup used to
log "Model loaded + providers registered" with model=None, flip the phase to
ready, and every readiness surface answered "Model still loading" forever.

Contract (main's mechanism):
  - Permanent failures are recorded in ``MODEL_LOAD_STATE`` (status/error):
    ``_note_model_load_error`` names the reason while the caller decides
    terminality, ``_record_model_load_failure`` makes it terminal.
  - ``_get_model_status`` reports the terminal state as ``E_MODEL_LOAD_FAILED``
    with the real reason; ``E_MODEL_LOADING`` is reserved for a load genuinely
    in flight.
  - The PGQ completion callback records a crashed worker too — a process that
    dies records nothing of its own.

Overlap: ``tests/server/test_startup_routers.py::TestAutoloadOutcome`` owns the
``_finish_autoload`` materialization/no-materialization and bootstrap-off exits;
the bootstrap-off case is deliberately NOT repeated here.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from routers.inference import _get_model_status
from startup_progress import MODEL_LOAD_STATE, STARTUP_PHASE


@pytest.fixture(autouse=True)
def _reset_process_state():
    """STARTUP_PHASE/MODEL_LOAD_STATE and ``state`` are process-wide —
    snapshot/restore around each test so verdicts never leak between tests.

    Baseline for every test: the load is in flight (status ``loading``, no
    terminal error); tests that need a terminal state seed it explicitly.
    """
    import state as server_state

    saved_phase = dict(STARTUP_PHASE)
    saved_load = dict(MODEL_LOAD_STATE)
    missing = object()
    saved_attrs = {
        name: getattr(server_state, name, missing)
        for name in ("model", "provider", "model_type", "tokenizer")
    }
    MODEL_LOAD_STATE.clear()
    MODEL_LOAD_STATE.update(status="loading", error=None)
    try:
        yield
    finally:
        STARTUP_PHASE.clear()
        STARTUP_PHASE.update(saved_phase)
        MODEL_LOAD_STATE.clear()
        MODEL_LOAD_STATE.update(saved_load)
        for name, value in saved_attrs.items():
            if value is missing:
                server_state.__dict__.pop(name, None)
            else:
                setattr(server_state, name, value)


class TestGetModelStatusVerdict:
    """_get_model_status must tell starting, failed, and in-flight apart."""

    def test_ready_short_circuits(self):
        with patch("routers.inference._model_ready", return_value=True):
            assert _get_model_status() == {"ready": True}

    def test_autoload_disabled_is_no_model(self):
        with (
            patch("routers.inference._model_ready", return_value=False),
            patch("config.ServerConfig.from_env") as mock_cfg,
        ):
            mock_cfg.return_value = SimpleNamespace(autoload_model="none", default_model="qwen")
            STARTUP_PHASE.update(phase="ready")
            status = _get_model_status()
        assert status["ready"] is False
        assert status["code"] == "E_NO_MODEL"
        assert "Autoload is disabled" in status["reason"]

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
        assert "step 3/9" in status["reason"]

    def test_terminal_failure_is_load_failed_with_real_reason(self):
        """THE bug: phase ready + autoload on + load permanently failed."""
        MODEL_LOAD_STATE.update(
            status="failed",
            error="No local weights for Qwen/Qwen2.5-0.5B-Instruct and SLO_BOOTSTRAP_HF is off",
        )
        STARTUP_PHASE.update(phase="ready")
        with (
            patch("routers.inference._model_ready", return_value=False),
            patch("config.ServerConfig.from_env") as mock_cfg,
        ):
            mock_cfg.return_value = SimpleNamespace(autoload_model="Qwen/Qwen2.5-0.5B-Instruct")
            status = _get_model_status()
        assert status["ready"] is False
        assert status["code"] == "E_MODEL_LOAD_FAILED"
        assert "SLO_BOOTSTRAP_HF" in status["reason"]
        assert status["status"] == 503

    def test_genuinely_in_flight_is_model_loading(self):
        """E_MODEL_LOADING stays for a real in-flight load (nothing terminal)."""
        STARTUP_PHASE.update(phase="ready")
        with (
            patch("routers.inference._model_ready", return_value=False),
            patch("config.ServerConfig.from_env") as mock_cfg,
        ):
            mock_cfg.return_value = SimpleNamespace(autoload_model="Qwen/Qwen2.5-0.5B-Instruct")
            status = _get_model_status()
        assert status["ready"] is False
        assert status["code"] == "E_MODEL_LOADING"


class TestAutoloadRecordsFailureReason:
    """Every permanent-failure exit of _autoload_model names its reason.

    ``_autoload_model`` only *notes* the reason (``_note_model_load_error``);
    ``_finish_autoload`` decides terminality — asserted by
    ``TestAutoloadOutcome`` in tests/server/test_startup_routers.py, as is the
    bootstrap-off give-up exit, so those are not repeated here.
    """

    def _cfg(self, autoload_model="Qwen/Qwen2.5-0.5B-Instruct"):
        return SimpleNamespace(
            native_soul_path="",
            autoload_model=autoload_model,
            autoload_device="auto",
            quantize_slonet=False,
            quant_bits=8,
            quant_mode="symmetric",
        )

    @patch("state.model", None)
    @patch("state.provider", None)
    @patch("time.sleep")
    @patch("domain.infrastructure.hf_hub.download_hf_model", side_effect=RuntimeError("no network"))
    @patch("domain.infrastructure.model_loader.ModelLoader.load")
    def test_hf_download_failure_records_error(self, mock_load, _dl, _sleep):
        """Bootstrap on but the download itself failed → reason recorded."""
        mock_load.return_value = MagicMock(success=False, error="No .slnc or .soul file found")
        from infrastructure.startup import _autoload_model

        with patch.dict("os.environ", {"SLO_BOOTSTRAP_HF": "1"}):
            result = _autoload_model(self._cfg())

        assert result is None
        assert "no network" in (MODEL_LOAD_STATE.get("error") or "")
        assert MODEL_LOAD_STATE.get("status") != "failed"  # caller decides terminality

    @patch("state.model", None)
    @patch("state.provider", None)
    @patch("time.sleep")
    @patch("domain.infrastructure.model_loader.ModelLoader.load")
    def test_post_download_retries_exhausted_records_error(self, mock_load, _sleep):
        """Download succeeded but every load attempt failed → reason recorded."""
        mock_load.return_value = MagicMock(success=False, error="corrupt safetensors")
        from infrastructure.startup import _autoload_model

        with (
            patch.dict("os.environ", {"SLO_BOOTSTRAP_HF": "1"}),
            patch("domain.infrastructure.hf_hub.download_hf_model"),
        ):
            result = _autoload_model(self._cfg())

        assert result is None
        assert "corrupt safetensors" in (MODEL_LOAD_STATE.get("error") or "")
        assert MODEL_LOAD_STATE.get("status") != "failed"

    @patch("state.model", None)
    @patch("state.provider", None)
    @patch("domain.infrastructure.model_loader.ModelLoader.load")
    def test_successful_load_heals_recorded_failure(self, mock_load):
        """A later success must clear a stale terminal verdict.

        Otherwise E_MODEL_LOAD_FAILED sticks for a model that loaded fine on
        the next attempt — the mirror image of the original bug.
        """
        from infrastructure.startup import _autoload_model, _finish_autoload

        MODEL_LOAD_STATE.update(status="failed", error="stale failure from a previous attempt")
        provider = MagicMock()
        mock_load.return_value = MagicMock(
            success=True,
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            model_type="slonet",
            provider=provider,
            model=MagicMock(),
            tokenizer=MagicMock(),
        )

        result = _autoload_model(self._cfg())
        assert result is provider

        with (
            patch("infrastructure.startup._build_guard_for_model", return_value="guard"),
            patch("infrastructure.startup._register_loaded"),
        ):
            ok = _finish_autoload(self._cfg(), provider)

        assert ok is True
        assert MODEL_LOAD_STATE["status"] == "ready"
        assert MODEL_LOAD_STATE["error"] is None


class TestPgqLoadDoneRecordsFailure:
    """The PGQ completion callback's failure branch records the verdict too.

    A crashed worker exits the callback without ever touching
    MODEL_LOAD_STATE — without this record the gate answered E_MODEL_LOADING
    forever for a load that had already stopped (the eternal-503 variant).
    """

    def _orchestrator(self):
        from infrastructure.startup import StartupOrchestrator

        return StartupOrchestrator.__new__(StartupOrchestrator)

    @patch("infrastructure.staged_loader.get_staged_loader")
    def test_failed_proc_records_terminal_failure(self, mock_loader):
        from domain.infrastructure._internal.pugqeep.engine import ProcessStatus

        proc = MagicMock()
        proc.status = ProcessStatus.FAILED
        proc.error = "worker crashed"

        self._orchestrator()._on_pgq_model_load_done(proc)

        assert MODEL_LOAD_STATE["status"] == "failed"
        assert "worker crashed" in (MODEL_LOAD_STATE.get("error") or "")
        mock_loader.return_value.set_model_progress.assert_called()
