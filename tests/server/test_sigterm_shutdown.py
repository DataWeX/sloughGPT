"""SIGTERM must reach the host even with a live PGQ Engine — card 20261104_070434.

The card's minimal pair, as a test: the same uvicorn app exits promptly after
SIGTERM unless ``Engine()`` is constructed during lifespan. Before the fix the
engine's constructor replaced uvicorn's ``capture_signals`` handler with one
that only stops the engine — the process then ignored TERM entirely and could
only be SIGKILLed, which made every FastAPI shutdown hook dead code in
production (``_shutdown_process_guard``, ``_cancel_parent_preload``, …).

Two shapes, both must pass:
- default: ``Engine()`` must not touch the host's handlers at all;
- explicit install (standalone-runner shape): the engine's handler must stop
  the engine AND chain to uvicorn's ``handle_exit`` so graceful shutdown runs.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_RUNNER = textwrap.dedent(
    '''
    """Uvicorn host + PGQ engine under test (see test_sigterm_shutdown.py)."""
    import os
    from contextlib import asynccontextmanager
    from pathlib import Path

    import uvicorn
    from fastapi import FastAPI

    from domain.infrastructure._internal.pugqeep.engine import Engine

    @asynccontextmanager
    async def lifespan(app):
        engine = Engine("sigterm-e2e")
        if os.environ.get("SLO_SIGTERM_INSTALL") == "1":
            # standalone-runner shape: the owner opts into signals explicitly
            engine.install_signal_handlers()
        Path(os.environ["SLO_SIGTERM_STARTED"]).write_text("started")
        yield
        Path(os.environ["SLO_SIGTERM_SHUTDOWN"]).write_text("shutdown")

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    def root():
        return {"ok": True}

    if __name__ == "__main__":
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=int(os.environ["SLO_SIGTERM_PORT"]),
            log_level="warning",
        )
    '''
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _run_term_case(tmp_path: Path, install: bool) -> tuple[int | None, bool]:
    """Start the runner, wait until Engine() is live, send SIGTERM, report outcome."""
    port = _free_port()
    started = tmp_path / "started"
    shutdown = tmp_path / "shutdown"
    log_path = tmp_path / "runner.log"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]).rstrip(
            os.pathsep
        ),
        "SLO_SIGTERM_PORT": str(port),
        "SLO_SIGTERM_STARTED": str(started),
        "SLO_SIGTERM_SHUTDOWN": str(shutdown),
        "SLO_SIGTERM_INSTALL": "1" if install else "0",
    }
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            [sys.executable, str(tmp_path / "runner.py")],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    try:
        deadline = time.monotonic() + 30
        while not started.exists():
            if proc.poll() is not None:
                pytest.fail(
                    f"runner exited {proc.returncode} before startup:\n{log_path.read_text()}"
                )
            if time.monotonic() > deadline:
                pytest.fail(
                    "runner never reached startup (Engine() not constructed):\n"
                    f"{log_path.read_text()}"
                )
            time.sleep(0.1)

        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            pytest.fail(
                f"runner ignored SIGTERM for 5 s (engine signal hijack):\n{log_path.read_text()}"
            )
        return proc.returncode, shutdown.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


@pytest.mark.parametrize(
    "install",
    [False, True],
    ids=["default-no-install", "explicit-install-chains"],
)
def test_sigterm_reaches_uvicorn_host(tmp_path: Path, install: bool) -> None:
    """TERM must stop the process promptly AND run the host's shutdown hooks."""
    (tmp_path / "runner.py").write_text(_RUNNER)
    returncode, shutdown_ran = _run_term_case(tmp_path, install)
    assert shutdown_ran, "FastAPI shutdown hook never ran — the host never learned about SIGTERM"
    assert returncode is not None
