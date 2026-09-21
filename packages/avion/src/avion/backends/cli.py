"""CLI backend — spawn processes, send input, expect output.

Pexpect-style interaction using only the stdlib (subprocess + threads).
"""

from __future__ import annotations

import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass


@dataclass
class CliResult:
    """Outcome of an expect/read."""

    matched: bool
    output: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {"matched": self.matched, "output": self.output, "error": self.error}


class CliBackend:
    """Drive an interactive process.

    Usage::

        backend = CliBackend()
        await backend.start()
        await backend.spawn("cat")
        await backend.send("hello")
        result = await backend.expect("hello")
        assert result.matched
        await backend.stop()
    """

    name = "cli"

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._buffer = ""
        self._queue: queue.Queue[str] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._history: list[str] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        self.terminate()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def history(self) -> list[str]:
        return list(self._history)

    async def spawn(self, command: str | list[str]) -> None:
        """Start a process. String commands run through the shell."""
        self.terminate()
        if isinstance(command, str):
            proc = subprocess.Popen(
                command,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        else:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        self._proc = proc
        self._buffer = ""
        self._queue = queue.Queue()
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def _drain(self) -> None:
        try:
            assert self._proc is not None and self._proc.stdout is not None
            for line in self._proc.stdout:
                self._queue.put(line)
        except Exception:
            pass

    async def send(self, text: str, newline: bool = True) -> None:
        """Write to the process stdin."""
        if not self.running:
            raise RuntimeError("no process running")
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(text + ("\n" if newline else ""))
        self._proc.stdin.flush()
        self._history.append(text)

    def _collect(self) -> str:
        while True:
            try:
                self._buffer += self._queue.get_nowait()
            except queue.Empty:
                break
        return self._buffer

    async def expect(self, pattern: str, timeout: float = 10.0) -> CliResult:
        """Wait until pattern (regex) appears in output."""
        import asyncio

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            output = self._collect()
            if re.search(pattern, output):
                return CliResult(matched=True, output=output)
            if not self.running and self._queue.empty():
                return CliResult(matched=False, output=output, error="process exited")
            await asyncio.sleep(0.05)
        return CliResult(
            matched=False, output=self._collect(), error=f"timeout waiting for {pattern!r}"
        )

    async def read(self, timeout: float = 1.0) -> str:
        """Collect whatever output arrives within timeout."""
        import asyncio

        await asyncio.sleep(timeout)
        return self._collect()

    def terminate(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
