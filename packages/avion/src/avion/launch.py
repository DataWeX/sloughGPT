"""launch — avion owns the browser it drives.

A computer-use agent that must re-login after every restart is a demo, not
a library.  This module gives the browser a *home*: a named persistent
profile (a real ``--user-data-dir``) that survives restarts, so a login
performed once — by a human watching the window, or by a recorded flow
replayed via ``events/replay.py`` — is still there next time avion
connects.  The CDP backend already knew how to *speak* to a browser
(``backends/cdp.py``); what was missing is the launch half: find a
Chromium, start it against a profile, and wait for Chrome to write down
where it put its debug socket.

The handshake: Chrome started with ``--remote-debugging-port=0`` (pick a
free port) writes ``<user-data-dir>/DevToolsActivePort``::

    line 1: the port it actually bound
    line 2: the browser websocket path (``/devtools/browser/<uuid>``)

Chrome tells us where it is; we connect.  No port scanning, no guessing.

Usage::

    from avion.launch import Profile, launch_chromium

    launched = launch_chromium(Profile("x"), headless=False)
    try:
        await launched.backend.start()   # a connected CDP backend
        # ... log in once; the profile remembers ...
    finally:
        launched.stop()
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from avion.backends.cdp import CdpBackend

# Chromium binary names, most specific first.
_CHROMIUM_NAMES = (
    "chromium",
    "chromium-browser",
    "google-chrome-stable",
    "google-chrome",
    "chrome",
)

# Playwright's browser cache — a common home for a Chromium on dev boxes.
_PLAYWRIGHT_PATTERNS = (
    "chromium-*/chrome-linux/chrome",
    "chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell",
)


def profiles_root() -> Path:
    """Where named profiles live — env override, then XDG, then ~/.config."""
    env = os.environ.get("AVION_PROFILES_DIR")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "avion" / "profiles"


@dataclass
class Profile:
    """A named, persistent browser home (a real ``--user-data-dir``)."""

    name: str = "default"
    root: Path = field(default_factory=profiles_root)

    @property
    def dir(self) -> Path:
        return self.root / self.name

    def ensure(self) -> Path:
        """Create the profile directory if needed; return it."""
        self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir


@dataclass(frozen=True)
class DevToolsEndpoint:
    """Where Chrome put its debug socket (parsed from DevToolsActivePort)."""

    port: int
    browser_ws_path: str

    @property
    def base_ws_url(self) -> str:
        """``ws://127.0.0.1:<port>`` — the base ``CdpBackend`` expects."""
        return f"ws://127.0.0.1:{self.port}"

    @property
    def browser_ws_url(self) -> str:
        """The full browser websocket URL (the browser-level target)."""
        return self.base_ws_url + self.browser_ws_path


def read_devtools_endpoint(
    user_data_dir: Path, *, timeout: float = 15.0, poll: float = 0.1
) -> DevToolsEndpoint:
    """Parse ``<user-data-dir>/DevToolsActivePort`` — Chrome's own handshake.

    Polls until the file exists and parses (Chrome writes it asynchronously
    after binding its port).  Raises ``TimeoutError`` with the directory in
    the message so a failed launch is diagnosable, never silent.
    """
    port_file = Path(user_data_dir) / "DevToolsActivePort"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            lines = port_file.read_text().splitlines()
        except OSError:
            time.sleep(poll)
            continue
        if len(lines) >= 2 and lines[0].strip().isdigit():
            return DevToolsEndpoint(port=int(lines[0]), browser_ws_path=lines[1].strip())
        time.sleep(poll)
    raise TimeoutError(
        f"no readable DevToolsActivePort in {user_data_dir} after {timeout}s "
        "(did Chromium exit early?)"
    )


def find_chromium(binary: str | None = None) -> str:
    """Locate a Chromium/Chrome binary — explicit, then env, then PATH, then cache."""
    if binary:
        if not Path(binary).exists():
            raise FileNotFoundError(f"chromium binary not found: {binary}")
        return binary
    env = os.environ.get("AVION_CHROMIUM")
    if env and Path(env).exists():
        return env
    for name in _CHROMIUM_NAMES:
        hit = shutil.which(name)
        if hit:
            return hit
    cache = Path.home() / ".cache" / "ms-playwright"
    if cache.is_dir():
        for pattern in _PLAYWRIGHT_PATTERNS:
            hits = sorted(cache.glob(pattern))
            if hits:
                return str(hits[-1])
    raise RuntimeError(
        "no Chromium found — set AVION_CHROMIUM, install chromium, "
        "or install the playwright browser cache"
    )


@dataclass
class Launched:
    """A live Chromium avion owns: the process, its profile, a CDP backend."""

    process: subprocess.Popen
    backend: CdpBackend
    profile: Profile
    endpoint: DevToolsEndpoint
    argv: tuple[str, ...] = ()

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    def stop(self, *, grace: float = 5.0) -> None:
        """Terminate the browser, escalate to kill if it refuses to leave."""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=grace)

    def __enter__(self) -> Launched:
        return self

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False


def launch_chromium(
    profile: Profile | None = None,
    *,
    headless: bool = False,
    binary: str | None = None,
    extra_args: Sequence[str] = (),
    timeout: float = 15.0,
) -> Launched:
    """Start Chromium against a persistent profile; return a wired backend.

    Default is a VISIBLE window — a human logging in once is the point of
    persistence; scripts and tests pass ``headless=True``.  The returned
    backend is constructed (not connected): call ``await launched.backend.start()``
    when ready.  If the debug handshake times out, the process is stopped
    and the error carries the exact argv to reproduce by hand.
    """
    from avion.backends.cdp import CdpBackend

    profile = profile or Profile()
    profile.ensure()
    chromium = find_chromium(binary)
    # A previous run's handshake file would fool the poll below into
    # reporting the OLD (dead) port — delete it so the file's reappearance
    # is unambiguous evidence of THIS browser's bind. Seen in the wild:
    # relaunching the same profile got the same port number back and the
    # stale file answered first.
    try:
        (profile.dir / "DevToolsActivePort").unlink()
    except FileNotFoundError:
        pass
    argv = (
        chromium,
        f"--user-data-dir={profile.dir}",
        "--remote-debugging-port=0",
        "--no-first-run",
        "--no-default-browser-check",
        *(("--headless=new",) if headless else ()),
        *extra_args,
        "about:blank",
    )
    process = subprocess.Popen(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        endpoint = read_devtools_endpoint(profile.dir, timeout=timeout)
    except BaseException:
        process.terminate()
        raise
    return Launched(
        process=process,
        backend=CdpBackend(cdp_url=endpoint.base_ws_url),
        profile=profile,
        endpoint=endpoint,
        argv=argv,
    )
