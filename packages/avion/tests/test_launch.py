"""launch tests — profiles, the DevToolsActivePort handshake, Chromium launch.

Pure tests always run; the real-browser test degrades explicitly (stated
skip) when no Chromium is available — never a silent pass, never a crash.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import pytest
from avion.launch import (
    DevToolsEndpoint,
    Launched,
    Profile,
    find_chromium,
    launch_chromium,
    profiles_root,
    read_devtools_endpoint,
)


class TestProfilesRoot:
    def test_env_override_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AVION_PROFILES_DIR", str(tmp_path / "p"))
        assert profiles_root() == tmp_path / "p"

    def test_xdg_respected(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AVION_PROFILES_DIR", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        assert profiles_root() == tmp_path / "xdg" / "avion" / "profiles"

    def test_default_under_home(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AVION_PROFILES_DIR", raising=False)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        assert profiles_root() == tmp_path / ".config" / "avion" / "profiles"


class TestProfile:
    def test_dir_is_root_over_name(self, tmp_path):
        p = Profile("x", root=tmp_path)
        assert p.dir == tmp_path / "x"

    def test_profiles_are_isolated(self, tmp_path):
        a, b = Profile("a", root=tmp_path), Profile("b", root=tmp_path)
        assert a.dir != b.dir

    def test_ensure_creates(self, tmp_path):
        p = Profile("made", root=tmp_path)
        assert not p.dir.exists()
        assert p.ensure() == p.dir
        assert p.dir.is_dir()


class TestDevToolsEndpoint:
    def test_urls(self):
        ep = DevToolsEndpoint(port=9333, browser_ws_path="/devtools/browser/abc")
        assert ep.base_ws_url == "ws://127.0.0.1:9333"
        assert ep.browser_ws_url == "ws://127.0.0.1:9333/devtools/browser/abc"


class TestReadDevtoolsEndpoint:
    def _write(self, d: Path, text: str) -> Path:
        d.mkdir(parents=True, exist_ok=True)
        (d / "DevToolsActivePort").write_text(text)
        return d

    def test_parses_port_and_path(self, tmp_path):
        d = self._write(tmp_path, "9222\n/devtools/browser/uuid-1\n")
        ep = read_devtools_endpoint(d, timeout=0.5)
        assert ep.port == 9222
        assert ep.browser_ws_path == "/devtools/browser/uuid-1"

    def test_missing_file_times_out_with_directory(self, tmp_path):
        with pytest.raises(TimeoutError, match="no readable DevToolsActivePort"):
            read_devtools_endpoint(tmp_path / "absent", timeout=0.2)

    def test_malformed_never_silently_parses(self, tmp_path):
        d = self._write(tmp_path, "not-a-port\n/devtools/browser/x\n")
        with pytest.raises(TimeoutError):
            read_devtools_endpoint(d, timeout=0.3)

    def test_waits_for_chromes_async_write(self, tmp_path):
        """Chrome writes the file after binding — polling must catch up."""
        d = tmp_path / "slow"
        d.mkdir()

        def _late():
            time.sleep(0.3)
            (d / "DevToolsActivePort").write_text("9444\n/devtools/browser/late\n")

        t = threading.Thread(target=_late)
        t.start()
        ep = read_devtools_endpoint(d, timeout=5.0)
        t.join()
        assert ep.port == 9444


class TestFindChromium:
    def test_explicit_binary_honored(self, tmp_path):
        fake = tmp_path / "chrome"
        fake.touch()
        assert find_chromium(str(fake)) == str(fake)

    def test_explicit_binary_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_chromium(str(tmp_path / "nope"))

    def test_env_var_honored(self, monkeypatch, tmp_path):
        fake = tmp_path / "envchrome"
        fake.touch()
        monkeypatch.setenv("AVION_CHROMIUM", str(fake))
        assert find_chromium() == str(fake)

    def test_nothing_found_raises_with_hint(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AVION_CHROMIUM", raising=False)
        monkeypatch.setenv("PATH", str(tmp_path))  # nothing on PATH
        monkeypatch.setenv("HOME", str(tmp_path))  # no playwright cache
        with pytest.raises(RuntimeError, match="AVION_CHROMIUM"):
            find_chromium()


class _StubProcess:
    """A Popen stand-in for stop()/escalation tests — no real browser."""

    def __init__(self, dies_on_terminate: bool = True):
        self._dead = False
        self.terminated = False
        self.killed = False
        self._dies = dies_on_terminate

    def poll(self):
        return 0 if self._dead else None

    def terminate(self):
        self.terminated = True
        if self._dies:
            self._dead = True

    def kill(self):
        self.killed = True
        self._dead = True

    def wait(self, timeout=None):
        if not self._dead:
            raise subprocess.TimeoutExpired("proc", timeout or 0)
        return 0


def _launched(proc) -> Launched:
    return Launched(
        process=proc,
        backend=None,  # type: ignore[arg-type] — stop()/alive never touch it
        profile=Profile("t"),
        endpoint=DevToolsEndpoint(1, "/x"),
    )


class TestLaunchedStop:
    def test_terminate_suffices_when_it_dies(self):
        proc = _StubProcess(dies_on_terminate=True)
        _launched(proc).stop()
        assert proc.terminated and not proc.killed

    def test_escalates_to_kill(self):
        proc = _StubProcess(dies_on_terminate=False)
        _launched(proc).stop(grace=0.05)
        assert proc.terminated and proc.killed

    def test_already_dead_not_signaled(self):
        proc = _StubProcess()
        proc._dead = True
        _launched(proc).stop()
        assert not proc.terminated and not proc.killed

    def test_context_manager_stops(self):
        proc = _StubProcess()
        with _launched(proc) as launched:
            assert launched.alive
        assert proc.terminated


def _chromium_or_skip() -> str:
    try:
        return find_chromium()
    except RuntimeError:
        pytest.skip("no Chromium available — set AVION_CHROMIUM or install chromium")


class TestLaunchChromiumIntegration:
    """The real handshake: launch → DevToolsActivePort → debug HTTP answers."""

    def test_launch_profile_endpoint_stop(self, tmp_path):
        binary = _chromium_or_skip()
        profile = Profile("itest", root=tmp_path / "profiles")
        launched = launch_chromium(
            profile,
            headless=True,
            binary=binary,
            extra_args=("--no-sandbox", "--disable-gpu"),
            timeout=30.0,
        )
        try:
            assert launched.alive
            # The profile is a real user-data-dir the browser accepted.
            assert launched.profile.dir.is_dir()
            # The debug socket answers — the handshake is real end-to-end.
            url = f"http://127.0.0.1:{launched.endpoint.port}/json/version"
            with urllib.request.urlopen(url, timeout=5) as resp:
                version = json.loads(resp.read().decode())
            assert "Browser" in version
            assert launched.endpoint.base_ws_url.startswith("ws://127.0.0.1:")
        finally:
            launched.stop()
        assert not launched.alive

    def test_profile_persists_across_launches(self, tmp_path):
        """The whole point: the same profile dir is reused next launch."""
        binary = _chromium_or_skip()
        root = tmp_path / "profiles"
        for _ in range(2):
            launched = launch_chromium(
                Profile("sticky", root=root),
                headless=True,
                binary=binary,
                extra_args=("--no-sandbox", "--disable-gpu"),
                timeout=30.0,
            )
            launched.stop()
        # Cookies/local storage live under this dir — a login written here
        # by run 1 is what run 2 boots with.
        assert (root / "sticky" / "Default").is_dir()
