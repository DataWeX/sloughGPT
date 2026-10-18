"""Tests for phoneme_encoder_cli.py — CLI tool."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Repo root (packages/core-py/tests/<file> -> parents[3]). The CLI runs as a
# subprocess and resolves `domain` via its own sys.path — when the suite is
# launched from packages/core-py (CI's `cd packages/core-py && pytest tests/`)
# the cwd no longer carries the repo root, so the subprocess dies with
# ModuleNotFoundError before any assertion can see stdout.
REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def run_cli():
    """Helper to run the CLI tool."""

    def _run(*args):
        cmd = [sys.executable, "-m", "domain.multimodal._internal.phoneme_encoder_cli"] + list(args)
        env = {
            **os.environ,
            "PYTHONPATH": str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, env=env)
        return result

    return _run


class TestPhonemeCLI:
    def test_single_text(self, run_cli):
        result = run_cli("hello")
        assert result.returncode == 0
        assert "Input:" in result.stdout
        assert "Phonemes:" in result.stdout

    def test_batch_mode(self, run_cli):
        result = run_cli("--batch", "hello", "world")
        assert result.returncode == 0
        assert "Batch encoding" in result.stdout

    def test_language_flag(self, run_cli):
        result = run_cli("--lang", "de", "ich")
        assert result.returncode == 0
        assert "Input:" in result.stdout

    def test_detect_mode(self, run_cli):
        result = run_cli("--detect", "hello")
        assert result.returncode == 0
        assert "Detected language:" in result.stdout

    def test_no_args(self, run_cli):
        result = run_cli()
        assert result.returncode == 1
        assert "Usage:" in result.stdout

    def test_multiple_words(self, run_cli):
        result = run_cli("hello", "world")
        assert result.returncode == 0
        assert "Input:" in result.stdout

    def test_italian(self, run_cli):
        result = run_cli("--lang", "it", "ciao")
        assert result.returncode == 0
        assert "Input:" in result.stdout

    def test_portuguese(self, run_cli):
        result = run_cli("--lang", "pt", "ola")
        assert result.returncode == 0
        assert "Input:" in result.stdout
