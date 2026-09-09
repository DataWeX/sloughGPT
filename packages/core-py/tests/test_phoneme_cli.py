"""Tests for phoneme_encoder_cli.py — CLI tool."""

import subprocess
import sys
import pytest


@pytest.fixture
def run_cli():
    """Helper to run the CLI tool."""
    def _run(*args):
        cmd = [sys.executable, "-m", "domains.multimodal.phoneme_encoder_cli"] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
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
