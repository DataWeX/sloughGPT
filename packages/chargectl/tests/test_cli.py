"""Tests for chargectl.cli — argument handling and output shape."""

from __future__ import annotations

import pytest
from chargectl.cli import build_parser, main


def test_parser_defaults_to_status():
    args = build_parser().parse_args([])
    assert args.command is None


def test_parser_accepts_limit():
    args = build_parser().parse_args(["limit", "80"])
    assert args.command == "limit"
    assert args.percent == "80"


def test_status_prints_level(capsys: pytest.CaptureFixture[str]):
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "%" in out
    assert "state" in out


def test_advice_prints_action(capsys: pytest.CaptureFixture[str]):
    assert main(["advice"]) == 0
    out = capsys.readouterr().out
    assert "charge limit" in out


def test_limit_rejects_garbage(capsys: pytest.CaptureFixture[str]):
    assert main(["limit", "banana"]) == 2
    assert "invalid limit" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["off", "OFF", "clear", "100"])
def test_limit_off_aliases(value: str, capsys: pytest.CaptureFixture[str]):
    # never raises — returns 0 (applied) or 1 (unsupported), never a traceback
    assert main(["limit", value]) in (0, 1)
    capsys.readouterr()
