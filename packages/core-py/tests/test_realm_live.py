"""Tests for packages/core-py/domains/shell/realm_live.py — pure logic only."""

from __future__ import annotations

import argparse

import pytest

from domains.shell.realm_live import _parse_grid, main


# ── _parse_grid ─────────────────────────────────────────────────────────────

class TestParseGrid:
    def test_valid_grid(self):
        assert _parse_grid("24,12,24") == (24, 12, 24)

    def test_small_grid(self):
        assert _parse_grid("1,1,1") == (1, 1, 1)

    def test_large_grid(self):
        assert _parse_grid("100,50,100") == (100, 50, 100)

    def test_non_square(self):
        assert _parse_grid("10,5,20") == (10, 5, 20)

    def test_too_few_parts_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_grid("24,12")

    def test_too_many_parts_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_grid("24,12,24,8")

    def test_zero_dimension_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_grid("0,12,24")

    def test_negative_dimension_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_grid("-1,12,24")

    def test_single_zero_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_grid("24,0,24")

    def test_non_numeric_raises(self):
        with pytest.raises((ValueError, argparse.ArgumentTypeError)):
            _parse_grid("a,b,c")


# ── main argument parsing ────────────────────────────────────────────────────

class TestMainArgParsing:
    def test_help_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_invalid_grid_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--grid", "0,0,0"])
        assert exc_info.value.code != 0

    def test_invalid_population_exits(self):
        with pytest.raises(SystemExit):
            main(["--population", "abc"])


# ── main execution (lightweight) ─────────────────────────────────────────────

class TestMainExecution:
    def test_defaults_run(self, capsys):
        result = main(["--ticks", "2", "--fps", "0"])
        assert result == 0
        captured = capsys.readouterr()
        assert "realm run complete" in captured.out

    def test_custom_grid(self, capsys):
        result = main(["--grid", "8,4,8", "--ticks", "1", "--fps", "0"])
        assert result == 0

    def test_with_seasons(self, capsys):
        result = main([
            "--seasons", "--seasons-per-year", "4",
            "--ticks", "2", "--fps", "0",
        ])
        assert result == 0

    def test_custom_population(self, capsys):
        result = main(["--population", "4", "--ticks", "1", "--fps", "0"])
        assert result == 0

    def test_custom_seed(self, capsys):
        result = main(["--seed", "42", "--ticks", "1", "--fps", "0"])
        assert result == 0

    def test_output_contains_stats(self, capsys):
        main(["--ticks", "3", "--fps", "0"])
        captured = capsys.readouterr()
        assert "ticks=3" in captured.out
        assert "energy_total=" in captured.out
        assert "alive=" in captured.out
        assert "births=" in captured.out
        assert "deaths=" in captured.out

    def test_custom_day_ticks(self, capsys):
        result = main(["--day", "16", "--ticks", "2", "--fps", "0"])
        assert result == 0

    def test_custom_rate(self, capsys):
        result = main(["--rate", "0.6", "--ticks", "1", "--fps", "0"])
        assert result == 0

    def test_custom_seasonality(self, capsys):
        result = main([
            "--seasons", "--seasonality", "0.5",
            "--ticks", "2", "--fps", "0",
        ])
        assert result == 0

    def test_multiple_days(self, capsys):
        result = main(["--ticks", "48", "--day", "24", "--fps", "0"])
        assert result == 0
        captured = capsys.readouterr()
        assert "ticks=48" in captured.out

    def test_large_population(self, capsys):
        result = main(["--population", "16", "--ticks", "1", "--fps", "0"])
        assert result == 0

    def test_no_seasons_flag_zero_season_ticks(self, capsys):
        result = main(["--ticks", "1", "--fps", "0"])
        assert result == 0
