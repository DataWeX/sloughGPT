"""Tests for apps/cli/src/core/cli_group.py — SmartGroup fuzzy matching."""
import sys
import os
import pytest
import click
from click.testing import CliRunner

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def smart_group():
    """Create a SmartGroup with test commands."""
    from core.cli_group import SmartGroup

    @click.group(cls=SmartGroup)
    def cli():
        pass

    @cli.command()
    def models():
        """List models."""

    @cli.command()
    def chat():
        """Start chat."""

    @cli.command()
    def train():
        """Train model."""

    @cli.command()
    def health():
        """Check health."""

    return cli


class TestSmartGroup:
    def test_exact_match(self, smart_group):
        runner = CliRunner()
        result = runner.invoke(smart_group, ["models"])
        assert result.exit_code == 0

    def test_fuzzy_match_prefix(self, smart_group):
        runner = CliRunner()
        # "mod" should fuzzy-match to "models"
        result = runner.invoke(smart_group, ["mod"])
        assert result.exit_code == 0

    def test_fuzzy_match_single_char(self, smart_group):
        runner = CliRunner()
        # "ch" should match "chat"
        result = runner.invoke(smart_group, ["ch"])
        assert result.exit_code == 0

    def test_no_match_exits(self, smart_group):
        runner = CliRunner()
        result = runner.invoke(smart_group, ["nonexistent"])
        assert result.exit_code != 0

    def test_substring_match(self, smart_group):
        runner = CliRunner()
        # "hea" is substring of "health"
        result = runner.invoke(smart_group, ["hea"])
        assert result.exit_code == 0

    def test_help_lists_commands(self, smart_group):
        runner = CliRunner()
        result = runner.invoke(smart_group, ["--help"])
        assert result.exit_code == 0
        assert "models" in result.output.lower() or "Models" in result.output

    def test_get_command_returns_none_for_unknown(self, smart_group):
        runner = CliRunner()
        # Direct API test
        ctx = click.Context(smart_group)
        cmd = smart_group.get_command(ctx, "zzz")
        assert cmd is None

    def test_get_command_exact(self, smart_group):
        ctx = click.Context(smart_group)
        cmd = smart_group.get_command(ctx, "models")
        assert cmd is not None
        assert cmd.name == "models"

    def test_fuzzy_match_case_insensitive(self, smart_group):
        ctx = click.Context(smart_group)
        # "MODELS" should fuzzy-match via substring
        cmd = smart_group.get_command(ctx, "MODELS")
        assert cmd is not None
