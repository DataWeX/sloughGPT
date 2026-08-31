"""Tests for slo_cli — fuzzy matching, auto-correct, command routing."""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from core import slo_cli as click


@pytest.fixture
def cli_group():
    """Create a slo_cli Group with test commands."""
    @click.group()
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


class TestFuzzyMatching:
    def test_exact_match(self, cli_group):
        assert "models" in cli_group.commands
        assert "chat" in cli_group.commands

    def test_prefix_match(self, cli_group):
        matches = cli_group._fuzzy_match("mod")
        assert "models" in matches

    def test_single_char_match(self, cli_group):
        matches = cli_group._fuzzy_match("ch")
        assert "chat" in matches

    def test_no_match(self, cli_group):
        matches = cli_group._fuzzy_match("zzz")
        assert len(matches) == 0

    def test_substring_match(self, cli_group):
        matches = cli_group._fuzzy_match("hea")
        assert "health" in matches

    def test_case_insensitive(self, cli_group):
        matches = cli_group._fuzzy_match("MODELS")
        assert "models" in matches


class TestCommandRegistration:
    def test_commands_registered(self, cli_group):
        assert len(cli_group.commands) == 4
        assert "models" in cli_group.commands
        assert "chat" in cli_group.commands
        assert "train" in cli_group.commands
        assert "health" in cli_group.commands

    def test_command_help(self, cli_group):
        assert cli_group.commands["models"].help == "List models."

    def test_subgroup_registration(self, cli_group):
        @cli_group.group(help="Manage datasets")
        def dataset():
            pass

        @dataset.command("list", help="List datasets")
        def dataset_list():
            pass

        assert "dataset" in cli_group.groups
        assert "list" in cli_group.groups["dataset"].commands


class TestOptionDecorators:
    def test_option_decorator(self):
        @click.option("--name", default="world", help="Who to greet")
        def hello(name):
            pass

        assert hasattr(hello, "_options")
        assert len(hello._options) == 1
        assert hello._options[0].dest == "name"

    def test_multiple_options(self):
        @click.option("--name", default="world")
        @click.option("--count", type=int, default=1)
        def hello(name, count):
            pass

        assert len(hello._options) == 2

    def test_flag_option(self):
        @click.option("--verbose", is_flag=True)
        def hello(verbose):
            pass

        assert hello._options[0].is_flag is True

    def test_choice_option(self):
        c = click.Choice(["a", "b", "c"])
        @click.option("--mode", choice=c)
        def hello(mode):
            pass

        assert hello._options[0].choice is not None

    def test_argument_decorator(self):
        @click.argument("name")
        def hello(name):
            pass

        assert hasattr(hello, "_arguments")
        assert len(hello._arguments) == 1
        assert hello._arguments[0].name == "name"


class TestTypes:
    def test_choice_convert_valid(self):
        c = click.Choice(["a", "b", "c"])
        assert c.convert("b", "test") == "b"

    def test_choice_convert_invalid(self):
        c = click.Choice(["a", "b", "c"])
        with pytest.raises(click.BadParameter):
            c.convert("d", "test")

    def test_choice_case_insensitive(self):
        c = click.Choice(["A", "B", "C"])
        assert c.convert("a", "test") == "A"

    def test_path_exists(self):
        p = click.Path(exists=False)
        assert p.convert("/tmp", "test") == "/tmp"

    def test_path_not_exists(self):
        p = click.Path(exists=True)
        with pytest.raises(click.BadParameter):
            p.convert("/nonexistent_path_12345", "test")


class TestEchoAndConfirm:
    def test_echo(self, capsys):
        click.echo("hello")
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_echo_no_newline(self, capsys):
        click.echo("hello", nl=False)
        captured = capsys.readouterr()
        assert captured.out == "hello"


class TestContext:
    def test_context_obj(self):
        ctx = click.Context()
        ctx.ensure_object(dict)
        assert isinstance(ctx.obj, dict)

    def test_context_ensure_object_idempotent(self):
        ctx = click.Context(obj={"key": "value"})
        ctx.ensure_object(dict)
        assert ctx.obj == {"key": "value"}


class TestParser:
    def test_parse_simple_args(self):
        opts = [click.Option(["--name"], default="world")]
        args_def = []
        kwargs, extra = click._parse_args(["--name", "foo"], opts, args_def)
        assert kwargs["name"] == "foo"

    def test_parse_flag(self):
        opts = [click.Option(["--verbose"], is_flag=True)]
        kwargs, extra = click._parse_args(["--verbose"], opts, [])
        assert kwargs["verbose"] is True

    def test_parse_int(self):
        opts = [click.Option(["--port"], type=int, default=8000)]
        kwargs, extra = click._parse_args(["--port", "9000"], opts, [])
        assert kwargs["port"] == 9000

    def test_parse_positional(self):
        args_def = [click.Argument("name")]
        kwargs, extra = click._parse_args(["hello"], [], args_def)
        assert kwargs["name"] == "hello"

    def test_parse_unknown_option(self):
        opts = []
        with pytest.raises(click.BadParameter, match="Unknown option"):
            click._parse_args(["--unknown"], opts, [])


class TestGroupExecution:
    def test_run_command(self, cli_group, capsys):
        cli_group(["models"])
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_run_nonexistent(self, cli_group):
        with pytest.raises(SystemExit):
            cli_group(["nonexistent"])

    def test_help_flag(self, cli_group, capsys):
        cli_group(["--help"])
        captured = capsys.readouterr()
        assert "SloughGPT" in captured.out or "Commands" in captured.out


class TestSuggestions:
    def test_suggestions_dict(self):
        assert "model list" in click._SUGGESTIONS
        assert click._SUGGESTIONS["model list"] == "model status"
