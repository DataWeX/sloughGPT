"""Tests for the CLI framework built into cli.py."""

import os
import sys
import json
from pathlib import Path as StdPath
from unittest.mock import patch

import pytest

sys.path.insert(0, str(StdPath(__file__).resolve().parent.parent / "src"))

# Import the framework from cli.py
from cli import (
    Choice, CliPath, Option, Argument, Context, Command, Group,
    UsageError, BadParameter, _parse_args, _record_usage,
    run, option, argument, pass_context, version_option, echo, confirm, group, command,
    _TTY, _SUGGESTIONS,
)


# ── Choice type ─────────────────────────────────────────────────────────

class TestChoice:
    def test_valid_choice_case_insensitive(self):
        c = Choice(["bash", "zsh", "fish"])
        assert c.convert("bash", "shell") == "bash"
        assert c.convert("Bash", "shell") == "bash"
        assert c.convert("ZSH", "shell") == "zsh"

    def test_invalid_choice(self):
        c = Choice(["bash", "zsh", "fish"])
        with pytest.raises(BadParameter):
            c.convert("powershell", "shell")

    def test_case_sensitive(self):
        c = Choice(["bash", "zsh"], case_sensitive=True)
        assert c.convert("bash", "shell") == "bash"
        with pytest.raises(BadParameter):
            c.convert("Bash", "shell")


# ── Path type ───────────────────────────────────────────────────────────

class TestPath:
    def test_basic(self):
        p = CliPath()
        assert p.convert("/tmp", "file") == "/tmp"

    def test_exists_check(self):
        p = CliPath(exists=True)
        with pytest.raises(BadParameter):
            p.convert("/nonexistent/path/xyz", "file")

    def test_resolve_path(self):
        p = CliPath(resolve_path=True)
        result = p.convert(".", "file")
        assert os.path.isabs(result)


# ── Option ──────────────────────────────────────────────────────────────

class TestOption:
    def test_primary_name(self):
        opt = Option(["--host"], default="localhost")
        assert opt.primary == "--host"

    def test_short_name(self):
        opt = Option(["-c", "--config"])
        assert opt.short == "-c"

    def test_no_short_name(self):
        opt = Option(["--config"])
        assert opt.short is None

    def test_dest_from_flag(self):
        opt = Option(["--no-color"])
        assert opt.dest == "no_color"

    def test_dest_from_secondary(self):
        opt = Option(["--json", "output_json"])
        assert opt.dest == "output_json"


# ── Argument ────────────────────────────────────────────────────────────

class TestArgument:
    def test_required(self):
        arg = Argument("name", required=True)
        assert arg.required is True
        assert arg.name == "name"

    def test_optional_with_default(self):
        arg = Argument("name", required=False, default="fallback")
        assert arg.default == "fallback"


# ── Context ─────────────────────────────────────────────────────────────

class TestContext:
    def test_ensure_object(self):
        ctx = Context()
        assert ctx.obj == {}
        ctx.ensure_object(dict)
        assert ctx.obj == {}

    def test_ensure_object_preserves(self):
        ctx = Context(obj={"key": "value"})
        ctx.ensure_object(dict)
        assert ctx.obj == {"key": "value"}

    def test_invoke_callable(self):
        ctx = Context()
        called = []
        def my_func(x=1):
            called.append(x)
        ctx.invoke(my_func, x=42)
        assert called == [42]


# ── Parser ──────────────────────────────────────────────────────────────

class TestParser:
    def test_parse_options(self):
        opts = [Option(["--host"], default="localhost")]
        args = ["--host", "example.com"]
        kwargs, extra = _parse_args(args, opts, [])
        assert kwargs["host"] == "example.com"

    def test_parse_flags(self):
        opts = [Option(["--verbose"], is_flag=True)]
        args = ["--verbose"]
        kwargs, extra = _parse_args(args, opts, [])
        assert kwargs["verbose"] is True

    def test_parse_no_flag(self):
        opts = [Option(["--no-color"], is_flag=True)]
        args = ["--no-color"]
        kwargs, extra = _parse_args(args, opts, [])
        assert kwargs["no_color"] is True

    def test_parse_positional(self):
        args = ["hello", "world"]
        kwargs, extra = _parse_args(args, [], [])
        assert extra == ["hello", "world"]

    def test_parse_argument_mapping(self):
        args = ["my-model"]
        arguments = [Argument("model_name")]
        kwargs, extra = _parse_args(args, [], arguments)
        assert kwargs["model_name"] == "my-model"

    def test_parse_optional_argument_default(self):
        args = []
        arguments = [Argument("model_name", required=False, default="gpt2")]
        kwargs, extra = _parse_args(args, [], arguments)
        assert kwargs["model_name"] == "gpt2"

    def test_parse_required_argument_missing(self):
        args = []
        arguments = [Argument("model_name", required=True)]
        with pytest.raises(UsageError):
            _parse_args(args, [], arguments)

    def test_parse_int_option(self):
        opts = [Option(["--port"], type=int, default=8000)]
        args = ["--port", "9000"]
        kwargs, extra = _parse_args(args, opts, [])
        assert kwargs["port"] == 9000
        assert isinstance(kwargs["port"], int)

    def test_parse_float_option(self):
        opts = [Option(["--lr"], type=float, default=0.01)]
        args = ["--lr", "0.001"]
        kwargs, extra = _parse_args(args, opts, [])
        assert kwargs["lr"] == 0.001

    def test_parse_choice_option(self):
        choice = Choice(["cpu", "cuda", "mps"])
        opts = [Option(["--device"], choice=choice, default="cpu")]
        args = ["--device", "CUDA"]
        kwargs, extra = _parse_args(args, opts, [])
        assert kwargs["device"] == "cuda"

    def test_parse_equals_style(self):
        opts = [Option(["--host"], default="localhost")]
        args = ["--host=example.com"]
        kwargs, extra = _parse_args(args, opts, [])
        assert kwargs["host"] == "example.com"

    def test_parse_multiple(self):
        opts = [Option(["--metadata"], multiple=True)]
        args = ["--metadata", "key1=val1", "--metadata", "key2=val2"]
        kwargs, extra = _parse_args(args, opts, [])
        assert kwargs["metadata"] == ["key1=val1", "key2=val2"]

    def test_parse_unknown_option(self):
        opts = []
        args = ["--unknown"]
        with pytest.raises(UsageError):
            _parse_args(args, opts, [])

    def test_parse_variadic_argument(self):
        args = ["a", "b", "c"]
        arguments = [Argument("items", nargs=-1)]
        kwargs, extra = _parse_args(args, [], arguments)
        assert kwargs["items"] == ["a", "b", "c"]

    def test_parse_short_option(self):
        opts = [Option(["-q", "--quiet"], is_flag=True)]
        args = ["-q"]
        kwargs, extra = _parse_args(args, opts, [])
        assert kwargs["quiet"] is True


# ── Group ───────────────────────────────────────────────────────────────

class TestGroup:
    def test_register_command(self):
        g = Group("test")

        @g.command("hello", help="Say hello")
        def hello():
            pass

        assert "hello" in g.commands
        assert g.commands["hello"].help == "Say hello"

    def test_register_subgroup(self):
        g = Group("test")

        @g.group("sub", help="A subgroup")
        def sub():
            pass

        assert "sub" in g.groups
        assert g.groups["sub"].help == "A subgroup"

    def test_fuzzy_match_prefix(self):
        g = Group("test")
        g.commands["model"] = Command("model", lambda: None)
        g.commands["memory"] = Command("memory", lambda: None)

        matches = g._fuzzy_match("mo")
        assert "model" in matches

    def test_fuzzy_match_substring(self):
        g = Group("test")
        g.commands["checkpoint"] = Command("checkpoint", lambda: None)

        matches = g._fuzzy_match("point")
        assert "checkpoint" in matches

    def test_fuzzy_match_no_match(self):
        g = Group("test")
        g.commands["model"] = Command("model", lambda: None)

        matches = g._fuzzy_match("xyz")
        assert matches == []


# ── Decorators ──────────────────────────────────────────────────────────

class TestDecorators:
    def test_option_decorator(self):
        @option("--host", default="localhost", help="Host")
        def my_cmd(host):
            pass

        assert hasattr(my_cmd, "_options")
        assert len(my_cmd._options) == 1
        assert my_cmd._options[0].dest == "host"

    def test_option_flag(self):
        @option("--verbose", is_flag=True)
        def my_cmd(verbose):
            pass

        assert my_cmd._options[0].is_flag is True

    def test_argument_decorator(self):
        @argument("name", required=True)
        def my_cmd(name):
            pass

        assert hasattr(my_cmd, "_arguments")
        assert len(my_cmd._arguments) == 1
        assert my_cmd._arguments[0].name == "name"

    def test_pass_context_decorator(self):
        @pass_context
        def my_cmd(ctx):
            pass

        # pass_context is a no-op, just returns the function
        assert callable(my_cmd)

    def test_multiple_options(self):
        @option("--host", default="localhost")
        @option("--port", type=int, default=8000)
        def my_cmd(host, port):
            pass

        assert len(my_cmd._options) == 2


# ── echo / confirm ──────────────────────────────────────────────────────

class TestEcho:
    def test_echo(self, capsys):
        echo("hello")
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_echo_no_newline(self, capsys):
        echo("hello", nl=False)
        captured = capsys.readouterr()
        assert captured.out == "hello"

    def test_echo_stderr(self, capsys):
        echo("error", err=True)
        captured = capsys.readouterr()
        assert "error" in captured.err


class TestConfirm:
    def test_confirm_yes(self, capsys):
        with patch("builtins.input", return_value="y"):
            assert confirm("Proceed?") is True

    def test_confirm_no(self, capsys):
        with patch("builtins.input", return_value="n"):
            assert confirm("Proceed?") is False

    def test_confirm_abort(self):
        with patch("builtins.input", return_value="n"):
            with pytest.raises(SystemExit):
                confirm("Proceed?", abort=True)


# ── Usage tracking ──────────────────────────────────────────────────────

class TestUsageTracking:
    def test_record_usage(self, tmp_path):
        usage_file = tmp_path / "usage.json"
        import cli as _cli
        with patch.object(_cli, "_USAGE_PATH", usage_file):
            _record_usage("model list")
            _record_usage("model list")
            _record_usage("train start")

            data = json.loads(usage_file.read_text())
            assert data["model list"] == 2
            assert data["train start"] == 1


# ── Group decorator (top-level) ─────────────────────────────────────────

class TestGroupDecorator:
    def test_creates_group(self):
        @group(help="Test group")
        def my_cli():
            pass

        assert isinstance(my_cli, Group)
        assert my_cli.help == "Test group"

    def test_group_with_commands(self):
        @group()
        def my_cli():
            pass

        @my_cli.command("hello", help="Say hello")
        def hello():
            pass

        assert "hello" in my_cli.commands


# ── Run (integration) ──────────────────────────────────────────────────

class TestRun:
    def test_run_help(self, capsys):
        @group(invoke_without_command=True)
        def my_cli():
            pass

        @my_cli.command("test", help="Test command")
        def test_cmd():
            pass

        with patch("sys.argv", ["cli", "--help"]):
            run(my_cli)

        captured = capsys.readouterr()
        assert "Commands:" in captured.out

    def test_run_version(self, capsys):
        @group()
        def my_cli():
            pass

        with patch("sys.argv", ["cli", "--version"]):
            run(my_cli)

        captured = capsys.readouterr()
        assert "v0.1.0" in captured.out or "dev" in captured.out

    def test_run_command(self, capsys):
        @group()
        def my_cli():
            pass

        @my_cli.command("hello")
        def hello():
            echo("Hello, world!")

        with patch("sys.argv", ["cli", "hello"]):
            run(my_cli)

        captured = capsys.readouterr()
        assert "Hello, world!" in captured.out

    def test_run_subgroup(self, capsys):
        @group()
        def my_cli():
            pass

        @my_cli.group("model", help="Model commands")
        def model():
            pass

        @model.command("list", help="List models")
        def model_list():
            echo("model1\nmodel2")

        with patch("sys.argv", ["cli", "model", "list"]):
            run(my_cli)

        captured = capsys.readouterr()
        assert "model1" in captured.out

    def test_run_fuzzy_match(self, capsys):
        @group()
        def my_cli():
            pass

        @my_cli.command("model", help="Model commands")
        def model():
            echo("models here")

        # Non-interactive: auto-resolves silently
        with patch("sys.argv", ["cli", "mo"]):
            with patch("sys.stdin.isatty", return_value=False):
                run(my_cli)

        captured = capsys.readouterr()
        assert "models here" in captured.out

    def test_run_unknown_command(self, capsys):
        @group()
        def my_cli():
            pass

        @my_cli.command("test")
        def test_cmd():
            pass

        with patch("sys.argv", ["cli", "xyz"]):
            with patch("sys.stdin.isatty", return_value=False):
                run(my_cli)

        captured = capsys.readouterr()
        assert "Unknown command" in captured.out

    def test_run_with_options(self, capsys):
        @group()
        def my_cli():
            pass

        @my_cli.command("serve")
        @option("--port", type=int, default=8000)
        def serve(port):
            echo(f"Port: {port}")

        with patch("sys.argv", ["cli", "serve", "--port", "9000"]):
            run(my_cli)

        captured = capsys.readouterr()
        assert "Port: 9000" in captured.out

    def test_run_with_context(self, capsys):
        @group()
        @option("--host", default="localhost")
        def my_cli(host):
            pass

        @my_cli.command("test")
        @pass_context
        def test_cmd(ctx):
            echo(f"host={ctx.obj.get('host', 'none')}")

        with patch("sys.argv", ["cli", "--host", "example.com", "test"]):
            run(my_cli)

        captured = capsys.readouterr()
        assert "host=example.com" in captured.out

    def test_run_missing_argument(self, capsys):
        @group()
        def my_cli():
            pass

        @my_cli.command("download")
        @argument("model_id")
        def download(model_id):
            pass

        with patch("sys.argv", ["cli", "download"]):
            with pytest.raises(SystemExit):
                run(my_cli)

    def test_run_unknown_group_option_exits_cleanly(self, capsys):
        @group()
        def my_cli():
            pass

        @my_cli.command("test")
        def test_cmd():
            pass

        with patch("sys.argv", ["cli", "--bogus-flag"]):
            with pytest.raises(SystemExit) as exc_info:
                run(my_cli)
            assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "Unknown option: --bogus-flag" in captured.out
        assert "Traceback" not in captured.out

    def test_run_unknown_group_option_before_subcommand(self, capsys):
        @group()
        @option("--host", default="localhost")
        def my_cli(host):
            pass

        @my_cli.command("serve")
        def serve():
            echo("serving")

        with patch("sys.argv", ["cli", "--host", "example.com", "--invalid-opt", "serve"]):
            with pytest.raises(SystemExit) as exc_info:
                run(my_cli)
            assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "Unknown option: --invalid-opt" in captured.out


# ── Integration with real cli.py patterns ────────────────────────────────

class TestRealPatterns:
    def test_cli_group_with_options(self, capsys):
        """Test group with subcommand that has options."""
        @group()
        def cli():
            pass

        @cli.command("greet")
        @option("--name", default="world", help="Name to greet")
        @option("--count", default=1, type=int, help="Number of greetings")
        def greet_cmd(name, count):
            for _ in range(count):
                echo(f"Hello {name}")

        with patch("sys.argv", ["cli", "greet", "--name", "Alice", "--count", "2"]):
            run(cli)

        captured = capsys.readouterr()
        assert "Hello Alice" in captured.out
        assert captured.out.count("Hello Alice") == 2

    def test_subgroup_with_subcommands(self, capsys):
        """Test the model → list pattern used in cli.py."""
        @group(invoke_without_command=True)
        def cli():
            pass

        @cli.group("model", help="Model commands")
        @pass_context
        def model(ctx):
            pass

        @model.command("list", help="List models")
        @pass_context
        def model_list(ctx):
            echo("gpt2\nllama")

        @model.command("status", help="Model status")
        @pass_context
        def model_status(ctx):
            echo("All good")

        with patch("sys.argv", ["cli", "model", "list"]):
            run(cli)

        captured = capsys.readouterr()
        assert "gpt2" in captured.out

    def test_choice_type_in_option(self, capsys):
        """Test Choice type used in cli.py export format."""
        @group()
        def cli():
            pass

        @cli.command("export")
        @option("--format", "-f", "fmt",
            type=Choice(["safetensors", "onnx", "gguf"]),
            default="safetensors")
        def export(fmt):
            echo(f"Format: {fmt}")

        with patch("sys.argv", ["cli", "export", "--format", "onnx"]):
            run(cli)

        captured = capsys.readouterr()
        assert "Format: onnx" in captured.out
