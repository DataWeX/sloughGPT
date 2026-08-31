"""
Comprehensive tests for slo_cli — the custom CLI framework.
Tests: Group, Command, Option, Argument, parser, types, fuzzy matching,
auto-correct, echo, confirm, Context, help formatting, end-to-end runs.
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from io import StringIO
from unittest.mock import patch, MagicMock

import pytest

# Ensure the CLI src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.slo_cli import (
    Group, Command, Option, Argument, Context, Choice, Path as CliPath,
    IntRange, UsageError, BadParameter,
    option, argument, pass_context, version_option, group, command,
    echo, confirm,
    _parse_args, _parse_global_options, _format_help,
    _resolve_and_run, _run_command, _run_group, _show_error, run,
    _TTY, _c, _p,
)


# ── Group tests ─────────────────────────────────────────────────────────

class TestGroup:
    def test_create_group(self):
        g = Group("test", "Test help")
        assert g.name == "test"
        assert g.help == "Test help"
        assert g.commands == {}
        assert g.groups == {}

    def test_command_decorator(self):
        g = Group("root")

        @g.command("hello", help="Say hello")
        def hello():
            pass

        assert "hello" in g.commands
        assert g.commands["hello"].help == "Say hello"
        assert g.commands["hello"].func is hello

    def test_group_decorator(self):
        g = Group("root")

        @g.group("sub", help="Sub group")
        def sub():
            pass

        assert "sub" in g.groups
        assert g.groups["sub"].help == "Sub group"
        # group() returns the Group, not the function
        assert isinstance(g.groups["sub"], Group)

    def test_subgroup_commands_register(self):
        root = Group("root")

        @root.group("model")
        def model():
            pass

        @model.command("list", help="List models")
        def model_list():
            pass

        assert "list" in root.groups["model"].commands
        assert root.groups["model"].commands["list"].func is model_list

    def test_add_command(self):
        g = Group("root")
        cmd = Command("test", lambda: None)
        g.add_command(cmd, "test")
        assert "test" in g.commands

    def test_add_group(self):
        g = Group("root")
        sub = Group("sub")
        g.add_group(sub, "sub")
        assert "sub" in g.groups

    def test_fuzzy_match_prefix(self):
        g = Group("root")
        g.commands["model"] = Command("model", lambda: None)
        g.commands["train"] = Command("train", lambda: None)
        assert g._fuzzy_match("mo") == ["model"]
        assert g._fuzzy_match("tr") == ["train"]

    def test_fuzzy_match_substring(self):
        g = Group("root")
        g.commands["checkpoint"] = Command("checkpoint", lambda: None)
        assert g._fuzzy_match("check") == ["checkpoint"]

    def test_fuzzy_match_fuzzy(self):
        g = Group("root")
        g.commands["model"] = Command("model", lambda: None)
        # "modl" is close to "model"
        matches = g._fuzzy_match("modl")
        assert "model" in matches

    def test_fuzzy_match_no_match(self):
        g = Group("root")
        g.commands["model"] = Command("model", lambda: None)
        assert g._fuzzy_match("xyz") == []

    def test_callable_group(self):
        g = Group("root")
        # Group should be callable (runs the CLI)
        assert callable(g)


# ── Option tests ────────────────────────────────────────────────────────

class TestOption:
    def test_basic_option(self):
        opt = Option(["--name"], help="Your name", default="world")
        assert opt.names == ["--name"]
        assert opt.help == "Your name"
        assert opt.default == "world"
        assert opt.dest == "name"

    def test_short_option(self):
        opt = Option(["-n", "--name"], help="Name")
        assert opt.short == "-n"
        assert opt.dest == "name"  # prefers long option

    def test_flag_option(self):
        opt = Option(["--verbose"], is_flag=True)
        assert opt.is_flag is True

    def test_bool_flag_pair(self):
        opt = Option(["--tui", "--no-tui"], is_flag=True)
        assert opt.is_bool_flag is True
        assert opt.dest == "tui"

    def test_multiple_option(self):
        opt = Option(["--tag"], multiple=True)
        assert opt.multiple is True

    def test_required_option(self):
        opt = Option(["--name"], required=True)
        assert opt.required is True

    def test_choice_option(self):
        opt = Option(["--format"], choice=Choice(["json", "csv"]))
        assert opt.choice is not None


# ── Argument tests ──────────────────────────────────────────────────────

class TestArgument:
    def test_required_argument(self):
        arg = Argument("name")
        assert arg.name == "name"
        assert arg.required is True

    def test_optional_argument(self):
        arg = Argument("name", required=False, default="world")
        assert arg.required is False
        assert arg.default == "world"

    def test_variadic_argument(self):
        arg = Argument("files", nargs=-1)
        assert arg.nargs == -1


# ── Parser tests ────────────────────────────────────────────────────────

class TestParser:
    def test_parse_simple_args(self):
        opts = [Option(["--name"], default="world")]
        args = [Argument("prompt")]
        kwargs, extra = _parse_args(["hello", "--name", "test"], opts, args)
        assert kwargs["prompt"] == "hello"
        assert kwargs["name"] == "test"

    def test_parse_flag(self):
        opts = [Option(["--verbose"], is_flag=True)]
        kwargs, extra = _parse_args(["--verbose"], opts, [])
        assert kwargs["verbose"] is True

    def test_parse_no_flag(self):
        opts = [Option(["--verbose"], is_flag=True)]
        kwargs, extra = _parse_args([], opts, [])
        assert kwargs["verbose"] is False

    def test_parse_bool_flag_pair(self):
        opts = [Option(["--tui", "--no-tui"], is_flag=True)]
        kwargs, extra = _parse_args(["--tui"], opts, [])
        assert kwargs["tui"] is True
        kwargs, extra = _parse_args(["--no-tui"], opts, [])
        assert kwargs["tui"] is False

    def test_parse_int_option(self):
        opts = [Option(["--port"], type=int, default=8000)]
        kwargs, extra = _parse_args(["--port", "9000"], opts, [])
        assert kwargs["port"] == 9000

    def test_parse_float_option(self):
        opts = [Option(["--temp"], type=float, default=0.5)]
        kwargs, extra = _parse_args(["--temp", "0.8"], opts, [])
        assert kwargs["temp"] == 0.8

    def test_parse_choice_option(self):
        opts = [Option(["--format"], choice=Choice(["json", "csv"]))]
        kwargs, extra = _parse_args(["--format", "json"], opts, [])
        assert kwargs["format"] == "json"

    def test_parse_invalid_choice(self):
        opts = [Option(["--format"], choice=Choice(["json", "csv"]))]
        with pytest.raises(UsageError):
            _parse_args(["--format", "xml"], opts, [])

    def test_parse_required_argument(self):
        opts = []
        args = [Argument("name")]
        with pytest.raises(UsageError):
            _parse_args([], opts, args)

    def test_parse_optional_argument(self):
        opts = []
        args = [Argument("name", required=False, default="world")]
        kwargs, extra = _parse_args([], opts, args)
        assert kwargs["name"] == "world"

    def test_parse_variadic_argument(self):
        opts = []
        args = [Argument("files", nargs=-1)]
        kwargs, extra = _parse_args(["a.txt", "b.txt", "c.txt"], opts, args)
        assert kwargs["files"] == ["a.txt", "b.txt", "c.txt"]

    def test_parse_eq_style(self):
        opts = [Option(["--name"])]
        kwargs, extra = _parse_args(["--name=test"], opts, [])
        assert kwargs["name"] == "test"

    def test_parse_unknown_option(self):
        opts = []
        with pytest.raises(UsageError):
            _parse_args(["--unknown"], opts, [])

    def test_parse_missing_option_value(self):
        opts = [Option(["--name"])]
        with pytest.raises(UsageError):
            _parse_args(["--name"], opts, [])

    def test_parse_multiple_option(self):
        opts = [Option(["--tag"], multiple=True)]
        kwargs, extra = _parse_args(["--tag", "a", "--tag", "b"], opts, [])
        assert kwargs["tag"] == ["a", "b"]

    def test_parse_shows_default(self):
        opts = [Option(["--port"], type=int, default=8000, show_default=True)]
        kwargs, extra = _parse_args([], opts, [])
        assert kwargs["port"] == 8000


# ── Type tests ──────────────────────────────────────────────────────────

class TestChoice:
    def test_valid_choice(self):
        c = Choice(["json", "csv"])
        assert c.convert("json", "--format") == "json"

    def test_case_insensitive(self):
        c = Choice(["JSON", "CSV"])
        assert c.convert("json", "--format") == "JSON"

    def test_invalid_choice(self):
        c = Choice(["json", "csv"])
        with pytest.raises(BadParameter):
            c.convert("xml", "--format")

    def test_case_sensitive(self):
        c = Choice(["JSON", "CSV"], case_sensitive=True)
        with pytest.raises(BadParameter):
            c.convert("json", "--format")


class TestPath:
    def test_path_exists(self):
        with tempfile.NamedTemporaryFile() as f:
            p = CliPath(exists=True)
            result = p.convert(f.name, "path")
            assert result == f.name

    def test_path_not_exists(self):
        p = CliPath(exists=True)
        with pytest.raises(BadParameter):
            p.convert("/nonexistent/path", "path")

    def test_path_file_okay(self):
        with tempfile.NamedTemporaryFile() as f:
            p = CliPath(file_okay=True)
            result = p.convert(f.name, "path")
            assert result == f.name

    def test_path_dir_not_okay(self):
        with tempfile.TemporaryDirectory() as d:
            p = CliPath(dir_okay=False)
            with pytest.raises(BadParameter):
                p.convert(d, "path")

    def test_path_resolve(self):
        p = CliPath(resolve_path=True)
        result = p.convert(".", "path")
        assert os.path.isabs(result)


class TestIntRange:
    def test_valid_int(self):
        r = IntRange(min=0, max=100)
        assert r.convert("50", "--port") == 50

    def test_below_min(self):
        r = IntRange(min=0, max=100)
        with pytest.raises(BadParameter):
            r.convert("-1", "--port")

    def test_above_max(self):
        r = IntRange(min=0, max=100)
        with pytest.raises(BadParameter):
            r.convert("101", "--port")

    def test_not_int(self):
        r = IntRange()
        with pytest.raises(BadParameter):
            r.convert("abc", "--port")


# ── Context tests ───────────────────────────────────────────────────────

class TestContext:
    def test_context_obj(self):
        ctx = Context({"host": "localhost"})
        assert ctx.obj["host"] == "localhost"

    def test_ensure_object(self):
        ctx = Context()
        ctx.ensure_object(dict)
        assert isinstance(ctx.obj, dict)

    def test_ensure_object_preserves(self):
        ctx = Context({"existing": True})
        ctx.ensure_object(dict)
        assert ctx.obj["existing"] is True

    def test_invoked_subcommand(self):
        ctx = Context()
        ctx.invoked_subcommand = "model"
        assert ctx.invoked_subcommand == "model"

    def test_invoke_command(self):
        ctx = Context()
        called = []

        def my_func(x):
            called.append(x)

        cmd = Command("test", my_func)
        ctx.invoke(cmd, x=42)
        assert called == [42]

    def test_invoke_callable(self):
        ctx = Context()
        called = []

        def my_func():
            called.append(True)

        ctx.invoke(my_func)
        assert called == [True]


# ── Decorator tests ─────────────────────────────────────────────────────

class TestDecorators:
    def test_option_decorator(self):
        @option("--name", help="Name", default="world")
        def greet():
            pass

        assert hasattr(greet, "_options")
        assert len(greet._options) == 1
        assert greet._options[0].dest == "name"

    def test_argument_decorator(self):
        @argument("prompt", required=True)
        def generate():
            pass

        assert hasattr(generate, "_arguments")
        assert len(generate._arguments) == 1
        assert generate._arguments[0].name == "prompt"

    def test_pass_context(self):
        @pass_context
        def my_cmd(ctx):
            pass

        # pass_context is a no-op, just returns the function
        assert callable(my_cmd)

    def test_version_option_decorator(self):
        @version_option(package_name="sloughgpt", prog_name="sloughgpt")
        def cli():
            pass

        assert hasattr(cli, "_version_option")
        assert cli._version_option is True

    def test_stacked_decorators(self):
        @option("--name", help="Name")
        @option("--count", type=int, default=1)
        @argument("prompt")
        def my_cmd():
            pass

        assert len(my_cmd._options) == 2
        assert len(my_cmd._arguments) == 1


# ── Global options parser ───────────────────────────────────────────────

class TestGlobalOptions:
    def test_parse_host(self):
        opts, remaining = _parse_global_options(["--host", "0.0.0.0", "model", "list"])
        assert opts["host"] == "0.0.0.0"
        assert remaining == ["model", "list"]

    def test_parse_port(self):
        opts, remaining = _parse_global_options(["--port", "9000", "model"])
        assert opts["port"] == 9000

    def test_parse_json_flag(self):
        opts, remaining = _parse_global_options(["--json", "model"])
        assert opts["json"] is True

    def test_parse_quiet_flag(self):
        opts, remaining = _parse_global_options(["-q", "model"])
        assert opts["quiet"] is True

    def test_parse_no_color(self):
        opts, remaining = _parse_global_options(["--no-color", "model"])
        assert opts["no_color"] is True

    def test_parse_version(self):
        opts, remaining = _parse_global_options(["--version"])
        assert opts["version"] is True

    def test_parse_help(self):
        opts, remaining = _parse_global_options(["--help"])
        assert opts["help"] is True

    def test_parse_yes(self):
        opts, remaining = _parse_global_options(["-y", "model"])
        assert opts["yes"] is True

    def test_parse_timeout(self):
        opts, remaining = _parse_global_options(["--timeout", "30", "model"])
        assert opts["timeout"] == 30

    def test_defaults(self):
        opts, remaining = _parse_global_options([])
        assert opts["host"] == "localhost"
        assert opts["port"] == 8000
        assert opts["timeout"] == 10

    def test_config_short(self):
        opts, remaining = _parse_global_options(["-c", "my.yaml", "model"])
        assert opts["config"] == "my.yaml"

    def test_config_long(self):
        opts, remaining = _parse_global_options(["--config", "my.yaml", "model"])
        assert opts["config"] == "my.yaml"


# ── Echo / Confirm tests ───────────────────────────────────────────────

class TestEcho:
    def test_echo(self, capsys):
        echo("hello")
        captured = capsys.readouterr()
        assert captured.out == "hello\n"

    def test_echo_no_nl(self, capsys):
        echo("hello", nl=False)
        captured = capsys.readouterr()
        assert captured.out == "hello"

    def test_echo_err(self, capsys):
        echo("error", err=True)
        captured = capsys.readouterr()
        assert captured.err == "error\n"

    def test_echo_empty(self, capsys):
        echo()
        captured = capsys.readouterr()
        assert captured.out == "\n"

    def test_echo_int(self, capsys):
        echo(42)
        captured = capsys.readouterr()
        assert captured.out == "42\n"


class TestConfirm:
    def test_confirm_yes(self):
        with patch("builtins.input", return_value="y"):
            assert confirm("Continue?") is True

    def test_confirm_no(self):
        with patch("builtins.input", return_value="n"):
            assert confirm("Continue?") is False

    def test_confirm_empty(self):
        with patch("builtins.input", return_value=""):
            assert confirm("Continue?") is False

    def test_confirm_abort_on_no(self):
        with patch("builtins.input", return_value="n"):
            with pytest.raises(SystemExit):
                confirm("Continue?", abort=True)

    def test_confirm_eof(self):
        with patch("builtins.input", side_effect=EOFError):
            assert confirm("Continue?") is False

    def test_confirm_abort_on_eof(self):
        with patch("builtins.input", side_effect=EOFError):
            with pytest.raises(SystemExit):
                confirm("Continue?", abort=True)


# ── Error handling tests ────────────────────────────────────────────────

class TestErrors:
    def test_usage_error(self):
        e = UsageError("bad option")
        assert str(e) == "bad option"
        assert e.message == "bad option"

    def test_bad_parameter(self):
        e = BadParameter("invalid value")
        assert isinstance(e, UsageError)


# ── End-to-end run tests ────────────────────────────────────────────────

class TestEndToEnd:
    def test_run_with_help(self, capsys):
        root = Group("root", invoke_without_command=True)

        @root.command("hello")
        def hello():
            echo("hello world")

        with patch("sys.argv", ["prog", "--help"]):
            run(root)

        captured = capsys.readouterr()
        assert "Commands:" in captured.out

    def test_run_command(self, capsys):
        root = Group("root")

        @root.command("hello")
        def hello():
            echo("hello world")

        with patch("sys.argv", ["prog", "hello"]):
            run(root)

        captured = capsys.readouterr()
        assert "hello world" in captured.out

    def test_run_subgroup(self, capsys):
        root = Group("root")

        @root.group("model")
        def model():
            pass

        @model.command("list")
        def model_list():
            echo("listing models")

        with patch("sys.argv", ["prog", "model", "list"]):
            run(root)

        captured = capsys.readouterr()
        assert "listing models" in captured.out

    def test_run_with_options(self, capsys):
        root = Group("root")

        @root.command("greet")
        @option("--name", default="world")
        def greet(name):
            echo(f"hello {name}")

        with patch("sys.argv", ["prog", "greet", "--name", "test"]):
            run(root)

        captured = capsys.readouterr()
        assert "hello test" in captured.out

    def test_run_with_arguments(self, capsys):
        root = Group("root")

        @root.command("echo")
        @argument("text")
        def echo_cmd(text):
            echo(text)

        with patch("sys.argv", ["prog", "echo", "hello"]):
            run(root)

        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_run_with_context(self, capsys):
        root = Group("root")

        @root.command("check")
        @pass_context
        def check(ctx):
            echo(f"host={ctx.obj.get('host', 'none')}")

        with patch("sys.argv", ["prog", "check"]):
            run(root)

        captured = capsys.readouterr()
        assert "host=localhost" in captured.out

    def test_run_global_option_passed_to_ctx(self, capsys):
        root = Group("root")

        @root.command("check")
        @pass_context
        def check(ctx):
            echo(f"host={ctx.obj.get('host')}")

        with patch("sys.argv", ["prog", "--host", "0.0.0.0", "check"]):
            run(root)

        captured = capsys.readouterr()
        assert "host=0.0.0.0" in captured.out

    def test_run_version(self, capsys):
        root = Group("root")

        @root.command("hello")
        def hello():
            echo("hello")

        with patch("sys.argv", ["prog", "--version"]):
            run(root)

        captured = capsys.readouterr()
        assert "v" in captured.out or "sloughgpt" in captured.out

    def test_run_no_command_shows_help(self, capsys):
        root = Group("root", invoke_without_command=True)

        @root.command("hello")
        def hello():
            echo("hello")

        with patch("sys.argv", ["prog"]):
            run(root)

        captured = capsys.readouterr()
        assert "Commands:" in captured.out

    def test_run_unknown_command(self, capsys):
        root = Group("root")

        @root.command("hello")
        def hello():
            echo("hello")

        with patch("sys.argv", ["prog", "xyz"]):
            run(root)

        captured = capsys.readouterr()
        assert "Unknown command" in captured.out

    def test_run_with_choice(self, capsys):
        root = Group("root")

        @root.command("fmt")
        @option("--format", choice=Choice(["json", "csv"]))
        def fmt(format):
            echo(f"format={format}")

        with patch("sys.argv", ["prog", "fmt", "--format", "json"]):
            run(root)

        captured = capsys.readouterr()
        assert "format=json" in captured.out

    def test_run_multiple_options(self, capsys):
        root = Group("root")

        @root.command("test")
        @option("--a", default="1")
        @option("--b", default="2")
        def test(a, b):
            echo(f"{a},{b}")

        with patch("sys.argv", ["prog", "test", "--a", "x", "--b", "y"]):
            run(root)

        captured = capsys.readouterr()
        assert "x,y" in captured.out

    def test_run_flag(self, capsys):
        root = Group("root")

        @root.command("test")
        @option("--verbose", is_flag=True)
        def test(verbose):
            echo(f"verbose={verbose}")

        with patch("sys.argv", ["prog", "test", "--verbose"]):
            run(root)

        captured = capsys.readouterr()
        assert "verbose=True" in captured.out

    def test_run_bool_flag_pair(self, capsys):
        root = Group("root")

        @root.command("test")
        @option("--tui/--no-tui", is_flag=True)
        def test(tui):
            echo(f"tui={tui}")

        with patch("sys.argv", ["prog", "test", "--tui"]):
            run(root)

        captured = capsys.readouterr()
        assert "tui=True" in captured.out

    def test_run_hidden_command_not_in_help(self, capsys):
        root = Group("root")

        @root.command("secret", hidden=True)
        def secret():
            echo("secret")

        @root.command("public")
        def public():
            echo("public")

        with patch("sys.argv", ["prog", "--help"]):
            run(root)

        captured = capsys.readouterr()
        assert "secret" not in captured.out
        assert "public" in captured.out

    def test_run_add_command(self, capsys):
        root = Group("root")

        def my_func():
            echo("added")

        cmd = Command("added", my_func)
        root.add_command(cmd, "added")

        with patch("sys.argv", ["prog", "added"]):
            run(root)

        captured = capsys.readouterr()
        assert "added" in captured.out


# ── Usage tracking tests ────────────────────────────────────────────────

class TestUsageTracking:
    def test_record_usage(self, tmp_path):
        from core import slo_cli
        usage_file = tmp_path / "usage.json"
        old_path = slo_cli._USAGE_PATH
        slo_cli._USAGE_PATH = usage_file
        try:
            slo_cli._record_usage("model")
            data = json.loads(usage_file.read_text())
            assert data["model"] == 1

            slo_cli._record_usage("model")
            data = json.loads(usage_file.read_text())
            assert data["model"] == 2
        finally:
            slo_cli._USAGE_PATH = old_path


# ── Module-level decorators test ────────────────────────────────────────

class TestModuleLevelDecorators:
    def test_module_group(self):
        @group(help="Test group")
        def mygroup():
            pass

        assert isinstance(mygroup, Group)
        assert mygroup.help == "Test group"

    def test_module_group_with_commands(self):
        @group()
        def root():
            pass

        @root.command("hello")
        def hello():
            pass

        assert "hello" in root.commands

    def test_module_command(self):
        @command("test", help="Test cmd")
        def test():
            pass

        assert isinstance(test, Command)
        assert test.name == "test"
        assert test.help == "Test cmd"
