"""Coverage for sloughgpt_sdk.cli."""
import builtins
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "sdk-py"))

import sloughgpt_sdk.cli as cli  # noqa: E402


class TestFormatJson:
    def test_indents_and_str_fallback(self):
        out = cli.format_json({"a": 1, "b": object()})
        assert "{\n" in out
        assert '"a": 1' in out


def _argv(main, *args):
    return ["sloughgpt-cli", "--url", "http://localhost:8000", *args]


def _run(args, client):
    mock_cls = Mock(return_value=client)
    with (
        patch("sloughgpt_sdk.cli.SloughGPTClient", mock_cls),
        patch.object(sys, "argv", args),
    ):
        return cli.main()


class TestMainHealth:
    def test_plain_output(self, capsys):
        client = Mock()
        client.health.return_value = SimpleNamespace(
            raw={"s": "ok"}, status="ok", version="1.0",
            model_loaded=True, model_name="gpt2", device="cpu",
        )
        assert _run(_argv(None, "health"), client) == 0
        out = capsys.readouterr().out
        assert "Status: ok" in out
        assert "Version: 1.0" in out
        assert "Model Loaded: True" in out
        assert "Model: gpt2" in out
        assert "Device: cpu" in out

    def test_json_output(self, capsys):
        client = Mock()
        client.health.return_value = SimpleNamespace(raw={"a": "b"}, status="ok", version="1.0", model_loaded=False, model_name=None, device="cpu")
        _run(_argv(None, "--json", "health"), client)
        out = capsys.readouterr().out
        assert '"a": "b"' in out


class TestMainInfo:
    def test_plain_output(self, capsys):
        client = Mock()
        client.info.return_value = SimpleNamespace(
            raw={}, version="1.0", pytorch_version="2.0",
            cuda_available=True, cuda={"device": "gpu0"}, cpu_count=8,
        )
        _run(_argv(None, "info"), client)
        out = capsys.readouterr().out
        assert "Version: 1.0" in out
        assert "GPU: gpu0" in out
        assert "CPU Cores: 8" in out

    def test_json_and_no_cuda(self, capsys):
        client = Mock()
        client.info.return_value = SimpleNamespace(version="1.0", pytorch_version="2.0", cuda_available=False, cuda=None, cpu_count=4, raw={"version": "1.0"})
        assert _run(_argv(None, "--json", "info"), client) == 0
        out = capsys.readouterr().out
        assert '"version"' in out


class TestMainGenerate:
    def test_plain(self, capsys):
        client = Mock()
        client.generate.return_value = SimpleNamespace(
            generated_text="hi", raw_response={"t": "hi"},
            tokens_generated=2, inference_time_ms=10.5,
        )
        _run(_argv(None, "generate", "hello"), client)
        out = capsys.readouterr().out
        assert out.strip() == "hi"

    def test_json_and_verbose(self, capsys):
        client = Mock()
        client.generate.return_value = SimpleNamespace(
            generated_text="hi", raw_response={"t": "hi"},
            tokens_generated=2, inference_time_ms=10.5,
        )
        assert _run(_argv(None, "--json", "--verbose", "generate", "hello"), client) == 0
        out = capsys.readouterr().out
        assert '"t": "hi"' in out
        assert "Tokens: 2" in out
        assert "Time: 10.50ms" in out

    def test_stream(self, capsys):
        client = Mock()
        client.generate_stream.return_value = ["a", "b"]
        assert _run(_argv(None, "generate", "hello", "--stream"), client) == 0
        out = capsys.readouterr().out
        assert "ab\n" in out


class TestMainChat:
    def test_without_system(self, capsys):
        client = Mock()
        client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content="reply"), raw_response={"r": 1})
        _run(_argv(None, "chat", "hi"), client)
        out = capsys.readouterr().out
        assert out.strip() == "reply"

    def test_with_system_and_json(self, capsys):
        client = Mock()
        client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content="reply"), raw_response={"r": 1})
        assert _run(_argv(None, "--json", "chat", "hi", "--system", "you are a bot"), client) == 0
        out = capsys.readouterr().out
        assert '"r": 1' in out
        body = client.chat.call_args.args[0]
        assert len(body) == 2
        assert body[0].content == "you are a bot"


class TestMainModels:
    def test_plain(self, capsys):
        client = Mock()
        client.list_models.return_value = [
            SimpleNamespace(id="m1", source="local", description="d1", raw={}),
            SimpleNamespace(id="m2", source=None, description=None, raw={}),
        ]
        assert _run(_argv(None, "models"), client) == 0
        out = capsys.readouterr().out
        assert "m1" in out and "Source: local" in out and "d1" in out
        assert "m2" in out

    def test_json(self, capsys):
        client = Mock()
        client.list_models.return_value = [SimpleNamespace(raw={"id": "m1"}, id="m1", source=None, description=None)]
        _run(_argv(None, "--json", "models"), client)
        out = capsys.readouterr().out
        assert '"id": "m1"' in out


class TestMainDatasets:
    def test_plain(self, capsys):
        client = Mock()
        client.list_datasets.return_value = [
            SimpleNamespace(id="ds1", description="set one", raw={}),
            SimpleNamespace(id="ds2", description=None, raw={}),
        ]
        _run(_argv(None, "datasets"), client)
        out = capsys.readouterr().out
        assert "ds1" in out and "set one" in out

    def test_json(self, capsys):
        client = Mock()
        client.list_datasets.return_value = [SimpleNamespace(raw={"id": "ds1"}, id="ds1", description=None)]
        assert _run(_argv(None, "--json", "datasets"), client) == 0
        assert '"id": "ds1"' in capsys.readouterr().out


class TestMainMetrics:
    def test_plain(self, capsys):
        client = Mock()
        client.metrics.return_value = SimpleNamespace(
            raw={}, requests_total=10, requests_success=8,
            requests_failed=2, cache_hits=3, cache_misses=7, avg_response_time_ms=1.5,
        )
        _run(_argv(None, "metrics"), client)
        out = capsys.readouterr().out
        assert "Total Requests: 10" in out
        assert "Cache Hits: 3" in out
        assert "Avg Response Time: 1.50ms" in out

    def test_prometheus(self, capsys):
        client = Mock()
        client.metrics_prometheus.return_value = "# HELP xyz"
        assert _run(_argv(None, "metrics", "--prometheus"), client) == 0
        assert "# HELP xyz" in capsys.readouterr().out

    def test_json(self, capsys):
        client = Mock()
        client.metrics.return_value = SimpleNamespace(
            raw={"r": "x"}, requests_total=1, requests_success=1, requests_failed=0,
            cache_hits=0, cache_misses=0, avg_response_time_ms=1.0,
        )
        _run(_argv(None, "--json", "metrics"), client)
        assert '"r": "x"' in capsys.readouterr().out


class TestMainRegistry:
    def test_list_plain(self, capsys):
        client = Mock()
        client.list_registry_models.return_value = [
            {"model_id": "m1", "status": "loaded", "model_type": "gpt2"},
        ]
        _run(_argv(None, "registry", "list"), client)
        out = capsys.readouterr().out
        assert "m1" in out
        assert "Status: loaded" in out
        assert "Type: gpt2" in out

    def test_list_json(self, capsys):
        client = Mock()
        client.list_registry_models.return_value = [{"model_id": "m1", "status": "s", "model_type": "t"}]
        _run(_argv(None, "--json", "registry", "list"), client)
        assert '"model_id": "m1"' in capsys.readouterr().out

    def test_list_empty(self, capsys):
        client = Mock()
        client.list_registry_models.return_value = []
        _run(_argv(None, "registry", "list"), client)
        assert "No models found." in capsys.readouterr().out

    def test_info_found(self, capsys):
        client = Mock()
        client.get_registry_model.return_value = {"model_id": "m1", "status": "loaded"}
        _run(_argv(None, "registry", "info", "m1"), client)
        out = capsys.readouterr().out
        assert "m1" in out and "status: loaded" in out

    def test_info_not_found(self, capsys):
        client = Mock()
        client.get_registry_model.return_value = None
        assert _run(_argv(None, "registry", "info", "ghost"), client) == 0
        assert "Model not found: ghost" in capsys.readouterr().out

    def test_info_json(self, capsys):
        client = Mock()
        client.get_registry_model.return_value = {"model_id": "m1"}
        _run(_argv(None, "--json", "registry", "info", "m1"), client)
        assert '"model_id": "m1"' in capsys.readouterr().out

    def test_best(self, capsys):
        client = Mock()
        client.get_registry_best.return_value = {"model_id": "best"}
        _run(_argv(None, "--json", "registry", "best"), client)
        assert '"model_id": "best"' in capsys.readouterr().out

    def test_best_plain(self, capsys):
        client = Mock()
        client.get_registry_best.return_value = "m1"
        assert _run(_argv(None, "registry", "best"), client) == 0
        assert "m1" in capsys.readouterr().out

    def test_stats(self, capsys):
        client = Mock()
        client.get_registry_stats.return_value = {"count": 2}
        assert _run(_argv(None, "registry", "stats"), client) == 0
        assert "2" in capsys.readouterr().out

    def test_no_action_prints_registry_help(self, capsys):
        _run(_argv(None, "registry"), Mock())
        out = capsys.readouterr().out
        assert "list" in out


class TestMainEdgeCases:
    def test_no_command_prints_help(self, capsys):
        client = Mock()
        assert _run(_argv(None), client) == 0
        out = capsys.readouterr().out
        assert "usage" in out.lower()

    def test_keyboard_interrupt(self, capsys):
        client = Mock()
        client.health.side_effect = KeyboardInterrupt
        assert _run(_argv(None, "health"), client) == 130
        assert "Interrupted" in capsys.readouterr().out

    def test_generic_error_non_verbose(self, capsys):
        client = Mock()
        client.health.side_effect = ValueError("bad url")
        assert _run(_argv(None, "health"), client) == 1
        assert "Error: bad url" in capsys.readouterr().err

    def test_generic_error_verbose(self, capsys):
        client = Mock()
        client.health.side_effect = ValueError("bad url")
        assert _run(_argv(None, "--verbose", "health"), client) == 1
        err = capsys.readouterr().err
        assert "Error: bad url" in err
        assert "Traceback" in err


class TestMainEntry:
    def _exec_entry(self, ns):
        src = Path(cli.__file__).read_text()
        exec(compile(src, str(cli.__file__), "exec"), ns)

    def test_import_failure_exits(self, capsys):
        codes = []
        original = builtins.__import__

        def blocking_import(name, *args, **kwargs):
            if name == "sloughgpt_sdk":
                raise ImportError("blocked")
            return original(name, *args, **kwargs)

        def fake_exit(code):
            codes.append(code)
            raise SystemExit(code)

        ns = {"__name__": "__main__", "__file__": str(cli.__file__)}
        with (
            patch("builtins.__import__", new=blocking_import),
            patch.object(sys, "exit", fake_exit),
        ):
            with pytest.raises(SystemExit) as excinfo:
                self._exec_entry(ns)
        assert excinfo.value.code == 1
        assert "not found" in capsys.readouterr().out

    def test_main_dispatches_health(self):
        import sloughgpt_sdk

        codes = []
        fake_client = Mock()
        fake_client.health.return_value = SimpleNamespace(
            status="ok",
            version="0.1",
            model_loaded=True,
            model_name="gpt2",
            device="cpu",
            raw={"s": "ok"},
        )
        ns = {"__name__": "__main__", "__file__": str(cli.__file__)}
        with (
            patch.object(sloughgpt_sdk, "SloughGPTClient", Mock(return_value=fake_client)),
            patch.object(sys, "argv", ["sloughgpt-cli", "--url", "http://x", "health"]),
            patch.object(sys, "exit", codes.append),
        ):
            self._exec_entry(ns)
        assert codes == [0]