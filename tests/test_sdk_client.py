"""Coverage for sloughgpt_sdk.client."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import requests

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "sdk-py"))

from sloughgpt_sdk import client as client_module  # noqa: E402
from sloughgpt_sdk.client import (  # noqa: E402
    AsyncSloughGPTClient,
    SimpleTracker,
    SloughGPTClient,
    _build_training_start_payload,
    _coerce_training_jobs_list,
    _unwrap_response,
)
from sloughgpt_sdk.exceptions import SloughGPTError  # noqa: E402
from sloughgpt_sdk.models import ChatMessage, ChatRequest  # noqa: E402


def _resp(data=None, text="", status=200):
    r = requests.Response()
    r.status_code = status
    if data is not None:
        r._content = json.dumps(data).encode()
    else:
        r._content = text.encode()
    return r


def _mock_client(data=None, text="", status=200):
    client = SloughGPTClient(base_url="http://localhost:8000/")
    session = Mock()
    session.request.return_value = _resp(data, text, status)
    client._session = session
    return client, session


class TestHelpers:
    def test_unwrap_response(self):
        assert _unwrap_response({"status": "success", "data": [1, 2]}) == [1, 2]
        assert _unwrap_response({"status": "ok", "data": [1]}) == {"status": "ok", "data": [1]}
        assert _unwrap_response([1, 2]) == [1, 2]

    def test_build_training_start_payload_defaults(self):
        p = _build_training_start_payload("m1", "ds1")
        assert p["name"] == "m1-training"
        assert p["model"] == "m1"
        assert p["dataset"] == "ds1"
        assert p["epochs"] == 3
        assert p["batch_size"] == 8
        assert p["learning_rate"] == 5e-5

    def test_build_training_start_payload_manifest(self):
        p = _build_training_start_payload("m1", "ds1", manifest_uri="s3://x")
        assert p["manifest_uri"] == "s3://x"
        assert "dataset" not in p

    def test_build_training_start_payload_dataset_ref(self):
        p = _build_training_start_payload("m1", "ds1", dataset_ref="ref-1")
        assert p["dataset_ref"] == "ref-1"
        assert "dataset" not in p

    def test_build_training_start_payload_kwargs_and_name(self):
        p = _build_training_start_payload("m1", "ds1", name="custom", extra=9)
        assert p["name"] == "custom"
        assert p["extra"] == 9

    def test_coerce_training_jobs_list(self):
        assert _coerce_training_jobs_list([{"id": 1}]) == [{"id": 1}]
        assert _coerce_training_jobs_list({"jobs": [{"id": 1}]}) == [{"id": 1}]
        assert _coerce_training_jobs_list({"jobs": "nope"}) == []
        assert _coerce_training_jobs_list("nope") == []


class TestInit:
    def test_trailing_slash_stripped(self):
        c = SloughGPTClient(base_url="http://x:1/")
        assert c.base_url == "http://x:1"

    def test_api_key_header(self):
        c = SloughGPTClient(base_url="http://x:1", api_key="k")
        assert c._headers["X-API-Key"] == "k"

    def test_custom_headers(self):
        c = SloughGPTClient(base_url="http://x:1", headers={"X-A": "1"})
        assert c._headers["X-A"] == "1"

    def test_request_defaults(self):
        c, session = _mock_client(data={"ok": True})
        c._request("GET", "/x")
        kwargs = session.request.call_args
        assert kwargs.args[:2] == ("GET", "http://localhost:8000/x")
        assert kwargs.kwargs["timeout"] == 30
        assert kwargs.kwargs["verify"] is True

    def test_request_raises_on_http_error(self):
        c = SloughGPTClient()
        c._session = Mock()
        c._session.request.return_value = _resp(data={}, status=404)
        with pytest.raises(requests.HTTPError):
            c._request("GET", "/missing")


class TestHealthAndInfo:
    def test_health(self):
        c, _ = _mock_client(data={"status": "ok", "version": "1", "model_loaded": True})
        h = c.health()
        assert h.status == "ok"
        assert h.is_healthy
        assert h.model_loaded is True
        assert h.version == "1"

    def test_liveness_readiness_detailed(self):
        c, _ = _mock_client(data={"alive": True})
        assert c.liveness() == {"alive": True}
        c, _ = _mock_client(data={"ready": True})
        assert c.readiness() == {"ready": True}
        c, _ = _mock_client(data={"cpu": 1})
        assert c.detailed_health() == {"cpu": 1}

    def test_info(self):
        c, _ = _mock_client(data={"version": "1.0", "cuda_available": False, "cpu_count": 4})
        info = c.info()
        assert info.version == "1.0"
        assert info.cpu_count == 4
        assert info.cuda_available is False


class TestGeneration:
    def test_generate(self):
        c, session = _mock_client(
            data={"generated_text": "hi", "model": "gpt2", "tokens_generated": 2}
        )
        result = c.generate("hello", max_new_tokens=5)
        body = session.request.call_args.kwargs["json"]
        assert body["prompt"] == "hello"
        assert body["max_new_tokens"] == 5
        assert result.generated_text == "hi"
        assert result.model == "gpt2"
        assert result.prompt == "hello"

    def test_generate_fills_inference_time(self):
        c, _ = _mock_client(data={"text": "out"})
        result = c.generate("p")
        assert result.inference_time_ms is not None
        assert result.inference_time_ms >= 0

    def test_generate_keeps_server_inference_time(self):
        c, _ = _mock_client(data={"text": "out", "inference_time_ms": 12.5})
        result = c.generate("p")
        assert result.inference_time_ms == 12.5

    def test_generate_stream(self):
        c = SloughGPTClient()
        resp = Mock()
        resp.raise_for_status.return_value = None
        resp.iter_lines.return_value = [
            "data: one",
            "data: two",
            "data: [DONE]",
            "ignored",
        ]
        c._session = Mock()
        c._session.request.return_value = resp
        tokens = list(c.generate_stream("p"))
        assert tokens == ["one", "two"]

    def test_generate_stream_blank_data_line(self):
        c = SloughGPTClient()
        resp = Mock()
        resp.raise_for_status.return_value = None
        resp.iter_lines.return_value = ["data:", "data: tok"]
        c._session = Mock()
        c._session.request.return_value = resp
        assert list(c.generate_stream("p")) == ["tok"]

    def test_generate_stream_recovers_json_decode_error(self):
        c = SloughGPTClient()
        resp = Mock()
        resp.raise_for_status.return_value = None
        resp.iter_lines.return_value = ["data: hello", "data: [DONE]"]
        c._session = Mock()
        c._session.request.return_value = resp
        gen = c.generate_stream("p")
        assert next(gen) == "hello"
        assert gen.throw(json.JSONDecodeError("x", "doc", 0)) == "hello"
        with pytest.raises(StopIteration):
            next(gen)


class TestChat:
    def test_chat_with_message_objects(self):
        c, session = _mock_client(data={"text": "hi there"})
        result = c.chat([client_module.ChatMessage.user("hey")])
        body = session.request.call_args.kwargs["json"]
        assert body["messages"] == [{"role": "user", "content": "hey"}]
        assert result.message.content == "hi there"

    def test_chat_with_dicts(self):
        c, session = _mock_client(data={"text": "yo"})
        result = c.chat([{"role": "user", "content": "hello"}, {"content": "no role"}])
        body = session.request.call_args.kwargs["json"]
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][1]["role"] == "user"
        assert result.message.content == "yo"

    def test_chat_choices_format(self):
        c, _ = _mock_client(
            data={"choices": [{"message": {"content": "from choices"}}], "model": "gpt2"}
        )
        result = c.chat([ChatMessage.user("q")])
        assert result.message.content == "from choices"
        assert result.model == "gpt2"

    def test_chat_error_raises(self):
        c, _ = _mock_client(data={"error": "boom", "text": ""})
        with pytest.raises(SloughGPTError, match="boom"):
            c.chat([ChatMessage.user("q")])

    def test_chat_error_ignored_when_text_present(self):
        c, _ = _mock_client(data={"error": "boom", "text": "still here"})
        result = c.chat([ChatMessage.user("q")])
        assert result.message.content == "still here"

    def test_chat_stream_tokens(self):
        c = SloughGPTClient()
        resp = Mock()
        resp.raise_for_status.return_value = None
        resp.iter_lines.return_value = [
            "data: " + json.dumps({"token": "a"}),
            "data: " + json.dumps({"token": "b"}),
            "data: " + json.dumps({}),
            "",
            "data: " + json.dumps({"error": "stop"}),
            "data: " + json.dumps({"token": "c"}),
        ]
        c._session = Mock()
        c._session.request.return_value = resp
        assert list(c.chat_stream([ChatMessage.user("q")])) == ["a", "b"]

    def test_chat_stream_skips_bad_json(self):
        c = SloughGPTClient()
        resp = Mock()
        resp.raise_for_status.return_value = None
        resp.iter_lines.return_value = ["data: not-json", "data: " + json.dumps({"token": "x"})]
        c._session = Mock()
        c._session.request.return_value = resp
        assert list(c.chat_stream([{"role": "user", "content": "q"}])) == ["x"]

    def test_chat_stream_with_instance_messages(self):
        c = SloughGPTClient()
        resp = Mock()
        resp.raise_for_status.return_value = None
        resp.iter_lines.return_value = ["data: " + json.dumps({"token": "z"})]
        c._session = Mock()
        c._session.request.return_value = resp
        assert list(c.chat_stream([client_module.ChatMessage.user("q")])) == ["z"]
        body = c._session.request.call_args.kwargs["json"]
        assert body["messages"] == [{"role": "user", "content": "q"}]


class TestModels:
    def test_list_models(self):
        c, _ = _mock_client(data={"models": [{"id": "gpt2", "name": "GPT-2"}]})
        models = c.list_models()
        assert models[0].id == "gpt2"
        assert models[0].name == "GPT-2"

    def test_list_models_bare_list(self):
        c, _ = _mock_client(data=[{"id": "a"}])
        assert c.list_models()[0].id == "a"

    def test_load_unload_current(self):
        c, session = _mock_client(data={"status": "ok"})
        c.load_model("gpt2")
        assert session.request.call_args.kwargs["json"] == {"model_id": "gpt2"}
        c.unload_model()
        c.get_current_model()

    def test_list_hf_models(self):
        c, session = _mock_client(data={"models": [{"id": "x"}]})
        result = c.list_hf_models()
        assert session.request.call_args.kwargs["params"] == {"limit": 10}
        assert result == [{"id": "x"}]

    def test_list_hf_models_with_query(self):
        c, session = _mock_client(data={"models": []})
        c.list_hf_models(query="llama", limit=5)
        assert session.request.call_args.kwargs["params"] == {"limit": 5, "q": "llama"}


class TestSessions:
    def test_session_crud(self):
        c, _ = _mock_client(data={"id": "s1"})
        assert c.create_session() == {"id": "s1"}
        c, session = _mock_client(data={"sessions": [{"id": "s1"}]})
        assert c.list_sessions() == [{"id": "s1"}]
        assert session.request.call_args.args[1] == "http://localhost:8000/chat/sessions"
        c, _ = _mock_client(data={"id": "s1"})
        assert c.get_session("s1") == {"id": "s1"}
        c, _ = _mock_client(data={"ok": True})
        assert c.delete_session("s1") == {"ok": True}

    def test_session_context(self):
        c, session = _mock_client(data={"ok": True})
        c.save_session_context("s1", {"messages": []})
        body = session.request.call_args.kwargs["json"]
        assert body == {"messages": []}
        assert "/session/s1/context" in session.request.call_args.args[1]

    def test_get_session_messages(self):
        c, _ = _mock_client(data={"messages": [{"role": "user"}]})
        assert c.get_session_messages("s1") == [{"role": "user"}]
        c, _ = _mock_client(data=[{"role": "user"}])
        assert c.get_session_messages("s1") == [{"role": "user"}]


class TestSouls:
    def test_list_souls(self):
        c, _ = _mock_client(data={"souls": [{"name": "warm"}]})
        assert c.list_souls() == [{"name": "warm"}]
        c, _ = _mock_client(data=[{"name": "warm"}])
        assert c.list_souls() == [{"name": "warm"}]

    def test_get_current_soul(self):
        c, _ = _mock_client(data={"name": "warm"})
        assert c.get_current_soul() == {"name": "warm"}

    def test_switch_soul(self):
        c, session = _mock_client(data={"ok": True})
        c.switch_soul("warm")
        assert session.request.call_args.kwargs["json"] == {}
        c, session = _mock_client(data={"ok": True})
        c.switch_soul("warm", checkpoint_name="cp1")
        assert session.request.call_args.kwargs["json"] == {"checkpoint_name": "cp1"}


class TestKnowledge:
    def test_list_knowledge(self):
        c, _ = _mock_client(data={"items": [{"content": "x"}]})
        assert c.list_knowledge() == [{"content": "x"}]
        c, _ = _mock_client(data=[{"content": "x"}])
        assert c.list_knowledge() == [{"content": "x"}]

    def test_add_knowledge(self):
        c, session = _mock_client(data={"id": "1"})
        c.add_knowledge("hi")
        assert session.request.call_args.kwargs["json"] == {"content": "hi"}
        c, session = _mock_client(data={"id": "1"})
        c.add_knowledge("hi", topic="t")
        assert session.request.call_args.kwargs["json"] == {"content": "hi", "topic": "t"}

    def test_delete_search_stats_topics_ingest(self):
        c, _ = _mock_client(data={"ok": True})
        assert c.delete_knowledge("1") == {"ok": True}
        c, session = _mock_client(data={"results": [{"content": "x"}]})
        assert c.search_knowledge("q") == [{"content": "x"}]
        assert session.request.call_args.kwargs["params"] == {"q": "q"}
        c, _ = _mock_client(data={"count": 1})
        assert c.get_knowledge_stats() == {"count": 1}
        c, _ = _mock_client(data={"topics": ["a"]})
        assert c.get_knowledge_topics() == ["a"]
        c, session = _mock_client(data={"ok": True})
        c.ingest_knowledge_url("http://e")
        assert session.request.call_args.kwargs["json"] == {"url": "http://e"}


class TestTokenizer:
    def test_stats_and_tokenize(self):
        c, _ = _mock_client(data={"vocab": 10})
        assert c.get_tokenizer_stats() == {"vocab": 10}
        c, session = _mock_client(data={"tokens": [1]})
        assert c.tokenize("hi") == {"tokens": [1]}
        assert session.request.call_args.kwargs["json"] == {"text": "hi"}

    def test_train_tokenizer(self):
        c, session = _mock_client(data={"ok": True})
        c.train_tokenizer("abc")
        assert session.request.call_args.kwargs["json"] == {"text": "abc"}
        c, session = _mock_client(data={"ok": True})
        c.train_tokenizer("abc", vocab_size=100)
        assert session.request.call_args.kwargs["json"] == {"text": "abc", "vocab_size": 100}


class TestSystemAndCompanion:
    def test_system(self):
        c, _ = _mock_client(data={"cpu": 1})
        assert c.get_system_metrics() == {"cpu": 1}
        c, _ = _mock_client(data={"os": "linux"})
        assert c.get_system_info() == {"os": "linux"}
        c, _ = _mock_client(data={"disk": 2})
        assert c.get_system_disk() == {"disk": 2}

    def test_companion(self):
        c, _ = _mock_client(data={"personalities": ["warm"]})
        assert c.get_personalities() == ["warm"]
        c, session = _mock_client(data={"ok": True})
        c.set_personality("warm")
        assert session.request.call_args.kwargs["json"] == {"personality": "warm"}
        c, _ = _mock_client(data={"prompt": "p"})
        assert c.get_companion_prompt() == {"prompt": "p"}
        c, _ = _mock_client(data={"presets": [{"name": "warm"}]})
        assert c.list_companion_presets() == [{"name": "warm"}]
        c, _ = _mock_client(data=[{"name": "warm"}])
        assert c.list_companion_presets() == [{"name": "warm"}]


class TestDatasets:
    def test_list_datasets(self):
        c, _ = _mock_client(data={"datasets": [{"id": "d1"}]})
        assert c.list_datasets()[0].id == "d1"
        c, _ = _mock_client(data=[{"id": "d1"}])
        assert c.list_datasets()[0].id == "d1"

    def test_get_dataset(self):
        c, _ = _mock_client(data={"id": "d1", "name": "N"})
        d = c.get_dataset("d1")
        assert d.id == "d1"
        assert d.name == "N"

    def test_dataset_stats(self):
        c, _ = _mock_client(data={"rows": 3})
        assert c.get_dataset_stats("d1") == {"rows": 3}

    def test_imports(self):
        c, session = _mock_client(data={"ok": True})
        c.import_dataset_local("/p")
        assert session.request.call_args.kwargs["json"] == {"path": "/p"}
        c, session = _mock_client(data={"ok": True})
        c.import_dataset_local("/p", name="n")
        assert session.request.call_args.kwargs["json"] == {"path": "/p", "name": "n"}
        c, session = _mock_client(data={"ok": True})
        c.import_dataset_github("u/r")
        assert session.request.call_args.kwargs["json"] == {"repo": "u/r"}
        c, session = _mock_client(data={"ok": True})
        c.import_dataset_github("u/r", name="n")
        assert session.request.call_args.kwargs["json"] == {"repo": "u/r", "name": "n"}
        c, session = _mock_client(data={"ok": True})
        c.import_dataset_url("http://u")
        assert session.request.call_args.kwargs["json"] == {"url": "http://u"}
        c, session = _mock_client(data={"ok": True})
        c.import_dataset_url("http://u", name="n")
        assert session.request.call_args.kwargs["json"] == {"url": "http://u", "name": "n"}


class TestMetrics:
    def test_metrics(self):
        c, _ = _mock_client(data={"requests_total": 5, "requests_success": 4})
        m = c.metrics()
        assert m.requests_total == 5
        assert m.requests_success == 4

    def test_metrics_prometheus(self):
        c, _ = _mock_client(text="# HELP ...")
        assert c.metrics_prometheus() == "# HELP ..."


class TestTraining:
    def test_start_training(self):
        c, session = _mock_client(data={"job_id": "j1"})
        assert c.start_training("m", "ds", epochs=2) == {"job_id": "j1"}
        body = session.request.call_args.kwargs["json"]
        assert body["model"] == "m"
        assert body["epochs"] == 2

    def test_get_and_list_jobs(self):
        c, _ = _mock_client(data={"status": "running"})
        assert c.get_training_status("j1") == {"status": "running"}
        c, _ = _mock_client(data={"jobs": [{"id": "j1"}]})
        assert c.list_training_jobs() == [{"id": "j1"}]
        c, _ = _mock_client(data=[{"id": "j1"}])
        assert c.list_training_jobs() == [{"id": "j1"}]
        c, _ = _mock_client(data={})
        assert c.list_training_jobs() == []

    def test_delete_and_control(self):
        c, _ = _mock_client(data={"ok": True})
        assert c.delete_training_job("j1") == {"ok": True}
        c, _ = _mock_client(data={"ok": True})
        assert c.stop_training() == {"ok": True}
        c, _ = _mock_client(data={"ok": True})
        assert c.pause_training() == {"ok": True}
        c, _ = _mock_client(data={"ok": True})
        assert c.resume_training() == {"ok": True}

    def test_recovery(self):
        c, _ = _mock_client(data={"recoverable": 1})
        assert c.get_training_recovery_stats() == {"recoverable": 1}
        c, _ = _mock_client(data={"ok": True})
        assert c.abandon_recovery("j1") == {"ok": True}


class TestAutoTrain:
    def test_start_stop_status(self):
        c, session = _mock_client(data={"ok": True})
        c.start_auto_train({"epochs": 1})
        assert session.request.call_args.kwargs["json"] == {"epochs": 1}
        c, _ = _mock_client(data={"ok": True})
        assert c.stop_auto_train() == {"ok": True}
        c, _ = _mock_client(data={"phase": "TRAIN"})
        assert c.get_auto_train_status() == {"phase": "TRAIN"}

    def test_checkpoints(self):
        c, _ = _mock_client(data={"checkpoints": [{"name": "c"}]})
        assert c.list_auto_train_checkpoints() == [{"name": "c"}]
        c, _ = _mock_client(data=[{"name": "c"}])
        assert c.list_auto_train_checkpoints() == [{"name": "c"}]
        c, _ = _mock_client(data={"ok": True})
        assert c.delete_auto_train_checkpoint("c") == {"ok": True}
        c, _ = _mock_client(data={"ok": True})
        assert c.load_auto_train_checkpoint("c") == {"ok": True}


class TestFeedbackWorkflow:
    def test_record_feedback(self):
        c, session = _mock_client(data={"ok": True})
        c.record_feedback("s", "m", 1)
        assert session.request.call_args.kwargs["json"] == {
            "session_id": "s", "message_id": "m", "score": 1
        }
        c, session = _mock_client(data={"ok": True})
        c.record_feedback("s", "m", 1, tags=["a"])
        assert session.request.call_args.kwargs["json"]["tags"] == ["a"]

    def test_feedback_and_workflow_status(self):
        c, _ = _mock_client(data={"count": 1})
        assert c.get_feedback_stats() == {"count": 1}
        c, _ = _mock_client(data={"healthy": True})
        assert c.get_workflow_status() == {"healthy": True}


class TestExperiments:
    def test_crud(self):
        c, session = _mock_client(data={"id": "e1"})
        assert c.create_experiment("n") == {"id": "e1"}
        assert session.request.call_args.kwargs["json"] == {"name": "n", "description": ""}
        c, _ = _mock_client(data={"experiments": [{"id": "e1"}]})
        assert c.list_experiments() == [{"id": "e1"}]
        c, _ = _mock_client(data={"id": "e1"})
        assert c.get_experiment("e1") == {"id": "e1"}

    def test_log_metric_param(self):
        c, session = _mock_client(data={"ok": True})
        c.log_metric("e1", "loss", 0.1)
        assert session.request.call_args.kwargs["json"] == {"metric": "loss", "value": 0.1}
        c, session = _mock_client(data={"ok": True})
        c.log_metric("e1", "loss", 0.1, step=2)
        assert session.request.call_args.kwargs["json"] == {"metric": "loss", "value": 0.1, "step": 2}
        c, session = _mock_client(data={"ok": True})
        c.log_param("e1", "lr", 1e-3)
        assert session.request.call_args.kwargs["json"] == {"param": "lr", "value": 1e-3}


class TestRateLimitSecurityRegistry:
    def test_rate_limit(self):
        c, _ = _mock_client(data={"limited": False})
        assert c.get_rate_limit_status() == {"limited": False}
        c, _ = _mock_client(data={"allowed": True})
        assert c.check_rate_limit() == {"allowed": True}

    def test_audit_log(self):
        c, _ = _mock_client(data={"status": "success", "data": [{"event": "x"}]})
        assert c.get_audit_log() == [{"event": "x"}]
        c, _ = _mock_client(data={"logs": [{"event": "x"}]})
        assert c.get_audit_log() == [{"event": "x"}]
        c, _ = _mock_client(data={"status": "success", "data": {"keys": [{"id": "1"}]}})
        assert c.get_security_keys() == [{"id": "1"}]
        c, _ = _mock_client(data={"keys": [{"id": "1"}]})
        assert c.get_security_keys() == [{"id": "1"}]

    def test_registry(self):
        client_module.ChatMessage  # ensure client module imported
        c, _ = _mock_client(data={"status": "success", "data": {"models": [{"id": "m"}]}})
        assert c.list_registry_models() == [{"id": "m"}]
        c, _ = _mock_client(data={"status": "success", "data": {"id": "m"}})
        assert c.get_registry_model("m") == {"id": "m"}
        c, _ = _mock_client(data={"status": "success", "data": {"id": "best"}})
        assert c.get_registry_best() == {"id": "best"}
        c, _ = _mock_client(data={"status": "success", "data": {"count": 2}})
        assert c.get_registry_stats() == {"count": 2}


class TestBenchmark:
    def test_run_benchmark(self):
        c, session = _mock_client(data={"ok": True})
        c.run_benchmark({"name": "b"})
        assert session.request.call_args.kwargs["json"] == {"name": "b"}

    def test_benchmark_metrics_stats(self):
        c, _ = _mock_client(data=[{"name": "b"}])
        assert c.get_benchmark_metrics() == [{"name": "b"}]
        c, _ = _mock_client(data={"metrics": [{"name": "b"}]})
        assert c.get_benchmark_metrics() == [{"name": "b"}]
        c, _ = _mock_client(data={"count": 1})
        assert c.get_benchmark_stats() == {"count": 1}


class TestConvenience:
    def test_context_manager(self):
        c = SloughGPTClient()
        c._session = Mock()
        with c as client:
            assert client is c
        c._session.close.assert_called_once()

    def test_quick_generate(self):
        c, _ = _mock_client(data={"generated_text": "out"})
        assert c.quick_generate("p") == "out"

    def test_quick_chat(self):
        c, _ = _mock_client(data={"text": "reply"})
        assert c.quick_chat("hi") == "reply"


class TestSimpleTracker:
    def test_tracker(self):
        c = SloughGPTClient()
        tracker = SimpleTracker(c, "bench")
        with tracker as t:
            t.log("loss", 0.1)
            t.next_step()
            assert t._step == 1
            t.finish()
        assert tracker._name == "bench"


class TestAsyncClient:
    @pytest.mark.asyncio
    async def test_init_headers(self):
        client = AsyncSloughGPTClient(base_url="http://x:1/", api_key="k", headers={"X-A": "1"})
        assert client.base_url == "http://x:1"
        assert client._headers["X-API-Key"] == "k"
        assert client._headers["X-A"] == "1"

    @pytest.mark.asyncio
    async def test_health(self):
        client = AsyncSloughGPTClient()
        client._request = AsyncMock(return_value={"status": "ok"})
        h = await client.health()
        assert h.is_healthy

    @pytest.mark.asyncio
    async def test_generate(self):
        client = AsyncSloughGPTClient()
        client._request = AsyncMock(return_value={"generated_text": "hi", "model": "m"})
        result = await client.generate("p")
        assert result.generated_text == "hi"

    @pytest.mark.asyncio
    async def test_chat(self):
        client = AsyncSloughGPTClient()
        client._request = AsyncMock(return_value={"text": "reply"})
        result = await client.chat([ChatMessage.user("q")])
        assert result.message.content == "reply"

    @pytest.mark.asyncio
    async def test_chat_error(self):
        client = AsyncSloughGPTClient()
        client._request = AsyncMock(return_value={"error": "boom", "text": ""})
        with pytest.raises(SloughGPTError, match="boom"):
            await client.chat([ChatMessage.user("q")])

    @pytest.mark.asyncio
    async def test_chat_dict_messages(self):
        client = AsyncSloughGPTClient()
        captured = {}

        async def fake_request(method, endpoint, **kw):
            captured.update(kw)
            return {"text": "ok"}

        client._request = fake_request
        await client.chat([{"role": "user", "content": "q"}])
        assert captured["json"]["messages"] == [{"role": "user", "content": "q"}]

    @pytest.mark.asyncio
    async def test_list_models(self):
        client = AsyncSloughGPTClient()
        client._request = AsyncMock(return_value={"models": [{"id": "gpt2"}]})
        models = await client.list_models()
        assert models[0].id == "gpt2"

    @pytest.mark.asyncio
    async def test_list_souls_switch(self):
        client = AsyncSloughGPTClient()
        client._request = AsyncMock(return_value={"souls": [{"name": "warm"}]})
        assert await client.list_souls() == [{"name": "warm"}]
        client._request = AsyncMock(return_value={"ok": True})
        captured = {}

        async def fake_request(method, endpoint, **kw):
            captured.update(kw)
            return {"ok": True}

        client._request = fake_request
        await client.switch_soul("warm", checkpoint_name="c")
        assert captured["json"] == {"checkpoint_name": "c"}

    @pytest.mark.asyncio
    async def test_knowledge(self):
        client = AsyncSloughGPTClient()
        client._request = AsyncMock(return_value={"items": [{"content": "x"}]})
        assert await client.list_knowledge() == [{"content": "x"}]
        captured = {}

        async def fake_request(method, endpoint, **kw):
            captured.update(kw)
            return {"id": "1"}

        client._request = fake_request
        assert await client.add_knowledge("hi", topic="t") == {"id": "1"}
        assert captured["json"] == {"content": "hi", "topic": "t"}
        client._request = AsyncMock(return_value={"results": ["a"]})
        assert await client.search_knowledge("q") == ["a"]

    @pytest.mark.asyncio
    async def test_metrics_and_system(self):
        client = AsyncSloughGPTClient()
        client._request = AsyncMock(return_value={"requests_total": 3})
        m = await client.metrics()
        assert m.requests_total == 3
        client._request = AsyncMock(return_value={"cpu": 1})
        assert await client.get_system_metrics() == {"cpu": 1}

    @pytest.mark.asyncio
    async def test_workflow_and_feedback(self):
        client = AsyncSloughGPTClient()
        client._request = AsyncMock(return_value={"healthy": True})
        assert await client.get_workflow_status() == {"healthy": True}
        captured = {}

        async def fake_request(method, endpoint, **kw):
            captured.update(kw)
            return {"ok": True}

        client._request = fake_request
        await client.record_feedback("s", "m", 1, tags=["a"])
        assert captured["json"]["tags"] == ["a"]

    @pytest.mark.asyncio
    async def test_training(self):
        client = AsyncSloughGPTClient()
        captured = {}

        async def fake_request(method, endpoint, **kw):
            captured.update(kw)
            return {"job_id": "j1"}

        client._request = fake_request
        assert await client.start_training("m", "ds") == {"job_id": "j1"}
        assert captured["json"]["dataset"] == "ds"
        client._request = AsyncMock(return_value={"status": "done"})
        assert await client.get_training_status("j1") == {"status": "done"}
        client._request = AsyncMock(return_value={"jobs": [{"id": "j1"}]})
        assert await client.list_training_jobs() == [{"id": "j1"}]
        client._request = AsyncMock(return_value=[{"id": "j1"}])
        assert await client.list_training_jobs() == [{"id": "j1"}]

    @pytest.mark.asyncio
    async def test_experiments(self):
        client = AsyncSloughGPTClient()
        captured = {}

        async def fake_request(method, endpoint, **kw):
            captured.update(kw)
            return {"id": "e1"}

        client._request = fake_request
        assert await client.create_experiment("n", extra=1) == {"id": "e1"}
        assert captured["json"]["extra"] == 1
        client._request = AsyncMock(return_value={"experiments": [{"id": "e1"}]})
        assert await client.list_experiments() == [{"id": "e1"}]
        client._request = AsyncMock(return_value={"id": "e1"})
        assert await client.get_experiment("e1") == {"id": "e1"}
        client._request = fake_request
        assert await client.log_metric("e1", "loss", 0.1) == {"id": "e1"}
        assert captured["json"]["metric"] == "loss"

    @pytest.mark.asyncio
    async def test_tokenizer_and_checkpoints(self):
        client = AsyncSloughGPTClient()
        client._request = AsyncMock(return_value={"vocab": 1})
        assert await client.get_tokenizer_stats() == {"vocab": 1}
        client._request = AsyncMock(return_value={"checkpoints": [{"name": "c"}]})
        assert await client.list_auto_train_checkpoints() == [{"name": "c"}]

    @pytest.mark.asyncio
    async def test_security_and_registry(self):
        client = AsyncSloughGPTClient()
        client._request = AsyncMock(return_value={"status": "success", "data": {"keys": ["k"]}})
        assert await client.get_security_keys() == ["k"]
        client._request = AsyncMock(return_value={"status": "success", "data": {"models": ["m"]}})
        assert await client.list_registry_models() == ["m"]
        client._request = AsyncMock(return_value={"status": "success", "data": {"id": "m"}})
        assert await client.get_registry_model("m") == {"id": "m"}
        client._request = AsyncMock(return_value={"status": "success", "data": {"id": "best"}})
        assert await client.get_registry_best() == {"id": "best"}
        client._request = AsyncMock(return_value={"status": "success", "data": {"n": 1}})
        assert await client.get_registry_stats() == {"n": 1}

    @pytest.mark.asyncio
    async def test_aenter_aexit(self):
        client = AsyncSloughGPTClient()
        async with client as c:
            assert c is client

    @pytest.mark.asyncio
    async def test_request_httpx_path(self):
        fake_response = SimpleNamespace(
            status_code=200,
            json=lambda: {"status": "success", "data": {"version": "1"}},
            raise_for_status=lambda: None,
        )

        class FakeAsyncClient:
            last = None

            def __init__(self, **kw):
                self.kw = kw
                FakeAsyncClient.last = self

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def request(self, method, url, **kw):
                return fake_response

        fake_httpx = SimpleNamespace(AsyncClient=FakeAsyncClient)
        client = AsyncSloughGPTClient(api_key="k")
        with patch.dict(sys.modules, {"httpx": fake_httpx}):
            data = await client._request(
                "POST", "/x", json={"a": 1}, extra_headers={"X-A": "1"}
            )
        assert data == {"status": "success", "data": {"version": "1"}}
        inst = FakeAsyncClient.last
        assert inst.kw["headers"]["X-API-Key"] == "k"
        assert inst.kw["headers"]["X-A"] == "1"
        assert inst.kw["verify"] is True