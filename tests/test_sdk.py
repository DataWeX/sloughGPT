"""
SloughGPT SDK Tests
Unit tests for the Python SDK.
"""

import os
import sys
import json
import time
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "packages" / "sdk-py"))

from sloughgpt_sdk import (
    SloughGPTClient,
    ChatMessage,
    GenerateRequest,
    GenerationResult,
)


class TestChatMessage(unittest.TestCase):
    """Tests for ChatMessage model."""

    def test_user_message(self):
        """Test creating user message."""
        msg = ChatMessage.user("Hello!")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello!")

    def test_assistant_message(self):
        """Test creating assistant message."""
        msg = ChatMessage.assistant("Hi there!")
        self.assertEqual(msg.role, "assistant")
        self.assertEqual(msg.content, "Hi there!")

    def test_system_message(self):
        """Test creating system message."""
        msg = ChatMessage.system("You are helpful.")
        self.assertEqual(msg.role, "system")
        self.assertEqual(msg.content, "You are helpful.")

    def test_to_dict(self):
        """Test converting to dictionary."""
        msg = ChatMessage.user("Test")
        data = msg.to_dict()
        self.assertEqual(data["role"], "user")
        self.assertEqual(data["content"], "Test")


class TestGenerateRequest(unittest.TestCase):
    """Tests for GenerateRequest model."""

    def test_default_values(self):
        """Test default request values."""
        req = GenerateRequest(prompt="Hello")
        self.assertEqual(req.prompt, "Hello")
        self.assertEqual(req.max_new_tokens, 100)
        self.assertEqual(req.temperature, 0.8)
        self.assertEqual(req.top_p, 0.9)

    def test_custom_values(self):
        """Test custom request values."""
        req = GenerateRequest(
            prompt="Test",
            max_new_tokens=50,
            temperature=0.5,
            top_k=20,
        )
        self.assertEqual(req.max_new_tokens, 50)
        self.assertEqual(req.temperature, 0.5)
        self.assertEqual(req.top_k, 20)

    def test_to_dict(self):
        """Test converting to dictionary."""
        req = GenerateRequest(prompt="Hello", max_new_tokens=50)
        data = req.to_dict()
        self.assertEqual(data["prompt"], "Hello")
        self.assertEqual(data["max_new_tokens"], 50)


class TestSloughGPTClient(unittest.TestCase):
    """Tests for the SloughGPT client."""

    @patch('requests.Session')
    def test_client_initialization(self, mock_session):
        """Test client initialization."""
        client = SloughGPTClient(base_url="http://localhost:8000")

        self.assertEqual(client.base_url, "http://localhost:8000")
        self.assertEqual(client.timeout, 30)

    @patch('requests.Session')
    def test_client_with_api_key(self, mock_session):
        """Test client with API key."""
        client = SloughGPTClient(
            base_url="http://localhost:8000",
            api_key="test_key",
        )

        self.assertEqual(client._headers.get("X-API-Key"), "test_key")


class TestBenchmark(unittest.TestCase):
    """Tests for benchmark utilities."""

    def test_benchmark_result_str(self):
        """Test BenchmarkResult string representation."""
        from sloughgpt_sdk.benchmarks import BenchmarkResult

        result = BenchmarkResult(
            name="Test",
            iterations=100,
            total_time_ms=100,
            avg_time_ms=1.0,
            min_time_ms=0.5,
            max_time_ms=2.0,
            median_time_ms=1.0,
            std_dev_ms=0.3,
            ops_per_second=1000,
        )

        self.assertIn("Test", str(result))
        self.assertIn("1000.00", str(result))

    def test_benchmark_result_to_dict(self):
        """Test BenchmarkResult to_dict."""
        from sloughgpt_sdk.benchmarks import BenchmarkResult

        result = BenchmarkResult(
            name="Test",
            iterations=100,
            total_time_ms=100,
            avg_time_ms=1.0,
            min_time_ms=0.5,
            max_time_ms=2.0,
            median_time_ms=1.0,
            std_dev_ms=0.3,
            ops_per_second=1000,
        )

        data = result.to_dict()
        self.assertEqual(data["name"], "Test")
        self.assertEqual(data["iterations"], 100)
        self.assertEqual(data["ops_per_second"], 1000)

    def test_run_benchmark(self):
        """Test running a simple benchmark."""
        from sloughgpt_sdk.benchmarks import Benchmark

        bench = Benchmark()
        result = bench.run(
            name="String concat",
            func=lambda: "hello" + "world",
            iterations=100,
        )

        self.assertEqual(result.name, "String concat")
        self.assertEqual(result.iterations, 100)
        self.assertGreater(result.ops_per_second, 0)

    def test_percentile_calculation(self):
        """Test percentile calculation."""
        from sloughgpt_sdk.benchmarks import percentile

        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        p50 = percentile(data, 50)
        self.assertGreaterEqual(p50, 5)
        self.assertLessEqual(p50, 6)
        self.assertEqual(percentile(data, 95), 10)
        self.assertEqual(percentile(data, 99), 10)

    def test_load_test_result(self):
        """Test LoadTestResult."""
        from sloughgpt_sdk.benchmarks import LoadTestResult

        result = LoadTestResult(
            name="Load Test",
            concurrent_workers=10,
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            total_time_ms=1000,
            requests_per_second=100,
            avg_latency_ms=10,
            min_latency_ms=5,
            max_latency_ms=50,
            median_latency_ms=8,
            p95_latency_ms=20,
            p99_latency_ms=30,
            success_rate=0.95,
        )

        self.assertEqual(result.name, "Load Test")
        self.assertEqual(result.success_rate, 0.95)
        self.assertEqual(result.total_requests, 100)

    def test_profiler(self):
        """Test profiler decorator."""
        from sloughgpt_sdk.benchmarks import Profiler

        profiler = Profiler()

        @profiler.profile("test_func")
        def test_func():
            return 1 + 1

        test_func()
        test_func()

        report = profiler.get_report()
        self.assertIn("test_func", report)
        self.assertEqual(report["test_func"]["calls"], 2)


class TestNewSDKMethods(unittest.TestCase):
    """Tests for new SDK methods."""

    def test_training_methods_exist(self):
        """Test training methods exist on client."""
        from sloughgpt_sdk import SloughGPTClient

        client = SloughGPTClient()

        self.assertTrue(hasattr(client, 'start_training'))
        self.assertTrue(hasattr(client, 'get_training_status'))
        self.assertTrue(hasattr(client, 'list_training_jobs'))
        self.assertTrue(hasattr(client, 'delete_training_job'))
        self.assertTrue(hasattr(client, 'stop_training'))
        self.assertTrue(hasattr(client, 'pause_training'))
        self.assertTrue(hasattr(client, 'resume_training'))

    def test_async_client_exported(self):
        """AsyncSloughGPTClient is public API (README lists it)."""
        from sloughgpt_sdk import AsyncSloughGPTClient

        self.assertTrue(callable(AsyncSloughGPTClient))

    def test_coerce_training_jobs_list_accepts_array_or_wrapper(self):
        """GET /training/jobs returns a list; some proxies may wrap {jobs: [...]}."""
        from sloughgpt_sdk.client import _coerce_training_jobs_list

        self.assertEqual(_coerce_training_jobs_list([{"id": "a"}]), [{"id": "a"}])
        self.assertEqual(
            _coerce_training_jobs_list({"jobs": [{"id": "b"}]}),
            [{"id": "b"}],
        )
        self.assertEqual(_coerce_training_jobs_list({}), [])
        self.assertEqual(_coerce_training_jobs_list("bad"), [])

    def test_build_training_start_payload_forwards_loop_options(self) -> None:
        """POST /training/start accepts log_interval / eval_interval (server TrainingRequest)."""
        from sloughgpt_sdk.client import _build_training_start_payload

        payload = _build_training_start_payload(
            "sloughgpt",
            "openwebtext",
            epochs=2,
            log_interval=5,
            eval_interval=90,
        )
        self.assertEqual(payload["model"], "sloughgpt")
        self.assertEqual(payload["dataset"], "openwebtext")
        self.assertEqual(payload["epochs"], 2)
        self.assertEqual(payload["log_interval"], 5)
        self.assertEqual(payload["eval_interval"], 90)

    def test_build_training_start_payload_forwards_trainer_hyperparams(self) -> None:
        """Extended TrainingRequest fields pass through kwargs."""
        from sloughgpt_sdk.client import _build_training_start_payload

        payload = _build_training_start_payload(
            "m",
            "d",
            weight_decay=0.02,
            scheduler="linear",
            device="cpu",
        )
        self.assertEqual(payload["weight_decay"], 0.02)
        self.assertEqual(payload["scheduler"], "linear")
        self.assertEqual(payload["device"], "cpu")

    def test_experiment_methods_exist(self):
        """Test experiment methods exist on client."""
        from sloughgpt_sdk import SloughGPTClient

        client = SloughGPTClient()

        self.assertTrue(hasattr(client, 'create_experiment'))
        self.assertTrue(hasattr(client, 'list_experiments'))
        self.assertTrue(hasattr(client, 'get_experiment'))
        self.assertTrue(hasattr(client, 'log_metric'))
        self.assertTrue(hasattr(client, 'log_param'))

    def test_rate_limit_methods_exist(self):
        """Test rate limit methods exist on client."""
        from sloughgpt_sdk import SloughGPTClient

        client = SloughGPTClient()

        self.assertTrue(hasattr(client, 'get_rate_limit_status'))
        self.assertTrue(hasattr(client, 'check_rate_limit'))

    def test_personality_methods_exist(self):
        """Test personality methods exist on client."""
        from sloughgpt_sdk import SloughGPTClient

        client = SloughGPTClient()

        self.assertTrue(hasattr(client, 'get_personalities'))
        self.assertTrue(hasattr(client, 'set_personality'))



class TestNewSDKEndpoints(unittest.TestCase):
    """Integration tests for all new SDK endpoints (mocked HTTP)."""

    def setUp(self):
        self.client_patcher = patch('requests.Session.request')
        self.mock_request = self.client_patcher.start()
        self.mock_response = MagicMock()
        self.mock_response.ok = True
        self.mock_response.status_code = 200
        self.mock_response.json.return_value = {}
        self.mock_request.return_value = self.mock_response

    def tearDown(self):
        self.client_patcher.stop()

    def _client(self):
        from sloughgpt_sdk import SloughGPTClient
        return SloughGPTClient()

    def _assert_called(self, method, path):
        self.assertEqual(len(self.mock_request.call_args_list), 1,
                         f"Expected 1 call, got {len(self.mock_request.call_args_list)}")
        args, kwargs = self.mock_request.call_args
        self.assertEqual(args[0], method, f"Expected method {method}, got {args[0]}")
        url = str(args[1])
        self.assertIn(path, url, f"Path {path} not in URL {url}")
        self.assertIn('timeout', kwargs)
        self.assertIn('verify', kwargs)

    # === Souls ===

    def test_list_souls(self):
        self._client().list_souls()
        self._assert_called('GET', '/souls')

    def test_get_current_soul(self):
        self._client().get_current_soul()
        self._assert_called('GET', '/souls/current')

    def test_switch_soul(self):
        self._client().switch_soul('friendly')
        self._assert_called('POST', '/souls/switch')
        call_body = self.mock_request.call_args[1].get('json', {})
        self.assertEqual(call_body.get('name'), 'friendly')

    def test_switch_soul_with_checkpoint(self):
        self._client().switch_soul('friendly', 'ckpt-v2')
        self._assert_called('POST', '/souls/switch')
        call_body = self.mock_request.call_args[1].get('json', {})
        self.assertEqual(call_body.get('name'), 'friendly')
        self.assertEqual(call_body.get('checkpoint_name'), 'ckpt-v2')

    # === Knowledge ===

    def test_list_knowledge(self):
        self._client().list_knowledge()
        self._assert_called('GET', '/knowledge')

    def test_add_knowledge(self):
        self._client().add_knowledge('Paris is capital', 'geo')
        self._assert_called('POST', '/knowledge')

    def test_delete_knowledge(self):
        self._client().delete_knowledge('k1')
        self._assert_called('DELETE', '/knowledge/k1')

    def test_search_knowledge(self):
        self._client().search_knowledge('paris')
        self._assert_called('GET', '/knowledge/search')

    def test_get_knowledge_stats(self):
        self._client().get_knowledge_stats()
        self._assert_called('GET', '/knowledge/stats')

    def test_get_knowledge_topics(self):
        self._client().get_knowledge_topics()
        self._assert_called('GET', '/knowledge/topics')

    def test_ingest_knowledge_url(self):
        self._client().ingest_knowledge_url('https://example.com')
        self._assert_called('POST', '/knowledge/ingest-url')

    # === Tokenizer ===

    def test_get_tokenizer_stats(self):
        self._client().get_tokenizer_stats()
        self._assert_called('GET', '/tokenizer/stats')

    def test_tokenize(self):
        self._client().tokenize('hello world')
        self._assert_called('POST', '/tokenizer/tokenize')

    def test_train_tokenizer(self):
        self._client().train_tokenizer('training text', 32000)
        self._assert_called('POST', '/tokenizer/train')

    # === System ===

    def test_get_system_metrics(self):
        self._client().get_system_metrics()
        self._assert_called('GET', '/system/metrics')

    def test_get_system_info(self):
        self._client().get_system_info()
        self._assert_called('GET', '/system/info')

    def test_get_system_disk(self):
        self._client().get_system_disk()
        self._assert_called('GET', '/system/disk')

    # === Companion ===

    def test_get_companion_prompt(self):
        self._client().get_companion_prompt()
        self._assert_called('GET', '/companion/prompt')

    def test_list_companion_presets(self):
        self._client().list_companion_presets()
        self._assert_called('GET', '/companion/presets')

    # === Generation endpoints ===

    def test_generate_uses_inference_path(self):
        self._client().generate('hello')
        self._assert_called('POST', '/inference/generate')

    def test_generate_stream_uses_inference_path(self):
        self.mock_response.iter_lines.return_value = []
        list(self._client().generate_stream('hello'))
        self._assert_called('POST', '/inference/generate/stream')

    # === Training Control ===

    def test_stop_training(self):
        self._client().stop_training()
        self._assert_called('POST', '/training/control/stop')

    def test_pause_training(self):
        self._client().pause_training()
        self._assert_called('POST', '/training/control/pause')

    def test_resume_training(self):
        self._client().resume_training()
        self._assert_called('POST', '/training/control/resume')

    def test_delete_training_job(self):
        self._client().delete_training_job('job-1')
        self._assert_called('DELETE', '/training/jobs/job-1')

    def test_get_training_recovery_stats(self):
        self._client().get_training_recovery_stats()
        self._assert_called('GET', '/recovery/stats')

    # === Auto-Train ===

    def test_start_auto_train(self):
        self._client().start_auto_train({'soul': 'friendly'})
        self._assert_called('POST', '/training/start')

    def test_stop_auto_train(self):
        self._client().stop_auto_train()
        self._assert_called('POST', '/training/stop')

    def test_get_auto_train_status(self):
        self._client().get_auto_train_status()
        self._assert_called('GET', '/training/status')

    def test_list_auto_train_checkpoints(self):
        self._client().list_auto_train_checkpoints()
        self._assert_called('GET', '/training/checkpoints')

    def test_delete_auto_train_checkpoint(self):
        self._client().delete_auto_train_checkpoint('ckpt-1')
        self._assert_called('DELETE', '/training/checkpoints/ckpt-1')

    def test_load_auto_train_checkpoint(self):
        self._client().load_auto_train_checkpoint('ckpt-1')
        self._assert_called('POST', '/training/checkpoints/ckpt-1/load')

    # === Feedback / Workflow ===

    def test_record_feedback(self):
        self._client().record_feedback('s1', 'm1', 1)
        self._assert_called('POST', '/feedback/workflow-record')

    def test_get_feedback_stats(self):
        self._client().get_feedback_stats()
        self._assert_called('GET', '/feedback/stats/summary')

    def test_get_workflow_status(self):
        self._client().get_workflow_status()
        self._assert_called('GET', '/workflow/status')

    # === Sessions ===

    def test_save_session_context(self):
        self._client().save_session_context('sess-1', {'ctx': 'data'})
        self._assert_called('POST', '/session/sess-1/context')

    def test_get_session_messages(self):
        self._client().get_session_messages('sess-1')
        self._assert_called('GET', '/session/sess-1/messages')

    # === Models ===

    def test_unload_model(self):
        self._client().unload_model()
        self._assert_called('POST', '/models/unload')

    def test_get_current_model(self):
        self._client().get_current_model()
        self._assert_called('GET', '/models/current')

    # === Datasets ===

    def test_import_dataset_local(self):
        self._client().import_dataset_local('/path', 'ds')
        self._assert_called('POST', '/datasets/import/local')

    def test_import_dataset_github(self):
        self._client().import_dataset_github('user/repo', 'ds')
        self._assert_called('POST', '/datasets/import/github')

    def test_import_dataset_url(self):
        self._client().import_dataset_url('https://example.com', 'ds')
        self._assert_called('POST', '/datasets/import/url')

    # === Benchmark ===

    def test_get_benchmark_metrics(self):
        self._client().get_benchmark_metrics()
        self._assert_called('GET', '/benchmark/metrics')

    def test_get_benchmark_stats(self):
        self._client().get_benchmark_stats()
        self._assert_called('GET', '/benchmark/stats')

    # === Security ===

    def test_get_audit_log(self):
        self._client().get_audit_log()
        self._assert_called('GET', '/security/audit')

    def test_get_security_keys(self):
        self._client().get_security_keys()
        self._assert_called('GET', '/security/keys')

    def test_get_security_keys_unwraps_keys_field(self):
        self.mock_response.json.return_value = {"status": "success", "data": {"keys": [{"id": "k1"}]}}
        result = self._client().get_security_keys()
        self.assertEqual(result, [{"id": "k1"}])

    def test_get_security_keys_accepts_flat_list(self):
        self.mock_response.json.return_value = [{"id": "k1"}]
        result = self._client().get_security_keys()
        self.assertEqual(result, [{"id": "k1"}])

    # === Registry ===

    def test_list_registry_models(self):
        self._client().list_registry_models()
        self._assert_called('GET', '/registry/models')

    def test_list_registry_models_unwraps_models_field(self):
        self.mock_response.json.return_value = {
            "status": "success",
            "data": {"models": [{"model_id": "gpt2"}], "count": 1},
        }
        result = self._client().list_registry_models()
        self.assertEqual(result, [{"model_id": "gpt2"}])

    def test_get_registry_model(self):
        self._client().get_registry_model('gpt2')
        self._assert_called('GET', '/registry/models/gpt2')

    def test_get_registry_best(self):
        self.mock_response.json.return_value = {
            "status": "success",
            "data": {"models": 2, "loaded": 1},
        }
        result = self._client().get_registry_best()
        self.assertEqual(result, {"models": 2, "loaded": 1})

    def test_get_registry_stats(self):
        self._client().get_registry_stats()
        self._assert_called('GET', '/registry/stats')

    # === Detailed Health ===

    def test_detailed_health(self):
        self._client().detailed_health()
        self._assert_called('GET', '/health/detailed')

    # === Set personality uses companion endpoint ===

    def test_set_personality_uses_companion(self):
        self._client().set_personality('friendly')
        self._assert_called('POST', '/companion/personality')


if __name__ == "__main__":
    unittest.main(verbosity=2)
