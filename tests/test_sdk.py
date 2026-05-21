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
    APIKeyManager,
    APIKey,
    KeyTier,
    WebhookManager,
    WebhookEvent,
    BillingManager,
    BillingCycle,
    Plan,
    Subscription,
    UsageDashboard,
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


class TestAPIKeyManager(unittest.TestCase):
    """Tests for API key management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_file.close()
        self.manager = APIKeyManager(storage_path=self.temp_file.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass
    
    def test_create_key(self):
        """Test creating an API key."""
        key, key_data = self.manager.create_key(name="Test Key")
        
        self.assertTrue(key.startswith("sk_"))
        self.assertIsNotNone(key_data)
        self.assertEqual(key_data.name, "Test Key")
        self.assertEqual(key_data.tier, KeyTier.FREE)
        self.assertTrue(key_data.is_active)
    
    def test_create_key_with_tier(self):
        """Test creating key with specific tier."""
        key, key_data = self.manager.create_key(
            name="Pro Key",
            tier=KeyTier.PRO,
        )
        
        self.assertEqual(key_data.tier, KeyTier.PRO)
        self.assertEqual(key_data.quota_daily, 10000)
    
    def test_validate_key(self):
        """Test validating an API key."""
        key, key_data = self.manager.create_key(name="Test")
        
        is_valid, reason, validated_key = self.manager.validate_key(key)
        
        self.assertTrue(is_valid)
        self.assertEqual(reason, "OK")
        self.assertIsNotNone(validated_key)
    
    def test_validate_invalid_key(self):
        """Test validating an invalid key."""
        is_valid, reason, key = self.manager.validate_key("invalid_key")
        
        self.assertFalse(is_valid)
        self.assertEqual(reason, "Invalid API key")
        self.assertIsNone(key)
    
    def test_revoke_key(self):
        """Test revoking an API key."""
        key, key_data = self.manager.create_key(name="Test")
        
        result = self.manager.revoke_key(key_data.key_id)
        
        self.assertTrue(result)
        
        is_valid, reason, _ = self.manager.validate_key(key)
        self.assertFalse(is_valid)
        self.assertEqual(reason, "API key is deactivated")
    
    def test_record_usage(self):
        """Test recording usage."""
        key, key_data = self.manager.create_key(name="Test")
        
        result = self.manager.record_usage(key, requests_count=5)
        
        self.assertTrue(result)
        
        stats = self.manager.get_usage_stats(key_data.key_id)
        self.assertEqual(stats["total_requests"], 5)
    
    def test_list_keys(self):
        """Test listing keys."""
        self.manager.create_key(name="Key 1")
        self.manager.create_key(name="Key 2")
        
        keys = self.manager.list_keys()
        
        self.assertEqual(len(keys), 2)
    
    def test_rotate_key(self):
        """Test key rotation."""
        key1, key_data1 = self.manager.create_key(name="Test")
        
        new_key, new_key_data = self.manager.rotate_key(key_data1.key_id)
        
        self.assertNotEqual(key1, new_key)
        self.assertEqual(new_key_data.name, "Test")
        
        is_valid, _, _ = self.manager.validate_key(key1)
        self.assertFalse(is_valid)
        
        is_valid, _, _ = self.manager.validate_key(new_key)
        self.assertTrue(is_valid)


class TestWebhookManager(unittest.TestCase):
    """Tests for webhook management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_file.close()
        self.manager = WebhookManager(storage_path=self.temp_file.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass
    
    def test_create_webhook(self):
        """Test creating a webhook."""
        webhook = self.manager.create_webhook(
            url="https://example.com/webhook",
            events=[WebhookEvent.KEY_CREATED],
        )
        
        self.assertTrue(webhook.id.startswith("wh_"))
        self.assertEqual(webhook.url, "https://example.com/webhook")
        self.assertIn(WebhookEvent.KEY_CREATED, webhook.events)
        self.assertTrue(webhook.is_active)
    
    def test_get_webhook(self):
        """Test getting a webhook."""
        webhook = self.manager.create_webhook(
            url="https://example.com/webhook",
            events=[WebhookEvent.KEY_CREATED],
        )
        
        retrieved = self.manager.get_webhook(webhook.id)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, webhook.id)
    
    def test_list_webhooks(self):
        """Test listing webhooks."""
        self.manager.create_webhook(
            url="https://example1.com/webhook",
            events=[WebhookEvent.KEY_CREATED],
        )
        self.manager.create_webhook(
            url="https://example2.com/webhook",
            events=[WebhookEvent.QUOTA_EXCEEDED],
        )
        
        webhooks = self.manager.list_webhooks()
        self.assertEqual(len(webhooks), 2)
        
        filtered = self.manager.list_webhooks(WebhookEvent.KEY_CREATED)
        self.assertEqual(len(filtered), 1)
    
    def test_delete_webhook(self):
        """Test deleting a webhook."""
        webhook = self.manager.create_webhook(
            url="https://example.com/webhook",
            events=[WebhookEvent.KEY_CREATED],
        )
        
        result = self.manager.delete_webhook(webhook.id)
        
        self.assertTrue(result)
        self.assertIsNone(self.manager.get_webhook(webhook.id))
    
    def test_verify_signature(self):
        """Test signature verification."""
        payload = '{"test": "data"}'
        secret = "test_secret"
        
        signature = WebhookManager._generate_signature(payload, secret)
        
        self.assertTrue(signature.startswith("sha256="))
        self.assertTrue(WebhookManager.verify_signature(signature, payload, secret))
        self.assertFalse(WebhookManager.verify_signature("invalid", payload, secret))


class TestBillingManager(unittest.TestCase):
    """Tests for billing management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_file.close()
        self.manager = BillingManager(storage_path=self.temp_file.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass
    
    def test_default_plans(self):
        """Test that default plans are created."""
        plans = self.manager.list_plans()
        
        self.assertGreaterEqual(len(plans), 4)
        
        plan_ids = [p.id for p in plans]
        self.assertIn("free", plan_ids)
        self.assertIn("starter", plan_ids)
        self.assertIn("pro", plan_ids)
        self.assertIn("enterprise", plan_ids)
    
    def test_create_customer(self):
        """Test creating a customer."""
        customer = self.manager.create_customer(
            email="test@example.com",
            name="Test User",
        )
        
        self.assertTrue(customer["id"].startswith("cus_"))
        self.assertEqual(customer["email"], "test@example.com")
        self.assertEqual(customer["name"], "Test User")
    
    def test_create_subscription(self):
        """Test creating a subscription."""
        customer = self.manager.create_customer(
            email="test@example.com",
            name="Test User",
        )
        
        subscription = self.manager.create_subscription(
            customer_id=customer["id"],
            plan_id="pro",
            billing_cycle=BillingCycle.MONTHLY,
        )
        
        self.assertTrue(subscription.id.startswith("sub_"))
        self.assertEqual(subscription.plan_id, "pro")
        self.assertEqual(subscription.billing_cycle, BillingCycle.MONTHLY)
        self.assertTrue(subscription.is_active())
    
    def test_cancel_subscription(self):
        """Test cancelling a subscription."""
        customer = self.manager.create_customer(
            email="test@example.com",
            name="Test User",
        )
        
        subscription = self.manager.create_subscription(
            customer_id=customer["id"],
            plan_id="pro",
        )
        
        result = self.manager.cancel_subscription(subscription.id)
        
        self.assertTrue(result)
        self.assertFalse(subscription.is_active())
    
    def test_record_usage(self):
        """Test recording usage."""
        customer = self.manager.create_customer(
            email="test@example.com",
            name="Test User",
        )
        
        record = self.manager.record_usage(
            customer_id=customer["id"],
            key_id="sk_test",
            requests=10,
            tokens=500,
        )
        
        self.assertTrue(record.id.startswith("use_"))
        self.assertEqual(record.requests, 10)
        self.assertEqual(record.tokens_used, 500)
    
    def test_get_customer_subscription(self):
        """Test getting customer subscription."""
        customer = self.manager.create_customer(
            email="test@example.com",
            name="Test User",
        )
        
        subscription = self.manager.create_subscription(
            customer_id=customer["id"],
            plan_id="pro",
        )
        
        retrieved = self.manager.get_customer_subscription(customer["id"])
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, subscription.id)


class TestUsageDashboard(unittest.TestCase):
    """Tests for usage dashboard."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_file.close()
        self.dashboard = UsageDashboard(storage_path=self.temp_file.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass
    
    def test_record_request(self):
        """Test recording a request."""
        self.dashboard.record_request(
            key_id="sk_test",
            customer_id="cus_test",
            tokens=100,
            latency_ms=50,
            success=True,
        )
        
        metrics = self.dashboard.get_metrics("1d")
        
        self.assertEqual(metrics.total_requests, 1)
        self.assertEqual(metrics.total_tokens, 100)
    
    def test_get_metrics(self):
        """Test getting metrics."""
        for i in range(5):
            self.dashboard.record_request(
                key_id="sk_test",
                customer_id="cus_test",
                tokens=100,
                latency_ms=50,
            )
        
        metrics = self.dashboard.get_metrics("1d")
        
        self.assertEqual(metrics.total_requests, 5)
        self.assertEqual(metrics.total_tokens, 500)
        self.assertGreater(metrics.success_rate, 0)
    
    def test_get_usage_breakdown(self):
        """Test getting usage breakdown."""
        self.dashboard.record_request(
            key_id="sk_test",
            customer_id="cus_test",
            endpoint="/generate",
        )
        self.dashboard.record_request(
            key_id="sk_test",
            customer_id="cus_test",
            endpoint="/chat",
        )
        
        breakdown = self.dashboard.get_usage_breakdown("1d", "endpoint")
        
        self.assertEqual(len(breakdown), 2)
    
    def test_generate_report(self):
        """Test generating a report."""
        self.dashboard.record_request(
            key_id="sk_test",
            customer_id="cus_test",
            tokens=100,
        )
        
        report = self.dashboard.generate_report("7d")
        
        self.assertIn("metrics", report)
        self.assertIn("breakdown", report)
        self.assertIn("top_customers", report)
        self.assertIn("top_keys", report)
    
    def test_export_csv(self):
        """Test CSV export."""
        self.dashboard.record_request(
            key_id="sk_test",
            customer_id="cus_test",
            tokens=100,
        )
        
        csv = self.dashboard.export_csv("1d")
        
        self.assertIn("timestamp", csv)
        self.assertIn("customer_id", csv)
        self.assertIn("sk_test", csv)


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
            use_mixed_precision=True,
            mixed_precision_dtype="fp16",
            scheduler="linear",
            device="cpu",
        )
        self.assertEqual(payload["weight_decay"], 0.02)
        self.assertTrue(payload["use_mixed_precision"])
        self.assertEqual(payload["mixed_precision_dtype"], "fp16")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
