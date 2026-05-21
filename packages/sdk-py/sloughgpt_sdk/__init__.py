"""
SloughGPT Python SDK
A Python client library for the SloughGPT API.

Usage:
    from sloughgpt_sdk import SloughGPTClient
    from sloughgpt_sdk.models import GenerationResult, ChatMessage
    
    # Or for WebSocket streaming
    from sloughgpt_sdk.websocket import WebSocketClient
    
    # Or for caching
    from sloughgpt_sdk.cache import InMemoryCache
    
    # CLI
    # sloughgpt-cli generate "Hello"
"""

__version__ = "1.0.0"
__author__ = "SloughGPT"
__email__ = "dev@sloughgpt.ai"
__url__ = "https://github.com/iamtowbee/sloughGPT"

import sys
import importlib.util
import os

_package_dir = os.path.dirname(__file__)
_models_path = os.path.join(_package_dir, "models.py")

_spec = importlib.util.spec_from_file_location("sloughgpt_sdk.models", _models_path)
_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_models)

sys.modules["sloughgpt_sdk.models"] = _models

GenerateRequest = _models.GenerateRequest
ChatMessage = _models.ChatMessage
ChatRequest = _models.ChatRequest
BatchRequest = _models.BatchRequest
BatchResult = _models.BatchResult
GenerationResult = _models.GenerationResult
ChatResult = _models.ChatResult
ModelInfo = _models.ModelInfo
DatasetInfo = _models.DatasetInfo
HealthStatus = _models.HealthStatus
SystemInfo = _models.SystemInfo
MetricsData = _models.MetricsData

import sys
sys.modules["sloughgpt_sdk"].GenerateRequest = GenerateRequest
sys.modules["sloughgpt_sdk"].ChatMessage = ChatMessage
sys.modules["sloughgpt_sdk"].ChatRequest = ChatRequest
sys.modules["sloughgpt_sdk"].BatchRequest = BatchRequest
sys.modules["sloughgpt_sdk"].BatchResult = BatchResult
sys.modules["sloughgpt_sdk"].GenerationResult = GenerationResult
sys.modules["sloughgpt_sdk"].ChatResult = ChatResult
sys.modules["sloughgpt_sdk"].ModelInfo = ModelInfo
sys.modules["sloughgpt_sdk"].DatasetInfo = DatasetInfo
sys.modules["sloughgpt_sdk"].HealthStatus = HealthStatus
sys.modules["sloughgpt_sdk"].SystemInfo = SystemInfo
sys.modules["sloughgpt_sdk"].MetricsData = MetricsData

from sloughgpt_sdk.client import SloughGPTClient, AsyncSloughGPTClient

_auth_path = os.path.join(_package_dir, "auth.py")
_auth_spec = importlib.util.spec_from_file_location("sloughgpt_sdk.auth", _auth_path)
_auth = importlib.util.module_from_spec(_auth_spec)
_auth_spec.loader.exec_module(_auth)
sys.modules["sloughgpt_sdk.auth"] = _auth

APIKeyManager = _auth.APIKeyManager
APIKey = _auth.APIKey
APIKeyMiddleware = _auth.APIKeyMiddleware
KeyTier = _auth.KeyTier

_webhooks_path = os.path.join(_package_dir, "webhooks.py")
_webhooks_spec = importlib.util.spec_from_file_location("sloughgpt_sdk.webhooks", _webhooks_path)
_webhooks = importlib.util.module_from_spec(_webhooks_spec)
_webhooks_spec.loader.exec_module(_webhooks)
sys.modules["sloughgpt_sdk.webhooks"] = _webhooks

WebhookManager = _webhooks.WebhookManager
Webhook = _webhooks.Webhook
WebhookEvent = _webhooks.WebhookEvent
WebhookDelivery = _webhooks.WebhookDelivery

_billing_path = os.path.join(_package_dir, "billing.py")
_billing_spec = importlib.util.spec_from_file_location("sloughgpt_sdk.billing", _billing_path)
_billing = importlib.util.module_from_spec(_billing_spec)
_billing_spec.loader.exec_module(_billing)
sys.modules["sloughgpt_sdk.billing"] = _billing

BillingManager = _billing.BillingManager
Plan = _billing.Plan
Subscription = _billing.Subscription
Invoice = _billing.Invoice
BillingCycle = _billing.BillingCycle

_dashboard_path = os.path.join(_package_dir, "dashboard.py")
_dashboard_spec = importlib.util.spec_from_file_location("sloughgpt_sdk.dashboard", _dashboard_path)
_dashboard = importlib.util.module_from_spec(_dashboard_spec)
_dashboard_spec.loader.exec_module(_dashboard)
sys.modules["sloughgpt_sdk.dashboard"] = _dashboard

UsageDashboard = _dashboard.UsageDashboard
DashboardMetrics = _dashboard.DashboardMetrics

_registry_path = os.path.join(_package_dir, "registry.py")
_registry_spec = importlib.util.spec_from_file_location("sloughgpt_sdk.registry", _registry_path)
_registry = importlib.util.module_from_spec(_registry_spec)
_registry_spec.loader.exec_module(_registry)
sys.modules["sloughgpt_sdk.registry"] = _registry

ModelRegistry = _registry.ModelRegistry
ModelSelector = _registry.ModelSelector
ModelStatus = _registry.ModelStatus
ModelTag = _registry.ModelTag

_http_path = os.path.join(_package_dir, "http_client.py")
_http_spec = importlib.util.spec_from_file_location("sloughgpt_sdk.http", _http_path)
_http = importlib.util.module_from_spec(_http_spec)
_http_spec.loader.exec_module(_http)
sys.modules["sloughgpt_sdk.http"] = _http

HTTPClient = _http.HTTPClient
Sanitizer = _http.Sanitizer
RequestInterceptor = _http.RequestInterceptor
LoggingInterceptor = _http.LoggingInterceptor
AuthInterceptor = _http.AuthInterceptor
RetryInterceptor = _http.RetryInterceptor
ResponseHandler = _http.ResponseHandler
ErrorHandler = _http.ErrorHandler
JSONParser = _http.JSONParser
RequestConfig = _http.RequestConfig
RequestContext = _http.RequestContext
ResponseContext = _http.ResponseContext
with_retry = _http.with_retry
with_timeout = _http.with_timeout
sanitize_request = _http.sanitize_request

__all__ = [
    "SloughGPTClient",
    "AsyncSloughGPTClient",
    "GenerateRequest",
    "ChatMessage",
    "ChatRequest",
    "BatchRequest",
    "BatchResult",
    "GenerationResult",
    "ChatResult",
    "ModelInfo",
    "DatasetInfo",
    "HealthStatus",
    "SystemInfo",
    "MetricsData",
    "APIKeyManager",
    "APIKey",
    "APIKeyMiddleware",
    "KeyTier",
    "WebhookManager",
    "Webhook",
    "WebhookEvent",
    "WebhookDelivery",
    "BillingManager",
    "Plan",
    "Subscription",
    "Invoice",
    "UsageDashboard",
    "DashboardMetrics",
    "ModelRegistry",
    "ModelSelector",
    "ModelStatus",
    "ModelTag",
    "HTTPClient",
    "Sanitizer",
    "RequestInterceptor",
    "LoggingInterceptor",
    "AuthInterceptor",
    "RetryInterceptor",
    "ResponseHandler",
    "ErrorHandler",
    "JSONParser",
    "RequestConfig",
    "RequestContext",
    "ResponseContext",
    "with_retry",
    "with_timeout",
    "sanitize_request",
]
