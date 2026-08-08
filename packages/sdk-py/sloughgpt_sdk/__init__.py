"""
SloughGPT Python SDK
A Python client library for the SloughGPT API.

Usage:
    from sloughgpt_sdk import SloughGPTClient
    from sloughgpt_sdk.models import GenerationResult, ChatMessage

    # Or for caching
    from sloughgpt_sdk.cache import InMemoryCache

    # CLI
    # sloughgpt-cli generate "Hello"
"""

__version__ = "1.1.0"
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
