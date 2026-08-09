"""Coverage for sloughgpt_sdk.http_client."""
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "sdk-py"))

from sloughgpt_sdk.http_client import (  # noqa: E402
    AuthInterceptor,
    ErrorHandler,
    HTTPClient,
    JSONParser,
    LoggingInterceptor,
    RequestConfig,
    RequestContext,
    RequestInterceptor,
    ResponseContext,
    ResponseHandler,
    RetryHandler,
    RetryInterceptor,
    Sanitizer,
    sanitize_request,
    with_timeout,
    with_retry,
)


class TestRequestContextDataclasses:
    def test_request_config_defaults(self):
        cfg = RequestConfig()
        assert cfg.timeout == 30
        assert cfg.retry_count == 3
        assert cfg.retry_backoff == 1.5
        assert cfg.retry_max_delay == 30.0
        assert cfg.validate_ssl is True

    def test_request_config_overrides(self):
        cfg = RequestConfig(timeout=5, retry_count=1, validate_ssl=False)
        assert cfg.timeout == 5
        assert cfg.retry_count == 1
        assert cfg.validate_ssl is False

    def test_request_context_default_attempt(self):
        ctx = RequestContext("GET", "http://x", {}, None, 100.0)
        assert ctx.attempt == 1
        assert ctx.method == "GET"
        assert ctx.body is None
        assert ctx.timestamp == 100.0

    def test_response_context_fields(self):
        req = RequestContext("POST", "http://x", {}, "b", 1.0)
        resp = ResponseContext(200, {"a": "1"}, "ok", 3.5, req)
        assert resp.status_code == 200
        assert resp.elapsed_ms == 3.5
        assert resp.request is req


class TestSanitizer:
    def test_sensitive_header_set(self):
        for name in ("authorization", "cookie", "x-api-key", "x-auth-token", "x-access-token", "proxy-authorization"):
            assert name in Sanitizer.SENSITIVE_HEADERS

    def test_sanitize_headers_masks_sensitive_case_insensitive(self):
        out = Sanitizer.sanitize_headers({"Authorization": "Bearer abc", "content-type": "json"})
        assert out["Authorization"] == "***"
        assert out["content-type"] == "json"

    def test_sanitize_body_empty_passthrough(self):
        assert Sanitizer.sanitize_body(None) is None
        assert Sanitizer.sanitize_body("") == ""
        assert Sanitizer.sanitize_body(0) == 0

    def test_sanitize_body_string(self):
        out = Sanitizer.sanitize_body('password="hunter2" token="abc123"')
        assert "hunter2" not in out
        assert "abc123" not in out

    def test_sanitize_body_string_bearer(self):
        out = Sanitizer.sanitize_body("Authorization: Bearer abc.def-gh")
        assert "abc.def-gh" not in out
        assert "Bearer ***" in out

    def test_sanitize_body_dict(self):
        body = {"password": "p", "nested": {"api_key": "k"}, "name": "ok"}
        out = Sanitizer.sanitize_body(body)
        assert out["password"] == "***"
        assert out["nested"]["api_key"] == "***"
        assert out["name"] == "ok"

    def test_sanitize_body_dict_apiKey_variant(self):
        out = Sanitizer.sanitize_body({"API_KEY": "x", "aUth": "y"})
        assert out["API_KEY"] == "***"
        assert out["aUth"] == "***"

    def test_sanitize_body_list(self):
        out = Sanitizer.sanitize_body(["password=1", {"secret": "s"}])
        assert "password=1" in out
        assert out[1]["secret"] == "***"

    def test_sanitize_body_other_type_passthrough(self):
        assert Sanitizer.sanitize_body(123) == 123

    def test_sanitize_string_leaves_plain_text(self):
        assert Sanitizer._sanitize_string("hello world") == "hello world"

    def test_sanitize_dict_recurses_scalars(self):
        out = Sanitizer._sanitize_dict({"count": 3})
        assert out == {"count": 3}


class TestRequestInterceptor:
    def test_add_returns_self_and_appends(self):
        interceptor = RequestInterceptor()
        fn = lambda ctx: ctx
        assert interceptor.add(fn) is interceptor
        assert len(interceptor._interceptors) == 1

    def test_remove_existing_returns_true(self):
        interceptor = RequestInterceptor()
        fn = lambda ctx: ctx
        interceptor.add(fn)
        assert interceptor.remove(fn) is True
        assert interceptor._interceptors == []

    def test_remove_missing_returns_false(self):
        interceptor = RequestInterceptor()
        assert interceptor.remove(lambda ctx: ctx) is False

    def test_intercept_chains(self):
        interceptor = RequestInterceptor()
        interceptor.add(lambda ctx: RequestContext("PATCH", "http://2", ctx.headers, ctx.body, ctx.timestamp))
        out = interceptor.intercept(RequestContext("GET", "http://1", {}, None, 0.0))
        assert out.method == "PATCH"

    def test_intercept_swallows_exceptions(self):
        interceptor = RequestInterceptor()
        interceptor.add(lambda ctx: (_ for _ in ()).throw(ValueError("boom")))
        out = interceptor.intercept(RequestContext("GET", "http://1", {}, None, 0.0))
        assert out.method == "GET"

    def test_intercept_empty_returns_same_context(self):
        interceptor = RequestInterceptor()
        ctx = RequestContext("GET", "http://1", {}, None, 0.0)
        assert interceptor.intercept(ctx) is ctx


class TestLoggingInterceptor:
    def test_calls_logger(self):
        logger = Mock()
        interceptor = LoggingInterceptor(logger=logger, level=logging.DEBUG)
        ctx = RequestContext("GET", "http://x", {}, None, 0.0)
        out = interceptor(ctx)
        assert out is ctx
        logger.log.assert_called_once()
        args = logger.log.call_args.args
        assert "Request: GET http://x (attempt 1)" in args[1]

    def test_default_logger_created(self):
        interceptor = LoggingInterceptor()
        assert interceptor.logger.name == "slo_sdk.http"


class TestAuthInterceptor:
    def test_adds_header(self):
        interceptor = AuthInterceptor("key123")
        ctx = RequestContext("GET", "http://x", {}, None, 0.0)
        out = interceptor(ctx)
        assert out.headers["X-API-Key"] == "key123"


class TestRetryInterceptor:
    def test_should_retry_within_limit(self):
        ri = RetryInterceptor(max_retries=3)
        assert ri.should_retry(500, 1) is True
        assert ri.should_retry(200, 1) is False

    def test_should_retry_stops_at_max(self):
        ri = RetryInterceptor(max_retries=3)
        assert ri.should_retry(500, 3) is False

    def test_custom_retry_on(self):
        ri = RetryInterceptor(retry_on=[418])
        assert ri.should_retry(418, 1) is True
        assert ri.should_retry(500, 1) is False

    def test_default_retry_statuses(self):
        ri = RetryInterceptor()
        assert ri.retry_on == [429, 500, 502, 503, 504]

    def test_get_delay_capped(self):
        ri = RetryInterceptor(backoff_factor=100.0, max_delay=10.0)
        assert ri.get_delay(1) == 10.0

    def test_get_delay_ramps(self):
        ri = RetryInterceptor(backoff_factor=2.0, max_delay=100.0)
        assert ri.get_delay(1) == 2.0
        assert ri.get_delay(3) == 8.0
        assert ri.get_delay(10) == 100.0


class TestResponseHandler:
    def test_add_returns_self(self):
        handler = ResponseHandler()
        assert handler.add(lambda ctx: ctx) is handler

    def test_handle_chains(self):
        handler = ResponseHandler()
        handler.add(lambda ctx: SimpleNamespace(status_code=ctx.status_code, body="a"))
        handler.add(lambda ctx: SimpleNamespace(status_code=ctx.status_code, body=ctx.body + "b"))
        ctx = SimpleNamespace(status_code=200, body="")
        out = handler.handle(ctx)
        assert out.body == "ab"

    def test_handle_swallows_handler_errors(self):
        handler = ResponseHandler()
        handler.add(lambda ctx: (_ for _ in ()).throw(RuntimeError("x")))
        ctx = SimpleNamespace(status_code=200)
        assert handler.handle(ctx) is ctx


class TestErrorHandler:
    def _ctx(self, status, body):
        return ResponseContext(status, {}, body, 1.0, Mock())

    def test_ok_passthrough(self):
        handler = ErrorHandler()
        ctx = self._ctx(200, "ok")
        assert handler(ctx) is ctx

    def test_error_with_detail(self):
        handler = ErrorHandler(error_class=ValueError)
        with pytest.raises(ValueError, match="bad thing"):
            handler(self._ctx(400, {"detail": "bad thing"}))

    def test_error_with_error_field(self):
        handler = ErrorHandler(error_class=ValueError)
        with pytest.raises(ValueError, match="oops"):
            handler(self._ctx(500, {"error": "oops"}))

    def test_error_fallback_http_message(self):
        handler = ErrorHandler(error_class=RuntimeError)
        with pytest.raises(RuntimeError, match="HTTP 503"):
            handler(self._ctx(503, None))


class TestJSONParser:
    def _ctx(self, body):
        return SimpleNamespace(body=body)

    def test_parses_json_string(self):
        parser = JSONParser()
        ctx = self._ctx('{"a": 1}')
        parser(ctx)
        assert ctx.body == {"a": 1}

    def test_invalid_json_passthrough(self):
        parser = JSONParser()
        ctx = self._ctx("not-json")
        parser(ctx)
        assert ctx.body == "not-json"

    def test_non_string_passthrough(self):
        parser = JSONParser()
        ctx = self._ctx({"already": True})
        parser(ctx)
        assert ctx.body == {"already": True}

    def test_empty_body_noop(self):
        parser = JSONParser()
        ctx = self._ctx("")
        parser(ctx)
        assert ctx.body == ""


class TestRetryHandler:
    def test_delegates(self):
        ri = RetryInterceptor(max_retries=2, backoff_factor=3.0, max_delay=99.0)
        rh = RetryHandler(ri)
        assert rh.should_retry(self._resp_ctx(500), 1) is True
        assert rh.get_delay(1) == 3.0

    def test_on_retry_callback(self):
        ri = RetryInterceptor()
        cb = Mock()
        rh = RetryHandler(ri, on_retry=cb)
        assert rh.on_retry is cb

    @staticmethod
    def _resp_ctx(status):
        return ResponseContext(status, {}, None, 0.0, Mock())


class TestWithRetry:
    def _fake(self, status):
        return SimpleNamespace(status_code=status)

    def test_success_no_retry(self):
        mock = Mock(return_value=self._fake(200))
        calls = []

        @with_retry(max_retries=3)
        def request():
            calls.append(1)
            return mock()

        with patch("sloughgpt_sdk.http_client.time.sleep"):
            out = request()
        assert out.status_code == 200
        assert len(calls) == 1

    def test_retries_non_retryable_then_exhausts(self):
        mock = Mock(return_value=self._fake(500))
        calls = []

        @with_retry(max_retries=3)
        def request():
            calls.append(1)
            return mock()

        with patch("sloughgpt_sdk.http_client.time.sleep") as sleep:
            out = request()
        assert len(calls) == 3
        assert sleep.call_count == 2
        assert out.status_code == 500

    def test_exception_retries_then_raises(self):
        mock = Mock(side_effect=ValueError("down"))

        @with_retry(max_retries=2)
        def request():
            return mock()

        with patch("sloughgpt_sdk.http_client.time.sleep"):
            with pytest.raises(ValueError, match="down"):
                request()
        assert mock.call_count == 2

    def test_max_delay_cap(self):
        mock = Mock(side_effect=ValueError("down"))

        @with_retry(max_retries=3, backoff=1000.0, max_delay=5.0)
        def request():
            return mock()

        with patch("sloughgpt_sdk.http_client.time.sleep") as sleep:
            with pytest.raises(ValueError):
                request()
        delays = [c.args[0] for c in sleep.call_args_list]
        assert delays == [1.0, 5.0]


class TestWithTimeout:
    def test_wrapper_runs_func_and_clears_alarm(self):
        calls = []

        @with_timeout(timeout=10)
        def work():
            calls.append(1)
            return "done"

        with (
            patch("signal.signal") as signal,
            patch("signal.alarm") as alarm,
        ):
            assert work() == "done"
        assert alarm.call_args_list == [call(10), call(0)]
        assert calls == [1]

    def test_timeout_handler_raises(self):
        with patch("signal.signal") as signal, patch("signal.alarm"):
            @with_timeout(timeout=1)
            def work():
                return 1

            work()
            handler = signal.call_args.args[1]
            with pytest.raises(TimeoutError, match="timed out"):
                handler(None, None)


class TestSanitizeRequest:
    def test_sanitizes_kwargs_before_call(self):
        captured = {}

        @sanitize_request
        def send(**kwargs):
            captured.update(kwargs)
            return True

        send(payload={"password": "hunter2"}, safe="ok")
        assert captured["payload"]["password"] == "***"
        assert captured["safe"] == "ok"


def jsonstr(data):
    import json

    return json.dumps(data)


def _fake_ok(**overrides):
    default = {"status_code": 200, "headers": {"content-type": "application/json"}, "text": jsonstr({"x": 1})}
    default.update(overrides)
    return SimpleNamespace(**default)


class TestHTTPClient:
    def test_init_base_url_and_config(self):
        cfg = RequestConfig(timeout=4)
        client = HTTPClient(base_url="http://host:8000/", config=cfg)
        assert client.base_url == "http://host:8000"
        assert client.config is cfg

    def test_init_api_key_adds_auth_interceptor(self):
        client = HTTPClient(api_key="secret")
        assert any(isinstance(i, AuthInterceptor) for i in client.interceptors._interceptors)

    def test_init_no_api_key_no_auth(self):
        client = HTTPClient()
        assert not any(isinstance(i, AuthInterceptor) for i in client.interceptors._interceptors)

    def test_get_session_creates_once(self):
        client = HTTPClient()
        fake = Mock()
        with patch("requests.Session", return_value=fake):
            s1 = client._get_session()
            s2 = client._get_session()
        assert s1 is s2
        assert client._session is s1
        fake.headers.update.assert_called_with({"User-Agent": "SloughGPT-SDK/1.0"})

    def test_create_context(self):
        client = HTTPClient()
        ctx = client._create_context("get", "http://x", {"a": "b"}, {"body": 1})
        assert ctx.method == "GET"
        assert ctx.url == "http://x"
        assert ctx.headers == {"a": "b"}
        assert ctx.body == {"body": 1}

    def test_create_context_defaults_headers(self):
        client = HTTPClient()
        ctx = client._create_context("GET", "http://x")
        assert ctx.headers == {}

    def test_create_response(self):
        client = HTTPClient()
        req = client._create_context("GET", "http://x")
        resp = client._create_response(req, 200, {"h": "1"}, ["b"], 2.5)
        assert resp.status_code == 200
        assert resp.elapsed_ms == 2.5
        assert resp.request is req

    def test_request_success(self):
        client = HTTPClient()
        session = Mock()
        session.request = Mock(return_value=_fake_ok())
        client._session = session
        out = client.request("GET", "/health")
        assert out == {"x": 1}
        kwargs = session.request.call_args.kwargs
        assert kwargs["url"] == "http://localhost:8000/health"
        assert kwargs["timeout"] == 30
        assert kwargs["verify"] is True

    def test_request_retries_then_returns_last_body(self):
        client = HTTPClient()
        session = Mock()
        session.request = Mock(return_value=_fake_ok(status_code=500, text='{"detail": "boom"}'))
        client._session = session
        out = client.request("GET", "/health")
        assert out == {"detail": "boom"}
        assert session.request.call_count == 1

    def test_request_exception_repeats(self):
        client = HTTPClient()
        session = Mock()
        session.request = Mock(side_effect=ConnectionError("refused"))
        client._session = session
        with patch("sloughgpt_sdk.http_client.time.sleep"):
            with pytest.raises(ConnectionError, match="refused"):
                client.request("GET", "/health")
        assert session.request.call_count == 3

    def test_request_headers_merged_with_auth(self):
        client = HTTPClient(api_key="k")
        session = Mock()
        session.request = Mock(return_value=_fake_ok())
        client._session = session
        client.request("GET", "/x", headers={"X-Custom": "v"})
        merged = session.headers.update.call_args.args[0]
        assert merged["X-Custom"] == "v"
        assert merged["X-API-Key"] == "k"
        kw = session.request.call_args.kwargs
        assert kw["timeout"] == 30
        assert kw["verify"] is True

    def test_verb_shortcuts(self):
        client = HTTPClient()
        session = Mock()
        session.request = Mock(return_value=_fake_ok())
        client._session = session
        for verb in ("get", "post", "put", "delete", "patch"):
            getattr(client, verb)("/r")
        methods = [c.kwargs["method"] for c in session.request.call_args_list]
        assert methods == ["GET", "POST", "PUT", "DELETE", "PATCH"]

    def test_close_and_context_manager(self):
        client = HTTPClient()
        session = Mock()
        client._session = session
        client.close()
        session.close.assert_called_once()
        assert client._session is None

        client2 = HTTPClient()
        client2._session = Mock()
        with client2 as entered:
            assert entered is client2
        assert client2._session is None