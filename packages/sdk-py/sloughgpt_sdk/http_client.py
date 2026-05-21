"""
SloughGPT SDK - HTTP Client with Request/Response Handling
Request sanitization, interceptors, and response handlers.
"""

import time
import re
import logging
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from functools import wraps
import threading


@dataclass
class RequestConfig:
    """Request configuration."""
    timeout: int = 30
    retry_count: int = 3
    retry_backoff: float = 1.5
    retry_max_delay: float = 30.0
    validate_ssl: bool = True


@dataclass
class RequestContext:
    """Context for a request."""
    method: str
    url: str
    headers: Dict[str, str]
    body: Any
    timestamp: float
    attempt: int = 1


@dataclass
class ResponseContext:
    """Context for a response."""
    status_code: int
    headers: Dict[str, str]
    body: Any
    elapsed_ms: float
    request: RequestContext


class Sanitizer:
    """Sanitize requests and responses."""
    
    SENSITIVE_HEADERS = {
        "authorization", "cookie", "x-api-key", 
        "x-auth-token", "x-access-token", "proxy-authorization"
    }
    
    SENSITIVE_PATTERNS = [
        (r'password["\']?\s*[:=]\s*["\'][^"\']+["\']', 'password":"***"'),
        (r'api[_-]?key["\']?\s*[:=]\s*["\'][^"\']+["\']', 'api_key":"***"'),
        (r'token["\']?\s*[:=]\s*["\'][^"\']+["\']', 'token":"***"'),
        (r'secret["\']?\s*[:=]\s*["\'][^"\']+["\']', 'secret":"***"'),
        (r'bearer\s+[a-zA-Z0-9._-]+', 'Bearer ***'),
    ]
    
    @classmethod
    def sanitize_headers(cls, headers: Dict[str, str]) -> Dict[str, str]:
        """Remove sensitive headers."""
        return {
            k: "***" if k.lower() in cls.SENSITIVE_HEADERS else v
            for k, v in headers.items()
        }
    
    @classmethod
    def sanitize_body(cls, body: Any) -> Any:
        """Remove sensitive data from body."""
        if not body:
            return body
        
        if isinstance(body, str):
            return cls._sanitize_string(body)
        
        if isinstance(body, dict):
            return cls._sanitize_dict(body)
        
        if isinstance(body, list):
            return [cls.sanitize_body(item) for item in body]
        
        return body
    
    @classmethod
    def _sanitize_string(cls, text: str) -> str:
        """Sanitize sensitive patterns in string."""
        result = text
        for pattern, replacement in cls.SENSITIVE_PATTERNS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result
    
    @classmethod
    def _sanitize_dict(cls, data: dict) -> dict:
        """Sanitize sensitive fields in dict."""
        sensitive_keys = {"password", "api_key", "apiKey", "token", "secret", "auth"}
        return {
            k: "***" if k.lower() in sensitive_keys else cls.sanitize_body(v)
            for k, v in data.items()
        }


class RequestInterceptor:
    """
    Intercept and modify requests.
    
    Example:
    
    ```python
    from sloughgpt_sdk.http import RequestInterceptor, LoggingInterceptor
    
    interceptor = RequestInterceptor()
    interceptor.add(LogInterceptor())
    interceptor.add(CustomInterceptor(lambda req: add_header(req, "X-Custom", "value")))
    ```
    """
    
    def __init__(self):
        """Initialize interceptor."""
        self._interceptors: List[Callable] = []
        self._lock = threading.Lock()
    
    def add(self, interceptor: Callable) -> "RequestInterceptor":
        """Add an interceptor."""
        with self._lock:
            self._interceptors.append(interceptor)
        return self
    
    def remove(self, interceptor: Callable) -> bool:
        """Remove an interceptor."""
        with self._lock:
            try:
                self._interceptors.remove(interceptor)
                return True
            except ValueError:
                return False
    
    def intercept(self, context: RequestContext) -> RequestContext:
        """Run all interceptors."""
        ctx = context
        for interceptor in self._interceptors:
            try:
                ctx = interceptor(ctx)
            except Exception:
                pass
        return ctx


class LoggingInterceptor:
    """Log requests and responses."""
    
    def __init__(self, logger: Optional[logging.Logger] = None, level: int = logging.INFO):
        self.logger = logger or logging.getLogger("sloughgpt_sdk.http")
        self.level = level
    
    def __call__(self, context: RequestContext) -> RequestContext:
        """Log request."""
        self.logger.log(
            self.level,
            f"Request: {context.method} {context.url} "
            f"(attempt {context.attempt})"
        )
        return context


class AuthInterceptor:
    """Add authentication headers."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def __call__(self, context: RequestContext) -> RequestContext:
        """Add auth header."""
        context.headers["X-API-Key"] = self.api_key
        return context


class RetryInterceptor:
    """Handle retries with exponential backoff."""
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        max_delay: float = 30.0,
        retry_on: Optional[List[int]] = None,
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.retry_on = retry_on or [429, 500, 502, 503, 504]
    
    def should_retry(self, status_code: int, attempt: int) -> bool:
        """Check if should retry."""
        if attempt >= self.max_retries:
            return False
        return status_code in self.retry_on
    
    def get_delay(self, attempt: int) -> float:
        """Calculate backoff delay."""
        delay = self.backoff_factor ** attempt
        return min(delay, self.max_delay)


class ResponseHandler:
    """
    Handle and transform responses.
    
    Example:
    
    ```python
    handler = ResponseHandler()
    handler.add(ErrorHandler())
    handler.add(JSONParser())
    ```
    """
    
    def __init__(self):
        """Initialize response handler."""
        self._handlers: List[Callable] = []
        self._lock = threading.Lock()
    
    def add(self, handler: Callable) -> "ResponseHandler":
        """Add a response handler."""
        with self._lock:
            self._handlers.append(handler)
        return self
    
    def handle(self, context: ResponseContext) -> ResponseContext:
        """Run all handlers."""
        ctx = context
        for handler in self._handlers:
            try:
                ctx = handler(ctx)
            except Exception:
                pass
        return ctx


class ErrorHandler:
    """Convert error responses to exceptions."""
    
    def __init__(self, error_class: type = Exception):
        self.error_class = error_class
    
    def __call__(self, context: ResponseContext) -> ResponseContext:
        """Check for errors."""
        if context.status_code >= 400:
            message = f"HTTP {context.status_code}"
            if context.body and isinstance(context.body, dict):
                message = context.body.get("detail", context.body.get("error", message))
            raise self.error_class(message)
        return context


class JSONParser:
    """Parse JSON responses."""
    
    def __call__(self, context: ResponseContext) -> ResponseContext:
        """Parse JSON body."""
        if context.body and isinstance(context.body, str):
            try:
                import json
                context.body = json.loads(context.body)
            except (json.JSONDecodeError, ValueError):
                pass
        return context


class RetryHandler:
    """Handle retries based on response."""
    
    def __init__(
        self,
        interceptor: RetryInterceptor,
        on_retry: Optional[Callable] = None,
    ):
        self.interceptor = interceptor
        self.on_retry = on_retry
    
    def should_retry(self, context: ResponseContext, attempt: int) -> bool:
        """Check if should retry."""
        return self.interceptor.should_retry(context.status_code, attempt)
    
    def get_delay(self, attempt: int) -> float:
        """Get backoff delay."""
        return self.interceptor.get_delay(attempt)


def with_retry(
    max_retries: int = 3,
    backoff: float = 1.5,
    max_delay: float = 30.0,
    retry_on: Optional[List[int]] = None,
):
    """
    Decorator for retry logic.
    
    Example:
    
    ```python
    @with_retry(max_retries=3)
    def make_request():
        return requests.get(url)
    ```
    """
    retry_on = retry_on or [429, 500, 502, 503, 504]
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    response = func(*args, **kwargs)
                    
                    if response.status_code not in retry_on:
                        return response
                    
                    if attempt < max_retries - 1:
                        delay = min(backoff ** attempt, max_delay)
                        time.sleep(delay)
                        
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = min(backoff ** attempt, max_delay)
                        time.sleep(delay)
            
            if last_exception:
                raise last_exception
            
            return response
            
        return wrapper
    return decorator


def with_timeout(timeout: int = 30):
    """Decorator to add timeout to requests."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Request timed out after {timeout}s")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            
            return result
        return wrapper
    return decorator


def sanitize_request(func: Callable) -> Callable:
    """Decorator to sanitize request data."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        sanitized_kwargs = {
            k: Sanitizer.sanitize_body(v) 
            for k, v in kwargs.items()
        }
        return func(*args, **sanitized_kwargs)
    return wrapper


class HTTPClient:
    """
    HTTP client with request/response handling.
    
    Example:
    
    ```python
    from sloughgpt_sdk.http import HTTPClient, LoggingInterceptor, ErrorHandler
    
    client = HTTPClient(base_url="http://localhost:8000")
    client.interceptors.add(LoggingInterceptor())
    client.response_handlers.add(ErrorHandler())
    
    response = client.get("/health")
    ```
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        config: Optional[RequestConfig] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize HTTP client."""
        self.base_url = base_url.rstrip("/")
        self.config = config or RequestConfig()
        self.interceptors = RequestInterceptor()
        self.response_handlers = ResponseHandler()
        self._session = None
        
        if api_key:
            self.interceptors.add(AuthInterceptor(api_key))
        
        self.response_handlers.add(JSONParser())
        self.response_handlers.add(ErrorHandler())
    
    def _get_session(self):
        """Get or create session."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "SloughGPT-SDK/1.0"})
        return self._session
    
    def _create_context(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        body: Any = None,
    ) -> RequestContext:
        """Create request context."""
        headers = headers or {}
        return RequestContext(
            method=method.upper(),
            url=url,
            headers=headers,
            body=body,
            timestamp=time.time(),
        )
    
    def _create_response(
        self,
        request: RequestContext,
        status_code: int,
        headers: Dict[str, str],
        body: Any,
        elapsed_ms: float,
    ) -> ResponseContext:
        """Create response context."""
        return ResponseContext(
            status_code=status_code,
            headers=headers,
            body=body,
            elapsed_ms=elapsed_ms,
            request=request,
        )
    
    def request(
        self,
        method: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Any:
        """Make HTTP request with handling."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        context = self._create_context(method, url, headers, kwargs.get("json") or kwargs.get("data"))
        context = self.interceptors.intercept(context)
        
        session = self._get_session()
        session.headers.update(context.headers)
        
        start_time = time.time()
        
        retry_handler = RetryHandler(RetryInterceptor(
            max_retries=self.config.retry_count,
            backoff_factor=self.config.retry_backoff,
            max_delay=self.config.retry_max_delay,
        ))
        
        last_response = None
        
        for attempt in range(self.config.retry_count):
            context.attempt = attempt + 1
            
            try:
                response = session.request(
                    method=context.method,
                    url=context.url,
                    timeout=self.config.timeout,
                    verify=self.config.validate_ssl,
                    **kwargs
                )
                
                elapsed_ms = (time.time() - start_time) * 1000
                
                resp_context = self._create_response(
                    context,
                    response.status_code,
                    dict(response.headers),
                    response.text,
                    elapsed_ms,
                )
                
                resp_context = self.response_handlers.handle(resp_context)
                
                return resp_context.body
                
            except Exception as e:
                last_response = e
                
                if attempt < self.config.retry_count - 1:
                    delay = retry_handler.get_delay(attempt)
                    time.sleep(delay)
        
        if last_response:
            raise last_response
    
    def get(self, endpoint: str, **kwargs) -> Any:
        """GET request."""
        return self.request("GET", endpoint, **kwargs)
    
    def post(self, endpoint: str, **kwargs) -> Any:
        """POST request."""
        return self.request("POST", endpoint, **kwargs)
    
    def put(self, endpoint: str, **kwargs) -> Any:
        """PUT request."""
        return self.request("PUT", endpoint, **kwargs)
    
    def delete(self, endpoint: str, **kwargs) -> Any:
        """DELETE request."""
        return self.request("DELETE", endpoint, **kwargs)
    
    def patch(self, endpoint: str, **kwargs) -> Any:
        """PATCH request."""
        return self.request("PATCH", endpoint, **kwargs)
    
    def close(self):
        """Close the session."""
        if self._session:
            self._session.close()
            self._session = None
    
    def __enter__(self):
        """Context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()
