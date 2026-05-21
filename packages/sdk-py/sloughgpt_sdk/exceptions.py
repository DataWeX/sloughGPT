"""
SloughGPT SDK - Exceptions
Custom exceptions for the SloughGPT SDK.
"""


class SloughGPTError(Exception):
    """Base exception for SloughGPT SDK."""
    
    def __init__(self, message: str, code: int = 0, details: dict = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
    
    def __str__(self):
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message


class APIError(SloughGPTError):
    """Exception raised for API errors."""
    
    def __init__(self, message: str, status_code: int = 0, response: dict = None):
        super().__init__(message, code=status_code)
        self.status_code = status_code
        self.response = response


class AuthenticationError(SloughGPTError):
    """Exception raised for authentication failures."""
    pass


class RateLimitError(SloughGPTError):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60):
        super().__init__(message, code=429)
        self.retry_after = retry_after


class ValidationError(SloughGPTError):
    """Exception raised for request validation errors."""
    pass


class TimeoutError(SloughGPTError):
    """Exception raised when request times out."""
    pass


class ConnectionError(SloughGPTError):
    """Exception raised for connection errors."""
    pass


class ModelNotFoundError(SloughGPTError):
    """Exception raised when model is not found."""
    pass


class ModelNotLoadedError(SloughGPTError):
    """Exception raised when model is not loaded."""
    pass


class CacheError(SloughGPTError):
    """Exception raised for cache errors."""
    pass
