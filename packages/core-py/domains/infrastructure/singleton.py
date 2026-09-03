"""
Singleton metaclass for thread-safe singleton pattern.

Provides a metaclass that ensures only one instance of a class exists,
with thread-safe double-checked locking.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")


def make_singleton(factory: Callable[[], T]) -> Callable[[], T]:
    """Create a thread-safe singleton getter from a factory function.

    Usage:
        def _create_db():
            return Database()

        get_db = make_singleton(_create_db)
        db = get_db()  # Thread-safe, returns same instance
    """
    _instance: Optional[T] = None
    _lock = threading.Lock()

    def getter() -> T:
        nonlocal _instance
        if _instance is None:
            with _lock:
                if _instance is None:
                    _instance = factory()
        return _instance

    return getter


class SingletonMeta(type):
    """Metaclass that implements the singleton pattern with thread safety.

    Usage:
        class MyClass(metaclass=SingletonMeta):
            def __init__(self, value: int):
                self.value = value

        a = MyClass(1)
        b = MyClass(2)
        assert a is b  # Same instance
        assert a.value == 1  # First call's args are used
    """
    _instances: Dict[type, Any] = {}
    _locks: Dict[type, threading.Lock] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            lock = cls._locks.setdefault(cls, threading.Lock())
            with lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

    @classmethod
    def clear_instance(mcs, cls: type) -> None:
        """Remove the cached instance for a class (for testing)."""
        with mcs._locks.setdefault(cls, threading.Lock()):
            mcs._instances.pop(cls, None)
