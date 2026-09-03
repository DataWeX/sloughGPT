"""Tests for singleton pattern — thread-safe singletons."""
from __future__ import annotations

import threading

from domains.infrastructure.singleton import SingletonMeta, make_singleton


class TestMakeSingleton:
    def test_returns_same_instance(self):
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"value": call_count}

        get = make_singleton(factory)
        a = get()
        b = get()
        assert a is b
        assert call_count == 1

    def test_thread_safety(self):
        instances = []

        def factory():
            return object()

        get = make_singleton(factory)

        def worker():
            instances.append(get())

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should be the same instance
        assert len(set(id(i) for i in instances)) == 1


class TestSingletonMeta:
    def test_same_instance(self):
        class MyClass(metaclass=SingletonMeta):
            def __init__(self, value):
                self.value = value

        SingletonMeta.clear_instance(MyClass)
        a = MyClass(1)
        b = MyClass(2)
        assert a is b
        assert a.value == 1  # First call wins

    def test_different_classes_different_instances(self):
        class A(metaclass=SingletonMeta):
            pass

        class B(metaclass=SingletonMeta):
            pass

        SingletonMeta.clear_instance(A)
        SingletonMeta.clear_instance(B)
        a = A()
        b = B()
        assert a is not b

    def test_clear_instance(self):
        class MyClass(metaclass=SingletonMeta):
            def __init__(self, value):
                self.value = value

        SingletonMeta.clear_instance(MyClass)
        a = MyClass(1)
        SingletonMeta.clear_instance(MyClass)
        b = MyClass(2)
        assert a is not b
        assert b.value == 2

    def test_thread_safety(self):
        class MyClass(metaclass=SingletonMeta):
            def __init__(self):
                self.created = True

        SingletonMeta.clear_instance(MyClass)
        instances = []

        def worker():
            instances.append(MyClass())

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(id(i) for i in instances)) == 1
