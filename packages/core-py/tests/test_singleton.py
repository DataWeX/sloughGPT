"""Tests for singleton utilities."""

import threading
import time

from domains.infrastructure.singleton import make_singleton, SingletonMeta


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
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            instances.append(get())

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(instances) == 10
        assert all(i is instances[0] for i in instances)

    def test_factory_called_once_with_different_types(self):
        get_int = make_singleton(lambda: 42)
        get_str = make_singleton(lambda: "hello")
        get_list = make_singleton(lambda: [1, 2, 3])

        assert get_int() == 42
        assert get_str() == "hello"
        assert get_list() == [1, 2, 3]
        assert get_int() is get_int()

    def test_factory_receives_no_args(self):
        call_args = []

        def factory():
            call_args.append(1)
            return "result"

        get = make_singleton(factory)
        get()
        get()
        assert len(call_args) == 1

    def test_factory_returning_none_not_cached(self):
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return None

        get = make_singleton(factory)
        a = get()
        b = get()
        assert a is None
        assert b is None
        assert call_count == 2

    def test_factory_returning_false(self):
        get = make_singleton(lambda: False)
        assert get() is False
        assert get() is False

    def test_factory_returning_zero(self):
        get = make_singleton(lambda: 0)
        assert get() == 0
        assert get() == 0

    def test_factory_returning_empty_string(self):
        get = make_singleton(lambda: "")
        assert get() == ""

    def test_factory_returning_empty_dict(self):
        get = make_singleton(lambda: {})
        a = get()
        b = get()
        assert a is b
        assert a == {}

    def test_factory_returning_empty_list(self):
        get = make_singleton(lambda: [])
        a = get()
        b = get()
        assert a is b
        assert a == []

    def test_multiple_singletons_independent(self):
        get_a = make_singleton(lambda: {"a": 1})
        get_b = make_singleton(lambda: {"b": 2})

        a = get_a()
        b = get_b()
        assert a is not b
        assert a == {"a": 1}
        assert b == {"b": 2}

    def test_singleton_with_complex_object(self):
        class Complex:
            def __init__(self):
                self.state = "initialized"

        get = make_singleton(Complex)
        obj = get()
        assert obj.state == "initialized"
        obj.state = "modified"
        assert get().state == "modified"

    def test_factory_exception_not_cached(self):
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("first call fails")
            return "ok"

        get = make_singleton(factory)
        try:
            get()
        except ValueError:
            pass
        result = get()
        assert result == "ok"
        assert call_count == 2

    def test_factory_called_at_most_once(self):
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return call_count

        get = make_singleton(factory)
        for _ in range(100):
            get()
        assert call_count == 1

    def test_getter_is_callable(self):
        get = make_singleton(lambda: 42)
        assert callable(get)

    def test_concurrent_reads_no_lock_contention(self):
        results = []

        def factory():
            time.sleep(0.01)
            return "done"

        get = make_singleton(factory)

        def worker():
            results.append(get())

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert all(r == "done" for r in results)

    def test_singleton_with_threading_lock_factory(self):
        lock = threading.Lock()
        get = make_singleton(lambda: lock)
        assert get() is lock

    def test_factory_side_effect_once(self):
        side_effects = []

        def factory():
            side_effects.append("created")
            return "value"

        get = make_singleton(factory)
        get()
        get()
        get()
        assert side_effects == ["created"]

    def test_factory_with_closure_state(self):
        counter = [0]

        def factory():
            counter[0] += 1
            return counter[0]

        get = make_singleton(factory)
        assert get() == 1
        assert get() == 1
        assert counter[0] == 1


class TestSingletonMeta:
    def test_same_instance(self):
        class MyClass(metaclass=SingletonMeta):
            def __init__(self, value):
                self.value = value

        a = MyClass(1)
        b = MyClass(2)
        assert a is b
        assert a.value == 1

    def test_clear_instance(self):
        class MyClass(metaclass=SingletonMeta):
            def __init__(self, value):
                self.value = value

        a = MyClass(1)
        SingletonMeta.clear_instance(MyClass)
        b = MyClass(2)
        assert a is not b
        assert b.value == 2

    def test_different_classes_independent(self):
        class A(metaclass=SingletonMeta):
            pass

        class B(metaclass=SingletonMeta):
            pass

        a = A()
        b = B()
        assert a is not b

    def test_no_init_class(self):
        class Simple(metaclass=SingletonMeta):
            pass

        a = Simple()
        b = Simple()
        assert a is b

    def test_init_with_kwargs(self):
        class Config(metaclass=SingletonMeta):
            def __init__(self, host="localhost", port=8080):
                self.host = host
                self.port = port

        c = Config(host="example.com", port=443)
        c2 = Config()
        assert c is c2
        assert c.host == "example.com"
        assert c.port == 443

    def test_init_with_default_only(self):
        class Settings(metaclass=SingletonMeta):
            def __init__(self, debug=False):
                self.debug = debug

        s = Settings()
        s2 = Settings(debug=True)
        assert s is s2
        assert s.debug is False

    def test_clear_and_recreate(self):
        class Service(metaclass=SingletonMeta):
            def __init__(self, name):
                self.name = name

        s1 = Service("first")
        SingletonMeta.clear_instance(Service)
        s2 = Service("second")
        assert s1 is not s2
        assert s2.name == "second"

    def test_clear_nonexistent_class(self):
        class Fresh(metaclass=SingletonMeta):
            pass

        SingletonMeta.clear_instance(Fresh)
        obj = Fresh()
        assert obj is not None

    def test_inheritance_not_singleton(self):
        class Base(metaclass=SingletonMeta):
            def __init__(self):
                self.kind = "base"

        class Child(Base):
            def __init__(self):
                super().__init__()
                self.kind = "child"

        base = Base()
        child = Child()
        assert base is not child

    def test_class_with_properties(self):
        class WithProps(metaclass=SingletonMeta):
            def __init__(self, x):
                self._x = x

            @property
            def x(self):
                return self._x

        obj = WithProps(42)
        obj2 = WithProps(99)
        assert obj is obj2
        assert obj.x == 42

    def test_class_with_classmethod(self):
        class WithClassMethod(metaclass=SingletonMeta):
            _count = 0

            def __init__(self):
                WithClassMethod._count += 1

            @classmethod
            def instance_count(cls):
                return cls._count

        a = WithClassMethod()
        b = WithClassMethod()
        assert a is b
        assert WithClassMethod.instance_count() == 1

    def test_class_with_staticmethod(self):
        class WithStatic(metaclass=SingletonMeta):
            @staticmethod
            def helper():
                return 42

        obj = WithStatic()
        assert WithStatic.helper() == 42

    def test_class_with_del(self):
        deleted = []

        class WithDel(metaclass=SingletonMeta):
            def __init__(self, v):
                self.v = v
            def __del__(self):
                deleted.append(self.v)

        a = WithDel(1)
        SingletonMeta.clear_instance(WithDel)
        b = WithDel(2)
        assert a is not b
        assert b.v == 2

    def test_multiple_clears(self):
        class Multi(metaclass=SingletonMeta):
            def __init__(self, v):
                self.v = v

        for i in range(5):
            SingletonMeta.clear_instance(Multi)
            obj = Multi(i)
            assert obj.v == i

    def test_thread_safety(self):
        instances = []

        class Threaded(metaclass=SingletonMeta):
            pass

        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            instances.append(Threaded())

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(instances) == 10
        assert all(i is instances[0] for i in instances)

    def test_clear_during_concurrent_access(self):
        class Concurrent(metaclass=SingletonMeta):
            def __init__(self, v):
                self.v = v

        results = []
        errors = []

        def creator(val):
            try:
                results.append(Concurrent(val))
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=creator, args=(1,))
        t2 = threading.Thread(target=lambda: SingletonMeta.clear_instance(Concurrent))
        t3 = threading.Thread(target=creator, args=(2,))

        t1.start()
        t2.start()
        t3.start()
        t1.join()
        t2.join()
        t3.join()

        assert len(results) == 2
        assert errors == []

    def test_isinstance_check(self):
        class Base:
            pass

        class Singleton(Base, metaclass=SingletonMeta):
            pass

        obj = Singleton()
        assert isinstance(obj, Base)
        assert isinstance(obj, Singleton)

    def test_repr(self):
        class Reprable(metaclass=SingletonMeta):
            def __repr__(self):
                return "Reprable()"

        obj = Reprable()
        assert repr(obj) == "Reprable()"

    def test_eq_same_class(self):
        class EqTest(metaclass=SingletonMeta):
            pass

        a = EqTest()
        b = EqTest()
        assert a == b

    def test_hash_consistent(self):
        class Hashable(metaclass=SingletonMeta):
            pass

        obj = Hashable()
        assert hash(obj) == hash(obj)

    def test_context_manager_pattern(self):
        class Contextual(metaclass=SingletonMeta):
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        with Contextual() as ctx:
            with Contextual() as ctx2:
                assert ctx is ctx2

    def test_delete_attribute(self):
        class WithAttr(metaclass=SingletonMeta):
            def __init__(self):
                self.temp = "value"

        obj = WithAttr()
        assert obj.temp == "value"
        del obj.temp
        assert not hasattr(obj, "temp")

    def test_class_vars_shared(self):
        class Shared(metaclass=SingletonMeta):
            class_var = "shared"

        a = Shared()
        Shared.class_var = "changed"
        b = Shared()
        assert a.class_var == "changed"
        assert b.class_var == "changed"

    def test_module_level_singletons_isolated(self):
        class ModA(metaclass=SingletonMeta):
            pass

        class ModB(metaclass=SingletonMeta):
            pass

        a1 = ModA()
        b1 = ModB()
        assert a1 is not b1

    def test_clear_preserves_class(self):
        class Preserved(metaclass=SingletonMeta):
            def __init__(self, v):
                self.v = v

        Preserved(10)
        SingletonMeta.clear_instance(Preserved)
        obj = Preserved(20)
        assert obj.v == 20

    def test_kwargs_first_call_sticks(self):
        class Kwargs(metaclass=SingletonMeta):
            def __init__(self, x=0, y=0):
                self.x = x
                self.y = y

        obj = Kwargs(x=1, y=2)
        obj2 = Kwargs(x=99, y=99)
        assert obj is obj2
        assert obj.x == 1
        assert obj.y == 2

    def test_class_with_slots(self):
        class Slotted(metaclass=SingletonMeta):
            __slots__ = ("val",)
            def __init__(self, val):
                self.val = val

        a = Slotted(42)
        b = Slotted(99)
        assert a is b
        assert a.val == 42

    def test_class_with_delattr(self):
        class DelAttr(metaclass=SingletonMeta):
            def __init__(self):
                self.data = {"a": 1}

        obj = DelAttr()
        del obj.data
        assert not hasattr(obj, "data")

    def test_class_with_setattr(self):
        class SetAttr(metaclass=SingletonMeta):
            pass

        obj = SetAttr()
        obj.dynamic = "added"
        assert obj.dynamic == "added"
        obj2 = SetAttr()
        assert obj2.dynamic == "added"

    def test_class_with_iter(self):
        class Iterable(metaclass=SingletonMeta):
            def __init__(self):
                self.items = [1, 2, 3]
            def __iter__(self):
                return iter(self.items)

        obj = Iterable()
        assert list(obj) == [1, 2, 3]
        obj2 = Iterable()
        assert list(obj2) == [1, 2, 3]
        assert obj is obj2

    def test_class_with_len(self):
        class Sized(metaclass=SingletonMeta):
            def __init__(self):
                self.data = [1, 2, 3, 4, 5]
            def __len__(self):
                return len(self.data)

        obj = Sized()
        assert len(obj) == 5
        obj2 = Sized()
        assert obj is obj2

    def test_class_with_getitem(self):
        class Subscriptable(metaclass=SingletonMeta):
            def __init__(self):
                self.data = {"a": 1, "b": 2}
            def __getitem__(self, key):
                return self.data[key]

        obj = Subscriptable()
        assert obj["a"] == 1
        obj2 = Subscriptable()
        assert obj2["b"] == 2
        assert obj is obj2

    def test_class_with_call(self):
        class Callable(metaclass=SingletonMeta):
            def __init__(self):
                self.call_count = 0
            def __call__(self, x):
                self.call_count += 1
                return x * 2

        obj = Callable()
        assert obj(5) == 10
        assert obj.call_count == 1
        obj2 = Callable()
        assert obj2(3) == 6
        assert obj2.call_count == 2
        assert obj is obj2
