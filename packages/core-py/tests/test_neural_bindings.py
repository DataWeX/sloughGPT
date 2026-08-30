"""Tests for domains.shell.addons.neural_bindings — Property descriptor."""

import pytest
from domains.shell.addons.neural_bindings import Property


class TestProperty:
    def test_set_name(self):
        class Obj:
            x = Property("_x")

        p = Obj.__dict__["x"]
        assert p._name == "x"

    def test_get_requires_addon(self):
        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_x", require_addon=True)

        obj = Obj()
        obj._x = 42
        assert obj.x == 42

    def test_get_no_require(self):
        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_x", require_addon=False)

        obj = Obj()
        obj._x = 99
        assert obj.x == 99

    def test_get_missing_returns_none(self):
        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_x")

        obj = Obj()
        assert obj.x is None

    def test_get_default_factory(self):
        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_x", default_factory=lambda: [1, 2, 3])

        obj = Obj()
        assert obj.x == [1, 2, 3]

    def test_get_class_returns_descriptor(self):
        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_x")

        assert isinstance(Obj.x, Property)

    def test_get_none_obj_returns_descriptor(self):
        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_x")

        assert isinstance(Obj.__dict__["x"].__get__(None, Obj), Property)

    def test_default_factory_called_each_access(self):
        call_count = 0
        def factory():
            nonlocal call_count
            call_count += 1
            return []

        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_x", default_factory=factory)

        obj = Obj()
        _ = obj.x
        _ = obj.x
        assert call_count == 2

    def test_default_factory_not_used_when_set(self):
        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_x", default_factory=lambda: [1])

        obj = Obj()
        obj._x = [99]
        assert obj.x == [99]

    def test_different_instances_different_values(self):
        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_x")

        obj1 = Obj()
        obj2 = Obj()
        obj1._x = 10
        obj2._x = 20
        assert obj1.x == 10
        assert obj2.x == 20

    def test_require_addon_called(self):
        calls = []
        class Obj:
            _require_addon = lambda self, name: calls.append(name)
            x = Property("_x", require_addon=True)

        obj = Obj()
        obj._x = 42
        _ = obj.x
        assert calls == ["neural"]

    def test_require_addon_not_called_when_false(self):
        calls = []
        class Obj:
            _require_addon = lambda self, name: calls.append(name)
            x = Property("_x", require_addon=False)

        obj = Obj()
        obj._x = 42
        _ = obj.x
        assert calls == []

    def test_multiple_properties(self):
        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_x")
            y = Property("_y")
            z = Property("_z")

        obj = Obj()
        obj._x = 1
        obj._y = 2
        obj._z = 3
        assert obj.x == 1
        assert obj.y == 2
        assert obj.z == 3

    def test_attr_name_underscore_prefix(self):
        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_internal_x")

        obj = Obj()
        obj._internal_x = 42
        assert obj.x == 42

    def test_none_value_returned(self):
        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_x")

        obj = Obj()
        obj._x = None
        assert obj.x is None

    def test_false_value_with_default_factory(self):
        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_x", default_factory=lambda: [1])

        obj = Obj()
        obj._x = False
        # False is falsy, so default_factory is used
        assert obj.x == [1]

    def test_zero_with_default_factory(self):
        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_x", default_factory=lambda: [1])

        obj = Obj()
        obj._x = 0
        # 0 is falsy, so default_factory is used
        assert obj.x == [1]

    def test_empty_string_with_default_factory(self):
        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_x", default_factory=lambda: [1])

        obj = Obj()
        obj._x = ""
        # Empty string is falsy, so default_factory is used
        assert obj.x == [1]

    def test_empty_list_with_default_factory(self):
        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_x", default_factory=lambda: [1])

        obj = Obj()
        obj._x = []
        # Empty list is falsy, so default_factory is used
        assert obj.x == [1]

    def test_set_new_value(self):
        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_x")

        obj = Obj()
        obj._x = 42
        obj._x = 100
        assert obj.x == 100

    def test_descriptor_on_class_level(self):
        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_x")

        assert isinstance(Obj.__dict__["x"], Property)
        assert Obj.__dict__["x"]._attr == "_x"

    def test_inheritance(self):
        class Base:
            _require_addon = lambda self, name: None
            x = Property("_x")

        class Child(Base):
            pass

        obj = Child()
        obj._x = 42
        assert obj.x == 42

    def test_overridden_require_addon(self):
        calls = []
        class Base:
            _require_addon = lambda self, name: calls.append("base")
            x = Property("_x", require_addon=True)

        class Child(Base):
            _require_addon = lambda self, name: calls.append("child")

        obj = Child()
        obj._x = 42
        _ = obj.x
        assert calls == ["child"]

    def test_property_with_numpy_array(self):
        import numpy as np
        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_x")

        obj = Obj()
        arr = np.array([1, 2, 3])
        obj._x = arr
        assert (obj.x == arr).all()

    def test_property_with_dict(self):
        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_x")

        obj = Obj()
        d = {"key": "value", "nested": [1, 2, 3]}
        obj._x = d
        assert obj.x == d

    def test_property_with_complex_default_factory(self):
        def complex_factory():
            return {"level1": {"level2": [1, 2, 3]}}

        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_x", default_factory=complex_factory)

        obj = Obj()
        result = obj.x
        assert result["level1"]["level2"] == [1, 2, 3]

    def test_multiple_instances_independent(self):
        class Obj:
            _require_addon = lambda self, name: None
            x = Property("_x")

        instances = [Obj() for _ in range(5)]
        for i, inst in enumerate(instances):
            inst._x = i * 10
        
        for i, inst in enumerate(instances):
            assert inst.x == i * 10

    def test_set_name_multiple_classes(self):
        class A:
            x = Property("_x")
        
        class B:
            x = Property("_x")

        assert A.__dict__["x"]._name == "x"
        assert B.__dict__["x"]._name == "x"
