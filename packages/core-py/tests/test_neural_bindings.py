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
