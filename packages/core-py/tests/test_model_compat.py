"""Tests for model_compat — universal model adapter."""

import pytest
from domains.infrastructure.model_compat import (
    wrap_model,
    detect_model_type,
    ModelType,
    UniversalModel,
)


@pytest.fixture(scope="session")
def engine():
    """Load NumpyEngine once for entire test session."""
    from domains.infrastructure.numpy_engine import NumpyEngine
    return NumpyEngine.from_pretrained("gpt2")


@pytest.fixture(scope="session")
def wrapped(engine):
    """Wrap engine once for entire test session."""
    return wrap_model(engine, model_id="gpt2")


class TestDetectModelType:
    """Type detection tests."""

    def test_none_is_unknown(self):
        assert detect_model_type(None) == ModelType.UNKNOWN

    def test_numpy_engine(self, engine):
        assert detect_model_type(engine) == ModelType.NUMPY_ENGINE

    def test_unknown_object(self):
        assert detect_model_type("not a model") == ModelType.UNKNOWN

    def test_dict_is_unknown(self):
        assert detect_model_type({}) == ModelType.UNKNOWN


class TestWrapModel:
    """Wrap model tests."""

    def test_numpy_engine_wrapped(self, wrapped):
        assert wrapped.model_type == ModelType.NUMPY_ENGINE
        assert hasattr(wrapped, "generate_text")
        assert hasattr(wrapped, "generate_stream")
        assert hasattr(wrapped, "info")

    def test_numpy_engine_generate_text(self, wrapped):
        result = wrapped.generate_text("Hello", max_new_tokens=5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_numpy_engine_info(self, wrapped):
        info = wrapped.info()
        assert "arch" in info
        assert info["arch"] == "GPT2LMHeadModel"

    def test_unknown_model_uses_generic(self):
        wrapped = wrap_model("not a model", model_id="test")
        assert wrapped.model_type == ModelType.UNKNOWN
        assert hasattr(wrapped, "generate_text")

    def test_numpy_engine_stream(self, wrapped):
        tokens = list(wrapped.generate_stream("Hello", max_new_tokens=3))
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)


class TestModelType:
    """ModelType enum tests."""

    def test_all_values_unique(self):
        values = [t.value for t in ModelType]
        assert len(values) == len(set(values))

    def test_numpy_engine_exists(self):
        assert ModelType.NUMPY_ENGINE.value == "numpy_engine"
