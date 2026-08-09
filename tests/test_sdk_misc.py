"""Coverage for sloughgpt_sdk.exceptions and sloughgpt_sdk.setup."""
import importlib.util
import io
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "sdk-py"))

from sloughgpt_sdk.exceptions import (  # noqa: E402
    APIError,
    AuthenticationError,
    CacheError,
    ConnectionError as SDKConnectionError,
    ModelNotFoundError,
    ModelNotLoadedError,
    RateLimitError,
    SloughGPTError,
    TimeoutError as SDKTimeoutError,
    ValidationError,
)


def _setup_root() -> Path:
    return _REPO_ROOT / "packages" / "sdk-py" / "sloughgpt_sdk"


class TestSloughGPTError:
    def test_str_plain(self):
        e = SloughGPTError("boom")
        assert str(e) == "boom"
        assert e.message == "boom"
        assert e.code == 0
        assert e.details == {}

    def test_str_with_code(self):
        assert str(SloughGPTError("boom", code=500)) == "[500] boom"

    def test_details_passed_through(self):
        assert SloughGPTError("boom", details={"x": 1}).details == {"x": 1}

    def test_details_none_becomes_empty(self):
        assert SloughGPTError("boom", details=None).details == {}

    def test_is_exception(self):
        assert issubclass(SloughGPTError, Exception)


class TestAPIError:
    def test_attributes(self):
        e = APIError("bad request", status_code=400, response={"err": "x"})
        assert e.status_code == 400
        assert e.response == {"err": "x"}
        assert e.code == 400
        assert e.message == "bad request"

    def test_response_defaults_none(self):
        assert APIError("bad request", status_code=500).response is None


class TestDerivedSloughGPTErrors:
    def test_subclass_relationship(self):
        for cls in (
            APIError,
            AuthenticationError,
            RateLimitError,
            ValidationError,
            SDKTimeoutError,
            SDKConnectionError,
            ModelNotFoundError,
            ModelNotLoadedError,
            CacheError,
        ):
            assert issubclass(cls, SloughGPTError), cls

    def test_rate_limit_defaults(self):
        e = RateLimitError()
        assert e.retry_after == 60
        assert e.code == 429
        assert str(e) == "[429] Rate limit exceeded"

    def test_rate_limit_custom(self):
        e = RateLimitError("slow down", retry_after=5)
        assert e.retry_after == 5
        assert e.message == "slow down"

    def test_base_style_raising_on_empty_subclass(self):
        e = ModelNotFoundError("missing")
        assert str(e) == "missing"
        assert e.code == 0

    def test_type_naming_readable(self):
        assert SDKConnectionError.__name__ == "ConnectionError"
        assert SDKTimeoutError.__name__ == "TimeoutError"


class TestSetupModule:
    @pytest.fixture
    def fake_setuptools(self):
        captured = {}

        class FakeSetuptools:
            def setup(self, **kwargs):
                captured.update(kwargs)

            def find_packages(self, *args, **kwargs):
                return []

        fake = FakeSetuptools()
        return fake, captured

    def test_setup_called_with_expected_metadata(self, fake_setuptools):
        fake, captured = fake_setuptools
        readme_sentinel = io.StringIO("# README")

        setup_path = _setup_root() / "setup.py"
        with (
            patch("builtins.open", return_value=readme_sentinel) as mopen,
            patch.dict(sys.modules, {"setuptools": fake}, clear=False),
        ):
            spec = importlib.util.spec_from_file_location("sdk_setup_probe", setup_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)

        assert captured["name"] == "sloughgpt-sdk"
        assert captured["version"] == "1.1.0"
        assert captured["python_requires"].startswith(">=")
        assert "requests>=2.25.0" in captured["install_requires"]
        assert captured["entry_points"]["console_scripts"] == [
            "sloughgpt-cli=sloughgpt_sdk.cli:main",
        ]
        assert captured["package_data"] == {"sloughgpt_sdk": ["py.typed"]}
        assert captured["long_description"] == "# README"
        assert captured["long_description_content_type"] == "text/markdown"

    def test_setup_reads_readme_utf8(self, fake_setuptools):
        fake, _ = fake_setuptools
        setup_path = _setup_root() / "setup.py"

        with (
            patch("builtins.open", return_value=io.StringIO("x")) as mopen,
            patch.dict(sys.modules, {"setuptools": fake}, clear=False),
        ):
            spec = importlib.util.spec_from_file_location("sdk_setup_probe2", setup_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)

        open_args = mopen.call_args.args[0]
        assert str(open_args).endswith("README.md")
        assert mopen.call_args.kwargs["encoding"] == "utf-8"