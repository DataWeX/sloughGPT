import importlib.util as _importlib_util
import sys as _sys
from pathlib import Path as _Path


def _preload_core_tests_conftest():
    """Ensure `_core_tests_conftest` in sys.modules (deterministic import).

    Several directories are named `tests/` and several contain
    `conftest.py`, so bare `from tests.conftest import X` / `from conftest
    import X` resolve to whichever portion wins sys.path order — a
    different module depending on collection (and on which pytest.ini
    sets the rootdir). This file loads before any test module under
    packages/core-py in every rootdir mode, so registering the canonical
    helper here makes `from _core_tests_conftest import ...` resolve
    identically everywhere.
    """
    if "_core_tests_conftest" in _sys.modules:
        return
    path = _Path(__file__).resolve().parent / "tests" / "conftest.py"
    spec = _importlib_util.spec_from_file_location("_core_tests_conftest", path)
    module = _importlib_util.module_from_spec(spec)
    _sys.modules["_core_tests_conftest"] = module
    spec.loader.exec_module(module)


_preload_core_tests_conftest()
del _preload_core_tests_conftest

collect_ignore = [
    "tests/test_numpy_engine.py",
    "tests/test_point_library.py",
    "tests/test_safetensors_loader.py",
    "tests/test_morph_tokenizer.py",
    "tests/test_neural_e2e.py",
]

from _core_tests_conftest import build_test_app as build_test_app  # noqa: F401
