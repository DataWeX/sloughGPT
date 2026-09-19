import importlib.util as _importlib_util
from pathlib import Path as _Path


def _load_root_build_test_app():
    """Load build_test_app from the repo-root tests/conftest.py by path.

    A plain ``from tests.conftest import ...`` is unreliable: ``tests`` is
    a namespace package spanning tests/, packages/*/tests/, and the first
    portion on sys.path wins (often packages/downcraft/tests, which has
    no build_test_app). Path-explicit loading is deterministic.
    """
    conftest_path = _Path(__file__).resolve().parents[2] / "tests" / "conftest.py"
    spec = _importlib_util.spec_from_file_location(
        "_root_tests_conftest", conftest_path
    )
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_test_app


collect_ignore = [
    "tests/test_numpy_engine.py",
    "tests/test_point_library.py",
    "tests/test_safetensors_loader.py",
    "tests/test_morph_tokenizer.py",
    "tests/test_neural_e2e.py",
]

build_test_app = _load_root_build_test_app()
del _load_root_build_test_app
