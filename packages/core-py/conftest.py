collect_ignore = [
    "tests/test_numpy_engine.py",
    "tests/test_point_library.py",
    "tests/test_safetensors_loader.py",
    "tests/test_morph_tokenizer.py",
    "tests/test_neural_e2e.py",
]

import importlib.util
from pathlib import Path as _Path

# Load the REPO-ROOT tests/conftest.py by path: several packages ship their
# own namespace `tests/` directory, so `from tests.conftest import ...` can
# resolve to the wrong one (e.g. downcraft's, which lacks build_test_app).
_root_conftest = _Path(__file__).resolve().parents[2] / "tests" / "conftest.py"
_spec = importlib.util.spec_from_file_location("_root_tests_conftest", _root_conftest)
assert _spec is not None and _spec.loader is not None
_root_conftest_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_conftest_mod)

build_test_app = _root_conftest_mod.build_test_app  # noqa: F401
