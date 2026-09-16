collect_ignore = [
    "tests/test_numpy_engine.py",
    "tests/test_point_library.py",
    "tests/test_safetensors_loader.py",
    "tests/test_morph_tokenizer.py",
    "tests/test_neural_e2e.py",
]

from tests.conftest import build_test_app as build_test_app  # noqa: F401
