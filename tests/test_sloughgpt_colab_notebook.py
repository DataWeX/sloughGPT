"""Regression checks for Colab smoke: notebook, shell script, ``pyproject`` ``notebook`` extra, ``Makefile``, ``.gitignore`` rules (nested repo copy, ``data/vector_store/``), ``verify.sh`` hints.

Notebook blob asserts a single ``_asyncio_run`` / ``ThreadPoolExecutor`` import (§11 cognitive cell merged).

Mostly static checks; also runs ``bash scripts/run_colab_notebook_smoke.sh --help`` when bash is on ``PATH``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _colab_cell_sources(nb: dict) -> str:
    parts: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") not in ("code", "markdown", "raw"):
            continue
        src = cell.get("source", [])
        if isinstance(src, list):
            src = "".join(src)
        parts.append(str(src))
    return "\n".join(parts)


def test_makefile_colab_targets_present() -> None:
    mk = _repo_root() / "Makefile"
    assert mk.is_file(), f"missing {mk}"
    text = mk.read_text(encoding="utf-8")
    assert "colab-smoke:" in text
    assert "colab-test:" in text


def test_gitignore_excludes_nested_sloughgpt_copy() -> None:
    """Avoid committing an accidental clone-inside-clone tree (./sloughGPT/)."""
    text = (_repo_root() / ".gitignore").read_text(encoding="utf-8")
    assert "sloughGPT/" in text


def test_gitignore_excludes_data_vector_store() -> None:
    """Local Chroma / vector persistence should stay untracked."""
    text = (_repo_root() / ".gitignore").read_text(encoding="utf-8")
    assert "data/vector_store/" in text


def test_verify_sh_lists_colab_make_shortcuts() -> None:
    verify = _repo_root() / "scripts" / "verify.sh"
    assert verify.is_file(), f"missing {verify}"
    text = verify.read_text(encoding="utf-8")
    assert "make colab-test" in text
    assert "make help" in text


def test_pyproject_defines_notebook_optional_extra() -> None:
    """Keep ``pip install -e ".[notebook]"`` documented installs in sync with pyproject."""
    text = (_repo_root() / "pyproject.toml").read_text(encoding="utf-8")
    assert "notebook = [" in text
    for pkg in ("jupyter", "nbclient", "nbformat"):
        assert pkg in text, f"pyproject optional notebook extra should mention {pkg!r}"


def test_sloughgpt_colab_smoke_script_help_runs() -> None:
    root = _repo_root()
    script = root / "scripts" / "run_colab_notebook_smoke.sh"
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash not on PATH")
    proc = subprocess.run(
        [bash, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Usage: run_colab_notebook_smoke.sh" in proc.stdout
    assert "Git Bash" in proc.stdout or "WSL" in proc.stdout


def test_sloughgpt_colab_smoke_script_present() -> None:
    root = _repo_root()
    script = root / "scripts" / "run_colab_notebook_smoke.sh"
    assert script.is_file(), f"missing {script}"
    text = script.read_text(encoding="utf-8")
    assert "Usage: run_colab_notebook_smoke.sh" in text
    assert "--help" in text
    assert "nbconvert" in text
    assert "python3 -m nbconvert" in text or "jupyter nbconvert" in text
    assert "SLOUGH_NOTEBOOK_TRAIN_CAP" in text
    assert "WSL" in text or "Git Bash" in text


def test_sloughgpt_colab_notebook_is_valid_nbformat_v4_with_smoke_hooks() -> None:
    path = _repo_root() / "sloughgpt_colab.ipynb"
    assert path.is_file(), f"missing {path}"

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("nbformat") == 4
    assert len(data.get("cells", [])) >= 20

    blob = _colab_cell_sources(data)
    assert "run_colab_notebook_smoke.sh" in blob
    assert "make colab-smoke" in blob
    assert "make colab-test" in blob
    assert "make help" in blob
    assert "--help" in blob
    assert "nbclient nbformat" in blob
    assert ".[notebook]" in blob
    assert "_ensure_sloughgpt_repo" in blob
    assert "SLOUGH_NOTEBOOK_TRAIN_CAP" in blob
    assert "SLOUGH_NOTEBOOK_FORCE_CPU" in blob
    assert "ThreadPoolExecutor" in blob
    assert "_asyncio_run" in blob
    assert blob.count("def _asyncio_run(coro):") == 1
    assert blob.count("from concurrent.futures import ThreadPoolExecutor") == 1
