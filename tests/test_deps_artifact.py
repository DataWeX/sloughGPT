"""Engineering-library landscape artifact — generator, freshness, grouping (viz-2, card 20261111_110106).

The library inventory is a BUILD-TIME read: manifests are read in the same
process as the build, so no HTTP endpoint is earned (boundary test: same
process -> function call). This gates ``scripts/gen_deps_artifact.py`` the way
``tests/contract/test_ts_contracts.py`` gates ``gen_ts_contracts``:

1. **Freshness** — the committed artifact matches what the manifests produce
   right now (``--check`` in a subprocess: a fresh interpreter reads exactly
   what a build would).
2. **Determinism** — the same tree yields byte-identical output, or the
   freshness gate would flap on ordering alone.
3. **The gate discriminates** — change a manifest and the output changes; a
   gate that cannot fail is not a gate.
4. **Grouping is honest** — duplicates (same package, >1 manifest), drift
   (same package, >1 declared version) and cross-tree presence are computed
   from normalized names, never from string equality on raw specs.
5. **Generator owns the bytes** — every output is in ``.prettierignore`` so
   lint-staged cannot requote the artifact and turn the gate red.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gen_deps_artifact.py"

sys.path.insert(0, str(ROOT / "scripts"))
import gen_deps_artifact as gen  # noqa: E402


# ── freshness on the real tree ─────────────────────────────────────────


def test_artifact_is_fresh_on_this_tree():
    """The committed artifact must equal what the manifests produce right now."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"stale artifact — run `python scripts/gen_deps_artifact.py`\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )


def test_build_is_deterministic():
    """Same tree, byte-identical output — ordering must not flap the gate."""
    assert gen.build(ROOT) == gen.build(ROOT)


def test_artifact_shape_is_stable():
    artifact = json.loads(gen.build(ROOT).split("export const DEPS: DepsArtifact = ", 1)[1])
    assert set(artifact) == {"manifests", "entries", "duplicates", "drift", "cross_tree"}
    assert artifact["manifests"], "no manifests read — the collector is broken"
    for entry in artifact["entries"]:
        assert set(entry) == {"manifest", "name", "key", "tree", "kind", "version"}
        assert entry["tree"] in ("python", "js")
        assert entry["kind"] in ("runtime", "optional", "dev", "peer")
    # every grouped entry must exist flat, and grouping must be by normalized key
    flat = {(e["manifest"], e["key"], e["version"]) for e in artifact["entries"]}
    for group in artifact["duplicates"] + artifact["drift"]:
        assert all((e["manifest"], e["key"], e["version"]) in flat for e in group["entries"])


def test_prettierignore_covers_every_output():
    """lint-staged would requote the generator's bytes and flap the gate."""
    ignore = (ROOT / ".prettierignore").read_text(encoding="utf-8")
    for out in gen.OUTPUTS:
        assert out.relative_to(ROOT).as_posix() in ignore, f"{out} missing from .prettierignore"


# ── the gate discriminates (synthetic tree) ────────────────────────────


def _write_tree(root: Path, *, requirements: str, web_deps: dict[str, str]) -> None:
    (root / "pyproject.toml").write_text(
        '[project]\nname = "fixture"\nversion = "0"\n'
        'dependencies = ["numpy>=1.24.0", "requests>=2.28.0"]\n',
        encoding="utf-8",
    )
    (root / "requirements.txt").write_text(requirements, encoding="utf-8")
    (root / "package.json").write_text(
        json.dumps({"name": "fixture-root", "dependencies": {"react": "^18.2.0"}}),
        encoding="utf-8",
    )
    (root / "apps").mkdir()
    (root / "apps" / "web").mkdir()
    (root / "apps" / "web" / "package.json").write_text(
        json.dumps({"name": "fixture-web", "dependencies": web_deps}),
        encoding="utf-8",
    )


def test_manifest_change_changes_the_output(tmp_path: Path):
    """A manifest edit must change the artifact — otherwise --check can never fail."""
    _write_tree(tmp_path, requirements="numpy>=1.26.0\n", web_deps={"react": "^19.0.0"})
    before = gen.build(tmp_path)
    (tmp_path / "requirements.txt").write_text("numpy>=1.27.0\n", encoding="utf-8")
    assert gen.build(tmp_path) != before


def test_duplicates_drift_and_cross_tree_are_detected(tmp_path: Path):
    _write_tree(
        tmp_path,
        requirements="numpy>=1.26.0\n",
        web_deps={"react": "^19.0.0", "requests": "^2.0.0"},
    )
    artifact = json.loads(gen.build(tmp_path).split("export const DEPS: DepsArtifact = ", 1)[1])

    drift_keys = {g["key"] for g in artifact["drift"]}
    assert "numpy" in drift_keys, "1.24 vs 1.26 must read as drift"
    assert "react" in drift_keys, "^18 vs ^19 must read as drift"

    dup_keys = {g["key"] for g in artifact["duplicates"]}
    assert {"numpy", "react", "requests"} <= dup_keys

    # requests is declared by BOTH trees (python requirements + js package.json)
    assert "requests" in artifact["cross_tree"]
    assert artifact["cross_tree"] == sorted(artifact["cross_tree"])


# ── spec parsing (the honesty layer) ───────────────────────────────────


@pytest.mark.parametrize(
    ("spec", "name", "version"),
    [
        ("numpy>=1.24.0", "numpy", ">=1.24.0"),
        ("uvicorn[standard]>=0.24.0", "uvicorn", ">=0.24.0"),
        ("pydantic==2.13.4", "pydantic", "==2.13.4"),
        ('foo>=1.0; python_version < "3.12"', "foo", ">=1.0"),
        ("colorama", "colorama", ""),
        ("Python_YAML>=6.0.0", "Python_YAML", ">=6.0.0"),
    ],
)
def test_python_spec_parsing(spec: str, name: str, version: str):
    assert gen.parse_python_spec(spec) == (name, version)


def test_normalize_is_pep503():
    assert gen.normalize("Python.YAML") == gen.normalize("python_yaml") == "python-yaml"
    assert gen.normalize("React") == "react"
