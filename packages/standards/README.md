## Standards

`packages/standards/` holds versioned API and dataset contracts (Markdown + JSON Schema under **`standards/`**).

Example validation runs in CI (**`standards-schemas`** in **`.github/workflows/ci_cd.yml`**) via **`scripts/validate_standards_schemas.py`** (**`jsonschema`** is included in **`python3 -m pip install -e ".[dev]"`** from the repo root).

See **`standards/SLOUGHGPT_STANDARD_V1.md`** (and **`packages/standards/standards/v1/`** for schemas). Native trainer **`step_*.pt`** char vocabulary (outside these JSON schemas) is documented in **`docs/policies/CONTRIBUTING.md`** (*Checkpoint vocabulary*).
