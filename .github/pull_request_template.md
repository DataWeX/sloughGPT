## Summary

What does this change and why?

## Checklist

- [ ] Installed from repo root as needed: `python3 -m pip install -e ".[dev]"` (see **QUICKSTART.md**)
- [ ] `./verify.sh` passes (path checks + ruff smoke when `ruff` is available)
- [ ] `python3 -m pytest tests/` passes (or the subset you touched; CI uses **`.github/workflows/reusable-ci-core.yml`**)
- [ ] If **repo root `package.json`** changed (e.g. **`dev:stack`** / **`concurrently`**): `npm run test:repo-root` (after `npm install` at repo root), `make test-repo-root`, or `python3 -m pytest tests/test_repo_root_package_json.py -q`
- [ ] If **`apps/web/`** changed: `cd apps/web && npm ci && npm run ci`
- [ ] If **`packages/strui/`** changed: `cd packages/strui && npm ci && npm run ci`
- [ ] If **`packages/sdk-ts/typescript-sdk/`** changed: `cd packages/sdk-ts/typescript-sdk && npm ci && npm run ci`
- [ ] If **`packages/sdk-py/sloughgpt_sdk/`** changed: `python3 -m pytest tests/test_sdk.py -q`
- [ ] If **`packages/standards/`** or **`scripts/validate_standards_schemas.py`** changed: `python3 scripts/validate_standards_schemas.py` (ensure **`jsonschema`**: **`python3 -m pip install -e ".[dev]"`** or **`python3 -m pip install jsonschema`**)

## Notes

Optional: related issues, rollout notes, breaking changes.
