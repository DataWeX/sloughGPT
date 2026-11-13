## Summary

What does this change and why?

## Checklist

- [ ] `npx turbo run lint typecheck` passes from repo root
- [ ] `npx turbo run test` passes from repo root
- [ ] If **Python code** changed: `ruff check .` and `ruff format --check .` pass
- [ ] If **`apps/web/`** changed: `cd apps/web && npm ci && npm run ci`
- [ ] If **`packages/strui/`** changed: `cd packages/strui && npm ci && npm run lint && npm run typecheck && npm test`
- [ ] If **`packages/sdk-ts/typescript-sdk/`** changed: `cd packages/sdk-ts/typescript-sdk && npm ci && npm run ci`
- [ ] If **`packages/sdk-py/sloughgpt_sdk/`** changed: `python3 -m pytest tests/test_sdk.py -q`
- [ ] If **`packages/standards/`** or **`scripts/validate_standards_schemas.py`** changed: `python3 scripts/validate_standards_schemas.py`

## Notes

Optional: related issues, rollout notes, breaking changes.
