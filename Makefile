.PHONY: api web tsc lint test-py test-web test dev install precommit

# ── Dev Servers ──────────────────────────────────────────
api:
	.venv/bin/python3 apps/api/server/main.py

web:
	cd apps/web && npm run dev

stack:
	./scripts/dev-stack.sh

# ── Type Check ──────────────────────────────────────────
tsc:
	cd apps/web && npx tsc --noEmit

# ── Lint ────────────────────────────────────────────────
lint:
	cd apps/web && npm run lint

# ── Tests (Targeted) ────────────────────────────────────
test-py:
	cd packages/core-py && python3 -m pytest $(ARGS)

test-py-fast:
	cd packages/core-py && python3 -m pytest -n auto --dist loadgroup -x -q $(ARGS)

test-web:
	cd apps/web && npm run test $(ARGS)

test-web-lib:
	cd apps/web && npm run test:lib

test-web-components:
	cd apps/web && npm run test:components

test-web-hooks:
	cd apps/web && npm run test:hooks

test-web-changed:
	cd apps/web && npm run test:changed

test: test-py-fast test-web-lib

# ── Install ──────────────────────────────────────────────
install:
	cd apps/web && npm ci
	.venv/bin/pip install -e packages/core-py/

# ── Tooling Setup ───────────────────────────────────────
precommit:
	.venv/bin/pre-commit install
	.venv/bin/pre-commit install-hooks
	@echo "pre-commit hooks installed. Run 'make precommit-run' to check all files."

precommit-run:
	.venv/bin/pre-commit run --all-files

precommit-update:
	.venv/bin/pre-commit autoupdate

# ── Cleanup ─────────────────────────────────────────────
clean:
	cd apps/web && rm -rf .next node_modules/.vitest-cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
