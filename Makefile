.PHONY: api api-daemon api-stop web tsc lint test-py test-py-training test-web test dev install build precommit test-repo-root colab-smoke colab-test setup-git test-doctor test-doctor-health test-doctor-flake test-doctor-recent

# ── Dev Servers ──────────────────────────────────────────
api:
	.venv/bin/python3 apps/api/server/main.py

api-daemon:
	.venv/bin/python3 apps/api/server/main.py --daemon --port 8000
	@echo "Server daemon started on port 8000. Use 'make api-stop' to kill."

api-stop:
	@lsof -ti :8000 2>/dev/null | xargs kill 2>/dev/null || true
	@echo "Server stopped on port 8000"

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
	cd packages/core-py && .venv/bin/python -m pytest $(ARGS)

test-py-fast:
	cd packages/core-py && .venv/bin/python -m pytest -n auto --dist loadgroup -x -q $(ARGS)

test-py-slow:
	cd packages/core-py && .venv/bin/python -m pytest -m slow $(ARGS)

test-py-training:
	cd packages/core-py && python3 -m pytest tests/test_training_sequence.py tests/test_training_status.py tests/test_trainer_protocol.py tests/test_auto_trainer.py tests/test_distillation.py tests/test_distill_gpt2.py tests/test_video_trainer.py tests/test_mobile_training_store.py tests/test_pugqeep_checkpoint.py tests/test_pugqeep.py tests/test_pugqeep_generic.py -x -q $(ARGS)

test-py-all:
	cd packages/core-py && python3 -m pytest -m "" $(ARGS)

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

# ── Build ────────────────────────────────────────────────
build: quant-core

quant-core:
	@echo "Building AVX2 C GEMM extensions..."
	@gcc -O3 -mavx2 -shared -fPIC \
		-o packages/core-py/domains/infrastructure/quant_core/matmul_int8.dylib \
		packages/core-py/domains/infrastructure/quant_core/matmul_int8.c \
		2>/dev/null && echo "  ✓ matmul_int8.dylib" \
		|| echo "  ⚠  matmul_int8.c — gcc/AVX2 unavailable"
	@gcc -O3 -mavx2 -shared -fPIC \
		-o packages/core-py/domains/infrastructure/quant_core/matmul_int4.dylib \
		packages/core-py/domains/infrastructure/quant_core/matmul_int4.c \
		2>/dev/null && echo "  ✓ matmul_int4.dylib" \
		|| echo "  ⚠  matmul_int4.c — gcc/AVX2 unavailable"

# ── Buildroot (Linux Image for v86) ─────────────────────
buildroot:
	bash buildroot/build.sh build

buildroot-docker:
	bash buildroot/build.sh docker

buildroot-clean:
	bash buildroot/build.sh clean

buildroot-shell:
	bash buildroot/build.sh shell

buildroot-status:
	bash buildroot/build.sh status

# ── Install ──────────────────────────────────────────────
install: build setup-git
	cd apps/web && npm ci
	.venv/bin/pip install -e packages/core-py/
	.venv/bin/pip install -e packages/chargectl/

# Install the chargectl enforcement daemon. Both targets rewrite ExecStart to this
# checkout's .venv console script when it exists, so the installed unit runs exactly
# the code you are looking at — see packages/chargectl/systemd/chargectl.service.
CHARGE_UNIT := packages/chargectl/systemd/chargectl.service

# System unit (root): needs a real charge node, e.g. a laptop.
charge-svc:
	@test -f $(CHARGE_UNIT) || { echo "charge-svc: missing $(CHARGE_UNIT)"; exit 1; }
	@command -v systemctl >/dev/null 2>&1 || { echo "charge-svc: systemctl not found (this host does not run systemd)"; exit 1; }
	@if [ "$$(id -u)" -ne 0 ]; then echo "charge-svc: needs root — run: sudo make charge-svc"; exit 1; fi
	@tmp=$$(mktemp); \
	trap 'rm -f "$$tmp"' EXIT; \
	if [ -x .venv/bin/chargectl ]; then \
		sed "s|^ExecStart=.*|ExecStart=\"$(CURDIR)/.venv/bin/chargectl\" daemon|" $(CHARGE_UNIT) > "$$tmp"; \
		echo "charge-svc: ExecStart -> $(CURDIR)/.venv/bin/chargectl"; \
	else \
		cp $(CHARGE_UNIT) "$$tmp"; \
		echo "charge-svc: ExecStart -> /usr/local/bin/chargectl (no .venv — run 'make install' first)"; \
	fi; \
	install -m 644 "$$tmp" /etc/systemd/system/chargectl.service
	install -d -m 755 /etc/chargectl /var/lib/chargectl
	systemctl daemon-reload
	systemctl enable --now chargectl
	@systemctl --no-pager --lines=0 status chargectl || true

# User unit (no root): dev boxes and VMs — threshold writes are a dry run there anyway.
charge-svc-user:
	@test -f $(CHARGE_UNIT) || { echo "charge-svc-user: missing $(CHARGE_UNIT)"; exit 1; }
	@command -v systemctl >/dev/null 2>&1 || { echo "charge-svc-user: systemctl not found (this host does not run systemd)"; exit 1; }
	@systemctl --user is-system-running >/dev/null 2>&1 || { echo "charge-svc-user: no user systemd session (is $$XDG_RUNTIME_DIR set?)"; exit 1; }
	@test -x .venv/bin/chargectl || { echo "charge-svc-user: .venv/bin/chargectl missing — run 'make install' first"; exit 1; }
	@mkdir -p ~/.config/systemd/user
	@sed -e "s|^ExecStart=.*|ExecStart=\"$(CURDIR)/.venv/bin/chargectl\" daemon|" \
		-e "s|^Environment=CHARGECTL_POLICY=.*|Environment=CHARGECTL_POLICY=%h/.config/chargectl/policy.json|" \
		-e "s|^Environment=CHARGECTL_STATE=.*|Environment=CHARGECTL_STATE=%h/.config/chargectl/state.json|" \
		-e "s|^After=multi-user.target|After=default.target|" \
		-e "s|^WantedBy=.*|WantedBy=default.target|" \
		-e "/^ReadWritePaths=/d" -e "/^ProtectSystem=/d" -e "/^ProtectHome=/d" \
		$(CHARGE_UNIT) > ~/.config/systemd/user/chargectl.service
	@echo "charge-svc-user: ExecStart -> $(CURDIR)/.venv/bin/chargectl"
	systemctl --user daemon-reload
	systemctl --user enable --now chargectl
	@systemctl --user --no-pager --lines=0 status chargectl || true

# Remove both units (policy + state files are left alone — they are user data).
charge-svc-down:
	@-systemctl disable --now chargectl 2>/dev/null || true
	@-rm -f /etc/systemd/system/chargectl.service
	@-systemctl daemon-reload
	@-systemctl --user disable --now chargectl 2>/dev/null || true
	@-rm -f ~/.config/systemd/user/chargectl.service
	@-systemctl --user daemon-reload 2>/dev/null || true
	@echo "chargectl units removed (policy + state files left in place)"

# ── Tooling Setup ───────────────────────────────────────
setup-git:
	git config core.sshCommand "ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3"
	@echo "Git SSH keepalive configured for this repo."

precommit:
	.venv/bin/pre-commit install
	.venv/bin/pre-commit install-hooks
	@echo "pre-commit hooks installed. Run 'make precommit-run' to check all files."

precommit-run:
	.venv/bin/pre-commit run --all-files

precommit-update:
	.venv/bin/pre-commit autoupdate

# ── Repo Root Tests ─────────────────────────────────────
test-repo-root:
	.venv/bin/python3 -m pytest tests/test_repo_root_package_json.py -v

# ── Colab Smoke Tests ───────────────────────────────────
colab-smoke:
	bash scripts/run_colab_notebook_smoke.sh

colab-test:
	.venv/bin/python3 -m pytest tests/test_sloughgpt_colab_notebook.py -v

# ── Test Doctor (Internal Diagnostics) ──────────────────
test-doctor:
	python3 scripts/test-doctor.py $(ARGS)

test-doctor-health:
	python3 scripts/test-doctor.py

test-doctor-flake:
	python3 scripts/test-doctor.py --flake

test-doctor-recent:
	python3 scripts/test-doctor.py --recent

test-doctor-fix:
	python3 scripts/test-doctor.py --fix-hint $(TEST)

# ── Cleanup ─────────────────────────────────────────────
clean:
	cd apps/web && rm -rf .next node_modules/.vitest-cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

test-planner:
	PYTHONPATH=packages/app-planner/src:packages/mogdb/src python3 -m pytest packages/app-planner/tests/ -x -q $(ARGS)
