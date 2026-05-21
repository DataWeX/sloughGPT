#!/bin/bash
# Quick verification script for SloughGPT (monorepo paths)

echo "🔍 Verifying SloughGPT installation..."

# Check Python
echo "Checking Python..."
python3 --version

# Check Node (optional for web)
echo "Checking Node..."
if command -v node &>/dev/null; then
    node --version
else
    echo "(node not found — skip web checks)"
fi

echo ""
echo "Checking required files..."

files=(
    "package.json"
    "apps/api/server/main.py"
    "packages/core-py/domains/ui/api_server.py"
    "apps/web/package.json"
    "apps/web/app/(app)/page.tsx"
    "apps/web/app/(app)/chat/page.tsx"
    "sloughgpt_colab.ipynb"
    "Makefile"
    "pyproject.toml"
)

all_found=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file NOT FOUND"
        all_found=false
    fi
done

if [ "$all_found" = true ]; then
    echo ""
    echo "✅ Core paths present!"
    echo ""
    if python3 -m ruff --version &>/dev/null; then
        echo "Ruff smoke (same rules as CI)..."
        python3 -m ruff check tests/ apps/cli/ apps/api/server/ train_sloughgpt.py packages/core-py/domains/training/checkpoint_utils.py --select E9,F63,F7,F82 || {
            echo "❌ Ruff smoke failed"
            exit 1
        }
        echo "✓ Ruff smoke passed"
        echo ""
    else
        echo "(Optional: python3 -m pip install ruff to run the CI lint smoke check locally.)"
        echo ""
    fi

    if command -v node &>/dev/null && [ -d "apps/web/node_modules" ]; then
        echo "Web npm run ci (apps/web)..."
        (cd apps/web && npm run ci) || {
            echo "❌ Web npm run ci failed (clean .next, lint, typecheck, test, next build)"
            exit 1
        }
        echo "✓ Web npm run ci passed"
        echo ""
    elif command -v node &>/dev/null; then
        echo "(Web: run cd apps/web && npm ci to enable npm run ci in this script.)"
        echo ""
    fi

    if command -v node &>/dev/null && [ ! -d "node_modules" ]; then
        echo "(Optional: npm install at repo root pulls concurrently for npm run dev:stack — same API+web as ./scripts/dev-stack.sh.)"
        echo ""
    fi

    echo "Root package.json contract: npm run test:repo-root  |  make test-repo-root  |  python3 -m pytest tests/test_repo_root_package_json.py -q"
    echo ""

    echo "CLI onboarding:"
    echo "  python3 cli.py start"
    echo ""
    echo "To start training (char LM, after pip install -e .):"
    echo "  make train-demo"
    echo "  or: python3 cli.py train --dataset shakespeare --epochs 3 --checkpoint-dir checkpoints"
    echo ""
    echo "To start SloughGPT:"
    echo "  API + web (one terminal): ./scripts/dev-stack.sh  |  make dev-stack  |  npm install && npm run dev:stack"
    echo "  Or separately:"
    echo "  1. API:  SLOUGHGPT_API_PORT=8000 python3 apps/api/server/main.py"
    echo "     or:  cd apps/api/server && python3 -m uvicorn main:app --reload --port 8000"
    echo "  2. Web: cd apps/web && npm run dev"
    echo ""
    echo "Docker: docker compose -f infra/docker/docker-compose.yml up -d api"
    echo ""
    echo "Then open the web dev URL (often http://localhost:3000)"
    echo ""
    echo "With a repo .venv, you can prefix commands:"
    echo "  ./run.sh python3 -m pytest tests/ -q"
    echo ""
    echo "Colab notebook smoke (optional full execute; see README → Google Colab):"
    echo "  ./scripts/run_colab_notebook_smoke.sh --help"
    echo "  make help   # colab-smoke / colab-test shortcuts"
    echo "  python3 -m pip install -e \".[notebook]\" && ./scripts/run_colab_notebook_smoke.sh"
    echo "  make colab-test"
    echo ""
    echo "Web CI parity (job test-web in .github/workflows/ci_cd.yml):"
    echo "  cd apps/web && npm ci && npm run ci"
    echo ""
    echo "TypeScript SDK CI parity (job test-sdk-ts):"
    echo "  cd packages/sdk-ts/typescript-sdk && npm ci && npm run ci"
    echo ""
    echo "Python SDK CI parity (job sdk-test-py):"
    echo "  python3 -m pytest tests/test_sdk.py -q"
    echo ""
    echo "Python core CI parity (job test in reusable-ci-core.yml), training-focused copy/paste subset:"
    echo "  python3 -m pytest tests/test_checkpoint_utils.py tests/test_config.py tests/test_domains_errors.py tests/test_train_sloughgpt_generate_text.py tests/test_train_sloughgpt_resume.py tests/test_sloughgpt_trainer_smoke.py tests/test_sloughgpt_trainer_resume.py tests/test_sloughgpt_trainer_progress_callback.py tests/test_cli_train_export_stem.py tests/test_cli_train_api_payload.py tests/test_training_router_kwds.py tests/test_training_schemas.py tests/test_lm_eval_char.py tests/test_cli_local_soul_candidates.py tests/test_soul_engine_conversation.py tests/test_repo_root_package_json.py tests/test_sloughgpt_colab_notebook.py -q"
    echo "  python3 train_sloughgpt.py --help"
    echo "  Full pytest module list (adds quantization, optimizations, benchmarking, inference, api, server, security, …): .github/workflows/reusable-ci-core.yml → Run core tests"
    echo ""
    echo "Standards CI parity (job standards-schemas):"
    echo "  python3 scripts/validate_standards_schemas.py"
    echo "  (jsonschema: python3 -m pip install -e \".[dev]\" or python3 -m pip install jsonschema)"
else
    echo ""
    echo "❌ Some files are missing!"
    exit 1
fi
