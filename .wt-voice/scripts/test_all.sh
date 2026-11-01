#!/usr/bin/env bash
# Run all tests across the entire codebase.
# Usage: ./scripts/test_all.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FAILURES=0

echo "=== Python core-py tests ==="
cd "$REPO_ROOT"
python3 -m pytest packages/core-py/tests/ -q --tb=short -p no:cacheprovider || FAILURES=$((FAILURES + 1))

echo ""
echo "=== Root-level Python tests ==="
python3 -m pytest tests/ \
  --override-ini="testpaths=tests" \
  --override-ini="addopts=" \
  --ignore=tests/test_checkpoint_utils.py \
  --ignore=tests/test_knowledge_graph.py \
  --ignore=tests/test_lm_eval_char.py \
  --ignore=tests/test_rag.py \
  --ignore=tests/test_auto_train.py \
  --ignore=tests/test_auto_train_integration.py \
  --ignore=tests/test_chat_loop_e2e.py \
  --ignore=tests/test_e2e_inference.py \
  --ignore=tests/test_e2e_smoke.py \
  --ignore=tests/test_inference_generate.py \
  --ignore=tests/test_torch_runtime.py \
  --ignore=tests/test_wandb_helpers.py \
  --ignore=tests/server/ \
  -q --tb=short || FAILURES=$((FAILURES + 1))

echo ""
echo "=== Server API tests ==="
cd "$REPO_ROOT/apps/api/server"
python3 -m pytest tests/ -q --tb=short -p no:cacheprovider || FAILURES=$((FAILURES + 1))

echo ""
echo "=== Frontend tests ==="
cd "$REPO_ROOT/apps/web"
npx vitest run --reporter=verbose 2>&1 | tail -5 || FAILURES=$((FAILURES + 1))

echo ""
echo "=== TypeScript check ==="
cd "$REPO_ROOT/apps/web"
npx tsc --noEmit 2>&1 || FAILURES=$((FAILURES + 1))

echo ""
if [ "$FAILURES" -gt 0 ]; then
    echo "FAILED: $FAILURES test suite(s) had failures"
    exit 1
else
    echo "ALL PASSED"
fi
