"""Tests for the lora-eval API router (routers/lora_eval.py).

Covers: LoraEvalRouter run, history, aggregate.
All domain calls are mocked; only HTTP-level behavior is tested.

Note: the lora_eval router imports get_lora_evaluator inside handler body,
so we must patch at the domain module level. Also get_per_user_lora for aggregate.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.lora_eval import router  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval_result(**overrides):
    defaults = {
        "perplexity": 10.0,
        "bleu": 0.3,
        "personality_score": 0.7,
        "throughput": 50.0,
        "to_dict": lambda: {
            "perplexity": 10.0,
            "bleu": 0.3,
            "personality_score": 0.7,
            "throughput": 50.0,
        },
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_evaluator(**overrides):
    defaults = {
        "run": lambda adapter_path=None, soul_name="assistant", save=True: _make_eval_result(),
        "compare": lambda b, w: {"verdict": "better", "perplexity_delta": -2.0},
        "compare_with_report": lambda b, w: "Adapter improves perplexity by 2.0",
        "get_history": lambda limit=20: [_make_eval_result()],
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_store(**overrides):
    defaults = {
        "aggregate_best_adapters": lambda **kw: {
            "user_count": 3,
            "total_feedback": 20,
            "output_path": "/tmp/best.npz",
            "eval": {"delta": {"verdict": "better", "perplexity_delta": -1.5}},
        },
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _app():
    app = FastAPI()
    app.include_router(router)
    return app


# ---------------------------------------------------------------------------
# Tests — patch at domain module level (lazy imports inside handler body)
# ---------------------------------------------------------------------------

PATCH_EVALUATOR = "domain.feedback._internal.lora_eval.get_lora_evaluator"
PATCH_STORE = "domain.feedback._internal.per_user_lora.get_per_user_lora"
PATCH_PER_U_LORA = "domain.feedback._internal.per_user_lora.get_per_user_lora"


class TestRunEval:
    # The router binds Path at module scope, so patching pathlib.Path never
    # reaches it — the old mock silently did nothing and the handler's real
    # exists() ran against disk. This stub is bound where the handler looks.
    class _PathExists:
        def __init__(self, p):
            self._p = p

        def resolve(self):
            # Real resolve — absolute, so the confinement check still sees a
            # path genuinely under _ADAPTER_BASE.
            return Path(self._p).resolve()

        def exists(self):
            return True

    @patch(PATCH_EVALUATOR)
    def test_baseline_only(self, mock_get):
        evaluator = _make_evaluator()
        mock_get.return_value = evaluator
        client = TestClient(_app())
        # Relative, and therefore CWD-relative like _ADAPTER_BASE itself, so
        # the two agree wherever pytest runs. An absolute path cannot work:
        # _VALID_ADAPTER_PATH rejects the space in this repo's directory name.
        resp = client.post(
            "/lora-eval/run", params={"adapter_path": "data/user_adapters/missing.npz"}
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "baseline_only"
        assert "baseline" in data

    @patch(PATCH_EVALUATOR)
    def test_with_adapter_comparison(self, mock_get):
        evaluator = _make_evaluator()
        mock_get.return_value = evaluator
        client = TestClient(_app())
        with patch("routers.lora_eval.Path", self._PathExists):
            resp = client.post(
                "/lora-eval/run", params={"adapter_path": "data/user_adapters/best.npz"}
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "compared"
        assert "baseline" in data
        assert "with_adapter" in data


class TestGetEvalHistory:
    @patch(PATCH_EVALUATOR)
    def test_returns_results(self, mock_get):
        evaluator = _make_evaluator()
        mock_get.return_value = evaluator
        client = TestClient(_app())
        resp = client.get("/lora-eval/history")
        assert resp.status_code == 200
        results = resp.json()["data"]["results"]
        assert len(results) == 1

    @patch(PATCH_EVALUATOR)
    def test_custom_limit(self, mock_get):
        evaluator = _make_evaluator(
            get_history=lambda limit=20: [_make_eval_result() for _ in range(limit)]
        )
        mock_get.return_value = evaluator
        client = TestClient(_app())
        resp = client.get("/lora-eval/history", params={"limit": 5})
        assert resp.status_code == 200
        assert len(resp.json()["data"]["results"]) == 5


class TestTriggerAggregation:
    @patch(PATCH_PER_U_LORA)
    @patch(PATCH_EVALUATOR)
    def test_aggregate_with_eval(self, mock_eval_get, mock_store_get):
        mock_eval_get.return_value = _make_evaluator()
        mock_store_get.return_value = _make_store()
        client = TestClient(_app())
        resp = client.post(
            "/lora-eval/aggregate",
            params={
                "top_k": 5,
                "min_feedback": 3,
                "output_name": "best_v2",
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "aggregated_with_eval"
        assert data["user_count"] == 3

    @patch(PATCH_PER_U_LORA)
    @patch(PATCH_EVALUATOR)
    def test_aggregate_no_eval(self, mock_eval_get, mock_store_get):
        mock_eval_get.return_value = _make_evaluator()
        store = _make_store(
            aggregate_best_adapters=lambda **kw: {
                "user_count": 2,
                "total_feedback": 10,
                "output_path": "/tmp/best.npz",
                "eval": {"error": "no adapters"},
            }
        )
        mock_store_get.return_value = store
        client = TestClient(_app())
        resp = client.post("/lora-eval/aggregate")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "aggregated_no_eval"

    @patch(PATCH_PER_U_LORA)
    @patch(PATCH_EVALUATOR)
    def test_aggregate_error(self, mock_eval_get, mock_store_get):
        mock_eval_get.return_value = _make_evaluator()
        store = _make_store(aggregate_best_adapters=lambda **kw: {"error": "no adapters found"})
        mock_store_get.return_value = store
        client = TestClient(_app())
        resp = client.post("/lora-eval/aggregate")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "no_adapters"
