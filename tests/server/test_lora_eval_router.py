"""
Tests for the LoRA eval router — /lora-eval/run, /lora-eval/history, /lora-eval/aggregate.

Uses a standalone FastAPI app with only the router under test to avoid
lifespan / startup dependency issues.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.lora_eval import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def _mock_eval_result():
    """Return a mock eval result with to_dict()."""
    result = MagicMock()
    result.to_dict.return_value = {
        "perplexity": 12.5,
        "bleu": 0.35,
        "throughput": 85.0,
        "personality_score": 0.72,
    }
    return result


class TestRunEval:
    """GET /lora-eval/run"""

    @patch("domains.feedback.lora_eval.get_lora_evaluator")
    def test_baseline_only_no_adapter(self, mock_get_eval):
        evaluator = MagicMock()
        evaluator.run.return_value = _mock_eval_result()
        mock_get_eval.return_value = evaluator

        resp = client.get("/lora-eval/run", params={"soul": "assistant"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "baseline_only"
        assert "baseline" in data
        assert data["baseline"]["perplexity"] == 12.5

    @patch("pathlib.Path")
    @patch("domains.feedback.lora_eval.get_lora_evaluator")
    def test_compared_with_adapter(self, mock_get_eval, mock_path_cls):
        evaluator = MagicMock()
        baseline = _mock_eval_result()
        with_adapter = _mock_eval_result()
        with_adapter.to_dict.return_value["perplexity"] = 10.0
        evaluator.run.side_effect = [baseline, with_adapter]
        evaluator.compare.return_value = {"perplexity_delta": -2.5}
        evaluator.compare_with_report.return_value = "Improvement report"
        mock_get_eval.return_value = evaluator

        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_cls.return_value = mock_path_instance

        resp = client.get(
            "/lora-eval/run",
            params={"adapter_path": "data/user_adapters/best_aggregated.npz", "soul": "assistant"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "compared"
        assert "baseline" in data
        assert "with_adapter" in data
        assert "delta" in data
        assert "report" in data

    @patch("domains.feedback.lora_eval.get_lora_evaluator")
    def test_evaluator_import_error_returns_500(self, mock_get_eval):
        mock_get_eval.side_effect = ImportError("no module")

        resp = client.get("/lora-eval/run")
        assert resp.status_code == 500

    @patch("domains.feedback.lora_eval.get_lora_evaluator")
    def test_internal_error_returns_500(self, mock_get_eval):
        mock_get_eval.side_effect = RuntimeError("evaluator broken")

        resp = client.get("/lora-eval/run")
        assert resp.status_code == 500

    @patch("domains.feedback.lora_eval.get_lora_evaluator")
    def test_evaluator_run_fails(self, mock_get_eval):
        evaluator = MagicMock()
        evaluator.run.side_effect = RuntimeError("run failed")
        mock_get_eval.return_value = evaluator

        resp = client.get("/lora-eval/run")
        assert resp.status_code == 500

    @patch("pathlib.Path")
    @patch("domains.feedback.lora_eval.get_lora_evaluator")
    def test_adapter_path_not_found_falls_back(self, mock_get_eval, mock_path_cls):
        evaluator = MagicMock()
        evaluator.run.return_value = _mock_eval_result()
        mock_get_eval.return_value = evaluator

        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path_cls.return_value = mock_path_instance

        resp = client.get("/lora-eval/run")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "baseline_only"


class TestEvalHistory:
    """GET /lora-eval/history"""

    @patch("domains.feedback.lora_eval.get_lora_evaluator")
    def test_returns_results(self, mock_get_eval):
        evaluator = MagicMock()
        result1 = _mock_eval_result()
        result2 = _mock_eval_result()
        evaluator.get_history.return_value = [result1, result2]
        mock_get_eval.return_value = evaluator

        resp = client.get("/lora-eval/history")
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 2

    @patch("domains.feedback.lora_eval.get_lora_evaluator")
    def test_empty_history(self, mock_get_eval):
        evaluator = MagicMock()
        evaluator.get_history.return_value = []
        mock_get_eval.return_value = evaluator

        resp = client.get("/lora-eval/history")
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    @patch("domains.feedback.lora_eval.get_lora_evaluator")
    def test_passes_limit_param(self, mock_get_eval):
        evaluator = MagicMock()
        evaluator.get_history.return_value = [_mock_eval_result()]
        mock_get_eval.return_value = evaluator

        resp = client.get("/lora-eval/history", params={"limit": 5})
        assert resp.status_code == 200
        evaluator.get_history.assert_called_once_with(limit=5)

    @patch("domains.feedback.lora_eval.get_lora_evaluator")
    def test_error_returns_500(self, mock_get_eval):
        mock_get_eval.side_effect = RuntimeError("history broken")

        resp = client.get("/lora-eval/history")
        assert resp.status_code == 500


class TestAggregate:
    """POST /lora-eval/aggregate"""

    @patch("domains.feedback.per_user_lora.get_per_user_lora")
    def test_aggregated_with_eval(self, mock_get_store):
        store = MagicMock()
        store.aggregate_best_adapters.return_value = {
            "output_path": "data/user_adapters/best.npz",
            "user_count": 3,
            "total_feedback": 15,
            "eval": {
                "delta": {
                    "verdict": "improved",
                    "perplexity_delta": -1.5,
                    "bleu_delta": 0.08,
                    "throughput_delta": 5.0,
                },
                "report": "Good improvement",
            },
        }
        mock_get_store.return_value = store

        resp = client.post("/lora-eval/aggregate", params={"top_k": 5, "run_eval": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "aggregated_with_eval"
        assert data["output_path"] == "data/user_adapters/best.npz"
        assert data["user_count"] == 3
        assert data["eval"]["verdict"] == "improved"
        assert data["eval"]["perplexity_delta"] == -1.5

    @patch("domains.feedback.per_user_lora.get_per_user_lora")
    def test_no_adapters(self, mock_get_store):
        store = MagicMock()
        store.aggregate_best_adapters.return_value = {
            "error": "No adapters found",
        }
        mock_get_store.return_value = store

        resp = client.post("/lora-eval/aggregate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "no_adapters"

    @patch("domains.feedback.per_user_lora.get_per_user_lora")
    def test_aggregated_no_eval(self, mock_get_store):
        store = MagicMock()
        store.aggregate_best_adapters.return_value = {
            "output_path": "data/user_adapters/best.npz",
            "user_count": 2,
            "total_feedback": 10,
            "eval": {"error": "eval failed"},
        }
        mock_get_store.return_value = store

        resp = client.post("/lora-eval/aggregate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "aggregated_no_eval"

    @patch("domains.feedback.per_user_lora.get_per_user_lora")
    def test_passes_params(self, mock_get_store):
        store = MagicMock()
        store.aggregate_best_adapters.return_value = {"error": "none"}
        mock_get_store.return_value = store

        client.post(
            "/lora-eval/aggregate",
            params={"top_k": 15, "min_feedback": 10, "output_name": "custom", "run_eval": False},
        )
        store.aggregate_best_adapters.assert_called_once_with(
            top_k=15,
            min_feedback_count=10,
            output_name="custom",
            run_eval=False,
        )

    @patch("domains.feedback.per_user_lora.get_per_user_lora")
    def test_store_error_returns_500(self, mock_get_store):
        mock_get_store.side_effect = RuntimeError("store broken")

        resp = client.post("/lora-eval/aggregate")
        assert resp.status_code == 500

    @patch("domains.feedback.per_user_lora.get_per_user_lora")
    def test_import_error_returns_500(self, mock_get_store):
        mock_get_store.side_effect = ImportError("no module")

        resp = client.post("/lora-eval/aggregate")
        assert resp.status_code == 500
