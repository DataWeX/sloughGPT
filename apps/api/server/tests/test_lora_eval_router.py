"""Tests for the /lora-eval router (eval runs, history, aggregation)."""

from unittest.mock import patch, MagicMock
from test_support import get_test_client


def _data(resp):
    """Unwrap the success_response() envelope."""
    body = resp.json()
    return body.get("data", body)


def _mock_eval_result():
    """Create a mock EvalResult."""
    r = MagicMock()
    r.to_dict.return_value = {
        "perplexity": 2.5,
        "bleu": 0.35,
        "throughput": 120.0,
    }
    return r


def _mock_evaluator():
    """Create a mock LoRAEvaluator."""
    ev = MagicMock()
    ev.run.return_value = _mock_eval_result()
    ev.get_history.return_value = [_mock_eval_result()]
    ev.compare.return_value = {
        "verdict": "improved",
        "perplexity_delta": -0.3,
        "bleu_delta": 0.05,
        "throughput_delta": 15.0,
    }
    ev.compare_with_report.return_value = "Adapter shows improvement across all metrics."
    return ev


def _mock_per_user_lora():
    """Create a mock PerUserLoRAStore."""
    store = MagicMock()
    store.aggregate_best_adapters.return_value = {
        "output_path": "data/user_adapters/best_aggregated.npz",
        "user_count": 3,
        "total_feedback": 45,
        "eval": {
            "delta": {
                "verdict": "improved",
                "perplexity_delta": -0.2,
            },
            "report": "Aggregated adapter improved.",
        },
    }
    return store


class TestRunEval:
    def test_run_eval_baseline_only(self):
        client = get_test_client()
        ev = _mock_evaluator()
        ev.run.side_effect = [MagicMock(to_dict=lambda: {"perplexity": 2.5})]
        with patch("domains.feedback.lora_eval.get_lora_evaluator", return_value=ev):
            resp = client.get("/lora-eval/run")
        assert resp.status_code == 200
        data = _data(resp)
        assert "status" in data
        assert data["status"] in ("compared", "baseline_only")

    def test_run_eval_with_adapter(self):
        client = get_test_client()
        ev = _mock_evaluator()
        with patch("domains.feedback.lora_eval.get_lora_evaluator", return_value=ev), \
             patch("pathlib.Path.exists", return_value=True):
            resp = client.get("/lora-eval/run?adapter_path=/tmp/test.npz")
        assert resp.status_code == 200
        data = _data(resp)
        assert data["status"] == "compared"
        assert "baseline" in data
        assert "with_adapter" in data
        assert "delta" in data

    def test_run_eval_custom_soul(self):
        client = get_test_client()
        ev = _mock_evaluator()
        with patch("domains.feedback.lora_eval.get_lora_evaluator", return_value=ev):
            resp = client.get("/lora-eval/run?soul=custom_soul")
        assert resp.status_code == 200
        ev.run.assert_called()


class TestEvalHistory:
    def test_get_history(self):
        client = get_test_client()
        ev = _mock_evaluator()
        with patch("domains.feedback.lora_eval.get_lora_evaluator", return_value=ev):
            resp = client.get("/lora-eval/history")
        assert resp.status_code == 200
        data = _data(resp)
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_get_history_with_limit(self):
        client = get_test_client()
        ev = _mock_evaluator()
        with patch("domains.feedback.lora_eval.get_lora_evaluator", return_value=ev):
            resp = client.get("/lora-eval/history?limit=5")
        assert resp.status_code == 200
        ev.get_history.assert_called_with(limit=5)

    def test_get_history_limit_validation(self):
        client = get_test_client()
        resp = client.get("/lora-eval/history?limit=0")
        assert resp.status_code == 422

    def test_get_history_limit_over_max(self):
        client = get_test_client()
        resp = client.get("/lora-eval/history?limit=200")
        assert resp.status_code == 422


class TestTriggerAggregation:
    def test_aggregation_success(self):
        client = get_test_client()
        store = _mock_per_user_lora()
        with patch("domains.feedback.per_user_lora.get_per_user_lora", return_value=store):
            resp = client.post("/lora-eval/aggregate?top_k=5&min_feedback=3")
        assert resp.status_code == 200
        data = _data(resp)
        assert data["status"] == "aggregated_with_eval"
        assert "output_path" in data
        assert data["user_count"] == 3

    def test_aggregation_no_adapters(self):
        client = get_test_client()
        store = MagicMock()
        store.aggregate_best_adapters.return_value = {"error": "no adapters found"}
        with patch("domains.feedback.per_user_lora.get_per_user_lora", return_value=store):
            resp = client.post("/lora-eval/aggregate")
        assert resp.status_code == 200
        data = _data(resp)
        assert data["status"] == "no_adapters"

    def test_aggregation_params(self):
        client = get_test_client()
        store = _mock_per_user_lora()
        with patch("domains.feedback.per_user_lora.get_per_user_lora", return_value=store):
            client.post("/lora-eval/aggregate?top_k=10&min_feedback=5&output_name=test_agg")
        store.aggregate_best_adapters.assert_called_once_with(
            top_k=10, min_feedback_count=5, output_name="test_agg", run_eval=True
        )
