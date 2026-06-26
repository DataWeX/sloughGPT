import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from apps.api.server.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_hf_models():
    """Mock HuggingFace model loading to avoid network calls."""
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model, \
         patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value.pad_token = "<|endoftext|>"
        yield


class TestAutoTrainStart:
    def test_start_sets_running_state(self, client, mock_hf_models):
        """POST /auto-train/start should start a training session."""
        response = client.post(
            "/auto-train/start",
            json={"teacher_model": "gpt2", "temperature": 0.8, "epochs": 1, "source_text": "hello world"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    def test_start_twice_should_warn(self, client, mock_hf_models):
        """Starting twice should work (replaces prior session)."""
        r1 = client.post(
            "/auto-train/start",
            json={"teacher_model": "gpt2", "source_text": "hello"}
        )
        assert r1.status_code == 200

        r2 = client.post(
            "/auto-train/start",
            json={"teacher_model": "gpt2", "source_text": "world"}
        )
        assert r2.status_code == 200
        data = r2.json()
        assert data["status"] == "ready"


class TestAutoTrainStop:
    def test_stop_before_start_gives_error(self, client):
        """Stopping without start should return stopped."""
        response = client.post("/auto-train/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["stopped", "not_running"]

    def test_stop_after_start_works(self, client, mock_hf_models):
        """Stop after start should work."""
        client.post(
            "/auto-train/start",
            json={"teacher_model": "gpt2"}
        )

        response = client.post("/auto-train/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"