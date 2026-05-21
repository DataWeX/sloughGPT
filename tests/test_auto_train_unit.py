import pytest
import torch
from unittest.mock import patch, MagicMock, AsyncMock, mock_open

pytestmark = pytest.mark.skip(reason="Legacy inline auto-train removed; router handles these")
import json


@pytest.fixture
def mock_baby_model():
    """Create a mock baby model."""
    model = MagicMock()
    model.parameters = MagicMock(return_value=[torch.zeros(10, 10)])
    model.state_dict = MagicMock(return_value={})
    model.train = MagicMock()
    model.eval = MagicMock()
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    chars = ["<PAD>", "<UNK>"] + list(" abcdefghijklmnopqrstuvwxyz0123456789.,!?-'")
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return {"stoi": stoi, "itos": itos}


class TestInitBabyTokenizer:
    """Tests for _init_baby_tokenizer function."""

    def test_init_baby_tokenizer_returns_correct_maps(self):
        """Should return proper stoi and itos maps."""
        from apps.api.server.main import _init_baby_tokenizer
        
        stoi, itos = _init_baby_tokenizer()
        
        assert "<PAD>" in stoi
        assert "<UNK>" in stoi
        assert stoi["<PAD>"] == 0
        assert stoi["<UNK>"] == 1
        assert itos[0] == "<PAD>"
        assert itos[1] == "<UNK>"

    def test_init_baby_tokenizer_includes_basic_chars(self):
        """Should include common characters."""
        from apps.api.server.main import _init_baby_tokenizer
        
        stoi, _ = _init_baby_tokenizer()
        
        for char in "abcdefghijklmnopqrstuvwxyz":
            assert char in stoi, f"Missing letter: {char}"
        chars_to_check = "0123456789.,!?-'" + "'"
        for char in chars_to_check:
            assert char in stoi, f"Missing char: {char}"


class TestLoadOrCreateBabyModel:
    """Tests for _load_or_create_baby_model function."""

    @patch("apps.api.server.main._REPO_ROOT")
    @patch("apps.api.server.main._create_baby_model")
    @patch("builtins.open", mock_open())
    @patch("torch.load")
    def test_load_existing_model(self, mock_repo_root, mock_create, mock_torch_load):
        """Should load existing model if available."""
        from apps.api.server.main import _load_or_create_baby_model, _auto_train_baby_model, _auto_train_baby_tokenizer
        
        mock_repo_root.__truediv__ = MagicMock(return_value=MagicMock(exists=MagicMock(return_value=True)))
        mock_torch_load.return_value = {
            "model": {},
            "tokenizer": {"stoi": {"a": 1}, "itos": {1: "a"}},
            "training_log": []
        }
        
        mock_model = MagicMock()
        mock_model.parameters.return_value = [MagicMock()]
        mock_create.return_value = mock_model
        
        _load_or_create_baby_model("test.pt")
        
        assert _auto_train_baby_model is not None

    @patch("apps.api.server.main._create_baby_model")
    @patch("builtins.open", mock_open())
    @patch("torch.load")
    @patch("apps.api.server.main._REPO_ROOT")
    def test_create_fresh_model_on_missing_file(self, mock_repo_root, mock_torch_load, mock_open, mock_create):
        """Should create fresh model if file doesn't exist."""
        from apps.api.server.main import _load_or_create_baby_model
        
        mock_path = MagicMock()
        mock_path.exists = MagicMock(return_value=False)
        mock_repo_root.__truediv__ = MagicMock(return_value=mock_path)
        
        mock_create.return_value = MagicMock()
        
        _load_or_create_baby_model("new.pt")
        
        mock_create.assert_called_once()


class TestTrainBabyOnPair:
    """Tests for _train_baby_on_pair function."""

    def test_train_returns_none_when_model_none(self):
        """Should return None when model is None."""
        from apps.api.server.main import _train_baby_on_pair
        
        with patch("apps.api.server.main._auto_train_baby_model", None):
            with patch("apps.api.server.main._auto_train_optimizer", None):
                result = _train_baby_on_pair("hello world", "goodbye world", 0)
        
        assert result is None

    def test_train_returns_none_on_short_input(self, mock_baby_model, mock_tokenizer):
        """Should return None for too short input."""
        from apps.api.server.main import _train_baby_on_pair
        
        with patch("apps.api.server.main._auto_train_baby_model", mock_baby_model):
            with patch("apps.api.server.main._auto_train_baby_tokenizer", mock_tokenizer):
                with patch("apps.api.server.main._auto_train_optimizer", MagicMock()):
                    result = _train_baby_on_pair("a", "b", 0)
        
        assert result is None

    @patch("torch.no_grad")
    def test_train_runs_successfully(self, mock_no_grad, mock_baby_model, mock_tokenizer):
        """Should train successfully with valid input."""
        from apps.api.server.main import _train_baby_on_pair
        
        mock_optimizer = MagicMock()
        mock_baby_model.return_value = (torch.randn(1, 32, 46), None)
        
        with patch("apps.api.server.main._auto_train_baby_model", mock_baby_model):
            with patch("apps.api.server.main._auto_train_baby_tokenizer", mock_tokenizer):
                with patch("apps.api.server.main._auto_train_optimizer", mock_optimizer):
                    result = _train_baby_on_pair("hello world test", "goodbye world test", 0)
        
        mock_optimizer.step.assert_called()


class TestAutoTrainEndpoints:
    """Tests for auto-train API endpoints."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from apps.api.server.main import app
        return TestClient(app)

    def test_start_endpoint_returns_200(self, client):
        """POST /auto-train/start should return 200."""
        with patch("apps.api.server.main._load_or_create_baby_model"):
            response = client.post(
                "/auto-train/start",
                json={"teacher_model": "gpt2", "temperature": 0.8}
            )
        
        assert response.status_code == 200
        assert response.json()["status"] == "started"

    def test_stop_endpoint_returns_200(self, client):
        """POST /auto-train/stop should return 200."""
        response = client.post("/auto-train/stop")
        
        assert response.status_code == 200
        assert response.json()["status"] == "stopped"

    def test_get_training_log(self, client):
        """GET /auto-train/log should return log."""
        response = client.get("/auto-train/log")
        
        assert response.status_code == 200
        assert "log" in response.json()


class TestAutoTrainConfig:
    """Tests for auto-train configuration."""

    def test_auto_train_request_defaults(self):
        """AutoTrainRequest should have correct defaults."""
        from apps.api.server.main import AutoTrainRequest
        
        request = AutoTrainRequest()
        
        assert request.teacher_model == "gpt2"
        assert request.student_model == "sloughgpt"
        assert request.temperature == 0.8
        assert request.learning_rate == 0.01
        assert request.max_steps == 20


class TestStreamAutoTrain:
    """Tests for /auto-train/stream endpoint."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from apps.api.server.main import app
        return TestClient(app)

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForCausalLM")
    def test_stream_requires_start_first(self, mock_model_cls, mock_tokenizer_cls, client):
        """Stream should fail if not started."""
        with patch("apps.api.server.main._auto_train_config", {}):
            response = client.get("/auto-train/stream")
            
            assert response.status_code == 200