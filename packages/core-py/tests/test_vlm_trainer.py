"""Tests for the VLM multimodal training pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture
def minimal_jsonl():
    """Create a minimal JSONL dataset for VLM testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({
            "image_path": "/dev/null",
            "caption": "test image description",
        }) + "\n")
        f.write(json.dumps({
            "image_path": "/dev/null",
            "conversations": [
                {"from": "human", "value": "What is this?"},
                {"from": "gpt", "value": "This is a test"},
            ],
        }) + "\n")
        return Path(f.name)


class TestVLMConfig:
    """VLMConfig should have sensible defaults."""

    def test_defaults(self):
        from domains.training.multimodal import VLMConfig
        cfg = VLMConfig()
        assert cfg.vision_encoder == "google/siglip-base-patch16-224"
        assert cfg.llm == "Qwen/Qwen2.5-0.5B-Instruct"
        assert cfg.stage1_epochs == 1
        assert cfg.stage2_epochs == 2
        assert cfg.use_lora is True
        assert cfg.freeze_vision is True


class TestVLMDataset:
    """VLMDataset should load and parse JSONL correctly."""

    def test_loads_entries(self, minimal_jsonl):
        from PIL import Image
        from PIL import Image as PILImage
        with patch("PIL.Image.open", return_value=PILImage.new("RGB", (224, 224))):
            from transformers import AutoProcessor, AutoTokenizer
            try:
                processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224", trust_remote_code=True)
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            except Exception:
                pytest.skip("HF models not available for download")

            from domains.training.multimodal import VLMDataset
            ds = VLMDataset(
                data_path=str(minimal_jsonl),
                processor=processor,
                tokenizer=tokenizer,
                max_seq_length=128,
            )
            assert len(ds) == 2
            item = ds[0]
            assert "pixel_values" in item
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item

    def test_empty_dataset(self, minimal_jsonl):
        """If no valid entries, dataset is fine but training will error gracefully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("")

        path = Path(f.name)
        from domains.training.multimodal import VLMConfig, VLMTrainer
        cfg = VLMConfig()
        trainer = VLMTrainer(cfg)
        result = trainer.train(str(path))
        assert result["status"] == "error"

        path.unlink()


class TestVLMModel:
    """VLMModel should construct correctly with mock components."""

    def test_constructor(self):
        """Constructor should not raise."""
        from domains.training.multimodal import VLMConfig, VLMModel
        cfg = VLMConfig()
        with patch("domains.training.multimodal.AutoModel.from_pretrained") as mock_vision, \
             patch("domains.training.multimodal.AutoModelForCausalLM.from_pretrained") as mock_llm:
            mock_vision.return_value.config.hidden_size = 768
            mock_llm.return_value.config.hidden_size = 512

            model = VLMModel(cfg)
            assert model.connector is not None

    def test_forward(self):
        """Forward should run without error with mock components."""
        from domains.training.multimodal import VLMConfig, VLMModel
        import torch
        import torch.nn as nn

        cfg = VLMConfig()

        class FakeVisionConfig:
            hidden_size = 768

        class FakeVision(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = FakeVisionConfig()
                self.fc = nn.Linear(768, 768)

            def forward(self, pixel_values):
                B = pixel_values.shape[0]
                out = torch.randn(B, 197, 768)
                from types import SimpleNamespace
                return SimpleNamespace(last_hidden_state=out)

        with patch("domains.training.multimodal.AutoModel.from_pretrained") as mock_vision, \
             patch("domains.training.multimodal.AutoModelForCausalLM.from_pretrained") as mock_llm:
            mock_vision.return_value = FakeVision()

            # Mock LLM with real nn.Module
            class FakeLLM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.config = type("cfg", (), {"hidden_size": 512})()
                    self.embed = nn.Embedding(500, 512)

                def get_input_embeddings(self):
                    return self.embed

                def forward(self, inputs_embeds=None, attention_mask=None, labels=None, return_dict=True):
                    B = inputs_embeds.shape[0]
                    dummy_logits = torch.randn(B, 128, 512)
                    loss = torch.tensor(0.5, requires_grad=True)
                    from types import SimpleNamespace
                    return SimpleNamespace(loss=loss, logits=dummy_logits)

            mock_llm.return_value = FakeLLM()

            model = VLMModel(cfg)
            pixel_values = torch.randn(2, 3, 224, 224)
            input_ids = torch.randint(0, 100, (2, 128))
            attention_mask = torch.ones(2, 128)

            output = model(pixel_values, input_ids, attention_mask)
            assert "loss" in output
            assert output["loss"] is not None


class TestMLPConnector:
    """MLPConnector should project vision features to LLM dimension."""

    def test_projection_shape(self):
        from domains.training.multimodal import MLPConnector
        import torch

        connector = MLPConnector(vision_dim=768, llm_dim=512, hidden_dim=1024)
        x = torch.randn(2, 197, 768)
        out = connector(x)
        assert out.shape == (2, 197, 512)

    def test_grad_flow(self):
        from domains.training.multimodal import MLPConnector
        import torch

        connector = MLPConnector(vision_dim=768, llm_dim=256, hidden_dim=512)
        x = torch.randn(2, 50, 768, requires_grad=True)
        out = connector(x)
        loss = out.sum()
        loss.backward()
        for p in connector.parameters():
            assert p.grad is not None
            assert p.grad.abs().sum() > 0


class TestVLMRequestSchema:
    """VLMRequest pydantic model should validate correctly."""

    def test_defaults(self):
        from training.schemas import VLMRequest
        req = VLMRequest(dataset="coco")
        assert req.vision_encoder == "google/siglip-base-patch16-224"
        assert req.llm == "Qwen/Qwen2.5-0.5B-Instruct"
        assert req.stage1_epochs == 1
        assert req.batch_size == 4

    def test_overrides(self):
        from training.schemas import VLMRequest
        req = VLMRequest(
            dataset="coco",
            llm="Qwen/Qwen2.5-1.5B-Instruct",
            stage2_epochs=5,
            batch_size=2,
        )
        assert req.llm == "Qwen/Qwen2.5-1.5B-Instruct"
        assert req.stage2_epochs == 5
        assert req.batch_size == 2



