"""Tests for domains.training.data_import — ImportResult; domains.training.hf_lora_finetune — HFLoraConfig."""

from domains.training.data_import import ImportResult
from domains.training.hf_lora_finetune import HFLoraConfig


class TestImportResult:
    def test_fields(self):
        ir = ImportResult(
            success=True, name="test", source="url",
            files_imported=5, total_chars=1000, output_path="/tmp/out",
        )
        assert ir.success is True
        assert ir.files_imported == 5
        assert ir.error is None

    def test_error(self):
        ir = ImportResult(
            success=False, name="test", source="url",
            files_imported=0, total_chars=0, output_path="", error="timeout",
        )
        assert ir.success is False
        assert ir.error == "timeout"


class TestHFLoraConfig:
    def test_defaults(self):
        hc = HFLoraConfig()
        assert hc.rank == 8
        assert hc.alpha == 16.0
        assert hc.epochs == 3
        assert hc.batch_size == 8

    def test_custom(self):
        hc = HFLoraConfig(model_path="m.pt", rank=16, epochs=5)
        assert hc.model_path == "m.pt"
        assert hc.rank == 16
        assert hc.epochs == 5

    def test_auto_adapter_name(self):
        hc = HFLoraConfig(model_path="gpt2.safetensors")
        assert "gpt2" in hc.adapter_name
