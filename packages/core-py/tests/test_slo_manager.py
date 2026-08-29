"""Tests for domains.inference.slo_manager — SloInfo."""

from domains.inference.slo_manager import SloInfo


class TestSloInfo:
    def test_defaults(self):
        si = SloInfo(name="test", path="/tmp/test.soul")
        assert si.name == "test"
        assert si.path == "/tmp/test.soul"
        assert si.description == ""
        assert si.personality == {}
        assert si.traits == []
        assert si.loaded_at is None

    def test_custom(self):
        si = SloInfo(
            name="custom", path="/tmp/custom.soul", description="a custom soul",
            personality={"warmth": 0.8}, traits=["friendly", "curious"],
            version="2.0", size_mb=1.5,
        )
        assert si.description == "a custom soul"
        assert si.personality["warmth"] == 0.8
        assert "friendly" in si.traits
        assert si.version == "2.0"
        assert si.size_mb == 1.5

    def test_training_fields(self):
        si = SloInfo(
            name="trained", path="/tmp/t.soul",
            epochs_trained=10, final_train_loss=0.5, final_val_loss=0.6,
            lineage="nanogpt", base_model="gpt2",
        )
        assert si.epochs_trained == 10
        assert si.final_train_loss == 0.5
        assert si.lineage == "nanogpt"
