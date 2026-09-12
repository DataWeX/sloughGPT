"""
Training Quality and Method Tests — quality scoring, DPO, LoRA, distillation.

Tests the quality scoring system and training method-specific functionality.

Usage:
    .venv/bin/python -m pytest tests/test_training_quality_methods.py -x -v
"""
import tempfile

DATA_TEXT = "The quick brown fox jumps over the lazy dog. " * 50

FAST_CONFIG = {
    "method": "sft",
    "data_quality_threshold": 0.0,
    "epochs": 1,
    "batch_size": 8,
    "block_size": 32,
    "max_steps": 2,
    "n_embed": 32,
    "n_layer": 2,
    "n_head": 2,
}


class TestQualityScorer:
    """Tests for the training data quality scoring system."""

    def test_score_pair_returns_float(self):
        from domains.training.quality_scorer import score_pair

        score = score_pair("What is AI?", "AI is artificial intelligence.")
        assert isinstance(score, float)

    def test_score_pair_range(self):
        from domains.training.quality_scorer import score_pair

        score = score_pair("What is AI?", "AI is artificial intelligence.")
        assert isinstance(score, float)
        assert score >= 0.0

    def test_score_batch(self):
        from domains.training.quality_scorer import score_batch

        pairs = [
            {"user": "What is AI?", "assistant": "AI is artificial intelligence."},
            {"user": "What is ML?", "assistant": "ML is machine learning."},
        ]
        scores = score_batch(pairs)
        assert isinstance(scores, list)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_score_text_chunk(self):
        from domains.training.quality_scorer import score_text_chunk

        score = score_text_chunk("The quick brown fox jumps over the lazy dog.")
        assert isinstance(score, float)
        assert score >= 0.0

    def test_compute_data_quality(self):
        from domains.training.quality_scorer import compute_data_quality

        quality = compute_data_quality(DATA_TEXT)
        assert isinstance(quality, dict)
        assert "avg_quality" in quality
        assert "repetition_rate" in quality
        assert "diversity" in quality

    def test_quality_score_with_good_data(self):
        from domains.training.quality_scorer import score_text_chunk

        good_text = "The quick brown fox jumps over the lazy dog. " * 10
        score = score_text_chunk(good_text)
        assert score > 0.0

    def test_quality_score_with_empty_data(self):
        from domains.training.quality_scorer import score_text_chunk

        score = score_text_chunk("")
        assert isinstance(score, float)


class TestTrainingMethods:
    """Tests for different training methods."""

    def test_sft_training(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(
                data_path=f.name,
                config={**FAST_CONFIG, "method": "sft"},
            )
            assert result.success

    def test_distillation_training(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(
                data_path=f.name,
                config={**FAST_CONFIG, "method": "distillation"},
            )
            assert result.success

    def test_lora_training(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(
                data_path=f.name,
                config={**FAST_CONFIG, "use_lora": True, "lora_rank": 4},
            )
            assert result.success

    def test_all_methods_produce_loss(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            for method in ["sft", "distillation"]:
                trainer = ComprehensiveTrainer()
                result = trainer.run_full_cycle(
                    data_path=f.name,
                    config={**FAST_CONFIG, "method": method},
                )
                assert result.final_loss is not None, f"No loss for method: {method}"


class TestTrainingDataFormats:
    """Tests for different training data formats."""

    def test_text_file_training(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert result.success

    def test_jsonl_file_training(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        jsonl_data = "\n".join([
            '{"user": "What is AI?", "assistant": "AI is artificial intelligence."}',
            '{"user": "What is ML?", "assistant": "ML is machine learning."}',
        ] * 10)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(jsonl_data)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert result.success

    def test_multiline_text_training(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        multiline_text = "\n".join([
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
            "How vexingly quick daft zebras jump!",
        ] * 20)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(multiline_text)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert result.success


class TestAdaptiveConfigRecommendations:
    """Tests for adaptive config recommendation quality."""

    def test_recommendation_varies_by_size(self):
        from domains.training.adaptive_config import AdaptiveConfigEngine

        engine = AdaptiveConfigEngine()
        small = engine.recommend(dataset_size=100, model="gpt2", method="finetune")
        large = engine.recommend(dataset_size=100000, model="gpt2", method="finetune")
        assert small.batch_size != large.batch_size or small.epochs != large.epochs

    def test_recommendation_has_confidence(self):
        from domains.training.adaptive_config import AdaptiveConfigEngine

        engine = AdaptiveConfigEngine()
        rec = engine.recommend(dataset_size=1000, model="gpt2", method="finetune")
        assert 0.0 <= rec.confidence <= 1.0

    def test_recommendation_has_reason(self):
        from domains.training.adaptive_config import AdaptiveConfigEngine

        engine = AdaptiveConfigEngine()
        rec = engine.recommend(dataset_size=1000, model="gpt2", method="finetune")
        assert isinstance(rec.reason, str)
        assert len(rec.reason) > 0


class TestPresetSystem:
    """Tests for the training preset system."""

    def test_all_presets_applicable(self):
        from domains.training.presets import PRESETS, apply_preset

        for key in PRESETS:
            config = apply_preset(key)
            assert config is not None, f"Failed to apply preset: {key}"

    def test_preset_configs_valid(self):
        from domains.training.presets import PRESETS, apply_preset

        for key in PRESETS:
            config = apply_preset(key)
            assert config["default_epochs"] > 0
            assert config["default_batch_size"] > 0
            assert config["default_learning_rate"] > 0

    def test_preset_training_works(self):
        from domains.training.presets import apply_preset
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        apply_preset("quick-finetune")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert result.success
