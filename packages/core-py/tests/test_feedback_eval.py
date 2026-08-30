"""Tests for domains.feedback — WorkflowConfig, OnlineLoRAUpdater config, EvalResult, BLEUScorer, PersonalityScore."""

from domains.feedback.workflow import WorkflowConfig
from domains.feedback.online_train import LoRAConfig
from domains.feedback.lora_eval import EvalResult, BLEUScorer, PersonalityScore


class TestWorkflowConfig:
    def test_defaults(self):
        cfg = WorkflowConfig()
        assert cfg.aggregate_interval_minutes == 60
        assert cfg.background_training_enabled is True
        assert cfg.export_format == "dpo"

    def test_custom_aggregate_interval(self):
        cfg = WorkflowConfig(aggregate_interval_minutes=30)
        assert cfg.aggregate_interval_minutes == 30

    def test_custom_prune_interval(self):
        cfg = WorkflowConfig(prune_interval_minutes=45)
        assert cfg.prune_interval_minutes == 45

    def test_custom_export_interval(self):
        cfg = WorkflowConfig(export_interval_hours=12)
        assert cfg.export_interval_hours == 12

    def test_custom_health_check(self):
        cfg = WorkflowConfig(health_check_interval_seconds=60)
        assert cfg.health_check_interval_seconds == 60

    def test_custom_bg_training(self):
        cfg = WorkflowConfig(background_training_interval_seconds=600)
        assert cfg.background_training_interval_seconds == 600

    def test_auto_dpo_interval(self):
        cfg = WorkflowConfig(auto_dpo_interval_minutes=180)
        assert cfg.auto_dpo_interval_minutes == 180

    def test_background_training_disabled(self):
        cfg = WorkflowConfig(background_training_enabled=False)
        assert cfg.background_training_enabled is False

    def test_auto_aggregate_threshold(self):
        cfg = WorkflowConfig(auto_aggregate_threshold=25)
        assert cfg.auto_aggregate_threshold == 25

    def test_auto_prune_threshold(self):
        cfg = WorkflowConfig(auto_prune_threshold=200)
        assert cfg.auto_prune_threshold == 200

    def test_min_feedback_for_aggregation(self):
        cfg = WorkflowConfig(min_feedback_for_aggregation=5)
        assert cfg.min_feedback_for_aggregation == 5

    def test_export_format_custom(self):
        cfg = WorkflowConfig(export_format="sft")
        assert cfg.export_format == "sft"

    def test_export_path_custom(self):
        cfg = WorkflowConfig(export_path="/tmp/exports")
        assert cfg.export_path == "/tmp/exports"

    def test_all_fields_settable(self):
        cfg = WorkflowConfig(
            aggregate_interval_minutes=10,
            prune_interval_minutes=20,
            auto_dpo_interval_minutes=30,
            export_interval_hours=2,
            health_check_interval_seconds=5,
            background_training_interval_seconds=60,
            background_training_enabled=False,
            auto_aggregate_threshold=10,
            auto_prune_threshold=50,
            min_feedback_for_aggregation=2,
            export_format="sft",
            export_path="/tmp/test",
        )
        assert cfg.aggregate_interval_minutes == 10
        assert cfg.prune_interval_minutes == 20
        assert cfg.export_interval_hours == 2
        assert cfg.background_training_enabled is False

    def test_init_no_args(self):
        cfg = WorkflowConfig()
        assert cfg is not None
        assert isinstance(cfg, WorkflowConfig)


class TestOnlineLoRAConfig:
    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.rank == 8
        assert cfg.alpha == 16
        assert cfg.dropout == 0.0
        assert "attn.c_attn" in cfg.target_modules

    def test_custom_rank(self):
        cfg = LoRAConfig(rank=16)
        assert cfg.rank == 16

    def test_custom_alpha(self):
        cfg = LoRAConfig(alpha=32)
        assert cfg.alpha == 32

    def test_custom_dropout(self):
        cfg = LoRAConfig(dropout=0.1)
        assert cfg.dropout == 0.1

    def test_custom_target_modules(self):
        cfg = LoRAConfig(target_modules=["layer1", "layer2"])
        assert cfg.target_modules == ["layer1", "layer2"]

    def test_target_modules_default_value(self):
        cfg = LoRAConfig()
        assert len(cfg.target_modules) == 4
        assert "attn.c_proj" in cfg.target_modules
        assert "mlp.c_fc" in cfg.target_modules
        assert "mlp.c_proj" in cfg.target_modules

    def test_rank_zero(self):
        cfg = LoRAConfig(rank=0)
        assert cfg.rank == 0

    def test_dropout_max(self):
        cfg = LoRAConfig(dropout=1.0)
        assert cfg.dropout == 1.0

    def test_alpha_rank_ratio(self):
        cfg = LoRAConfig(rank=4, alpha=8)
        assert cfg.alpha / cfg.rank == 2.0


class TestEvalResult:
    def test_fields(self):
        er = EvalResult(
            timestamp="2024-01-01", adapter_path=None, prompts=10,
            references=10, perplexity=5.0, bleu=0.7, avg_response_len=20.0,
            inference_time_sec=1.0, tokens_per_sec=50.0, personality_score=0.8,
        )
        assert er.perplexity == 5.0
        assert er.bleu == 0.7

    def test_to_dict(self):
        er = EvalResult(
            timestamp="2024-01-01", adapter_path=None, prompts=10,
            references=10, perplexity=5.0, bleu=0.7, avg_response_len=20.0,
            inference_time_sec=1.0, tokens_per_sec=50.0, personality_score=0.8,
        )
        d = er.to_dict()
        assert isinstance(d, dict)
        assert "perplexity" in d

    def test_to_dict_excludes_quality_delta(self):
        er = EvalResult(
            timestamp="2024-01-01", adapter_path=None, prompts=5,
            references=5, perplexity=2.0, bleu=0.9, avg_response_len=15.0,
            inference_time_sec=0.5, tokens_per_sec=30.0, personality_score=0.6,
            quality_delta=0.1,
        )
        d = er.to_dict()
        assert "quality_delta" not in d

    def test_adapter_path_none(self):
        er = EvalResult(
            timestamp="2024-01-01", adapter_path=None, prompts=5,
            references=5, perplexity=1.0, bleu=0.5, avg_response_len=10.0,
            inference_time_sec=0.1, tokens_per_sec=20.0, personality_score=0.4,
        )
        assert er.adapter_path is None

    def test_adapter_path_set(self):
        er = EvalResult(
            timestamp="2024-01-01", adapter_path="/tmp/model.npz", prompts=5,
            references=5, perplexity=1.0, bleu=0.5, avg_response_len=10.0,
            inference_time_sec=0.1, tokens_per_sec=20.0, personality_score=0.4,
        )
        assert er.adapter_path == "/tmp/model.npz"

    def test_timestamp_format(self):
        er = EvalResult(
            timestamp="2024-01-01T12:00:00", adapter_path=None, prompts=1,
            references=1, perplexity=1.0, bleu=1.0, avg_response_len=5.0,
            inference_time_sec=0.1, tokens_per_sec=10.0, personality_score=0.5,
        )
        assert "T" in er.timestamp

    def test_perplexity_none(self):
        er = EvalResult(
            timestamp="2024-01-01", adapter_path=None, prompts=1,
            references=0, perplexity=None, bleu=None, avg_response_len=5.0,
            inference_time_sec=0.1, tokens_per_sec=10.0, personality_score=0.5,
        )
        assert er.perplexity is None
        assert er.bleu is None

    def test_to_dict_has_all_expected_keys(self):
        er = EvalResult(
            timestamp="2024-01-01", adapter_path=None, prompts=10,
            references=10, perplexity=5.0, bleu=0.7, avg_response_len=20.0,
            inference_time_sec=1.0, tokens_per_sec=50.0, personality_score=0.8,
        )
        d = er.to_dict()
        expected_keys = {
            "timestamp", "adapter_path", "prompts", "references",
            "perplexity", "bleu", "avg_response_len", "inference_time_sec",
            "tokens_per_sec", "personality_score",
        }
        assert expected_keys == set(d.keys())

    def test_prompts_zero(self):
        er = EvalResult(
            timestamp="2024-01-01", adapter_path=None, prompts=0,
            references=0, perplexity=None, bleu=None, avg_response_len=0.0,
            inference_time_sec=0.0, tokens_per_sec=None, personality_score=None,
        )
        assert er.prompts == 0

    def test_inference_time_zero(self):
        er = EvalResult(
            timestamp="2024-01-01", adapter_path=None, prompts=1,
            references=1, perplexity=1.0, bleu=1.0, avg_response_len=5.0,
            inference_time_sec=0.0, tokens_per_sec=10.0, personality_score=0.5,
        )
        assert er.inference_time_sec == 0.0

    def test_high_perplexity(self):
        er = EvalResult(
            timestamp="2024-01-01", adapter_path=None, prompts=10,
            references=10, perplexity=10000.0, bleu=0.0, avg_response_len=100.0,
            inference_time_sec=10.0, tokens_per_sec=1.0, personality_score=0.1,
        )
        assert er.perplexity == 10000.0


class TestBLEUScorer:
    def test_identical(self):
        score = BLEUScorer.score("hello world", "hello world")
        assert float(score) == 100.0
    def test_different(self):
        score = BLEUScorer.score("cat dog", "fish bird")
        assert float(score) == 0.0
    def test_empty(self):
        score = BLEUScorer.score("", "hello")
        assert float(score) == 0.0
    def test_empty_both(self):
        score = BLEUScorer.score("", "")
        assert float(score) == 0.0
    def test_candidate_subset(self):
        score = BLEUScorer.score("hello", "hello world")
        assert float(score) > 0.0
    def test_reference_subset(self):
        score = BLEUScorer.score("hello world", "hello")
        assert float(score) > 0.0
    def test_single_word_match(self):
        score = BLEUScorer.score("the", "the")
        assert float(score) == 100.0
    def test_partial_match(self):
        score = BLEUScorer.score("hello world", "hello there world")
        assert float(score) > 0.0
    def test_max_n_one(self):
        score = BLEUScorer.score("a b c", "a b c", max_n=1)
        assert float(score) > 0.0
    def test_max_n_two(self):
        score = BLEUScorer.score("a b c", "a b c", max_n=2)
        assert float(score) > 0.0
    def test_case_sensitive(self):
        score_upper = BLEUScorer.score("Hello World", "hello world")
        assert float(score_upper) < 100.0
    def test_long_candidate(self):
        score = BLEUScorer.score("a " * 50, "a")
        assert float(score) > 0.0
    def test_long_reference(self):
        score = BLEUScorer.score("a", "a " * 50)
        assert float(score) > 0.0
    def test_whitespace_only_candidate(self):
        score = BLEUScorer.score("   ", "hello")
        assert float(score) == 0.0
    def test_whitespace_only_reference(self):
        score = BLEUScorer.score("hello", "   ")
        assert float(score) == 0.0
    def test_score_is_float(self):
        score = BLEUScorer.score("hello world", "hello world")
        assert isinstance(score, float)
    def test_score_non_negative(self):
        score = BLEUScorer.score("random text", "other text")
        assert float(score) >= 0.0
    def test_perfect_score_upper_bound(self):
        score = BLEUScorer.score("exact match here", "exact match here")
        assert float(score) <= 100.0
    def test_min_score_lower_bound(self):
        score = BLEUScorer.score("xyz", "abc")
        assert float(score) >= 0.0
    def test_punctuation(self):
        score = BLEUScorer.score("hello, world!", "hello, world!")
        assert float(score) == 100.0
    def test_numeric_tokens(self):
        score = BLEUScorer.score("1 2 3", "1 2 3")
        assert float(score) == 100.0
    def test_special_characters(self):
        score = BLEUScorer.score("@#$", "@#$")
        assert float(score) == 100.0
    def test_empty_reference(self):
        score = BLEUScorer.score("hello", "")
        assert float(score) == 0.0


class TestPersonalityScore:
    def test_fields(self):
        ps = PersonalityScore(
            soul_name="test", warmth_score=0.8, creativity_score=0.7,
            formality_score=0.5, coherence_score=0.9, overall=0.75,
        )
        assert ps.soul_name == "test"
        assert ps.overall == 0.75

    def test_to_dict(self):
        ps = PersonalityScore(
            soul_name="test", warmth_score=0.8, creativity_score=0.7,
            formality_score=0.5, coherence_score=0.9, overall=0.75,
        )
        d = ps.to_dict()
        assert d["soul"] == "test"
        assert d["warmth"] == 0.8

    def test_to_dict_all_keys(self):
        ps = PersonalityScore(
            soul_name="soul", warmth_score=0.1, creativity_score=0.2,
            formality_score=0.3, coherence_score=0.4, overall=0.5,
        )
        d = ps.to_dict()
        expected_keys = {"soul", "warmth", "creativity", "formality", "coherence", "overall"}
        assert expected_keys == set(d.keys())

    def test_to_dict_values_match_fields(self):
        ps = PersonalityScore(
            soul_name="a", warmth_score=0.99, creativity_score=0.88,
            formality_score=0.77, coherence_score=0.66, overall=0.55,
        )
        d = ps.to_dict()
        assert d["warmth"] == ps.warmth_score
        assert d["creativity"] == ps.creativity_score
        assert d["formality"] == ps.formality_score
        assert d["coherence"] == ps.coherence_score
        assert d["overall"] == ps.overall

    def test_zero_scores(self):
        ps = PersonalityScore(
            soul_name="empty", warmth_score=0.0, creativity_score=0.0,
            formality_score=0.0, coherence_score=0.0, overall=0.0,
        )
        assert ps.overall == 0.0
        d = ps.to_dict()
        assert d["warmth"] == 0.0

    def test_max_scores(self):
        ps = PersonalityScore(
            soul_name="max", warmth_score=1.0, creativity_score=1.0,
            formality_score=1.0, coherence_score=1.0, overall=1.0,
        )
        assert ps.overall == 1.0

    def test_empty_soul_name(self):
        ps = PersonalityScore(
            soul_name="", warmth_score=0.5, creativity_score=0.5,
            formality_score=0.5, coherence_score=0.5, overall=0.5,
        )
        assert ps.soul_name == ""

    def test_overall_independent_of_traits(self):
        ps = PersonalityScore(
            soul_name="test", warmth_score=1.0, creativity_score=1.0,
            formality_score=1.0, coherence_score=1.0, overall=0.0,
        )
        assert ps.overall == 0.0
        d = ps.to_dict()
        assert d["overall"] == 0.0

    def test_negative_scores_accepted(self):
        ps = PersonalityScore(
            soul_name="neg", warmth_score=-0.1, creativity_score=-0.2,
            formality_score=-0.3, coherence_score=-0.4, overall=-0.5,
        )
        assert ps.overall == -0.5
