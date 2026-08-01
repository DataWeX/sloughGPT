"""Tests for domains.training.auto_config: dataset analysis and auto-config engine."""

import json

from domains.training.auto_config import (
    DatasetAnalysis,
    TrainingConfig,
    analyse_dataset,
    auto_configure,
    plain_language_verdict,
    _build_explanation,
    _pick_batch_size,
    _pick_epochs,
    _pick_lr,
    _pick_method,
    _pick_model,
    _pick_seq_length,
    _pick_warmup,
)


class TestDatasetAnalysis:
    def test_defaults(self):
        a = DatasetAnalysis(path="/x", format="text")
        assert a.sample_count == 0
        assert a.char_count == 0
        assert a.word_count == 0
        assert a.avg_line_length == 0.0
        assert a.preview_lines == []

    def test_is_dialogue_from_markers(self):
        a = DatasetAnalysis(path="/x", format="text", has_dialogue_markers=True)
        assert a.is_dialogue is True

    def test_is_dialogue_from_role_fields(self):
        a = DatasetAnalysis(path="/x", format="messages", has_role_fields=True)
        assert a.is_dialogue is True

    def test_not_dialogue(self):
        a = DatasetAnalysis(path="/x", format="text")
        assert a.is_dialogue is False

    def test_is_messages_format(self):
        assert DatasetAnalysis(path="/x", format="messages").is_messages_format is True
        assert DatasetAnalysis(path="/x", format="text").is_messages_format is False

    def test_size_categories(self):
        assert DatasetAnalysis(path="/x", format="text", word_count=999).size_category == "tiny"
        assert DatasetAnalysis(path="/x", format="text", word_count=1000).size_category == "small"
        assert DatasetAnalysis(path="/x", format="text", word_count=9_999).size_category == "small"
        assert DatasetAnalysis(path="/x", format="text", word_count=10_000).size_category == "medium"
        assert DatasetAnalysis(path="/x", format="text", word_count=99_999).size_category == "medium"
        assert DatasetAnalysis(path="/x", format="text", word_count=100_000).size_category == "large"


class TestTrainingConfig:
    def test_defaults(self):
        c = TrainingConfig()
        assert c.model == "gpt2"
        assert c.method == "finetune"
        assert c.epochs == 3
        assert c.batch_size == 4
        assert c.learning_rate == 2e-4
        assert c.max_seq_length == 512
        assert c.warmup_steps == 50
        assert c.weight_decay == 0.01
        assert c.use_lora is True
        assert c.lora_rank == 8
        assert c.lora_alpha == 16
        assert c.rl_post_train is False
        assert c.analysis is None
        assert c.explanation == ""

    def test_to_dict_keys(self):
        c = TrainingConfig(dataset="d", data_path="/p")
        d = c.to_dict()
        assert d["dataset"] == "d"
        assert d["data_path"] == "/p"
        assert d["model"] == "gpt2"
        assert d["epochs"] == 3
        assert d["use_lora"] is True
        assert d["rl_post_train"] is False
        assert set(d) == {
            "model", "dataset", "data_path", "epochs", "batch_size", "learning_rate",
            "max_seq_length", "warmup_steps", "weight_decay", "use_lora", "lora_rank",
            "lora_alpha", "rl_post_train", "rl_num_generations", "rl_learning_rate",
            "rl_kl_coef", "rl_reward_mode",
        }


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


class TestAnalyseDataset:
    def test_missing_path_unknown(self, tmp_path):
        a = analyse_dataset(str(tmp_path / "nope.txt"))
        assert a.format == "unknown"
        assert a.path == str(tmp_path / "nope.txt")

    def test_plain_text(self, tmp_path):
        p = _write(tmp_path / "input.txt", "hello world\nsecond line\n\nthird line\n")
        a = analyse_dataset(p)
        assert a.format == "text"
        assert a.sample_count == 3
        assert a.char_count == len("hello world\nsecond line\n\nthird line\n")
        assert a.word_count == 6
        assert a.preview_lines == ["hello world", "second line", "third line"]

    def test_jsonl_messages_format(self, tmp_path):
        lines = [json.dumps({"messages": [{"role": "user", "content": "hi"}]}) for _ in range(3)]
        p = _write(tmp_path / "chat.jsonl", "\n".join(lines) + "\n")
        a = analyse_dataset(p)
        assert a.format == "messages"
        assert a.has_role_fields is True
        assert a.sample_count == 3

    def test_jsonl_text_format(self, tmp_path):
        lines = [json.dumps({"text": "some plain text"}) for _ in range(3)]
        p = _write(tmp_path / "data.jsonl", "\n".join(lines) + "\n")
        a = analyse_dataset(p)
        assert a.format == "jsonl"
        assert a.has_role_fields is False

    def test_json_first_line_detected_in_txt(self, tmp_path):
        p = _write(tmp_path / "data.txt", json.dumps({"text": "x"}) + "\n")
        a = analyse_dataset(p)
        assert a.format == "jsonl"

    def test_bad_json_lines_skipped(self, tmp_path):
        p = _write(tmp_path / "data.jsonl", "{not json}\n")
        a = analyse_dataset(p)
        assert a.format == "jsonl"
        assert a.has_role_fields is False

    def test_dialogue_markers_detected(self, tmp_path):
        lines = ["User: hello", "Assistant: hi there", "User: how are you"]
        p = _write(tmp_path / "chat.txt", "\n".join(lines) + "\n")
        a = analyse_dataset(p)
        assert a.has_dialogue_markers is True
        assert a.is_dialogue is True

    def test_no_dialogue_markers(self, tmp_path):
        p = _write(tmp_path / "plain.txt", "the cat sat on the mat\n" * 20)
        a = analyse_dataset(p)
        assert a.has_dialogue_markers is False


class TestAutoConfigure:
    def test_preferred_model_override(self, tmp_path):
        p = _write(tmp_path / "input.txt", "some words here\n" * 10)
        cfg = auto_configure("mydata", p, available_models=["gpt2"], preferred_model="qwen2.5-0.5b-instruct")
        assert cfg.model == "qwen2.5-0.5b-instruct"
        assert cfg.dataset == "mydata"
        assert cfg.data_path == p

    def test_tiny_text_uses_distill(self, tmp_path):
        p = _write(tmp_path / "input.txt", "a few words\n")
        cfg = auto_configure("d", p, available_models=["gpt2"])
        assert cfg.method == "distill"
        assert cfg.epochs == 10
        assert cfg.batch_size == 2
        assert cfg.learning_rate == 1e-3
        assert cfg.use_lora is False

    def test_dialogue_picks_chat_model_and_finetune(self, tmp_path):
        p = _write(tmp_path / "chat.txt", "User: hi\nAssistant: hello\n" * 30)
        cfg = auto_configure("d", p, available_models=["gpt2", "qwen2.5-0.5b-instruct"])
        assert cfg.model == "qwen2.5-0.5b-instruct"
        assert cfg.method == "finetune"
        assert cfg.use_lora is True

    def test_messages_format_finetune(self, tmp_path):
        p = _write(tmp_path / "chat.jsonl", json.dumps({"messages": [{}]}) + "\n")
        cfg = auto_configure("d", p, available_models=["gpt2"])
        assert cfg.method == "finetune"
        assert cfg.use_lora is True

    def test_chat_model_enables_rl(self, tmp_path):
        p = _write(tmp_path / "data.jsonl", json.dumps({"messages": [{}]}) + "\n")
        cfg = auto_configure("d", p, available_models=["qwen2.5-0.5b-instruct"])
        assert cfg.rl_post_train is True

    def test_non_chat_model_no_rl(self, tmp_path):
        p = _write(tmp_path / "data.jsonl", json.dumps({"messages": [{}]}) + "\n")
        cfg = auto_configure("d", p, available_models=["gpt2"])
        assert cfg.rl_post_train is False

    def test_explanation_populated(self, tmp_path):
        p = _write(tmp_path / "input.txt", "the cat sat on the mat\n" * 50)
        cfg = auto_configure("d", p, available_models=["gpt2"])
        assert cfg.explanation
        assert "gpt2" in cfg.explanation or "teacher" in cfg.explanation

    def test_default_models_list(self, tmp_path):
        p = _write(tmp_path / "input.txt", "some words\n" * 100)
        cfg = auto_configure("d", p, available_models=None)
        assert cfg.model == "gpt2"

    def test_analysis_attached(self, tmp_path):
        p = _write(tmp_path / "input.txt", "the cat sat on the mat\n" * 100)
        cfg = auto_configure("d", p, available_models=["gpt2"])
        assert cfg.analysis is not None
        assert cfg.analysis.format == "text"


class TestPickHelpers:
    def test_pick_model_chat_for_dialogue(self):
        analysis = DatasetAnalysis(path="/x", format="messages", has_role_fields=True)
        assert _pick_model(analysis, ["gpt2", "qwen2.5-0.5b-instruct"]) == "qwen2.5-0.5b-instruct"

    def test_pick_model_small_for_tiny(self):
        analysis = DatasetAnalysis(path="/x", format="text", word_count=10)
        assert _pick_model(analysis, ["big-model", "gpt2"]) == "gpt2"

    def test_pick_model_prefers_qwen_otherwise(self):
        analysis = DatasetAnalysis(path="/x", format="text", word_count=50_000)
        assert _pick_model(analysis, ["gpt2", "qwen3-4b"]) == "qwen3-4b"

    def test_pick_model_first_available(self):
        analysis = DatasetAnalysis(path="/x", format="text", word_count=50_000)
        assert _pick_model(analysis, ["a-model", "b-model"]) == "a-model"

    def test_pick_method_messages(self):
        a = DatasetAnalysis(path="/x", format="messages", has_role_fields=True)
        assert _pick_method(a) == "finetune"

    def test_pick_method_dialogue(self):
        a = DatasetAnalysis(path="/x", format="text", has_dialogue_markers=True)
        assert _pick_method(a) == "finetune"

    def test_pick_method_tiny_distill(self):
        assert _pick_method(DatasetAnalysis(path="/x", format="text", word_count=100)) == "distill"

    def test_pick_method_large_finetune(self):
        assert _pick_method(DatasetAnalysis(path="/x", format="text", word_count=200_000)) == "finetune"

    def test_pick_epochs_mapping(self):
        assert _pick_epochs(DatasetAnalysis(path="/x", format="text", word_count=10)) == 10
        assert _pick_epochs(DatasetAnalysis(path="/x", format="text", word_count=5_000)) == 5
        assert _pick_epochs(DatasetAnalysis(path="/x", format="text", word_count=50_000)) == 3
        assert _pick_epochs(DatasetAnalysis(path="/x", format="text", word_count=500_000)) == 2

    def test_pick_batch_size_mapping(self):
        assert _pick_batch_size(DatasetAnalysis(path="/x", format="text", word_count=10)) == 2
        assert _pick_batch_size(DatasetAnalysis(path="/x", format="text", word_count=5_000)) == 4
        assert _pick_batch_size(DatasetAnalysis(path="/x", format="text", word_count=500_000)) == 8

    def test_pick_lr(self):
        assert _pick_lr("distill", DatasetAnalysis(path="/x", format="text", word_count=500_000)) == 1e-3
        assert _pick_lr("finetune", DatasetAnalysis(path="/x", format="text", word_count=10)) == 1e-5
        assert _pick_lr("finetune", DatasetAnalysis(path="/x", format="text", word_count=5_000)) == 5e-5
        assert _pick_lr("finetune", DatasetAnalysis(path="/x", format="text", word_count=50_000)) == 2e-4

    def test_pick_seq_length(self):
        a = DatasetAnalysis(path="/x", format="text", avg_line_length=10)
        assert _pick_seq_length(a) == 128
        a = DatasetAnalysis(path="/x", format="text", avg_line_length=100)
        assert _pick_seq_length(a) == 256
        a = DatasetAnalysis(path="/x", format="text", avg_line_length=500)
        assert _pick_seq_length(a) == 512

    def test_pick_warmup(self):
        assert _pick_warmup(DatasetAnalysis(path="/x", format="text", sample_count=10)) == 10
        assert _pick_warmup(DatasetAnalysis(path="/x", format="text", sample_count=100)) == 30
        assert _pick_warmup(DatasetAnalysis(path="/x", format="text", sample_count=1000)) == 50


class TestBuildExplanation:
    def test_messages(self):
        a = DatasetAnalysis(path="/x", format="messages", sample_count=42, word_count=0)
        text = _build_explanation(a, "finetune", "gpt2", 3, False)
        assert "42 conversation turns" in text
        assert "Fine-tuning gpt2" in text

    def test_dialogue_text(self):
        a = DatasetAnalysis(path="/x", format="text", has_dialogue_markers=True, word_count=1234)
        text = _build_explanation(a, "finetune", "gpt2", 3, False)
        assert "1,234 words" in text

    def test_plain_text(self):
        a = DatasetAnalysis(path="/x", format="text", word_count=999)
        text = _build_explanation(a, "finetune", "gpt2", 3, False)
        assert "999 words of text" in text

    def test_distill_mentions_teacher(self):
        a = DatasetAnalysis(path="/x", format="text", word_count=999)
        text = _build_explanation(a, "distill", "gpt2", 10, False)
        assert "teacher" in text

    def test_tiny_explains_epochs(self):
        a = DatasetAnalysis(path="/x", format="text", word_count=100)
        text = _build_explanation(a, "finetune", "gpt2", 10, False)
        assert "10 epochs" in text
        assert "Small dataset" in text

    def test_large_epochs(self):
        a = DatasetAnalysis(path="/x", format="text", word_count=500_000)
        text = _build_explanation(a, "finetune", "gpt2", 2, False)
        assert "2 epochs is enough" in text

    def test_rl_line(self):
        a = DatasetAnalysis(path="/x", format="text", word_count=5_000)
        text = _build_explanation(a, "finetune", "gpt2", 5, True)
        assert "personality reinforcement" in text


class TestPlainLanguageVerdict:
    def test_improved_base(self):
        text = plain_language_verdict({"verdict": "improved"})
        assert text == "Your AI learned to give better answers."

    def test_improved_with_metrics(self):
        text = plain_language_verdict({
            "verdict": "improved",
            "perplexity_improvement_pct": 12.4,
            "bleu_delta": 0.1,
            "personality_delta": 0.2,
        })
        assert "12% more coherent" in text
        assert "more relevant" in text
        assert "more consistent" in text

    def test_improved_personality_threshold(self):
        text = plain_language_verdict({
            "verdict": "improved",
            "perplexity_improvement_pct": 5.0,
            "bleu_delta": -1.0,
            "personality_delta": 0.03,
        })
        assert "more consistent" not in text
        assert "5% more coherent" in text

    def test_improved_no_perplexity(self):
        text = plain_language_verdict({"verdict": "improved", "perplexity_improvement_pct": None})
        assert text == "Your AI learned to give better answers."

    def test_degraded(self):
        text = plain_language_verdict({
            "verdict": "degraded",
            "perplexity_improvement_pct": -8.0,
        })
        assert "made some things worse" in text
        assert "Coherence dropped by 8%" in text
        assert "fewer epochs" in text

    def test_degraded_no_drop(self):
        text = plain_language_verdict({"verdict": "degraded", "perplexity_improvement_pct": None})
        assert "Try training for fewer epochs" in text

    def test_mixed(self):
        text = plain_language_verdict({
            "verdict": "mixed",
            "perplexity_improvement_pct": 3.0,
        })
        assert "Mixed results" in text
        assert "Coherence improved by 3%" in text
        assert "adjusting the training settings" in text

    def test_mixed_worsened(self):
        text = plain_language_verdict({
            "verdict": "mixed",
            "perplexity_improvement_pct": -4.0,
        })
        assert "worsened by 4%" in text

    def test_unknown_falls_to_mixed(self):
        text = plain_language_verdict({})
        assert "Mixed results" in text
