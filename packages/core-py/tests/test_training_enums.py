"""Tests for domains.training — RLHFMetric, RLHFConfig, TrackerBackend, TrackingConfig, LoRAType, LoRAConfig, DataFormat."""

from domains.training.rlhf import RLHFMetric, RLHFConfig
from domains.training.tracking import TrackerBackend, TrackingConfig
from domains.training.lora import LoRAType, LoRAConfig
from domains.training import DataFormat


class TestRLHFMetric:
    def test_all_members(self):
        assert len(RLHFMetric) == 6

    def test_values(self):
        assert RLHFMetric.REWARD.value == "reward"
        assert RLHFMetric.KL_DIVERGENCE.value == "kl_divergence"

    def test_value_loss(self):
        assert RLHFMetric.VALUE_LOSS.value == "value_loss"

    def test_policy_loss(self):
        assert RLHFMetric.POLICY_LOSS.value == "policy_loss"

    def test_entropy(self):
        assert RLHFMetric.ENTROPY.value == "entropy"

    def test_advantage(self):
        assert RLHFMetric.ADVANTAGE.value == "advantage"

    def test_iteration(self):
        names = [m.name for m in RLHFMetric]
        assert "REWARD" in names
        assert "KL_DIVERGENCE" in names

    def test_membership_check(self):
        assert RLHFMetric.REWARD in RLHFMetric

    def test_is_hashable(self):
        assert hash(RLHFMetric.REWARD) is not None

    def test_equality(self):
        assert RLHFMetric.REWARD == RLHFMetric.REWARD

    def test_inequality(self):
        assert RLHFMetric.REWARD != RLHFMetric.KL_DIVERGENCE

    def test_name_matches_value(self):
        for m in RLHFMetric:
            assert m.name.lower() == m.value


class TestRLHFConfig:
    def test_defaults(self):
        cfg = RLHFConfig()
        assert cfg.ppo_epochs == 4
        assert cfg.clip_epsilon == 0.2
        assert cfg.gamma == 1.0
        assert cfg.lam == 0.95
        assert cfg.use_ref_model is True

    def test_default_num_mini_batches(self):
        cfg = RLHFConfig()
        assert cfg.num_mini_batches == 4

    def test_default_value_loss_coef(self):
        cfg = RLHFConfig()
        assert cfg.value_loss_coef == 0.5

    def test_default_entropy_coef(self):
        cfg = RLHFConfig()
        assert cfg.entropy_coef == 0.01

    def test_default_max_grad_norm(self):
        cfg = RLHFConfig()
        assert cfg.max_grad_norm == 1.0

    def test_default_gen_max_length(self):
        cfg = RLHFConfig()
        assert cfg.gen_max_length == 512

    def test_default_gen_temperature(self):
        cfg = RLHFConfig()
        assert cfg.gen_temperature == 1.0

    def test_default_gen_top_p(self):
        cfg = RLHFConfig()
        assert cfg.gen_top_p == 0.9

    def test_default_reward_model_path(self):
        cfg = RLHFConfig()
        assert cfg.reward_model_path is None

    def test_default_ref_model_path(self):
        cfg = RLHFConfig()
        assert cfg.ref_model_path is None

    def test_custom_ppo_epochs(self):
        cfg = RLHFConfig(ppo_epochs=8)
        assert cfg.ppo_epochs == 8

    def test_custom_clip_epsilon(self):
        cfg = RLHFConfig(clip_epsilon=0.1)
        assert cfg.clip_epsilon == 0.1

    def test_custom_gamma(self):
        cfg = RLHFConfig(gamma=0.99)
        assert cfg.gamma == 0.99

    def test_custom_lam(self):
        cfg = RLHFConfig(lam=0.9)
        assert cfg.lam == 0.9

    def test_custom_use_ref_model(self):
        cfg = RLHFConfig(use_ref_model=False)
        assert cfg.use_ref_model is False

    def test_custom_reward_model_path(self):
        cfg = RLHFConfig(reward_model_path="/tmp/rm.pt")
        assert cfg.reward_model_path == "/tmp/rm.pt"

    def test_custom_ref_model_path(self):
        cfg = RLHFConfig(ref_model_path="/tmp/ref.pt")
        assert cfg.ref_model_path == "/tmp/ref.pt"

    def test_custom_gen_max_length(self):
        cfg = RLHFConfig(gen_max_length=1024)
        assert cfg.gen_max_length == 1024

    def test_custom_gen_temperature(self):
        cfg = RLHFConfig(gen_temperature=0.7)
        assert cfg.gen_temperature == 0.7

    def test_custom_gen_top_p(self):
        cfg = RLHFConfig(gen_top_p=0.95)
        assert cfg.gen_top_p == 0.95

    def test_dataclass_fields_count(self):
        import dataclasses
        fields = [f.name for f in dataclasses.fields(RLHFConfig)]
        assert len(fields) >= 12

    def test_dataclass_field_names(self):
        import dataclasses
        fields = {f.name for f in dataclasses.fields(RLHFConfig)}
        expected = {
            "ppo_epochs", "num_mini_batches", "clip_epsilon", "value_loss_coef",
            "entropy_coef", "max_grad_norm", "gamma", "lam", "reward_model_path",
            "ref_model_path", "use_ref_model", "gen_max_length", "gen_temperature",
            "gen_top_p",
        }
        assert expected.issubset(fields)


class TestTrackerBackend:
    def test_all_members(self):
        assert len(TrackerBackend) >= 3

    def test_values(self):
        assert TrackerBackend.MLFLOW.value == "mlflow"
        assert TrackerBackend.WANDB.value == "wandb"

    def test_comet_value(self):
        assert TrackerBackend.COMET.value == "comet"

    def test_none_value(self):
        assert TrackerBackend.NONE.value == "none"

    def test_iteration(self):
        names = [m.name for m in TrackerBackend]
        assert "MLFLOW" in names
        assert "WANDB" in names

    def test_membership_check(self):
        assert TrackerBackend.MLFLOW in TrackerBackend

    def test_equality(self):
        assert TrackerBackend.MLFLOW == TrackerBackend.MLFLOW

    def test_inequality(self):
        assert TrackerBackend.MLFLOW != TrackerBackend.WANDB

    def test_is_hashable(self):
        assert hash(TrackerBackend.MLFLOW) is not None

    def test_name_matches_value(self):
        for m in TrackerBackend:
            assert m.name.lower() == m.value


class TestTrackingConfig:
    def test_defaults(self):
        cfg = TrackingConfig()
        assert cfg.backend == TrackerBackend.NONE

    def test_default_experiment_name(self):
        cfg = TrackingConfig()
        assert cfg.experiment_name == "sloughgpt_experiment"

    def test_default_run_name(self):
        cfg = TrackingConfig()
        assert cfg.run_name is None

    def test_default_tracking_uri(self):
        cfg = TrackingConfig()
        assert cfg.tracking_uri is None

    def test_default_api_key(self):
        cfg = TrackingConfig()
        assert cfg.api_key is None

    def test_default_project(self):
        cfg = TrackingConfig()
        assert cfg.project == "sloughgpt"

    def test_default_entity(self):
        cfg = TrackingConfig()
        assert cfg.entity is None

    def test_default_job_type(self):
        cfg = TrackingConfig()
        assert cfg.job_type is None

    def test_default_tags(self):
        cfg = TrackingConfig()
        assert cfg.tags is None

    def test_custom_backend(self):
        cfg = TrackingConfig(backend=TrackerBackend.MLFLOW)
        assert cfg.backend == TrackerBackend.MLFLOW

    def test_custom_experiment_name(self):
        cfg = TrackingConfig(experiment_name="test_exp")
        assert cfg.experiment_name == "test_exp"

    def test_custom_run_name(self):
        cfg = TrackingConfig(run_name="run_1")
        assert cfg.run_name == "run_1"

    def test_custom_tracking_uri(self):
        cfg = TrackingConfig(tracking_uri="http://localhost:5000")
        assert cfg.tracking_uri == "http://localhost:5000"

    def test_custom_project(self):
        cfg = TrackingConfig(project="my_project")
        assert cfg.project == "my_project"

    def test_custom_entity(self):
        cfg = TrackingConfig(entity="my_team")
        assert cfg.entity == "my_team"

    def test_custom_job_type(self):
        cfg = TrackingConfig(job_type="train")
        assert cfg.job_type == "train"

    def test_custom_tags(self):
        cfg = TrackingConfig(tags=["tag1", "tag2"])
        assert cfg.tags == ["tag1", "tag2"]

    def test_dataclass_fields_count(self):
        import dataclasses
        fields = [f.name for f in dataclasses.fields(TrackingConfig)]
        assert len(fields) >= 8

    def test_dataclass_field_names(self):
        import dataclasses
        fields = {f.name for f in dataclasses.fields(TrackingConfig)}
        assert "backend" in fields
        assert "experiment_name" in fields
        assert "run_name" in fields
        assert "project" in fields


class TestLoRAType:
    def test_all_members(self):
        assert len(LoRAType) >= 2

    def test_values(self):
        assert LoRAType.LORA.value == "lora"
        assert LoRAType.IA3.value == "ia3"

    def test_lora_plus_value(self):
        assert LoRAType.LORA_PLUS.value == "lora_plus"

    def test_iteration(self):
        names = [m.name for m in LoRAType]
        assert "LORA" in names
        assert "IA3" in names
        assert "LORA_PLUS" in names

    def test_membership_check(self):
        assert LoRAType.LORA in LoRAType

    def test_equality(self):
        assert LoRAType.LORA == LoRAType.LORA

    def test_inequality(self):
        assert LoRAType.LORA != LoRAType.IA3

    def test_is_hashable(self):
        assert hash(LoRAType.LORA) is not None

    def test_name_matches_value(self):
        for m in LoRAType:
            assert m.name.lower() == m.value


class TestLoRAConfig:
    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.rank == 8
        assert cfg.alpha == 16.0
        assert cfg.dropout == 0.05

    def test_custom(self):
        cfg = LoRAConfig(rank=16, alpha=32.0)
        assert cfg.rank == 16
        assert cfg.alpha == 32.0

    def test_default_target_modules(self):
        cfg = LoRAConfig()
        assert cfg.target_modules == ["q_proj", "v_proj", "k_proj", "o_proj"]

    def test_default_lora_type(self):
        cfg = LoRAConfig()
        assert cfg.lora_type == LoRAType.LORA

    def test_default_bias(self):
        cfg = LoRAConfig()
        assert cfg.bias == "none"

    def test_default_task_type(self):
        cfg = LoRAConfig()
        assert cfg.task_type == "CAUSAL_LM"

    def test_custom_dropout(self):
        cfg = LoRAConfig(dropout=0.1)
        assert cfg.dropout == 0.1

    def test_custom_target_modules(self):
        cfg = LoRAConfig(target_modules=["W_q", "W_v"])
        assert cfg.target_modules == ["W_q", "W_v"]

    def test_custom_lora_type(self):
        cfg = LoRAConfig(lora_type=LoRAType.IA3)
        assert cfg.lora_type == LoRAType.IA3

    def test_custom_bias(self):
        cfg = LoRAConfig(bias="all")
        assert cfg.bias == "all"

    def test_custom_task_type(self):
        cfg = LoRAConfig(task_type="SEQ_CLS")
        assert cfg.task_type == "SEQ_CLS"

    def test_dataclass_fields_count(self):
        import dataclasses
        fields = [f.name for f in dataclasses.fields(LoRAConfig)]
        assert len(fields) == 7

    def test_dataclass_field_names(self):
        import dataclasses
        fields = {f.name for f in dataclasses.fields(LoRAConfig)}
        expected = {"rank", "alpha", "dropout", "target_modules", "lora_type", "bias", "task_type"}
        assert expected == fields

    def test_post_init_sets_target_modules(self):
        cfg = LoRAConfig(target_modules=None)
        assert cfg.target_modules == ["q_proj", "v_proj", "k_proj", "o_proj"]


class TestDataFormat:
    def test_all_members(self):
        assert len(DataFormat) == 3

    def test_values(self):
        assert DataFormat.JSON.value == "json"
        assert DataFormat.JSONL.value == "jsonl"
        assert DataFormat.CSV.value == "csv"

    def test_iteration(self):
        names = [m.name for m in DataFormat]
        assert "JSON" in names
        assert "JSONL" in names
        assert "CSV" in names

    def test_membership_check(self):
        assert DataFormat.JSON in DataFormat

    def test_equality(self):
        assert DataFormat.JSON == DataFormat.JSON

    def test_inequality(self):
        assert DataFormat.JSON != DataFormat.CSV

    def test_is_hashable(self):
        assert hash(DataFormat.JSON) is not None

    def test_name_matches_value(self):
        for m in DataFormat:
            assert m.name.lower() == m.value

    def test_count(self):
        assert len(DataFormat) == 3
