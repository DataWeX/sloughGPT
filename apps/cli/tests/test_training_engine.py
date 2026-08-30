"""Tests for TrainingEngine config, data sources, and dispatch logic.

Runs in the CLI venv (no fastapi dependency). Tests the pure domain logic
by importing config.py as a standalone module.
"""

import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Load config.py as a standalone module (no __init__.py involvement)
_repo_root = Path(__file__).resolve().parents[3]
_training_dir = _repo_root / "apps" / "api" / "server" / "training"

_config_spec = importlib.util.spec_from_file_location(
    "training_config", str(_training_dir / "config.py")
)
_config_mod = importlib.util.module_from_spec(_config_spec)
sys.modules["training_config"] = _config_mod
_config_spec.loader.exec_module(_config_mod)

DirectConfig = _config_mod.DirectConfig
DistillConfig = _config_mod.DistillConfig
LoRAConfig = _config_mod.LoRAConfig
VisualConfig = _config_mod.VisualConfig
TrainingConfig = _config_mod.TrainingConfig
DatasetSource = _config_mod.DatasetSource
ManifestSource = _config_mod.ManifestSource
SessionSource = _config_mod.SessionSource
SubprocessSource = _config_mod.SubprocessSource
TrainingMode = _config_mod.TrainingMode
DataSourceType = _config_mod.DataSourceType
TrainingJobRequest = _config_mod.TrainingJobRequest
ResumeConfig = _config_mod.ResumeConfig
DataSource = _config_mod.DataSource


# ─── Config Tests ──────────────────────────────────────────────────────


class TestDirectConfig:
    def test_defaults(self):
        config = DirectConfig()
        assert config.mode == TrainingMode.DIRECT
        assert config.epochs == 3
        assert config.batch_size == 32
        assert config.n_embed == 128
        assert config.use_lora is False
        assert config.stream is False

    def test_override(self):
        config = DirectConfig(epochs=10, batch_size=64, n_embed=256, name="custom")
        assert config.epochs == 10
        assert config.batch_size == 64
        assert config.n_embed == 256
        assert config.name == "custom"


class TestDistillConfig:
    def test_defaults(self):
        config = DistillConfig()
        assert config.mode == TrainingMode.DISTILL
        assert config.teacher_model == "gpt2"
        assert config.temperature == 4.0
        assert config.alpha == 0.5
        assert config.beta == 0.5

    def test_override(self):
        config = DistillConfig(teacher_model="gpt2-medium", temperature=8.0)
        assert config.teacher_model == "gpt2-medium"
        assert config.temperature == 8.0


class TestLoRAConfig:
    def test_defaults(self):
        config = LoRAConfig()
        assert config.model_path == ""
        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.output_dir == "models"

    def test_requires_model_path_for_validation(self):
        config = LoRAConfig(model_path="/path/to/model.slnc")
        assert config.model_path == "/path/to/model.slnc"


class TestVisualConfig:
    def test_defaults(self):
        config = VisualConfig()
        assert config.embed_dim == 256
        assert config.hidden_dim == 512
        assert config.n_vision_layers == 3
        assert config.max_frames == 8


# ─── DataSource Tests ──────────────────────────────────────────────────


class TestDatasetSource:
    def test_by_name(self):
        source = DatasetSource(name="tinyshakespeare")
        assert source.type == DataSourceType.DATASET
        assert source.name == "tinyshakespeare"
        assert source.path is None

    def test_by_path(self):
        source = DatasetSource(path="/explicit/path/data.txt")
        assert source.type == DataSourceType.DATASET
        assert source.path == "/explicit/path/data.txt"
        assert source.name == ""


class TestManifestSource:
    def test_defaults(self):
        source = ManifestSource(manifest_uri="https://example.com/manifest.json")
        assert source.type == DataSourceType.MANIFEST
        assert source.manifest_uri == "https://example.com/manifest.json"
        assert source.dataset_id is None


class TestSessionSource:
    def test_defaults(self):
        source = SessionSource()
        assert source.type == DataSourceType.SESSION
        assert source.min_pair_quality == 2.0
        assert source.max_pairs == 500

    def test_custom(self):
        source = SessionSource(min_pair_quality=3.5, max_pairs=1000)
        assert source.min_pair_quality == 3.5
        assert source.max_pairs == 1000


class TestSubprocessSource:
    def test_defaults(self):
        source = SubprocessSource()
        assert source.type == DataSourceType.SUBPROCESS
        assert source.script is None

    def test_custom(self):
        source = SubprocessSource(script="/path/to/script.py", script_args=["--lr", "0.001"])
        assert source.script == "/path/to/script.py"
        assert source.script_args == ["--lr", "0.001"]


# ─── TrainingJobRequest Tests ─────────────────────────────────────────────


class TestTrainingJobRequest:
    def test_direct_request(self):
        config = DirectConfig(name="test-job", epochs=5)
        source = DatasetSource(name="tinyshakespeare")
        request = TrainingJobRequest(config=config, source=source)
        assert request.config.name == "test-job"
        assert request.config.epochs == 5
        assert request.source.name == "tinyshakespeare"
        assert request.resume is None

    def test_distill_request_with_resume(self):
        config = DistillConfig(teacher_model="gpt2", temperature=8.0)
        source = DatasetSource(name="corpus")
        resume = ResumeConfig(checkpoint_name="ckpt-1")
        request = TrainingJobRequest(config=config, source=source, resume=resume)
        assert request.config.teacher_model == "gpt2"
        assert request.resume.checkpoint_name == "ckpt-1"

    def test_lora_request(self):
        config = LoRAConfig(model_path="/models/base.slnc", rank=16)
        source = DatasetSource(name="dataset")
        request = TrainingJobRequest(config=config, source=source)
        assert request.config.model_path == "/models/base.slnc"
        assert request.config.rank == 16

    def test_session_source_request(self):
        config = DirectConfig(name="session-train")
        source = SessionSource(min_pair_quality=3.0, max_pairs=200)
        request = TrainingJobRequest(config=config, source=source)
        assert request.source.min_pair_quality == 3.0
        assert request.source.max_pairs == 200


# ─── ResumeConfig Tests ────────────────────────────────────────────────


class TestResumeConfig:
    def test_from_checkpoint_name(self):
        resume = ResumeConfig(checkpoint_name="my-checkpoint")
        assert resume.checkpoint_name == "my-checkpoint"
        assert resume.resume_path is None

    def test_from_path(self):
        resume = ResumeConfig(resume_path="/path/to/checkpoint.soul")
        assert resume.resume_path == "/path/to/checkpoint.soul"
        assert resume.checkpoint_name is None

    def test_both(self):
        resume = ResumeConfig(checkpoint_name="ckpt", resume_path="/path/ckpt.soul")
        assert resume.checkpoint_name == "ckpt"
        assert resume.resume_path == "/path/ckpt.soul"
