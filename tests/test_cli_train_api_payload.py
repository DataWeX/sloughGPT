"""``cli.py train --api`` posts a ``TrainingRequest``-shaped JSON body."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
pytest.importorskip("torch")

from config_loader import Config, DataConfig, DeviceConfig, ModelConfig, TrainingConfig, merge_args_with_config


@pytest.fixture
def training_args_api() -> SimpleNamespace:
    return SimpleNamespace(
        api=True,
        host="127.0.0.1",
        port=8000,
        config="config.yaml",
        dataset="shakespeare",
        epochs=2,
        batch_size=8,
        lr=0.01,
        max_steps=40,
        soul_name="api-smoke-job",
        log_interval=7,
        eval_interval=99,
        optimized=False,
    )


@pytest.fixture
def tiny_config() -> Config:
    return Config(
        model=ModelConfig(
            name="cli-test-model",
            n_embed=64,
            n_layer=2,
            n_head=2,
            block_size=32,
        ),
        training=TrainingConfig(log_interval=10, eval_interval=100),
    )


def test_train_api_posts_json_training_request_shape(training_args_api, tiny_config) -> None:
    from apps.cli.src.commands.train import cmd_train

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"id": "job_1", "status": "running"}

    cfg = deepcopy(tiny_config)
    with patch("config_loader.load_config", return_value=cfg):
        with patch(
            "config_loader.merge_args_with_config",
            side_effect=merge_args_with_config,
        ):
            with patch("requests.post", return_value=mock_resp) as post:
                cmd_train(training_args_api)

    assert post.call_count == 1
    url = post.call_args[0][0]
    assert url.endswith("/training/start")

    body = post.call_args[1]["json"]
    assert body["name"] == "api-smoke-job"
    assert body["model"] == "cli-test-model"
    assert body["dataset"] == "shakespeare"
    assert body["epochs"] == 2
    assert body["batch_size"] == 8
    assert body["learning_rate"] == 0.01
    assert body["n_embed"] == 64
    assert body["n_layer"] == 2
    assert body["n_head"] == 2
    assert body["block_size"] == 32
    assert body["max_steps"] == 40
    assert body["log_interval"] == 7
    assert body["eval_interval"] == 99
    assert body["dropout"] == 0.1
    assert body["weight_decay"] == 0.01
    assert body["gradient_accumulation_steps"] == 1
    assert body["max_grad_norm"] == 1.0
    assert body["use_mixed_precision"] is True
    assert body["mixed_precision_dtype"] == "bf16"
    assert body["warmup_steps"] == 500
    assert body["min_lr"] == 1e-5
    assert body["scheduler"] == "cosine"
    assert body["use_lora"] is False
    assert body["lora_rank"] == 8
    assert body["lora_alpha"] == 16
    assert body["checkpoint_dir"] == "checkpoints"
    assert body["checkpoint_interval"] == 1000
    assert body["save_best_only"] is False
    assert body["max_checkpoints"] == 5
    assert "device" not in body

    assert post.call_args[1]["headers"]["Content-Type"] == "application/json"


def test_train_api_omits_max_steps_when_none(tiny_config) -> None:
    from apps.cli.src.commands.train import cmd_train

    cfg = deepcopy(tiny_config)
    cfg.training = TrainingConfig(log_interval=44, eval_interval=55)

    args = SimpleNamespace(
        api=True,
        host="127.0.0.1",
        port=8000,
        config="config.yaml",
        dataset="tiny",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        max_steps=None,
        soul_name=None,
        log_interval=None,
        eval_interval=None,
        optimized=False,
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"id": "job_2", "status": "running"}

    with patch("config_loader.load_config", return_value=cfg):
        with patch(
            "config_loader.merge_args_with_config",
            side_effect=merge_args_with_config,
        ):
            with patch("requests.post", return_value=mock_resp) as post:
                cmd_train(args)

    body = post.call_args[1]["json"]
    assert "max_steps" not in body
    assert body["name"] == "cli-test-model"
    assert body["log_interval"] == 44
    assert body["eval_interval"] == 55


def test_train_api_max_steps_from_config_when_cli_omitted(tiny_config) -> None:
    """``training.max_steps`` from YAML merge is sent when ``--max-steps`` is omitted."""
    from apps.cli.src.commands.train import cmd_train

    cfg = deepcopy(tiny_config)
    cfg.training = TrainingConfig(
        log_interval=10,
        eval_interval=100,
        max_steps=777,
    )

    args = SimpleNamespace(
        api=True,
        host="127.0.0.1",
        port=8000,
        config="config.yaml",
        dataset="tiny",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        max_steps=None,
        soul_name="n",
        log_interval=None,
        eval_interval=None,
        optimized=False,
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"id": "job_yaml_ms", "status": "running"}

    with patch("config_loader.load_config", return_value=cfg):
        with patch(
            "config_loader.merge_args_with_config",
            side_effect=merge_args_with_config,
        ):
            with patch("requests.post", return_value=mock_resp) as post:
                cmd_train(args)

    body = post.call_args[1]["json"]
    assert body["max_steps"] == 777


def test_train_api_epochs_batch_lr_from_merged_config_not_raw_zeros(tiny_config) -> None:
    """Falsy CLI numerics that merge skips must not override ``config.training`` in the JSON body."""
    from apps.cli.src.commands.train import cmd_train

    cfg = deepcopy(tiny_config)
    cfg.training = TrainingConfig(
        epochs=10,
        batch_size=64,
        learning_rate=0.02,
        log_interval=10,
        eval_interval=100,
    )

    args = SimpleNamespace(
        api=True,
        host="127.0.0.1",
        port=8000,
        config="config.yaml",
        dataset="tiny",
        epochs=0,
        batch_size=0,
        lr=0.0,
        max_steps=None,
        soul_name="x",
        log_interval=None,
        eval_interval=None,
        optimized=False,
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"id": "job_edge", "status": "running"}

    with patch("config_loader.load_config", return_value=cfg):
        with patch(
            "config_loader.merge_args_with_config",
            side_effect=merge_args_with_config,
        ):
            with patch("requests.post", return_value=mock_resp) as post:
                cmd_train(args)

    body = post.call_args[1]["json"]
    assert body["epochs"] == 10
    assert body["batch_size"] == 64
    assert body["learning_rate"] == 0.02


def test_train_api_job_name_from_model_soul_yaml(tiny_config) -> None:
    """``model.soul_name`` in config is used when ``--soul-name`` is omitted."""
    from apps.cli.src.commands.train import cmd_train

    cfg = deepcopy(tiny_config)
    cfg.model.soul_name = "named-from-yaml"

    args = SimpleNamespace(
        api=True,
        host="127.0.0.1",
        port=8000,
        config="config.yaml",
        dataset="tiny",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        max_steps=None,
        soul_name=None,
        log_interval=None,
        eval_interval=None,
        optimized=False,
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"id": "job_soul_yaml", "status": "running"}

    with patch("config_loader.load_config", return_value=cfg):
        with patch(
            "config_loader.merge_args_with_config",
            side_effect=merge_args_with_config,
        ):
            with patch("requests.post", return_value=mock_resp) as post:
                cmd_train(args)

    assert post.call_args[1]["json"]["name"] == "named-from-yaml"


def test_train_api_dataset_from_config_when_cli_dataset_empty(tiny_config) -> None:
    """Falsy ``dataset`` on args skips merge; body uses ``config.data.dataset``."""
    from apps.cli.src.commands.train import cmd_train

    cfg = deepcopy(tiny_config)
    cfg.data = DataConfig(
        dataset="from_yaml_only",
        data_path="datasets/from_yaml_only/input.txt",
    )

    args = SimpleNamespace(
        api=True,
        host="127.0.0.1",
        port=8000,
        config="config.yaml",
        dataset="",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        max_steps=None,
        soul_name=None,
        log_interval=None,
        eval_interval=None,
        optimized=False,
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"id": "job_ds_yaml", "status": "running"}

    with patch("config_loader.load_config", return_value=cfg):
        with patch(
            "config_loader.merge_args_with_config",
            side_effect=merge_args_with_config,
        ):
            with patch("requests.post", return_value=mock_resp) as post:
                cmd_train(args)

    assert post.call_args[1]["json"]["dataset"] == "from_yaml_only"


def test_train_api_optimized_sets_fp16_in_json(tiny_config) -> None:
    from apps.cli.src.commands.train import cmd_train

    cfg = deepcopy(tiny_config)
    args = SimpleNamespace(
        api=True,
        host="127.0.0.1",
        port=8000,
        config="config.yaml",
        dataset="tiny",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        max_steps=None,
        soul_name=None,
        log_interval=None,
        eval_interval=None,
        optimized=True,
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"id": "job_opt", "status": "running"}

    with patch("config_loader.load_config", return_value=cfg):
        with patch(
            "config_loader.merge_args_with_config",
            side_effect=merge_args_with_config,
        ):
            with patch("requests.post", return_value=mock_resp) as post:
                cmd_train(args)

    body = post.call_args[1]["json"]
    assert body["use_mixed_precision"] is True
    assert body["mixed_precision_dtype"] == "fp16"


def test_train_api_includes_device_when_not_auto(tiny_config) -> None:
    from apps.cli.src.commands.train import cmd_train

    cfg = deepcopy(tiny_config)
    cfg.device = DeviceConfig(type="cuda")
    args = SimpleNamespace(
        api=True,
        host="127.0.0.1",
        port=8000,
        config="config.yaml",
        dataset="tiny",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        max_steps=None,
        soul_name=None,
        log_interval=None,
        eval_interval=None,
        optimized=False,
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"id": "job_dev", "status": "running"}

    with patch("config_loader.load_config", return_value=cfg):
        with patch(
            "config_loader.merge_args_with_config",
            side_effect=merge_args_with_config,
        ):
            with patch("requests.post", return_value=mock_resp) as post:
                cmd_train(args)

    assert post.call_args[1]["json"]["device"] == "cuda"
