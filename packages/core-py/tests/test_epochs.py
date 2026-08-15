"""Tests for honest ``epochs_trained`` metadata in .soul checkpoints.

``SloughGPTTrainer.save`` must report the number of epochs actually completed
at save time — not the config's target epoch count. A save before any training
step claims zero epochs rather than a fabricated value, and a run that stops
mid-epoch (e.g. on max_steps) reports the completed epochs, never the entered
count.
"""

import pytest

from domains.inference.slo_format import load_soul
from domains.training.train_pipeline import SloughGPTTrainer, TrainerConfig

DATA_TEXT = (
    "the quick brown fox jumps over the lazy dog and runs across the meadow "
    "again and again while the dog sleeps soundly in the warm sun all day "
    "long. " * 8
)


@pytest.fixture
def data_path(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text(DATA_TEXT, encoding="utf-8")
    return str(p)


def tiny_config(tmp_path, **overrides):
    cfg = TrainerConfig(
        vocab_size=0,
        n_embed=16,
        n_layer=1,
        n_head=2,
        block_size=8,
        dropout=0.0,
        batch_size=4,
        epochs=3,
        max_steps=50,
        gradient_accumulation_steps=1,
        checkpoint_dir=str(tmp_path / "ckpts"),
        log_interval=1,
        eval_interval=1000,
        checkpoint_interval=1000,
        warmup_steps=1,
        min_lr=1e-5,
        max_checkpoints=5,
        scheduler_type="cosine",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_epochs_trained_matches_actual_completed_epochs(data_path, tmp_path):
    cfg = tiny_config(tmp_path)
    t = SloughGPTTrainer(data_path, config=cfg)
    t.train()

    steps_per_epoch = len(t.train_data) // cfg.block_size // cfg.batch_size
    expected_epochs = min(cfg.epochs, cfg.max_steps // steps_per_epoch)

    assert expected_epochs < cfg.epochs  # run stops before the config target
    assert t._completed_epochs == expected_epochs
    assert t._last_train_loss is not None  # at least one step ran

    profile, _ = load_soul(t._last_checkpoint_path)
    assert profile.epochs_trained == expected_epochs
    assert profile.epochs_trained != cfg.epochs  # not the config default


def test_epochs_trained_is_zero_for_mid_epoch_stop(data_path, tmp_path):
    cfg = tiny_config(tmp_path, max_steps=6)  # stops well inside epoch 0
    t = SloughGPTTrainer(data_path, config=cfg)
    t.train()

    assert t.global_step == 6
    assert t._last_train_loss is not None
    assert t._completed_epochs == 0  # no epoch fully completed

    profile, _ = load_soul(t._last_checkpoint_path)
    assert profile.epochs_trained == 0
    assert profile.epochs_trained != cfg.epochs


def test_epochs_trained_is_zero_before_training(data_path, tmp_path):
    t = SloughGPTTrainer(data_path, config=tiny_config(tmp_path))
    out = str(tmp_path / "fresh")
    t.save(out)

    profile, _ = load_soul(out + ".soul")
    assert profile.epochs_trained == 0


def test_completed_epochs_accumulate_across_resume(data_path, tmp_path):
    cfg1 = tiny_config(tmp_path, max_steps=40)
    t1 = SloughGPTTrainer(data_path, config=cfg1)
    t1.train()

    steps_per_epoch = len(t1.train_data) // cfg1.block_size // cfg1.batch_size
    exp1 = min(cfg1.epochs, 40 // steps_per_epoch)
    p1, _ = load_soul(t1._last_checkpoint_path)
    assert p1.epochs_trained == exp1

    cfg2 = tiny_config(tmp_path, max_steps=80)
    t2 = SloughGPTTrainer(data_path, config=cfg2)
    t2.train(resume=True, resume_path=t1._last_checkpoint_path)

    assert t2.global_step == 80
    exp2 = min(cfg2.epochs, 80 // steps_per_epoch)
    p2, _ = load_soul(t2._last_checkpoint_path)
    assert p2.epochs_trained == exp2
    assert p2.epochs_trained > p1.epochs_trained  # tally carried across resume


def test_completed_epochs_embedded_in_training_state(data_path, tmp_path):
    cfg = tiny_config(tmp_path, max_steps=40)
    t = SloughGPTTrainer(data_path, config=cfg)
    t.train()

    profile, _ = load_soul(t._last_checkpoint_path)
    training = profile.metadata["training_state"]
    steps_per_epoch = len(t.train_data) // cfg.block_size // cfg.batch_size
    assert training["completed_epochs"] == min(cfg.epochs, 40 // steps_per_epoch)
    assert training["step"] == 40
