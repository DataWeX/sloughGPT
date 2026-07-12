"""Tests for CLI train commands."""
import sys
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Add cli src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestDistillConfig:
    def test_config_defaults(self):
        from domains.training.distill_gpt2 import DistillConfig
        c = DistillConfig()
        assert c.n_embed == 128
        assert c.n_layer == 4
        assert c.n_head == 4
        assert c.block_size == 128
        assert c.epochs == 10
        assert c.lr == 3e-4
        assert c.temperature == 4.0
        assert c.resume_checkpoint is None

    def test_config_resume(self):
        from domains.training.distill_gpt2 import DistillConfig
        c = DistillConfig(resume_checkpoint="test.soul", resume_epoch=5)
        assert c.resume_checkpoint == "test.soul"
        assert c.resume_epoch == 5

    def test_config_custom_values(self):
        from domains.training.distill_gpt2 import DistillConfig
        c = DistillConfig(n_embed=64, n_layer=2, n_head=2, epochs=3, lr=1e-3)
        assert c.n_embed == 64
        assert c.n_layer == 2
        assert c.epochs == 3
        assert c.lr == 1e-3

    def test_config_temperature(self):
        from domains.training.distill_gpt2 import DistillConfig
        c = DistillConfig(temperature=2.0)
        assert c.temperature == 2.0


class TestCmdDistill:
    def test_missing_text_returns_early(self):
        from commands.train import cmd_distill
        args = MagicMock()
        args.text_source = None
        args.file = None
        args.api = False
        cmd_distill(args)

    def test_file_not_found_returns_early(self):
        from commands.train import cmd_distill
        args = MagicMock()
        args.text_source = None
        args.file = "/nonexistent/file.txt"
        args.api = False
        cmd_distill(args)

    def test_empty_text_returns_early(self):
        from commands.train import cmd_distill
        args = MagicMock()
        args.text_source = ""
        args.file = None
        args.api = False
        cmd_distill(args)
