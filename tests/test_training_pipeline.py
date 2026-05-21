"""Test TrainingDataPipeline."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec


def load_module():
    path = Path("/Users/mac/sloughGPT/packages/core-py/domains/infrastructure/training_pipeline.py")
    spec = spec_from_file_location("training_pipeline", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTrainingDataPipeline(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = self.module.TrainingDataPipeline(data_dir=self.temp_dir)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_add_conversation(self):
        """Test adding a conversation."""
        conv = self.pipeline.add_conversation(
            session_id="s1",
            user_message="Hello",
            assistant_message="Hi there!",
            model="gpt2",
        )
        self.assertIsNotNone(conv.id)
        self.assertEqual(conv.user_message, "Hello")

    def test_add_feedback(self):
        """Test adding feedback updates quality score."""
        conv = self.pipeline.add_conversation("s1", "Hello", "Hi", "gpt2")

        result = self.pipeline.add_feedback(conv.id, "up")
        self.assertTrue(result)

        pairs = self.pipeline.get_training_pairs()
        pair = next(p for p in pairs if p.conversation_id == conv.id)
        self.assertEqual(pair.quality_score, 1.0)

        # Test down feedback
        conv2 = self.pipeline.add_conversation("s1", "Test", "Response", "gpt2")
        self.pipeline.add_feedback(conv2.id, "down")

        pairs = self.pipeline.get_training_pairs()
        pair2 = next(p for p in pairs if p.conversation_id == conv2.id)
        self.assertEqual(pair2.quality_score, 0.0)

    def test_get_conversations_filtered(self):
        """Test filtering conversations."""
        self.pipeline.add_conversation("s1", "Hello", "Hi", "gpt2")
        self.pipeline.add_conversation("s2", "Test", "Response", "gpt2")

        s1_convs = self.pipeline.get_conversations(session_id="s1")
        self.assertEqual(len(s1_convs), 1)

    def test_get_training_pairs_quality_filter(self):
        """Test filtering by quality."""
        # No feedback = 0.5 quality
        self.pipeline.add_conversation("s1", "Hello", "Hi", "gpt2")

        # Up feedback = 1.0 quality
        conv = self.pipeline.add_conversation("s1", "Test", "Response", "gpt2")
        self.pipeline.add_feedback(conv.id, "up")

        # Get only high quality
        good_pairs = self.pipeline.get_training_pairs(min_quality=0.8)
        self.assertEqual(len(good_pairs), 1)

    def test_export_jsonl(self):
        """Test exporting to JSONL."""
        self.pipeline.add_conversation("s1", "Hello", "Hi", "gpt2")
        self.pipeline.add_conversation("s1", "How are you?", "I'm good", "gpt2")

        filepath = self.pipeline.export_training_data(min_quality=0.0, format="jsonl")

        self.assertTrue(os.path.exists(filepath))

        # Check content
        with open(filepath) as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)

            data = json.loads(lines[0])
            self.assertIn("prompt", data)
            self.assertIn("response", data)

    def test_mark_pairs_used(self):
        """Test marking pairs as used."""
        self.pipeline.add_conversation("s1", "Hello", "Hi", "gpt2")

        pairs = self.pipeline.get_training_pairs()
        pair_ids = [p.id for p in pairs]

        run = self.pipeline.create_training_run("v1", len(pair_ids), "gpt2")
        self.pipeline.mark_pairs_used(pair_ids, run.id)

        # Check they're marked
        used_pairs = self.pipeline.get_training_pairs(include_used=True)
        self.assertTrue(all(p.used_in_training for p in used_pairs))

    def test_training_runs(self):
        """Test training run tracking."""
        run = self.pipeline.create_training_run("v1", 100, "gpt2")
        self.assertEqual(run.status, "pending")

        self.pipeline.update_training_run(run.id, "completed", {"loss": 0.5})

        runs = self.pipeline.get_training_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].status, "completed")
        self.assertEqual(runs[0].metrics["loss"], 0.5)

    def test_stats(self):
        """Test statistics."""
        self.pipeline.add_conversation("s1", "Hello", "Hi", "gpt2")

        stats = self.pipeline.get_stats()
        self.assertEqual(stats["conversations_total"], 1)
        self.assertEqual(stats["training_pairs_total"], 1)

    def test_backup(self):
        """Test backup creation."""
        self.pipeline.add_conversation("s1", "Hello", "Hi", "gpt2")

        backup_path = self.pipeline.create_backup()

        self.assertTrue(os.path.exists(backup_path))
        self.assertTrue(os.path.exists(Path(backup_path) / "conversations.db"))

    def test_export_latest(self):
        """Test that latest.jsonl is created."""
        self.pipeline.add_conversation("s1", "Hello", "Hi", "gpt2")

        self.pipeline.export_training_data(min_quality=0.0)

        latest = Path(self.temp_dir) / "exports" / "latest.jsonl"
        self.assertTrue(latest.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
