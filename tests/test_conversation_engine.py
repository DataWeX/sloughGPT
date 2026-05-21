"""Test the ConversationDB - persistent database with incremental backup."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec


def load_conversation_engine():
    """Load conversation engine module directly."""
    ce_path = Path(
        "/Users/mac/sloughGPT/packages/core-py/domains/infrastructure/conversation_engine.py"
    )
    spec = spec_from_file_location("conversation_engine", ce_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestConversationDB(unittest.TestCase):
    def setUp(self):
        """Create temp directory for testing."""
        self.module = load_conversation_engine()
        self.temp_dir = tempfile.mkdtemp()
        self.db = self.module.ConversationDB(
            data_dir=self.temp_dir,
            db_name="test.db",
            backup_enabled=True,
        )

    def tearDown(self):
        """Clean up temp directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_add_message(self):
        """Test adding a message."""
        msg = self.db.add_message(
            session_id="s1",
            role="user",
            content="Hello",
            model="gpt2",
        )
        self.assertIsNotNone(msg.id)

        # Check file exists
        db_file = Path(self.temp_dir) / "test.db"
        self.assertTrue(db_file.exists())

    def test_get_session(self):
        """Test retrieving messages for a session."""
        self.db.add_message("s1", "user", "Hi", "gpt2")
        self.db.add_message("s1", "assistant", "Hello!", "gpt2")

        messages = self.db.get_session("s1")
        self.assertEqual(len(messages), 2)

    def test_update_message(self):
        """Test updating a message."""
        msg = self.db.add_message("s1", "user", "Test", "gpt2")

        result = self.db.update_message(msg.id, feedback="up")
        self.assertTrue(result)

        updated = self.db.get_message(msg.id)
        self.assertEqual(updated.feedback, "up")

    def test_delete_message(self):
        """Test deleting a message."""
        msg = self.db.add_message("s1", "user", "Test", "gpt2")

        result = self.db.delete_message(msg.id)
        self.assertTrue(result)

        deleted = self.db.get_message(msg.id)
        self.assertIsNone(deleted)

    def test_delete_session(self):
        """Test deleting all messages in a session."""
        self.db.add_message("s1", "user", "Hi", "gpt2")
        self.db.add_message("s1", "assistant", "Hello", "gpt2")
        self.db.add_message("s2", "user", "Different", "gpt2")

        deleted = self.db.delete_session("s1")
        self.assertEqual(deleted, 2)

        s1_messages = self.db.get_session("s1")
        self.assertEqual(len(s1_messages), 0)

    def test_search(self):
        """Test content search."""
        self.db.add_message("s1", "user", "Tell me about Python", "gpt2")
        self.db.add_message("s1", "assistant", "Python is great", "gpt2")

        results = self.db.search("Python")
        self.assertEqual(len(results), 2)

    def test_get_sessions(self):
        """Test getting all session IDs."""
        self.db.add_message("s1", "user", "Hi", "gpt2")
        self.db.add_message("s2", "user", "Hello", "gpt2")

        sessions = self.db.get_sessions()
        self.assertEqual(len(sessions), 2)
        self.assertIn("s1", sessions)
        self.assertIn("s2", sessions)

    def test_training_pairs(self):
        """Test extracting training pairs."""
        self.db.add_message("s1", "user", "What is AI?", "gpt2")
        self.db.add_message("s1", "assistant", "AI is intelligence", "gpt2")

        pairs = self.db.get_training_pairs()
        self.assertEqual(len(pairs), 1)

    def test_stats(self):
        """Test statistics."""
        self.db.add_message("s1", "user", "Hello", "gpt2")
        self.db.add_message("s1", "assistant", "Hi", "gpt2")

        stats = self.db.get_stats()
        self.assertEqual(stats["total_messages"], 2)
        self.assertEqual(stats["total_sessions"], 1)
        self.assertIn("db_size_bytes", stats)

    def test_export_json(self):
        """Test JSON export."""
        self.db.add_message("s1", "user", "Test", "gpt2")

        exported = self.db.export_json()
        data = json.loads(exported)
        self.assertEqual(len(data), 1)

    def test_export_csv(self):
        """Test CSV export."""
        self.db.add_message("s1", "user", "Test", "gpt2")

        csv = self.db.export_csv()
        self.assertIn("id,session_id", csv)
        self.assertIn("Test", csv)

    def test_backup(self):
        """Test creating a backup."""
        self.db.add_message("s1", "user", "Hello", "gpt2")

        backup_path = self.db.create_backup()
        self.assertTrue(os.path.exists(backup_path))

        backups = self.db.list_backups()
        self.assertEqual(len(backups), 1)

    def test_restore_from_backup(self):
        """Test restoring from backup."""
        self.db.add_message("s1", "user", "Hello", "gpt2")

        backup_path = self.db.create_backup()

        # Clear the DB
        self.db.clear()
        self.assertEqual(len(self.db.get_recent(limit=10)), 0)

        # Restore
        result = self.db.restore_from_backup(backup_path)
        self.assertTrue(result)

        messages = self.db.get_recent(limit=10)
        self.assertEqual(len(messages), 1)

    def test_compact(self):
        """Test compaction."""
        self.db.add_message("s1", "user", "Hello", "gpt2")
        self.db.add_message("s1", "assistant", "Hi", "gpt2")

        self.db.compact()

        stats = self.db.get_stats()
        self.assertEqual(stats["total_messages"], 2)

    def test_clear(self):
        """Test clearing all data."""
        self.db.add_message("s1", "user", "Test", "gpt2")
        self.db.clear()

        messages = self.db.get_recent(limit=10)
        self.assertEqual(len(messages), 0)

    def test_incremental_log(self):
        """Test that log is created for incremental updates."""
        # Add messages - log should be created
        self.db.add_message("s1", "user", "Hello", "gpt2")

        log_path = Path(self.temp_dir) / "conversations.log"
        self.assertTrue(log_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
