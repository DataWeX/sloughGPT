"""SloEngine multi-turn chat: history in prompt, chat() vs generate()."""

from __future__ import annotations

import unittest

from domains.core.soul import SloEngine


class TestSoulEngineConversation(unittest.TestCase):
    def test_generate_does_not_duplicate_current_user_in_history_block(self) -> None:
        eng = SloEngine()
        _t1, extra1 = eng.generate("hello", include_reasoning=False, return_reasoning=True)
        fp1 = extra1["full_prompt"]
        self.assertIn("User: hello", fp1)
        self.assertNotIn("[CONVERSATION_HISTORY]", fp1)

        _t2, extra2 = eng.generate("bye", include_reasoning=False, return_reasoning=True)
        fp2 = extra2["full_prompt"]
        self.assertIn("[CONVERSATION_HISTORY]", fp2)
        self.assertIn("User: hello", fp2)
        self.assertIn("User: bye", fp2)
        # Current turn appears exactly once as the trailing "User: bye" line, not twice inside history + current
        self.assertLessEqual(fp2.count("User: bye"), 1)

    def test_chat_replays_prior_messages_then_completes_last_user(self) -> None:
        eng = SloEngine()
        out = eng.chat(
            [
                {"role": "user", "content": "First?"},
                {"role": "assistant", "content": "First reply."},
                {"role": "user", "content": "Second?"},
            ],
            include_reasoning=False,
            return_reasoning=True,
        )
        self.assertIsInstance(out, tuple)
        text, extra = out
        self.assertIsInstance(text, str)
        fp = extra["full_prompt"]
        self.assertIn("User: First?", fp)
        self.assertIn("Assistant: First reply.", fp)
        self.assertIn("User: Second?", fp)
        # Completed pair stored
        self.assertEqual(eng._session_history[-2]["content"], "Second?")  # type: ignore[index]

    def test_chat_requires_last_user(self) -> None:
        eng = SloEngine()
        with self.assertRaises(ValueError):
            eng.chat([{"role": "assistant", "content": "only assistant"}])

    def test_clear_conversation(self) -> None:
        eng = SloEngine()
        eng.generate("x", include_reasoning=False)
        self.assertGreater(len(eng._session_history), 0)
        eng.clear_conversation()
        self.assertEqual(eng._session_history, [])
        self.assertEqual(eng._cognitive_state["session_turns"], 0)


if __name__ == "__main__":
    unittest.main()
