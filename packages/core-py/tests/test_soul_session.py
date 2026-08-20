"""Tests for domains.soul.cognitive — SessionMemory, EpisodicMemoryStore."""

from domains.soul.cognitive import SessionMemory, EpisodicMemoryStore


class TestSessionMemory:
    def test_init(self):
        sm = SessionMemory(max_turns=10)
        assert sm.max_turns == 10
        assert len(sm.conversation) == 0

    def test_add_message(self):
        sm = SessionMemory(max_turns=10)
        msg = sm.add("user", "hello")
        assert msg["role"] == "user"
        assert msg["content"] == "hello"
        assert len(sm.conversation) == 1

    def test_eviction(self):
        sm = SessionMemory(max_turns=3)
        for i in range(5):
            sm.add("user", f"msg {i}")
        assert len(sm.conversation) == 3

    def test_get_context(self):
        sm = SessionMemory(max_turns=10)
        for i in range(5):
            sm.add("user", f"msg {i}")
        ctx = sm.get_context(n=3)
        assert len(ctx) == 3

    def test_get_full_session(self):
        sm = SessionMemory()
        sm.add("user", "hello")
        sm.add("assistant", "hi")
        full = sm.get_full_session()
        assert len(full) == 2

    def test_clear(self):
        sm = SessionMemory()
        sm.add("user", "hello")
        old_id = sm.session_id
        sm.clear()
        assert len(sm.conversation) == 0
        assert sm.session_id != old_id

    def test_get_summary(self):
        sm = SessionMemory()
        sm.add("user", "hello")
        summary = sm.get_summary()
        assert summary["turns"] == 1
        assert "session_id" in summary


class TestEpisodicMemoryStore:
    def test_init(self):
        ems = EpisodicMemoryStore(max_episodes=10)
        assert ems.max_episodes == 10
        assert len(ems.episodes) == 0

    def test_save_episode(self):
        ems = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        ep_id = ems.save_episode("s1", conv)
        assert ep_id in ems.episodes

    def test_eviction(self):
        ems = EpisodicMemoryStore(max_episodes=3)
        for i in range(5):
            conv = [{"role": "user", "content": f"msg {i}"}]
            ems.save_episode(f"s{i}", conv)
        assert len(ems.episodes) <= 3

    def test_importance_calculation(self):
        ems = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "important remember this"}]
        ep_id = ems.save_episode("s1", conv)
        meta = ems.episode_metadata[ep_id]
        assert meta["importance"] > 0.5

    def test_empty_episode(self):
        ems = EpisodicMemoryStore()
        ep_id = ems.save_episode("s1", [])
        meta = ems.episode_metadata[ep_id]
        assert meta["importance"] == 0.0
