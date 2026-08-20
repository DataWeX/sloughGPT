"""Tests for domains.memory.consolidation — plan_consolidation, _embed_cache."""

from domains.memory.consolidation import plan_consolidation, _embed_cache


class TestEmbedCache:
    def test_empty(self):
        assert _embed_cache([]) == {}

    def test_basic(self):
        facts = [{"id": "a", "content": "hello world"}]
        cache = _embed_cache(facts)
        assert "a" in cache
        assert len(cache["a"]) > 0

    def test_skips_missing_id(self):
        facts = [{"content": "no id"}]
        cache = _embed_cache(facts)
        assert cache == {}


class TestPlanConsolidation:
    def test_empty(self):
        result = plan_consolidation([])
        assert result["keep_ids"] == []
        assert result["removed_count"] == 0

    def test_no_duplicates(self):
        facts = [
            {"id": "1", "content": "completely different topic A", "topic": "a"},
            {"id": "2", "content": "completely different topic B", "topic": "b"},
        ]
        result = plan_consolidation(facts, threshold=0.9)
        assert result["removed_count"] == 0

    def test_near_duplicates(self):
        facts = [
            {"id": "1", "content": "The user prefers Zed over VS Code", "topic": "editor"},
            {"id": "2", "content": "User prefers the editor Zed", "topic": "editor"},
        ]
        result = plan_consolidation(facts, threshold=0.3)
        assert result["removed_count"] == 1

    def test_keep_longest(self):
        facts = [
            {"id": "short", "content": "likes Zed", "topic": "editor"},
            {"id": "long", "content": "The user strongly prefers Zed editor over all others", "topic": "editor"},
        ]
        result = plan_consolidation(facts, threshold=0.2)
        assert "long" in result["keep_ids"]

    def test_groups_contain_keep_and_duplicates(self):
        facts = [
            {"id": "a", "content": "likes Zed editor a lot", "topic": "editor"},
            {"id": "b", "content": "likes Zed editor a lot actually", "topic": "editor"},
        ]
        result = plan_consolidation(facts, threshold=0.2)
        if result["groups"]:
            g = result["groups"][0]
            assert "keep" in g
            assert "duplicates" in g
