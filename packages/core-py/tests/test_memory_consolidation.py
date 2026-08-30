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

    def test_skips_empty_id(self):
        facts = [{"id": "", "content": "empty id"}]
        cache = _embed_cache(facts)
        assert cache == {}

    def test_multiple_facts(self):
        facts = [
            {"id": "a", "content": "hello"},
            {"id": "b", "content": "world"},
        ]
        cache = _embed_cache(facts)
        assert len(cache) == 2
        assert "a" in cache
        assert "b" in cache

    def test_embedding_is_vector(self):
        facts = [{"id": "x", "content": "test"}]
        cache = _embed_cache(facts)
        import numpy as np
        assert isinstance(cache["x"], np.ndarray)

    def test_same_content_same_embedding(self):
        facts = [
            {"id": "a", "content": "identical text"},
            {"id": "b", "content": "identical text"},
        ]
        cache = _embed_cache(facts)
        import numpy as np
        np.testing.assert_array_equal(cache["a"], cache["b"])

    def test_different_content_different_embedding(self):
        facts = [
            {"id": "a", "content": "cats are cute"},
            {"id": "b", "content": "nuclear physics"},
        ]
        cache = _embed_cache(facts)
        import numpy as np
        assert not np.array_equal(cache["a"], cache["b"])

    def test_content_key_used(self):
        facts = [{"id": "a", "content": "hello"}]
        cache = _embed_cache(facts)
        assert "a" in cache

    def test_extra_keys_ignored(self):
        facts = [{"id": "a", "content": "hello", "topic": "x", "extra": True}]
        cache = _embed_cache(facts)
        assert "a" in cache

    def test_mixed_valid_invalid(self):
        facts = [
            {"id": "a", "content": "valid"},
            {"content": "no id"},
            {"id": "b", "content": "also valid"},
        ]
        cache = _embed_cache(facts)
        assert len(cache) == 2
        assert set(cache.keys()) == {"a", "b"}

    def test_unicode_content(self):
        facts = [{"id": "u", "content": "日本語テスト"}]
        cache = _embed_cache(facts)
        assert "u" in cache
        assert len(cache["u"]) > 0

    def test_long_content(self):
        facts = [{"id": "long", "content": "word " * 1000}]
        cache = _embed_cache(facts)
        assert "long" in cache

    def test_empty_content(self):
        facts = [{"id": "empty", "content": ""}]
        cache = _embed_cache(facts)
        assert "empty" in cache

    def test_single_character_content(self):
        facts = [{"id": "single", "content": "a"}]
        cache = _embed_cache(facts)
        assert "single" in cache
        assert len(cache["single"]) > 0

    def test_embedding_consistent_across_calls(self):
        facts = [{"id": "a", "content": "consistent"}]
        c1 = _embed_cache(facts)
        c2 = _embed_cache(facts)
        import numpy as np
        np.testing.assert_array_equal(c1["a"], c2["a"])

    def test_fact_with_false_id(self):
        facts = [{"id": 0, "content": "zero id"}]
        cache = _embed_cache(facts)
        assert cache == {}

    def test_fact_with_none_id(self):
        facts = [{"id": None, "content": "none id"}]
        cache = _embed_cache(facts)
        assert cache == {}

    def test_many_facts(self):
        facts = [{"id": f"f{i}", "content": f"fact number {i}"} for i in range(100)]
        cache = _embed_cache(facts)
        assert len(cache) == 100


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

    def test_distinct_topics_no_merge(self):
        facts = [
            {"id": "a", "content": "same content", "topic": "x"},
            {"id": "b", "content": "same content", "topic": "y"},
        ]
        result = plan_consolidation(facts, threshold=0.5)
        assert result["removed_count"] == 0

    def test_threshold_zero(self):
        facts = [
            {"id": "a", "content": "aaa", "topic": "t"},
            {"id": "b", "content": "bbb", "topic": "t"},
        ]
        result = plan_consolidation(facts, threshold=0.0)
        assert result["removed_count"] == 1

    def test_threshold_one(self):
        facts = [
            {"id": "a", "content": "identical", "topic": "t"},
            {"id": "b", "content": "identical", "topic": "t"},
        ]
        result = plan_consolidation(facts, threshold=1.0)
        assert result["removed_count"] == 0

    def test_single_fact(self):
        facts = [{"id": "only", "content": "lone fact", "topic": "solo"}]
        result = plan_consolidation(facts, threshold=0.5)
        assert result["keep_ids"] == ["only"]
        assert result["removed_count"] == 0

    def test_result_structure(self):
        result = plan_consolidation([])
        assert set(result.keys()) == {"keep_ids", "remove_ids", "groups", "removed_count"}

    def test_keep_union_remove_covers_all(self):
        facts = [
            {"id": "a", "content": "short", "topic": "t"},
            {"id": "b", "content": "a bit longer content here", "topic": "t"},
            {"id": "c", "content": "completely different", "topic": "u"},
        ]
        result = plan_consolidation(facts, threshold=0.5)
        assert set(result["keep_ids"]) | set(result["remove_ids"]) == {"a", "b", "c"}

    def test_no_overlap_keep_remove(self):
        facts = [
            {"id": "a", "content": "same content here", "topic": "t"},
            {"id": "b", "content": "same content here", "topic": "t"},
        ]
        result = plan_consolidation(facts, threshold=0.5)
        assert set(result["keep_ids"]).isdisjoint(set(result["remove_ids"]))

    def test_removed_count_matches(self):
        facts = [
            {"id": "a", "content": "similar text one", "topic": "t"},
            {"id": "b", "content": "similar text two", "topic": "t"},
            {"id": "c", "content": "similar text three", "topic": "t"},
        ]
        result = plan_consolidation(facts, threshold=0.3)
        assert result["removed_count"] == len(result["remove_ids"])

    def test_three_facts_two_merge(self):
        facts = [
            {"id": "a", "content": "the user likes cats", "topic": "pets"},
            {"id": "b", "content": "the user likes felines", "topic": "pets"},
            {"id": "c", "content": "quantum computing is emerging", "topic": "tech"},
        ]
        result = plan_consolidation(facts, threshold=0.3)
        assert result["removed_count"] == 1

    def test_group_keys(self):
        facts = [
            {"id": "a", "content": "same text here", "topic": "t"},
            {"id": "b", "content": "same text here", "topic": "t"},
        ]
        result = plan_consolidation(facts, threshold=0.5)
        if result["groups"]:
            g = result["groups"][0]
            assert set(g.keys()) == {"keep", "duplicates"}

    def test_keep_is_longest_in_group(self):
        facts = [
            {"id": "short", "content": "hi", "topic": "t"},
            {"id": "medium", "content": "hello world", "topic": "t"},
            {"id": "long", "content": "hello world this is longer", "topic": "t"},
        ]
        result = plan_consolidation(facts, threshold=0.3)
        if result["groups"]:
            assert result["groups"][0]["keep"]["id"] == "long"

    def test_multiple_groups(self):
        facts = [
            {"id": "a1", "content": "cats are great", "topic": "pets"},
            {"id": "a2", "content": "cats are awesome", "topic": "pets"},
            {"id": "b1", "content": "cars are fast", "topic": "transport"},
            {"id": "b2", "content": "cars are speedy", "topic": "transport"},
        ]
        result = plan_consolidation(facts, threshold=0.3)
        assert result["removed_count"] == 2
        assert len(result["groups"]) == 2

    def test_empty_content_facts(self):
        facts = [
            {"id": "a", "content": "", "topic": "t"},
            {"id": "b", "content": "", "topic": "t"},
        ]
        result = plan_consolidation(facts, threshold=0.5)
        assert result["removed_count"] == 1

    def test_unicode_content(self):
        facts = [
            {"id": "a", "content": "日本語テスト", "topic": "lang"},
            {"id": "b", "content": "日本語テストです", "topic": "lang"},
        ]
        result = plan_consolidation(facts, threshold=0.3)
        assert result["removed_count"] == 1

    def test_many_facts_no_merge(self):
        facts = [{"id": f"f{i}", "content": f"unique fact {i} about topic {i}", "topic": f"t{i}"} for i in range(20)]
        result = plan_consolidation(facts, threshold=0.80)
        assert result["removed_count"] == 0
        assert len(result["keep_ids"]) == 20

    def test_groups_are_list(self):
        facts = [{"id": "a", "content": "same", "topic": "t"}, {"id": "b", "content": "same", "topic": "t"}]
        result = plan_consolidation(facts, threshold=0.5)
        assert isinstance(result["groups"], list)

    def test_keep_ids_is_list(self):
        result = plan_consolidation([])
        assert isinstance(result["keep_ids"], list)

    def test_remove_ids_is_list(self):
        result = plan_consolidation([])
        assert isinstance(result["remove_ids"], list)

    def test_removed_count_is_int(self):
        result = plan_consolidation([])
        assert isinstance(result["removed_count"], int)

    def test_default_threshold(self):
        facts = [{"id": "a", "content": "test content", "topic": "t"}]
        result = plan_consolidation(facts)
        assert result["removed_count"] == 0

    def test_same_length_different_content(self):
        facts = [
            {"id": "a", "content": "abcd", "topic": "t"},
            {"id": "b", "content": "wxyz", "topic": "t"},
        ]
        result = plan_consolidation(facts, threshold=0.5)
        assert result["removed_count"] == 0

    def test_partial_overlap(self):
        facts = [
            {"id": "a", "content": "the user likes python programming very much", "topic": "dev"},
            {"id": "b", "content": "the user enjoys python programming a lot", "topic": "dev"},
        ]
        result = plan_consolidation(facts, threshold=0.3)
        assert result["removed_count"] == 1

    def test_no_topic_field(self):
        facts = [{"id": "a", "content": "test"}, {"id": "b", "content": "test"}]
        result = plan_consolidation(facts, threshold=0.5)
        assert result["removed_count"] == 1

    def test_empty_topic_string(self):
        facts = [
            {"id": "a", "content": "same content", "topic": ""},
            {"id": "b", "content": "same content", "topic": ""},
        ]
        result = plan_consolidation(facts, threshold=0.5)
        assert result["removed_count"] == 1

    def test_groups_dedup_content(self):
        facts = [
            {"id": "a", "content": "likes cats", "topic": "t"},
            {"id": "b", "content": "likes cats", "topic": "t"},
            {"id": "c", "content": "likes dogs", "topic": "t"},
        ]
        result = plan_consolidation(facts, threshold=0.5)
        assert result["removed_count"] == 1
        assert len(result["groups"]) == 1

    def test_transitive_merge_three(self):
        facts = [
            {"id": "a", "content": "user likes red color", "topic": "color"},
            {"id": "b", "content": "user prefers red color shade", "topic": "color"},
            {"id": "c", "content": "user enjoys red color hue", "topic": "color"},
        ]
        result = plan_consolidation(facts, threshold=0.3)
        assert result["removed_count"] == 2
        assert len(result["groups"]) == 1

    def test_multiple_clusters_per_topic(self):
        facts = [
            {"id": "a", "content": "likes cats", "topic": "pets"},
            {"id": "b", "content": "likes felines", "topic": "pets"},
            {"id": "c", "content": "drives a red sedan car", "topic": "transport"},
            {"id": "d", "content": "owns a red sedan vehicle", "topic": "transport"},
        ]
        result = plan_consolidation(facts, threshold=0.3)
        assert result["removed_count"] == 2
        assert len(result["groups"]) == 2

    def test_keep_ids_input_order(self):
        facts = [
            {"id": "z", "content": "same content here", "topic": "t"},
            {"id": "a", "content": "same content here", "topic": "t"},
        ]
        result = plan_consolidation(facts, threshold=0.5)
        assert result["keep_ids"] == ["z"]
        assert result["remove_ids"] == ["a"]
