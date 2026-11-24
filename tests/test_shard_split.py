"""Shard-split mechanics (card 20261107_112103).

The contract CI depends on: shards are DISjoint and their union is the full
list, the spec parser fails loud on garbage (a misconfigured CI must never
silently run the whole suite in one shard), and unset env = exact no-op so
local/testmon runs are byte-identical to before the hook existed.
"""

from __future__ import annotations

import pytest

from tests import shard_split


class TestParse:
    def test_valid_spec(self):
        assert shard_split.parse("2/4") == (2, 4)
        assert shard_split.parse(" 1/1 ") == (1, 1)

    @pytest.mark.parametrize(
        "bad",
        ["", "4", "4/", "/4", "0/4", "5/4", "-1/4", "a/b", "2/0", "2.5/4"],
    )
    def test_invalid_specs_raise(self, bad):
        with pytest.raises(ValueError):
            shard_split.parse(bad)


class TestApply:
    ITEMS = list(range(10))

    def test_shards_are_disjoint_and_complete(self):
        n = 4
        shards = [shard_split.apply(self.ITEMS, f"{k}/{n}") for k in range(1, n + 1)]
        merged = sorted(x for s in shards for x in s)
        assert merged == self.ITEMS  # union == full list, no dupes, no loss

    def test_round_robin_membership(self):
        assert shard_split.apply(self.ITEMS, "1/4") == [0, 4, 8]
        # indices 3, 7 (11 is past the end); index 9 lands in group 2.
        assert shard_split.apply(self.ITEMS, "4/4") == [3, 7]

    def test_single_shard_is_everything(self):
        assert shard_split.apply(self.ITEMS, "1/1") == self.ITEMS

    def test_more_shards_than_items_yields_empty_not_error(self):
        assert shard_split.apply([1, 2], "4/4") == []


class TestEnabled:
    def test_unset_means_none(self, monkeypatch):
        monkeypatch.delenv("SLO_TEST_SHARD", raising=False)
        assert shard_split.enabled() is None

    def test_empty_string_means_none(self, monkeypatch):
        monkeypatch.setenv("SLO_TEST_SHARD", "")
        assert shard_split.enabled() is None

    def test_set_returns_raw_spec(self, monkeypatch):
        monkeypatch.setenv("SLO_TEST_SHARD", "3/4")
        assert shard_split.enabled() == "3/4"
