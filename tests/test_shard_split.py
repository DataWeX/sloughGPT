"""Shard-split mechanics (card 20261107_112103).

The contract CI depends on: shards are DISjoint and their union is the full
list, a file's tests never straddle two shards (shared state lives at file
scope — module globals, ordered disk chains; run 45361359978), the spec
parser fails loud on garbage (a misconfigured CI must never silently run
the whole suite in one shard), and unset env = exact no-op so local/testmon
runs are byte-identical to before the hook existed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests import shard_split


def _items(*nodeids: str) -> list[SimpleNamespace]:
    """Collected-item stand-ins: the split only reads ``item.nodeid``."""
    return [SimpleNamespace(nodeid=nid) for nid in nodeids]


def _file(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


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
    ITEMS = _items(
        "tests/a.py::test_one",
        "tests/a.py::test_two",
        "tests/b.py::test_one",
        "tests/b.py::test_two",
        "tests/c.py::test_one",
        "tests/d.py::test_one",
        "tests/e.py::test_one",
        "tests/f.py::test_one",
    )

    def test_shards_are_disjoint_and_complete(self):
        n = 4
        shards = [shard_split.apply(self.ITEMS, f"{k}/{n}") for k in range(1, n + 1)]
        merged = sorted(i.nodeid for s in shards for i in s)
        assert merged == sorted(i.nodeid for i in self.ITEMS)

    def test_file_never_straddles_shards(self):
        """Shared state lives at file scope — the split must respect it."""
        n = 4
        owners: dict[str, int] = {}
        for k in range(1, n + 1):
            for item in shard_split.apply(self.ITEMS, f"{k}/{n}"):
                f = _file(item.nodeid)
                assert owners.setdefault(f, k) == k, f"{f} straddles shards"
        assert set(owners) == {_file(i.nodeid) for i in self.ITEMS}

    def test_placement_ignores_collection_order(self):
        """Filesystem order differs per runner; placement must not."""

        def placement(items):
            return {
                _file(i.nodeid): k for k in range(1, 5) for i in shard_split.apply(items, f"{k}/4")
            }

        assert placement(self.ITEMS) == placement(list(reversed(self.ITEMS)))

    def test_lpt_keeps_shards_balanced(self):
        items = _items(*(f"tests/f{i}.py::test_{j}" for i in range(8) for j in range(5)))
        sizes = [len(shard_split.apply(items, f"{k}/4")) for k in range(1, 5)]
        assert sizes == [10, 10, 10, 10]

    def test_single_shard_is_everything(self):
        shard = shard_split.apply(self.ITEMS, "1/1")
        assert sorted(i.nodeid for i in shard) == sorted(i.nodeid for i in self.ITEMS)

    def test_more_shards_than_items_yields_empty_not_error(self):
        assert shard_split.apply(_items("a.py::t1", "b.py::t1"), "4/4") == []


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
