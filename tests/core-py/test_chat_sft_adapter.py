"""chat-sft adapter — format, renderer, split discipline (card 20261101_052145 step 1).

TDD pins for the reserved ``chat-sft`` task's first adapter:
  * registry — boot registers the spec with its BatchSampler contract;
  * renderer round-trip — tool_calls survive into the rendered text with
    the same role markers the existing chat text paths use;
  * held-out split — ``split="heldout"`` records never enter the default
    train BatchArray (training on the eval set fakes the eval);
  * BatchArray contract — sized, random-access, re-iterable.
"""

from __future__ import annotations

import json

import pytest

from domain.training._internal.adapter_registry import REGISTRY, boot
from domain.training._internal.chat_sft_adapter import (
    TrajectoryBatchArray,
    build,
    render_trajectory,
)


def _record(messages, **extra):
    return json.dumps({"messages": messages, **extra})


def _jsonl(tmp_path, *records, name="trajectories.jsonl"):
    f = tmp_path / name
    f.write_text("\n".join(records) + "\n", encoding="utf-8")
    return f


TOOL_TRAJ = [
    {"role": "system", "content": "you are an agent"},
    {"role": "user", "content": "check the weather"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "weather", "arguments": '{"city": "Oslo"}'}}],
    },
    {"role": "tool", "content": '{"temp": 3}'},
    {"role": "assistant", "content": "It is 3 degrees in Oslo."},
]


class TestRegistry:
    def test_boot_registers_chat_sft(self):
        boot()
        spec = REGISTRY.get("chat-sft")
        assert spec.task == "chat-sft"
        assert spec.sampler == "token-stream"
        assert spec.block_size is None
        assert spec.consumes == ("text-records",)

    def test_produces_is_batch_array(self):
        boot()
        spec = REGISTRY.get("chat-sft")
        for dunder in ("__len__", "__getitem__", "__iter__"):
            assert hasattr(spec.produces, dunder), f"produces lacks {dunder}"


class TestRenderer:
    def test_tool_calls_survive_rendering(self):
        text = render_trajectory(TOOL_TRAJ)
        assert "weather" in text
        assert '{"city": "Oslo"}' in text
        assert "It is 3 degrees in Oslo." in text

    def test_role_markers_match_chat_convention(self):
        """Same markers as ExperienceSampler/chat_trainer render today."""
        text = render_trajectory(TOOL_TRAJ)
        assert "System: you are an agent" in text
        assert "User: check the weather" in text
        assert 'Tool: {"temp": 3}' in text
        assert "Assistant: It is 3 degrees in Oslo." in text

    def test_unknown_role_fails_fast(self):
        with pytest.raises(ValueError, match="role"):
            render_trajectory([{"role": "narrator", "content": "hi"}])

    def test_empty_trajectory_fails_fast(self):
        with pytest.raises(ValueError, match="messages"):
            render_trajectory([])


class TestHeldOutSplit:
    def test_heldout_records_never_enter_default_train(self, tmp_path):
        path = _jsonl(
            tmp_path,
            _record([{"role": "user", "content": "train one"}]),
            _record([{"role": "user", "content": "held out"}], split="heldout"),
            _record([{"role": "user", "content": "train two"}]),
        )
        train = build(path)
        assert len(train) == 2
        assert all("held out" not in item for item in train)

    def test_heldout_split_loads_exactly_the_held_out(self, tmp_path):
        path = _jsonl(
            tmp_path,
            _record([{"role": "user", "content": "train one"}]),
            _record([{"role": "user", "content": "held out"}], split="heldout"),
        )
        held = build(path, split="heldout")
        assert len(held) == 1
        assert "held out" in held[0]

    def test_split_none_loads_everything(self, tmp_path):
        path = _jsonl(
            tmp_path,
            _record([{"role": "user", "content": "train"}]),
            _record([{"role": "user", "content": "held"}], split="heldout"),
        )
        assert len(build(path, split=None)) == 2

    def test_malformed_record_fails_fast(self, tmp_path):
        path = _jsonl(tmp_path, '{"not": "a trajectory"}')
        with pytest.raises(ValueError, match="messages"):
            build(path)

    def test_invalid_json_fails_fast_with_line_number(self, tmp_path):
        f = tmp_path / "bad.jsonl"
        f.write_text("{broken\n", encoding="utf-8")
        with pytest.raises(ValueError, match=":1:"):
            build(f)


class TestBatchArrayContract:
    def test_sized_random_access_reiterable(self, tmp_path):
        path = _jsonl(
            tmp_path,
            _record([{"role": "user", "content": "alpha"}]),
            _record([{"role": "user", "content": "beta"}]),
        )
        arr = build(path)
        assert isinstance(arr, TrajectoryBatchArray)
        assert len(arr) == 2
        first_pass = list(arr)
        assert first_pass == list(arr), "BatchArray must be re-iterable (epochs)"
        assert arr[0].startswith("User: alpha")
        assert arr[1].startswith("User: beta")
