"""chat-sft adapter — chat-format tool-use trajectories → ``BatchArray`` (ADR-002).

Card 20261101_052145 step 1: fills ADR-002's reserved ``chat-sft`` task —
a dataset adapter that renders JSONL trajectory records
(``{"messages": [...], "source": ..., "split": ...}``) into the ONE
trainer's token-stream path. No new training loop (the registry leaf rule
is gate-pinned — this module never imports the loop); v1 = full-coverage
loss over rendered chat text, exactly what the existing chat paths
(``ExperienceSampler`` / ``chat_trainer``) do today.

Split discipline: records may carry ``"split": "heldout"``; ``build()``
defaults to ``split="train"`` so a held-out eval set can never leak into
training by default (test-pinned).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from .adapter_registry import AdapterSpec, register

#: Rendered role markers — same convention as the existing chat text paths
#: (``ExperienceSampler`` / chat_trainer: "User: …\\nAssistant: …").
_ROLE_MARKERS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool",
}


def render_message(message: dict) -> list[str]:
    """One trajectory message → rendered lines (tool_calls survive verbatim)."""
    role = str(message.get("role", "")).lower()
    content = message.get("content") or ""
    marker = _ROLE_MARKERS.get(role)
    if marker is None:
        raise ValueError(f"unknown message role: {role!r}")
    if role == "assistant" and message.get("tool_calls"):
        lines = [f"{marker}: {content}"] if content else [f"{marker}:"]
        for call in message["tool_calls"]:
            fn = call.get("function", call)
            name = fn.get("name", "")
            arguments = fn.get("arguments", "")
            lines.append(f"Tool call: {name} {arguments}".rstrip())
        return lines
    return [f"{marker}: {content}"]


def render_trajectory(messages: list) -> str:
    """Trajectory → one training string (one rendered line per message)."""
    if not messages:
        raise ValueError("trajectory has no messages")
    lines: list[str] = []
    for message in messages:
        lines.extend(render_message(message))
    return "\n".join(lines) + "\n"


class TrajectoryBatchArray:
    """Map-style view over a JSONL trajectory file (one rendered string per record).

    Sized, random-access, re-iterable — the ``BatchArray`` contract the
    epochs/shuffle semantics require.
    """

    def __init__(self, path: Path | str, *, split: str | None = "train") -> None:
        self._rendered: list[str] = []
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{number}: invalid JSON trajectory record"
                    ) from exc
                if not isinstance(record, dict) or not isinstance(
                    record.get("messages"), list
                ):
                    raise ValueError(
                        f"{path}:{number}: trajectory record needs a messages list"
                    )
                if split is not None and record.get("split", "train") != split:
                    continue
                self._rendered.append(render_trajectory(record["messages"]))

    def __len__(self) -> int:
        return len(self._rendered)

    def __getitem__(self, i: int) -> str:
        return self._rendered[i]

    def __iter__(self) -> Iterator[str]:
        return iter(self._rendered)


def build(path: Path | str, *, split: str | None = "train") -> TrajectoryBatchArray:
    """JSONL trajectories → ``BatchArray`` (the adapter's declared out-contract)."""
    return TrajectoryBatchArray(path, split=split)


SPEC = AdapterSpec(
    name="chat-sft",
    task="chat-sft",
    consumes=("text-records",),
    produces=TrajectoryBatchArray,
    build=build,
    module=__name__,
    # BatchSampler contract: rendered trajectories are flat text the loop
    # tokenizes and cuts into blocks — block_size comes from the training
    # config at runtime (same as text-pretrain).
    sampler="token-stream",
    block_size=None,
)

register(SPEC)
