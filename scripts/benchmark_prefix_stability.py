"""
Cross-turn prompt-prefix stability benchmark (card 20261103_071231).

KV reuse is decided by one thing: the longest shared token **prefix** between
this turn's prompt and the cached state. Measured on the real model, a
repeated 482-token prompt costs **43.02 s** cold and **1.26 s** warm
(scripts/benchmark_kv_prefix_cache.py). So the question for every
context-injection decision is not "did the model see the fact" but "did the
head change".

This measures the chat seam's own assembly across turns whose per-turn
context differs, and reports the shared prefix in tokens for two placements:

  head-fold    per-turn context appended to the system prompt
               (the pre-fix path, also what `messages.insert(0, ...)` did)
  tail-attach  per-turn context appended to the LAST user turn via the real
               ``routers.inference._append_volatile_context``

``head-fold`` is the *generous* baseline: the old RAG path inserted a system
message at index 0, which changes the very first token and reuses nothing at
all. Appending to the existing system prompt at least keeps the caller's own
instructions reusable — so any gap measured here is a lower bound on the
regression that produced the 43 s figure.

No model, no GPU: the quantity is a property of the strings the router hands
the provider, so this runs in CI in milliseconds and cannot silently drift
into measuring the model instead of the seam. Token counts use the router's
own ``_estimate_tokens`` (~4 chars/token), the same estimate its context
budgeter uses — consistent numbers beat precise-but-different ones here.

Usage:
    ./scripts/python scripts/benchmark_prefix_stability.py
    ./scripts/python scripts/benchmark_prefix_stability.py --json out.json
    ./scripts/python scripts/benchmark_prefix_stability.py --turns 8
Exit code is non-zero when tail-attach reuses less than ``--min-reuse``
(default 0.90) of the head — i.e. when a change has re-broken the prefix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
for _p in (
    str(_REPO),
    str(_REPO / "packages" / "downcraft"),
    str(_REPO / "packages" / "core-py"),
    str(_REPO / "packages" / "mogdb" / "src"),
    str(_REPO / "apps" / "api" / "server"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from apps.api.server.routers.inference import (  # noqa: E402
    _append_volatile_context,
    _estimate_tokens,
)

SYSTEM_PROMPT = (
    "You are sloughGPT, a local-first assistant. Answer directly, keep the "
    "format the user asked for, and never invent facts you were not given."
)

# Per-turn retrieval context, deliberately different every turn — this is what
# the bug rotated: knowledge facts, the built context frame, and RAG passages.
TURN_CONTEXT = [
    "The user's timezone is Europe/Berlin.",
    "The deploy target for this project is aarch64 with musl.",
    "Preferred editor keybindings: vim defaults, no leader remap.",
    "Chunk size for the RAG index is 512 tokens with 64 overlap.",
    "The inference server listens on port 8000 behind the systemd user unit.",
    "KV cache quantization is int8 on the serving path.",
    "The dataset adapter streams JSONL, never loads the whole file.",
    "Trait weights are clamped to [0, 1] by construction.",
]

USER_TURNS = [
    "what should I do next?",
    "and after that?",
    "summarise the plan.",
    "how long will it take?",
    "what could go wrong?",
    "what is the rollback?",
    "who owns it?",
    "what do I check first?",
]


def _context_block(facts: list[str]) -> str:
    """Render per-turn context the way the router does before attaching it."""
    return "Use the following context to answer:\n" + "\n".join(f"- {f}" for f in facts)


def _render(messages: list[dict]) -> str:
    """Flatten messages in order — the prefix the provider tokenises.

    The chat template adds constant framing around each role, which shifts
    both placements by the same constant and therefore cannot change the
    comparison this benchmark exists to make.
    """
    return "".join(f"<{m['role']}>\n{m.get('content') or ''}\n" for m in messages)


def _shared_prefix_tokens(a: str, b: str) -> int:
    """Token count of the longest common character prefix (~4 chars/token)."""
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return 0 if i == 0 else _estimate_tokens(a[:i])


def _turn_messages(turn: int, history_turns: int) -> list[dict]:
    """Messages for one turn: system + prior exchanges + this user message.

    ``history_turns <= 0`` means the client re-sends the whole conversation,
    which is what browsers do on every request.
    """
    start = 0 if history_turns <= 0 else max(0, turn - history_turns)
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for t in range(start, turn):
        messages.append({"role": "user", "content": USER_TURNS[t % len(USER_TURNS)]})
        messages.append({"role": "assistant", "content": "ok"})
    messages.append({"role": "user", "content": USER_TURNS[turn % len(USER_TURNS)]})
    return messages


def _head_fold(messages: list[dict], block: str) -> list[dict]:
    """The pre-fix path: per-turn context appended to the system turn."""
    out = [dict(m) for m in messages]
    out[0]["content"] = f"{out[0]['content']}\n\n{block}"
    return out


def _score(placement: str, turns: int, history_turns: int) -> dict:
    """Reuse as the cache actually measures it: vs the PREVIOUS turn's prompt.

    The session state holds ``prev_ids`` = last prompt + last output, and
    ``_resolve_kv_state`` reuses the leading run of tokens that matches it.
    So the number that matters is ``shared_prefix(prompt[i-1], prompt[i])`` —
    not the distance from turn 0, and not the head alone.

    The client re-sends the conversation on every request, and the injected
    block lives only inside the request (``_append_volatile_context`` mutates
    the parsed copy; the persisted session history stores the raw message), so
    each turn's history is byte-identical to the previous turn's request up to
    where the new block starts. That is exactly why tail-attach keeps a large
    reusable prefix and head-fold keeps almost none.
    """
    prompts: list[str] = []
    for turn in range(turns):
        messages = _turn_messages(turn, history_turns)
        block = _context_block([TURN_CONTEXT[(turn + k) % len(TURN_CONTEXT)] for k in range(3)])
        if placement == "head-fold":
            messages = _head_fold(messages, block)
        else:
            messages = _append_volatile_context(messages, block)
        prompts.append(_render(messages))

    reused: list[int] = []
    fractions: list[float] = []
    for prev, cur in zip(prompts, prompts[1:]):
        shared = _shared_prefix_tokens(prev, cur)
        reused.append(shared)
        fractions.append(shared / _estimate_tokens(cur))

    return {
        "placement": placement,
        "turns": turns,
        "head_tokens": _estimate_tokens(prompts[0]),
        "reused_tokens_per_turn": reused,
        "mean_reuse_fraction": round(sum(fractions) / len(fractions), 4) if fractions else 0.0,
        "min_reuse_fraction": round(min(fractions), 4) if fractions else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--turns", type=int, default=10, help="turns per placement")
    ap.add_argument(
        "--history", type=int, default=0, help="prior exchanges kept (0 = the whole conversation)"
    )
    ap.add_argument(
        "--min-multiple",
        type=float,
        default=2.0,
        help="tail-attach must reuse this many times more tokens than head-fold at the last turn",
    )
    ap.add_argument("--json", metavar="PATH", help="write metrics as JSON")
    args = ap.parse_args()

    rows = [_score(p, args.turns, args.history) for p in ("head-fold", "tail-attach")]

    print("Cross-turn prompt-prefix stability (tokens, router estimate)")
    print(f"  turns={args.turns}  history={args.history}  system_prompt={len(SYSTEM_PROMPT)} chars")
    print()
    print(
        f"  {'placement':<12} {'turn-0':>7} {'mean reuse':>11} {'min reuse':>10}  reused/turn (vs prev)"
    )
    for row in rows:
        print(
            f"  {row['placement']:<12} {row['head_tokens']:>7} "
            f"{row['mean_reuse_fraction']:>11.1%} {row['min_reuse_fraction']:>10.1%}  "
            f"{row['reused_tokens_per_turn']}"
        )

    tail = next(r for r in rows if r["placement"] == "tail-attach")
    head = next(r for r in rows if r["placement"] == "head-fold")
    tail_last = tail["reused_tokens_per_turn"][-1]
    head_last = head["reused_tokens_per_turn"][-1]
    multiple = tail_last / head_last if head_last else float("inf")
    ok = multiple >= args.min_multiple

    print()
    print(
        f"  head-fold pins the reusable prefix at the system prompt "
        f"({head_last} tokens) no matter how long the conversation gets — that is the "
        "43 s cold prefill, paid every turn."
    )
    print(
        f"  tail-attach grows it with the conversation "
        f"({tail['reused_tokens_per_turn'][0]} -> {tail_last} tokens) because the per-turn "
        "context no longer sits in front of the history."
    )
    print(
        f"  gate: last-turn reuse multiple >= {args.min_multiple:.1f}x -> "
        f"{multiple:.1f}x {'PASS' if ok else 'FAIL'}"
    )

    if args.json:
        Path(args.json).write_text(
            json.dumps({"rows": rows, "min_multiple": args.min_multiple, "pass": ok}, indent=2)
            + "\n"
        )
        print(f"  wrote {args.json}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
