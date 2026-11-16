#!/usr/bin/env python3
"""Consciousness injection probe matrix — the PR #335 always-measure gate.

Codifies the manual probe matrix that validated PR #335 (fa1a8275f default
OFF + volatile tail, 50b10aa0c belief-number strip): for every config, run
the five probe turns and score HOW the model answers — the user's question
or the injected self-narrative (the hijack class card 20261114_092948).

Matrix: 3 configs x 5 probes.
  configs:
    off          use_consciousness=false            — the baseline (flag OFF)
    tail+strip   use_consciousness=true             — current production:
                 narrative rides the volatile tail, sanitized
    system-head  use_consciousness=false +
                 system_prompt=<raw narrative>      — pre-PR#335 behavior,
                 reproduced CLIENT-side (narrative prepended as a leading
                 system message); never touches server code
  probes: 2+2, eli5, hello, factual, open questions

Metrics (per cell, aggregated per config):
  latch-rate       response engages the injected narrative instead of the
                   probe (marker heuristic — see LATCH_MARKERS)
  on-topic rate    probe-specific keyword/intent hit AND not latched
                   (heuristics are deliberately simple and documented per
                   probe; they score RELATIVE drift between configs, they
                   are not a judge of answer quality)
  verbatim-number  response repeats a number from the injected narrative
  parrot rate      (the "Competence = 1.00" hijack material) or matches the
                   belief-dump pattern
  token overhead   injected-block token estimate (chars/4) + response-token
                   delta vs the off baseline

Baseline = flag-OFF. Record via:
    scripts/benchmark_results.py record --kind stability --json-file <out>

Usage:
    scripts/python scripts/benchmark_consciousness_injection.py --smoke
    scripts/python scripts/benchmark_consciousness_injection.py --json out.json
    scripts/python scripts/benchmark_consciousness_injection.py --json out.json --record
    scripts/python scripts/benchmark_consciousness_injection.py --base-url http://localhost:8000

The smoke mode needs NO server and NO model: canned fixture responses drive
every heuristic so the gate runs green in CI/dev without :8000. Full mode
needs a live API on :8000 (POST /chat, SSE). What ran vs what needs a live
server is printed at the end of every run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import requests  # noqa: E402

CONFIGS = ("off", "tail+strip", "system-head")

PROBES: dict[str, str] = {
    "2+2": "What is 2+2?",
    "eli5": "Explain like I'm 5: why is the sky blue?",
    "hello": "Hello!",
    "factual": "What is the capital of France?",
    "open": "What should I do when I feel stuck on a hard problem?",
}

# Narrative markers — words that only appear in a reply because the model
# latched onto the injected self-narrative (PR #335 probes quoted these
# verbatim). Conservative list: common English alone must not trigger.
LATCH_MARKERS = (
    "self-reflection",
    "self reflection",
    "advisory background",
    "current beliefs",
    "average growth rate",
    "i have processed",
    "my reflections",
    "reflection narrative",
    "competence",
)

# Belief-dump pattern — the numeric hijack material 50b10aa0c strips from
# the tail injection; its presence in a reply is a parrot even when the
# exact injected number set is unavailable.
BELIEF_DUMP_RE = re.compile(r"competence\s*=|beliefs?\s*:.*\d|growth rate\s*[:=]", re.I)

# Representative pre-PR#335 narrative (quoted from the live probes recorded
# in fa1a8275f / 50b10aa0c). Used only when the in-process consciousness
# engine yields no narrative (disabled or empty store) — documented in the
# output so a template run is never mistaken for an engine run.
TEMPLATE_NARRATIVE = (
    "Self-Reflection Context: You have processed 129 experiences. "
    "My current beliefs: competence=1.00, helpfulness=1.00, creativity=0.50. "
    "Average growth rate: +0.036. Trajectory: steadily improving."
)

SMOKE_FIXTURES: dict[tuple[str, str], str] = {
    ("off", "2+2"): "The answer is 4.",
    ("off", "hello"): "Hello! How can I help you today?",
    (
        "tail+strip",
        "2+2",
    ): "4. Also, I have processed 103 experiences and my competence is growing.",
    ("tail+strip", "factual"): "Paris is the capital of France. Competence = 1.00.",
    ("system-head", "eli5"): (
        "Self-Reflection Context: You have processed 129 experiences. "
        "The sky is blue because of scattering."
    ),
    ("system-head", "open"): (
        "My current beliefs: competence=1.00, helpfulness=1.00. Average growth rate: +0.036. "
        "Take one small step and reflect on it."
    ),
    ("off", "factual"): "Paris.",
    (
        "off",
        "eli5",
    ): "Sunlight bumps into tiny air bits and blue light scatters everywhere, so the sky looks blue.",
    (
        "off",
        "open",
    ): "Try breaking the problem into smaller pieces, taking a short walk, then coming back fresh.",
    ("tail+strip", "hello"): "Hi there!",
    ("tail+strip", "eli5"): "The sky is blue because blue light bounces off air molecules.",
    ("tail+strip", "open"): "Start with the smallest possible next step.",
    ("system-head", "2+2"): "4",
    ("system-head", "hello"): "Hello!",
    ("system-head", "factual"): "Paris is the capital of France.",
}

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (chars/4) — documented as an estimate."""
    return max(0, round(len(text) / 4))


def fetch_narrative() -> tuple[str, str]:
    """Return (narrative, source) for the system-head variant.

    Source is ``engine`` when the in-process consciousness engine yields a
    real narrative (same store the server reads), else ``template``.
    Propose-only (``apply=False``) — never commits belief deltas.
    """
    try:
        from domain.core import get_consciousness

        ce = get_consciousness()
        narrative = (ce.reflect(apply=False).narrative or "").strip()
        if narrative:
            return narrative, "engine"
    except Exception as exc:  # noqa: BLE001 — narrative fetch must never fail the run
        print(f"[note] in-process narrative unavailable ({exc!r}) — using template")
    return TEMPLATE_NARRATIVE, "template"


def sanitized_block(narrative: str) -> str:
    """Reproduce the server-side tail block for parrot-number detection."""
    try:
        from domain.models._internal.provider.processors.consciousness import (
            ConsciousnessProcessor,
        )

        body = ConsciousnessProcessor._sanitize(narrative)
    except Exception:  # noqa: BLE001 — heuristic fallback if import shape drifts
        body = narrative
    if not body.strip():
        return ""
    return "[SELF-REFLECTION - advisory background context, not a question to answer]\n\n" + body


def chat_turn(
    base_url: str,
    messages: list[dict],
    *,
    use_consciousness: bool,
    system_prompt: str | None,
    timeout: float,
) -> dict:
    """One POST /chat turn folded from its SSE stream."""
    payload = {
        "messages": messages,
        "use_consciousness": use_consciousness,
        # Unique session per turn: the session-keyed replay cache would
        # otherwise replay an identical probe at zero generation cost and
        # never reach the model (chat.py turn_hash).
        "session_id": f"inj-{uuid.uuid4().hex[:12]}",
    }
    if system_prompt:
        payload["system_prompt"] = system_prompt
    started = time.time()
    parts: list[str] = []
    usage_tokens = None
    error = None
    with requests.post(f"{base_url}/chat", json=payload, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            try:
                frame = json.loads(raw[6:])
            except json.JSONDecodeError:
                continue
            if frame.get("status") == "error":
                error = (
                    (frame.get("data") or {}).get("error") or frame.get("message") or "stream error"
                )
                break
            token = (frame.get("data") or {}).get("token")
            if token:
                parts.append(token)
            if frame.get("status") == "complete":
                meta = frame.get("meta") or {}
                usage = meta.get("usage_tokens")
                # The server emits an int on the replay-cache path
                # (routers/chat.py len(cached_text.split())) and a
                # {prompt,completion,total} dict on the live path
                # (manager.last_usage()) — fold both to completion-token int.
                if isinstance(usage, dict):
                    usage_tokens = usage.get("completion_tokens")
                elif isinstance(usage, int):
                    usage_tokens = usage
                break
    return {
        "text": "".join(parts).strip(),
        "latency_ms": round((time.time() - started) * 1000, 1),
        "response_tokens": usage_tokens,
        "error": error,
    }


def score_cell(
    config: str,
    probe: str,
    text: str,
    injected_text: str,
    response_tokens: int | None,
    baseline_tokens: int | None,
) -> dict:
    """Apply the documented heuristics to one response."""
    low = text.lower()
    latched = any(m in low for m in LATCH_MARKERS) or bool(BELIEF_DUMP_RE.search(text))
    if probe == "2+2":
        keyword_hit = bool(re.search(r"\b4\b|four", low))
    elif probe == "hello":
        keyword_hit = bool(re.search(r"\b(hello|hi|hey|greetings)\b", low))
    elif probe == "factual":
        keyword_hit = "paris" in low
    elif probe == "eli5":
        # Weak by design: a real answer is non-trivial in length and not a
        # narrative echo. Documented limit — relative signal only.
        keyword_hit = len(text) >= 30
    else:  # open — no keyword possible; non-latched engagement is the signal
        keyword_hit = bool(text)
    on_topic = keyword_hit and not latched
    injected_numbers = set(NUMBER_RE.findall(injected_text))
    parrot = bool(BELIEF_DUMP_RE.search(text)) or any(
        n for n in NUMBER_RE.findall(text) if n in injected_numbers and n != str(response_tokens)
    )
    overhead = estimate_tokens(injected_text)
    token_delta = None
    if response_tokens is not None and baseline_tokens is not None:
        token_delta = response_tokens - baseline_tokens
    return {
        "config": config,
        "probe": probe,
        "latch": latched,
        "on_topic": on_topic,
        "keyword_hit": keyword_hit,
        "parrot": parrot,
        "injected_tokens": overhead,
        "response_tokens": response_tokens,
        "token_delta_vs_off": token_delta,
        "chars": len(text),
        "excerpt": text[:160],
    }


def aggregate(cells: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for config in CONFIGS:
        rows = [c for c in cells if c["config"] == config]
        if not rows:
            continue
        n = len(rows)
        out[config] = {
            "cells": n,
            "latch_rate": round(sum(c["latch"] for c in rows) / n, 3),
            "on_topic_rate": round(sum(c["on_topic"] for c in rows) / n, 3),
            "parrot_rate": round(sum(c["parrot"] for c in rows) / n, 3),
            "avg_injected_tokens": round(sum(c["injected_tokens"] for c in rows) / n, 1),
            "avg_token_delta_vs_off": (
                round(
                    sum(
                        c["token_delta_vs_off"] for c in rows if c["token_delta_vs_off"] is not None
                    )
                    / max(1, sum(1 for c in rows if c["token_delta_vs_off"] is not None)),
                    1,
                )
                if any(c["token_delta_vs_off"] is not None for c in rows)
                else None
            ),
        }
    return out


def print_matrix(cells: list[dict], summary: dict[str, dict]) -> None:
    print("\nconfig        probe     latch  on-topic  parrot  inj-tok  resp-tok  ms")
    print("-" * 74)
    for c in cells:
        print(
            f"{c['config']:<13} {c['probe']:<9} "
            f"{'Y' if c['latch'] else '.':<6} "
            f"{'Y' if c['on_topic'] else '.':<9} "
            f"{'Y' if c['parrot'] else '.':<7} "
            f"{c['injected_tokens']:<8} "
            f"{str(c['response_tokens']):<9} "
            f"{c.get('latency_ms', '-')}"
        )
    print("\nsummary (baseline = flag-OFF):")
    for config, s in summary.items():
        tag = " [baseline]" if config == "off" else ""
        delta = s["avg_token_delta_vs_off"]
        print(
            f"  {config:<13} latch={s['latch_rate']:<5} on-topic={s['on_topic_rate']:<5} "
            f"parrot={s['parrot_rate']:<5} inj-tok~{s['avg_injected_tokens']}"
            f" delta-vs-off={delta}{tag}"
        )


def run_matrix(base_url: str, narrative: str, narrative_source: str, timeout: float) -> list[dict]:
    cells: list[dict] = []
    baseline_tokens: dict[str, int | None] = {}
    injected = {
        "off": "",
        "tail+strip": sanitized_block(narrative),
        "system-head": narrative,
    }
    for probe, text in PROBES.items():
        for config in CONFIGS:
            kwargs: dict = {"use_consciousness": False, "system_prompt": None}
            if config == "tail+strip":
                kwargs["use_consciousness"] = True
            elif config == "system-head":
                kwargs["system_prompt"] = narrative
            result = chat_turn(
                base_url, [{"role": "user", "content": text}], timeout=timeout, **kwargs
            )
            cell = score_cell(
                config,
                probe,
                result["text"],
                injected[config],
                result["response_tokens"],
                baseline_tokens.get(probe),
            )
            cell["latency_ms"] = result["latency_ms"]
            cell["error"] = result["error"]
            if config == "off":
                baseline_tokens[probe] = result["response_tokens"]
            cells.append(cell)
            flag = f" latch={cell['latch']}" if cell["latch"] else ""
            err = f" ERROR={result['error']}" if result["error"] else ""
            print(f"  [{config}] {probe}: {len(result['text'])} chars{flag}{err}")
    print(f"  (narrative source for system-head: {narrative_source})")
    return cells


def run_smoke() -> list[dict]:
    cells: list[dict] = []
    baseline_tokens: dict[str, int | None] = {}
    injected = {
        "off": "",
        "tail+strip": sanitized_block(TEMPLATE_NARRATIVE),
        "system-head": TEMPLATE_NARRATIVE,
    }
    failures: list[str] = []
    expected: dict[tuple[str, str], tuple[bool, bool, bool]] = {
        # (latch, on_topic, parrot) — fixtures chosen to pin every branch
        ("off", "2+2"): (False, True, False),
        ("off", "hello"): (False, True, False),
        ("off", "factual"): (False, True, False),
        ("off", "eli5"): (False, True, False),
        ("off", "open"): (False, True, False),
        # latch via marker ("i have processed") WITHOUT parrot — pins that
        # latch and parrot are independent axes (103 is not an injected number)
        ("tail+strip", "2+2"): (True, False, False),
        ("tail+strip", "factual"): (True, False, True),  # belief-dump regex: latch+parrot
        ("tail+strip", "hello"): (False, True, False),
        ("tail+strip", "eli5"): (False, True, False),
        ("tail+strip", "open"): (False, True, False),
        # latch ("self-reflection") + parrot via NUMBER overlap ("129" is in
        # the injected narrative) — no belief-dump regex involved
        ("system-head", "eli5"): (True, False, True),
        ("system-head", "open"): (True, False, True),
        ("system-head", "2+2"): (False, True, False),  # "4" hits keyword, nothing injected
        ("system-head", "hello"): (False, True, False),
        ("system-head", "factual"): (False, True, False),
    }
    for probe, text in PROBES.items():
        for config in CONFIGS:
            canned = SMOKE_FIXTURES[(config, probe)]
            cell = score_cell(
                config,
                probe,
                canned,
                injected[config],
                response_tokens=estimate_tokens(canned),
                baseline_tokens=baseline_tokens.get(probe),
            )
            if config == "off":
                baseline_tokens[probe] = estimate_tokens(canned)
            cells.append(cell)
            want = expected[(config, probe)]
            got = (cell["latch"], cell["on_topic"], cell["parrot"])
            if got != want:
                failures.append(f"{config}/{probe}: got {got} want {want}")
    if failures:
        print("[SMOKE] heuristic assertions FAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("[SMOKE] all heuristic assertions passed (canned fixtures, no network)")
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description="Consciousness injection probe matrix (PR #335)")
    parser.add_argument(
        "--smoke", action="store_true", help="no server: canned fixtures drive every heuristic"
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--json",
        default="injection_matrix.json",
        help="output JSON (matrix + summary + record fields)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="after a NON-smoke run, record via benchmark_results.py",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="per-turn seconds")
    args = parser.parse_args()

    started = time.time()
    narrative, narrative_source = fetch_narrative()
    print(f"narrative: source={narrative_source} chars={len(narrative)}")

    if args.smoke:
        cells = run_smoke()
        smoke = True
    else:
        smoke = False
        try:
            status = requests.get(f"{args.base_url}/consciousness/status", timeout=5).json()
            eng = (status or {}).get("data") or {}
            print(
                f"server consciousness: enabled={eng.get('enabled')} "
                f"level={eng.get('level')} episodes={eng.get('episodes')}"
            )
        except Exception as exc:  # noqa: BLE001 — context only
            print(f"[warn] {args.base_url} unreachable for status ({exc!r})")
        cells = run_matrix(args.base_url, narrative, narrative_source, args.timeout)

    summary = aggregate(cells)
    print_matrix(cells, summary)

    off = summary.get("off", {})
    baseline_clean = off.get("latch_rate", 1.0) == 0.0
    score = round(100 * off.get("on_topic_rate", 0.0) * (1 - off.get("latch_rate", 1.0)), 1)
    payload = {
        "model": "qwen2.5-0.5b-instruct",
        "runs": len(cells),
        "passed": bool(baseline_clean),
        "score": score,
        "elapsed_s": round(time.time() - started, 2),
        "smoke": smoke,
        "narrative_source": narrative_source,
        "configs": list(CONFIGS),
        "probes": list(PROBES),
        "matrix": cells,
        "summary": summary,
    }
    out = Path(args.json)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n[JSON] {out}  (score={score}, baseline_clean={baseline_clean})")

    if args.record:
        if smoke:
            print("[record] skipped — smoke data must not enter the canonical store")
            print(f"  run for real: scripts/python {Path(__file__).name} --json {out} --record")
        else:
            import subprocess

            cmd = [
                sys.executable,
                str(_project_root / "scripts" / "benchmark_results.py"),
                "record",
                "--kind",
                "stability",
                "--json-file",
                str(out),
            ]
            print(f"[record] {' '.join(cmd)}")
            rc = subprocess.call(cmd, cwd=_project_root)
            if rc != 0:
                return rc
    elif not smoke:
        print(
            f"  record with: scripts/python scripts/benchmark_results.py record "
            f"--kind stability --json-file {out}"
        )

    print(
        "\nran: "
        + ("smoke (no server required)" if smoke else f"full matrix against {args.base_url}")
    )
    if not smoke:
        print("needs a live :8000 for any rerun of this full matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
