#!/usr/bin/env python
"""quiz — build, answer, and score multiple-choice quizzes on our stack.

The runnable use case for the verifiable-reward loop:

  build    generate MCQs from a source document -> jsonl
  answer   fill a model answer letter per question -> jsonl
  score    accuracy against the answer key

Backends: --soul (local checkpoint, our engine, numpy sampling) or
--http [ --api-key ] (hosted chat-completions endpoint). Stdlib only.
The builder is source-grounded on purpose: questions come FROM the
material, so the key is checkable and the flywheel stays honest.

Usage::

    python scripts/quiz.py build material.md --count 5 -o quiz.jsonl --soul model.soul
    python scripts/quiz.py answer quiz.jsonl --soul model.soul
    python scripts/quiz.py score quiz.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import urllib.request
from pathlib import Path

# Run from anywhere: the repo root carries `domain/`, and scripts/ is not
# on sys.path the way `python -c` inherits cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_LETTER = re.compile(r"\b([ABCD])\b")

_BUILD_PROMPT = """Create {n} multiple-choice question(s) about the material below.
Each has exactly one correct answer and exactly four options.
Reply with ONLY a JSON array, no prose:
[{{"question": "...", "options": ["...", "...", "...", "..."], "answer": "A"}}]

Material:
{source}
"""

_ANSWER_PROMPT = """Answer the multiple-choice question. Reply with ONLY the letter A, B, C, or D.

Question: {question}

{options}
"""


def _chat(
    prompt: str,
    soul: str | None = None,
    http: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> str:
    """One completion through our stack — local soul or hosted endpoint."""
    if soul:
        from domain.inference._internal.slonet_provider import SloNetChatProvider

        client = SloNetChatProvider.from_soul(soul, model_id=soul)
        return asyncio.run(
            client.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
    if http:
        payload: dict = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if model:
            payload["model"] = model
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(
            http.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, None, 120) as resp:
            data = json.loads(resp.read())
        return str(data["choices"][0]["message"]["content"])
    raise SystemExit("need a backend: --soul PATH or --http URL [--api-key KEY]")


def _extract_json_array(text: str) -> list:
    """First JSON array in the reply — tolerant of prose around it."""
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        found = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return found if isinstance(found, list) else []


def cmd_build(args: argparse.Namespace) -> int:
    source = Path(args.source).read_text(encoding="utf-8")
    if len(source) > 24_000:
        source = source[:24_000] + "\n...(truncated)"
    reply = _chat(
        _BUILD_PROMPT.format(n=args.count, source=source),
        soul=args.soul,
        http=args.http,
        api_key=args.api_key or None,
        model=args.model,
    )
    rows = []
    for item in _extract_json_array(reply):
        if not isinstance(item, dict) or "question" not in item or "answer" not in item:
            continue
        options = [str(o) for o in item.get("options", [])[:4]]
        if len(options) != 4:
            continue
        answer = str(item["answer"]).strip()[:1].upper()
        if answer not in "ABCD":
            continue
        rows.append(
            {
                "source_id": Path(args.source).name,
                "question": str(item["question"]),
                "options": options,
                "answer": answer,
            }
        )
    if not rows:
        print("no parseable questions came back — raw reply:", file=sys.stderr)
        print(reply[:600], file=sys.stderr)
        return 1
    out = Path(args.output)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"built {len(rows)} question(s) -> {out}")
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    path = Path(args.quiz)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    answered = 0
    for row in rows:
        options = "\n".join(
            f"{letter}) {text}" for letter, text in zip("ABCD", row.get("options", []))
        )
        reply = _chat(
            _ANSWER_PROMPT.format(question=row["question"], options=options),
            soul=args.soul,
            http=args.http,
            api_key=args.api_key or None,
            model=args.model,
            max_tokens=8,
        )
        match = _LETTER.search(reply.strip().upper())
        row["answer_model"] = match.group(1) if match else ""
        answered += 1
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"answered {answered} question(s) in {path}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in Path(args.quiz).read_text().splitlines() if line.strip()]
    if not rows:
        print("no questions")
        return 1
    correct = sum(
        1
        for r in rows
        if str(r.get("answer", "")).upper() == str(r.get("answer_model", "")).upper()
    )
    total = len(rows)
    print(f"score: {correct}/{total} = {correct / total:.0%}")
    misses = [r for r in rows if r.get("answer") != r.get("answer_model")][:5]
    for r in misses:
        print(f"  miss: {r['question'][:70]}  key={r.get('answer')} got={r.get('answer_model')}")
    return 0 if correct == total else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_backend(sp):
        sp.add_argument("--soul", default=None, help="local .soul checkpoint")
        sp.add_argument("--http", default=None, help="chat-completions base URL (…/v1)")
        sp.add_argument("--api-key", default=None, help="API key (or env AVION_API_KEY)")
        sp.add_argument("--model", default=None, help="model id for hosted endpoints")

    sp = sub.add_parser("build", help="generate MCQs from a source document")
    sp.add_argument("source", help="source material (text file)")
    sp.add_argument("--count", type=int, default=5)
    sp.add_argument("-o", "--output", default="quiz.jsonl")
    add_backend(sp)
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("answer", help="fill a model answer per question")
    sp.add_argument("quiz", help="quiz jsonl from build")
    sp.add_argument("--limit", type=int, default=0, help="answer only the first N")
    add_backend(sp)
    sp.set_defaults(func=cmd_answer)

    sp = sub.add_parser("score", help="accuracy against the key")
    sp.add_argument("quiz", help="quiz jsonl from answer")
    sp.set_defaults(func=cmd_score)

    args = parser.parse_args(argv)
    if getattr(args, "api_key", None) is None:
        import os

        args.api_key = os.environ.get("AVION_API_KEY")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
