"""Declarative registry of the UX_FLOWS user journeys.

Every flow links to its spec heading in docs/UX_FLOWS.md (Flow.spec) — the
registry-integrity tests in tests/test_ux_flows_library.py fail the build if a
spec heading is renamed or a flow loses its link.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from domain.testing.ux_flows.steps import (
    s_click,
    s_exists,
    s_goto,
    s_send,
    s_upload,
    s_wait,
)


@dataclass
class Flow:
    id: str
    label: str
    url: str
    spec: str
    build: Callable[[], list]


def _chat_flow(
    flow_id: str,
    label: str,
    url: str,
    prompt: str,
    spec: str,
    pill: str | None = None,
    marker: str | None = None,
) -> Flow:
    def build():
        steps = [s_goto(url, verify="Chat")]
        if marker:
            steps.append(s_wait(marker))
        if pill:
            steps.append(s_click(pill))
        steps.append(s_send(prompt))
        return steps

    return Flow(flow_id, label, url, spec, build)


def _lease_file() -> str:
    path = os.path.join(
        os.environ.get("SLO_JOURNEY_CACHE", os.path.expanduser("~/.cache/slog-journeys")),
        "lease.txt",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(
            "LEASE AGREEMENT\n"
            "Section 4.2: The move-out notice period is 30 days in writing.\n"
            "Rent is due on the 1st of each month.\n"
        )
    return path


def flow_home() -> Flow:
    def build():
        return [s_goto("/"), s_wait("Teach me"), s_wait("Chat")]

    return Flow(
        "0-home",
        "Home screen + sidebar",
        "/",
        "## Navigation & Layout (Plain English)",
        build,
    )


def flow_read() -> Flow:
    def build():
        return [
            s_goto("/chat?mode=read", verify="Chat"),
            s_upload(_lease_file()),
            s_send("What's the move-out notice period?"),
        ]

    return Flow("3-read", "Read My Files", "/chat?mode=read", "### 3. Read My Files", build)


def flow_talk() -> Flow:
    def build():
        # Firefox has no Web Speech API — the app must open voice mode and
        # degrade with a plain-language notice (never a crash/jargon).
        return [
            s_goto("/chat?mode=talk", verify="Chat"),
            s_exists('button[aria-label="Exit voice mode"]', "voice mode open"),
            s_wait("Speech recognition not supported", timeout=15),
        ]

    return Flow("7-talk", "Talk Out Loud", "/chat?mode=talk", "### 7. Talk Out Loud", build)


def flow_training() -> Flow:
    def build():
        return [
            s_goto("/training", verify="Teach"),
            s_wait("Pick your data"),
            s_wait("Next: Configure"),
            s_click("Next: Configure", optional=True),
        ]

    return Flow(
        "12-training",
        "Train My AI (3-click)",
        "/training",
        "### Training: 3-Click Flow",
        build,
    )


FLOWS: list[Flow] = [
    flow_home(),
    _chat_flow(
        "1-chat",
        "Chat That Remembers Me",
        "/chat",
        "What was that recipe we talked about yesterday?",
        "### 1. Chat That Remembers Me",
    ),
    _chat_flow(
        "2-write",
        "Writing Assistant",
        "/chat?mode=write",
        "Tell my landlord the sink is broken and ask when he can fix it",
        "### 2. Writing Assistant",
        marker="Tone",
        pill="Friendly",
    ),
    flow_read(),
    _chat_flow(
        "4-brainstorm",
        "Brainstorm With Me",
        "/chat?mode=brainstorm",
        "Gift ideas for my dad's 60th birthday, he loves fishing and cooking",
        "### 4. Brainstorm With Me",
        marker="Topic",
        pill="Gift Ideas",
    ),
    _chat_flow(
        "5-rewrite",
        "Rewrite & Polish",
        "/chat?mode=rewrite",
        "Rewrite: teh quick bown fox dont jump over teh lazy dogg",
        "### 5. Rewrite & Polish",
        marker="Action",
        pill="Fix Grammar",
    ),
    _chat_flow(
        "6-create",
        "Create Images",
        "/chat?mode=create",
        "Create an image: a cozy cabin in the mountains at sunset",
        "### 6. Create Images",
        marker="Style",
        pill="Realistic",
    ),
    flow_talk(),
    _chat_flow(
        "8-translate",
        "Translate",
        "/chat?mode=translate",
        "How much does this cost?",
        "### 8. Translate",
        marker="To",
        pill="EN→ES",
    ),
    _chat_flow(
        "9-decide",
        "Help Me Decide",
        "/chat?mode=decide",
        "Should I take the job in New York or stay in my current role?",
        "### 9. Help Me Decide",
        marker="Output",
        pill="Pros & Cons",
    ),
    _chat_flow(
        "10-explain",
        "Explain Things Simply",
        "/chat?mode=explain",
        "How does the internet work?",
        "### 10. Explain Things Simply",
        marker="Level",
        pill="Simple",
    ),
    _chat_flow(
        "11-wellness",
        "Make Me Well (Wellness)",
        "/chat?mode=wellness",
        "I want a short sleep story about the ocean",
        "### 11. Make Me Well (Wellness)",
        marker="Type",
        pill="Sleep Story",
    ),
    flow_training(),
]

_BY_ID = {f.id: f for f in FLOWS}


def get_flow(flow_id: str) -> Flow:
    return _BY_ID[flow_id]


def list_flows() -> list[Flow]:
    return list(FLOWS)
