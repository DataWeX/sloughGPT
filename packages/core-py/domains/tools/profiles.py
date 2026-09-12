"""Tool Profiles — declarative definitions for the seven everyday tools.

Each tool describes its UI surface (params, options) and how to render a
prompt for the inference engine.  Definitions live here so both the API
router and the frontend (via ``GET /tools``) share one source of truth.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ToolOption",
    "ToolParam",
    "ToolProfile",
    "TOOL_PROFILES",
    "get_tool_profile",
]


@dataclass(frozen=True)
class ToolOption:
    """A selectable option for a tool (tone, type, difficulty, ...)."""

    id: str
    label: str
    description: str = ""


@dataclass(frozen=True)
class ToolParam:
    """A free-text parameter a tool accepts."""

    id: str
    label: str
    placeholder: str
    multiline: bool = False
    optional: bool = False


@dataclass(frozen=True)
class ToolProfile:
    """Declarative definition of a tool."""

    id: str
    name: str
    description: str
    icon: str
    params: list[ToolParam] = field(default_factory=list)
    options: dict[str, list[ToolOption]] = field(default_factory=dict)
    default_options: dict[str, str] = field(default_factory=dict)
    system_prompt: str = ""
    max_tokens: int = 700
    render_fn: Callable[[dict[str, Any]], str] | None = field(default=None, repr=False)

    def render_prompt(self, payload: dict[str, Any]) -> str:
        """Render the user prompt for this tool from a payload."""
        if self.render_fn is None:
            raise NotImplementedError
        return self.render_fn(payload)


TOOL_PROFILES: dict[str, ToolProfile] = {}


def _register(profile: ToolProfile) -> ToolProfile:
    TOOL_PROFILES[profile.id] = profile
    return profile


# ── Writing Assistant ────────────────────────────────────────────────
def _writing() -> ToolProfile:
    def render(payload: dict[str, Any]) -> str:
        action = payload.get("action", "write")
        tone = payload.get("tone", "professional")
        kind = payload.get("type", "email")
        text = payload.get("text", "").strip()

        tone_map = {
            "friendly": "friendly",
            "professional": "professional",
            "funny": "funny",
            "short": "short and concise",
            "detailed": "detailed and thorough",
        }
        tone_label = tone_map.get(tone, tone)
        type_label = kind.replace("-", " ")

        if action == "shorter":
            return (
                f"Make this {type_label} shorter while keeping the {tone_label} tone:\n\n{text}\n\n"
                f"Just output the shorter {type_label}."
            )
        if action == "funnier":
            return (
                f"Make this {type_label} funnier while keeping it as a {type_label}:\n\n{text}\n\n"
                f"Just output the funnier version."
            )
        if action == "rewrite":
            return (
                f"Rewrite this {type_label} in a different way keeping the {tone_label} tone:\n\n{text}\n\n"
                f"Just output the rewritten {type_label}."
            )
        return (
            f"Write a {tone_label} {type_label} based on this: \"{text}\"\n\n"
            f"Just output the {type_label} directly, no explanation."
        )

    return ToolProfile(
        id="writing",
        name="Writing Assistant",
        description="Write an email, social post, story, poem, letter or note in your tone of choice.",
        icon="document",
        params=[
            ToolParam("text", "What do you want to write?", "Tell me what you want to write about...", multiline=True),
        ],
        options={
            "tone": [
                ToolOption("friendly", "Friendly"),
                ToolOption("professional", "Professional"),
                ToolOption("funny", "Funny"),
                ToolOption("short", "Short"),
                ToolOption("detailed", "Detailed"),
            ],
            "type": [
                ToolOption("email", "Email"),
                ToolOption("social", "Social Post"),
                ToolOption("story", "Story"),
                ToolOption("poem", "Poem"),
                ToolOption("letter", "Letter"),
                ToolOption("note", "Note"),
            ],
        },
        default_options={"tone": "professional", "type": "email"},
        system_prompt=(
            "You are a helpful writing assistant. Produce clean, ready-to-use text "
            "matching the requested tone and format. No filler, no explanation."
        ),
        max_tokens=700,
        render_fn=render,
    )


_register(_writing())


# ── Translate ────────────────────────────────────────────────────────

def _translate() -> ToolProfile:
    def render(payload: dict[str, Any]) -> str:
        text = payload.get("text", "").strip()
        lang = payload.get("target_lang", "Spanish")
        return (
            f"Translate the following text to {lang}. Output ONLY the translation, "
            f"no explanation:\n\n{text}"
        )

    return ToolProfile(
        id="translate",
        name="Translate",
        description="Translate text between languages with auto-detection.",
        icon="chat",
        params=[
            ToolParam("text", "Source text", "Type or paste text to translate...", multiline=True),
            ToolParam("target_lang", "Target language", "Spanish", optional=True),
        ],
        options={
            "target_lang": [
                ToolOption(lang, lang) for lang in (
                    "Spanish", "French", "German", "Italian", "Portuguese",
                    "Chinese", "Japanese", "Korean", "Arabic", "Hindi",
                    "Russian", "Dutch", "Swedish", "Polish", "Turkish",
                )
            ],
        },
        default_options={"target_lang": "Spanish"},
        system_prompt=(
            "You are a professional translator. Output ONLY the translation, "
            "no commentary, no quotes."
        ),
        max_tokens=700,
        render_fn=render,
    )


# ── Rewrite & Polish ─────────────────────────────────────────────────

def _rewrite() -> ToolProfile:
    def render(payload: dict[str, Any]) -> str:
        action = payload.get("action", "grammar")
        text = payload.get("text", "").strip()

        prompts = {
            "grammar": "Fix spelling and grammar in this text. Output ONLY the corrected text:",
            "shorter": "Make this text more concise. Output ONLY the shorter version:",
            "friendlier": "Rewrite this in a friendly, casual tone. Output ONLY the rewritten text:",
            "professional": "Rewrite this in a professional tone. Output ONLY the rewritten text:",
            "sound-like-me": "Rewrite this to sound more natural and personal. Output ONLY the rewritten text:",
        }
        return f"{prompts.get(action, prompts['grammar'])}\n\n{text}"

    return ToolProfile(
        id="rewrite",
        name="Rewrite & Polish",
        description="Fix grammar, shorten, or change the tone of text you paste.",
        icon="sparkle",
        params=[
            ToolParam("text", "Original text", "Paste what you wrote...", multiline=True),
        ],
        options={
            "action": [
                ToolOption("grammar", "Fix Grammar", "Fix spelling and grammar only"),
                ToolOption("shorter", "Make Shorter", "Make it concise"),
                ToolOption("friendlier", "Make Friendlier", "Casual and warm"),
                ToolOption("professional", "Make Professional", "Formal and polished"),
                ToolOption("sound-like-me", "Sound Like Me", "Natural and personal"),
            ],
        },
        default_options={"action": "grammar"},
        system_prompt=(
            "You are a careful editor. Preserve meaning while improving the text "
            "according to the requested action. Output ONLY the rewritten text."
        ),
        max_tokens=700,
        render_fn=render,
    )


# ── Brainstorm ───────────────────────────────────────────────────────

def _brainstorm() -> ToolProfile:
    def render(payload: dict[str, Any]) -> str:
        messages = payload.get("history", [])
        lines = [
            f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
            for m in messages
        ]
        return "\n".join(lines) + "\nAssistant:"

    return ToolProfile(
        id="brainstorm",
        name="Brainstorm",
        description="Creative partner — ideas in bullet points, not essays.",
        icon="brain",
        params=[
            ToolParam("history", "Conversation history", "What's on your mind?", multiline=True),
        ],
        default_options={},
        system_prompt=(
            "You are a creative brainstorming partner. Respond with ideas in bullet "
            "points, not essays. Be concise and creative."
        ),
        max_tokens=700,
        render_fn=render,
    )


# ── Help Me Decide ───────────────────────────────────────────────────

def _decide() -> ToolProfile:
    def render(payload: dict[str, Any]) -> str:
        question = payload.get("question", "").strip()
        option_a = payload.get("option_a", "").strip()
        option_b = payload.get("option_b", "").strip()
        notes_a = payload.get("notes_a", "").strip()
        notes_b = payload.get("notes_b", "").strip()

        prompt = [
            "Help me decide. Create a pro/con table for each option, then a recommendation.",
            "",
            f"Question: {question}",
            f"Option A: {option_a}",
        ]
        if notes_a:
            prompt.append(f"Notes: {notes_a}")
        prompt.append("")
        prompt.append(f"Option B: {option_b}")
        if notes_b:
            prompt.append(f"Notes: {notes_b}")
        prompt += [
            "",
            "Format:",
            f"## Option A: {option_a}",
            "### Pros",
            "- ...",
            "### Cons",
            "- ...",
            "",
            f"## Option B: {option_b}",
            "### Pros",
            "- ...",
            "### Cons",
            "- ...",
            "",
            "## Recommendation",
            "...",
        ]
        return "\n".join(prompt)

    return ToolProfile(
        id="decide",
        name="Help Me Decide",
        description="Compare two options with pros, cons and a clear recommendation.",
        icon="chart",
        params=[
            ToolParam("question", "What are you deciding between?", "e.g. Should I take the job in New York or stay?"),
            ToolParam("option_a", "Option A", "First option"),
            ToolParam("option_b", "Option B", "Second option"),
            ToolParam("notes_a", "Notes for A", "Notes (optional)", optional=True),
            ToolParam("notes_b", "Notes for B", "Notes (optional)", optional=True),
        ],
        default_options={},
        system_prompt=(
            "You are an impartial decision helper. Build clear pro/con tables for "
            "each option, then give a concise recommendation."
        ),
        max_tokens=900,
        render_fn=render,
    )


# ── Explain Simply ───────────────────────────────────────────────────

def _explain() -> ToolProfile:
    def render(payload: dict[str, Any]) -> str:
        topic = payload.get("topic", "").strip()
        difficulty = payload.get("difficulty", "normal")

        difficulty_map = {
            "simple": (
                "Explain this like I'm 5 years old. Use simple words, analogies and "
                "examples a child would understand."
            ),
            "normal": "Explain this clearly and simply. Use everyday language and relatable examples.",
            "detailed": (
                "Explain this in depth. Include technical details, examples and "
                "real-world applications."
            ),
        }
        return f"{difficulty_map.get(difficulty, difficulty_map['normal'])}\n\nTopic: {topic}"

    return ToolProfile(
        id="explain",
        name="Explain Things Simply",
        description="Understand anything at the level you choose.",
        icon="search",
        params=[
            ToolParam("topic", "What do you want explained?", "e.g. How does the internet work?", multiline=True),
        ],
        options={
            "difficulty": [
                ToolOption("simple", "Simple", "Like explaining to a child"),
                ToolOption("normal", "Normal", "Clear and straightforward"),
                ToolOption("detailed", "Detailed", "In-depth with examples"),
            ],
        },
        default_options={"difficulty": "normal"},
        system_prompt=(
            "You are a gifted teacher. Explain the topic at the requested level using "
            "plain language, analogies and examples."
        ),
        max_tokens=900,
        render_fn=render,
    )


# ── Wellness ─────────────────────────────────────────────────────────

def _wellness() -> ToolProfile:
    def render(payload: dict[str, Any]) -> str:
        kind = payload.get("kind", "sleep")
        prefs = payload.get("preferences", "").strip()

        prompts = {
            "sleep": (
                "Tell me a gentle, calming sleep story. "
                + (f"The user wants: {prefs}." if prefs else "Make it about a peaceful natural setting.")
                + " The story should be soothing, with a slow pace and calming imagery. "
                "End with the words fading into silence."
            ),
            "meditate": (
                "Guide me through a short meditation. "
                + (f"Focus on: {prefs}." if prefs else "Focus on breathing and presence.")
                + " Speak slowly, calmly. Include pauses marked with \"...\". Help me feel grounded."
            ),
            "journal": (
                "Give me a thoughtful journal prompt to reflect on. "
                + (f"Theme: {prefs}." if prefs else "Make it about gratitude and growth.")
                + " The prompt should inspire deep reflection."
            ),
            "breathe": (
                "Guide me through a breathing exercise. "
                + (f"Style: {prefs}." if prefs else "Use 4-7-8 breathing.")
                + " Give me clear instructions with counts. Make it calming and rhythmic."
            ),
            "affirm": (
                "Give me 3 positive affirmations for today. "
                + (f"Theme: {prefs}." if prefs else "Make them empowering and kind.")
                + " Each should be short, present-tense and meaningful."
            ),
        }
        return prompts.get(kind, prompts["sleep"])

    return ToolProfile(
        id="wellness",
        name="Make Me Well",
        description="Sleep stories, meditations, journal prompts, breathing exercises and affirmations.",
        icon="sparkle",
        params=[
            ToolParam("preferences", "Any preferences?", "Optional — personalise the session", optional=True),
        ],
        options={
            "kind": [
                ToolOption("sleep", "Sleep Story", "A gentle story to help you drift off"),
                ToolOption("meditate", "Meditation", "Guided meditation for peace"),
                ToolOption("journal", "Journal Prompt", "Reflect on your day"),
                ToolOption("breathe", "Breathing Exercise", "Calm your mind with breath"),
                ToolOption("affirm", "Positive Affirmation", "Uplifting words for your day"),
            ],
        },
        default_options={"kind": "sleep"},
        system_prompt=(
            "You are a warm, calming guide. Respond with gentle, soothing, present-tense "
            "guidance. No medical claims, just feeling better and relaxing."
        ),
        max_tokens=700,
        render_fn=render,
    )


def get_tool_profile(tool_id: str) -> ToolProfile | None:
    """Look up a tool profile by id."""
    return TOOL_PROFILES.get(tool_id)

# ── Registry — evaluate every profile at import time ─────────────────
_register(_translate())
_register(_rewrite())
_register(_brainstorm())
_register(_decide())
_register(_explain())
_register(_wellness())
