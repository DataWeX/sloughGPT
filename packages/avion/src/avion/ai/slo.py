"""SloModel — drive Avion's agent loop with our own inference engine.

The engine is text-only (no vision), so the model grounds on the
accessibility tree + recent history + tool cards and replies with one
JSON action. Screenshots never leave the machine.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from avion.ai.models import (
    ACTION_CARDS,
    Action,
    ActionType,
    Step,
    tool_cards_text,
    validate_action,
)

_HISTORY_STEPS = 5
_TREE_CHARS = 2000


def build_prompt(
    task: str,
    accessibility_tree: dict[str, Any] | None = None,
    history: list[Step] | None = None,
    available_actions: list[ActionType] | None = None,
    tools=None,
    max_chars: int | None = None,
) -> str:
    """Build the text prompt asking the model for one JSON action.

    max_chars budgets the variable sections (tree + history) for
    small-context samplers; fixed instructions always survive.
    """
    tree_budget, history_budget = _split_budget(max_chars)
    lines = [
        "You operate a web browser. Given the task and page state,",
        "reply with exactly one JSON action, no other text.",
        "",
        f"Task: {task}",
        "",
        "Page (accessibility tree):",
        _tree_text(accessibility_tree, tree_budget),
        "",
        "Recent steps:",
        _history_text(history, history_budget),
        "",
        "Available actions:",
        _cards_text(available_actions),
        "",
    ]
    if tools is not None and len(tools):
        lines += [
            "Outside tools (use action_type tool_call):",
            tools.describe(),
            "",
        ]
    lines.append('Reply format: {"action_type": "<name>", "params": {...}, "reasoning": "<why>"}')
    return "\n".join(lines)


def _split_budget(max_chars: int | None) -> tuple[int | None, int | None]:
    if max_chars is None:
        return (None, None)
    # History carries the loop state; tree is verbose. Favor history.
    history_budget = min(max_chars // 2, 800)
    return (max(0, max_chars - history_budget), history_budget)


def _tree_text(tree: dict[str, Any] | None, budget: int | None = None) -> str:
    if not tree:
        return "(no accessibility tree)"
    text = json.dumps(tree, default=str)
    limit = budget if budget is not None else _TREE_CHARS
    if len(text) > limit:
        text = text[:limit] + "...[trimmed]"
    return text


def _history_text(history: list[Step] | None, budget: int | None = None) -> str:
    steps = history or []
    if not steps:
        return "(first step)"
    lines = []
    for s in steps[-_HISTORY_STEPS:]:
        if s.action is None:
            continue
        lines.append(f"- {s.action.action_type.value} {s.action.params} => {s.observation[:120]}")
    text = "\n".join(lines) or "(no actions yet)"
    if budget is not None and len(text) > budget:
        text = "...[trimmed]\n" + text[-budget:]
    return text


def _cards_text(available: list[ActionType] | None) -> str:
    if available:
        wanted = set(available) | {ActionType.DONE, ActionType.FAIL}
        lines = []
        for at in ActionType:
            if at in wanted and at in ACTION_CARDS:
                card = ACTION_CARDS[at]
                params = ", ".join(card.params) or "none"
                lines.append(f"- {at.value}: {card.description} Params({params})")
        return "\n".join(lines)
    return tool_cards_text()


def parse_action(text: str) -> Action:
    """Parse model output into an Action.

    Garbage never kills a run: unparseable output becomes a short WAIT
    (the no-progress guard stops persistent nonsense), unknown types
    become FAIL, invalid params become WAIT with the validation error.
    """
    data = _extract_json(text)
    if data is None:
        return Action(
            action_type=ActionType.WAIT,
            params={"ms": 500},
            reasoning=f"unparseable model output: {text[:120]}",
            confidence=0.2,
        )
    raw_type = str(data.get("action_type", ""))
    try:
        action_type = ActionType(raw_type)
    except ValueError:
        return Action(
            action_type=ActionType.FAIL,
            reasoning=f"unknown action_type {raw_type!r}",
            confidence=0.2,
        )
    action = Action(
        action_type=action_type,
        params=data.get("params", {}) or {},
        reasoning=str(data.get("reasoning", "")),
        confidence=float(data.get("confidence", 0.8)),
    )
    error = validate_action(action)
    if error:
        return Action(
            action_type=ActionType.WAIT,
            params={"ms": 500},
            reasoning=f"invalid action ({error}), retrying",
            confidence=0.2,
        )
    return action


def strip_echo(text: str, prompt: str) -> str:
    """Drop echoed prompt/template, keep the model's completion.

    Many samplers return prompt + completion. Parsing the whole thing
    lets prompt text (option numbers, the JSON format example) become
    phantom actions — always parse the completion only.
    """
    if prompt and prompt in text:
        return text.split(prompt)[-1]
    return text


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    # Balanced-brace candidates left to right; prefer one naming a known
    # action (prompt echoes carry the {"action_type": "<name>"} example).
    # Regex can't do nesting: non-greedy stops at the first inner "}",
    # greedy fuses separate blobs into one unparseable span.
    fallback = None
    for candidate in _json_candidates(text):
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if fallback is None:
            fallback = data
        try:
            ActionType(str(data.get("action_type", "")))
        except ValueError:
            continue
        return data
    return fallback


def _json_candidates(text: str) -> list[str]:
    """Substrings with balanced braces, in order of appearance."""
    out = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start : i + 1])
                    start = -1
    return out


class SloModel:
    """VisionModel backed by any async text generator.

    Usage::

        model = SloModel(generate=lambda prompt: engine.chat(prompt))
        agent = Agent(model=model)
    """

    def __init__(
        self,
        generate: Callable[[str], Awaitable[str]],
        model_name: str = "slo",
        max_tokens: int = 512,
        temperature: float = 0.3,
        tools=None,
        available_actions: list[ActionType] | None = None,
        max_prompt_chars: int | None = None,
    ):
        self._generate = generate
        self._name = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._tools = tools
        self._available_actions = available_actions
        self._max_prompt_chars = max_prompt_chars
        self.last_prompt = ""

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def supports_vision(self) -> bool:
        return False

    async def predict_action(
        self,
        task: str,
        screenshot_b64: str,
        accessibility_tree: dict[str, Any] | None = None,
        history: list[Step] | None = None,
        available_actions: list[ActionType] | None = None,
    ) -> Action:
        prompt = build_prompt(
            task,
            accessibility_tree,
            history,
            available_actions or self._available_actions,
            tools=self._tools,
            max_chars=self._max_prompt_chars,
        )
        self.last_prompt = prompt
        try:
            text = await self._generate(prompt)
        except Exception as e:
            return Action(
                action_type=ActionType.FAIL,
                reasoning=f"inference error: {e}",
                confidence=0.0,
            )
        return parse_action(strip_echo(text, prompt))

    @classmethod
    def from_inference_client(cls, client, **kwargs) -> SloModel:
        """Adapt an InferenceClient (duck-typed .chat) to a model."""
        name = getattr(client, "model_id", "slo-engine")

        async def generate(prompt: str) -> str:
            return await client.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.3),
            )

        return cls(generate=generate, model_name=str(name), **kwargs)

    @classmethod
    def from_soul(cls, soul_path: str, **kwargs) -> SloModel:
        """Drive the model from a local .soul checkpoint file.

        Imports the provider as an encapsulated package
        (``domains.inference.slonet_provider``) — no folder paths,
        no sys.path tricks. Sampling runs locally via numpy.
        """
        try:
            from domains.inference.slonet_provider import SloNetChatProvider
        except ImportError as e:
            raise ImportError(
                "SloNet provider package not installed; "
                "from_soul needs the sloughGPT core packages."
            ) from e
        provider = SloNetChatProvider.from_soul(soul_path, model_id=soul_path)
        return cls.from_inference_client(provider, **kwargs)

    @classmethod
    def from_http(
        cls, base_url: str, endpoint: str = "/infer", timeout: float = 120.0, **kwargs
    ) -> SloModel:
        """Drive the model through a running API server (no browser needed)."""
        from avion.backends.api import ApiBackend

        backend = ApiBackend(base_url=base_url, timeout=timeout)
        name = f"slo-http:{base_url}"

        async def generate(prompt: str) -> str:
            await backend.start()
            try:
                resp = await backend.post(endpoint, {"prompt": prompt})
            finally:
                await backend.stop()
            if not resp.ok:
                raise RuntimeError(f"HTTP {resp.status}: {resp.error}")
            body = resp.body
            if isinstance(body, dict) and "text" in body:
                return str(body["text"])
            return str(body)

        return cls(generate=generate, model_name=name, **kwargs)
