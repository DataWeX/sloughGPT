"""ToolUseProcessor — __slots__, module-level re, base64."""

from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("slo.models.provider")


@dataclass(slots=True)
class ToolDef:
    name: str
    provider_name: str
    description: str = ""


_BUILTIN_TOOLS = [
    ToolDef(
        name="describe_image",
        provider_name="multimodal",
        description=(
            "To see an image, output: [[TOOL: describe_image]] <base64_image_data>\n"
            "I will describe it and you can continue."
        ),
    ),
]


class ToolUseProcessor:
    __slots__ = ("_tools",)
    TOOL_RE = re.compile(r"\[\[TOOL:\s*(\w+)\s*\]\]\s*(\S+)")
    _PLACEHOLDER_ARG_RE = re.compile(r"^<[^>]*>$")

    def __init__(self, tools: list[ToolDef] | None = None):
        self._tools = tools or _BUILTIN_TOOLS

    @staticmethod
    def _has_image(messages: list) -> bool:
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return True
            elif isinstance(content, str) and "data:image/" in content:
                return True
        return False

    async def process(self, messages: list) -> list:
        if not self._has_image(messages):
            return messages
        tool_descriptions = "\n".join(f"- {t.description}" for t in self._tools)
        tool_prompt = f"You have access to these tools:\n{tool_descriptions}\n"
        has_system = any(m.get("role") == "system" for m in messages)
        if has_system:
            for m in messages:
                if m.get("role") == "system":
                    m["content"] = f"{m['content']}\n\n{tool_prompt}"
                    break
        else:
            messages.insert(0, {"role": "system", "content": tool_prompt})
        return messages

    def match_tool(self, text: str) -> tuple[str, str, str] | None:
        m = self.TOOL_RE.search(text)
        if m:
            tool_name, tool_arg = m.group(1), m.group(2)
            if self._PLACEHOLDER_ARG_RE.match(tool_arg):
                return None
            for tool in self._tools:
                if tool.name == tool_name:
                    return (tool_name, tool_arg, m.group(0))
        return None
