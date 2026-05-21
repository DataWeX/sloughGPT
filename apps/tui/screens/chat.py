"""Chat screen for SloughGPT TUI — streaming conversation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from rich.text import Text

from apps.tui.components import CONSOLE, Color, header, divider
from apps.tui.session import TuiSession
from apps.tui.screen import Screen
from apps.tui.bindings import Binding


@dataclass
class ChatMessage:
    role: str
    content: str


class ChatScreen(Screen):
    name = "chat"
    bindings = [Binding(["/"], "clear", "clear")]

    def __init__(self):
        super().__init__()
        self.messages: List[ChatMessage] = []
        self.input_buffer: str = ""
        self.streaming: bool = False

    def _render_messages(self):
        for msg in self.messages:
            role_label = "You" if msg.role == "user" else "Assistant"
            role_color = Color.PRIMARY if msg.role == "user" else Color.SUCCESS
            CONSOLE.print()
            CONSOLE.print(Text(f"  {role_label}", style=f"bold {role_color}"))
            terminal_w = CONSOLE.width - 4
            text = msg.content
            while text:
                CONSOLE.print(Text(f"  {text[:terminal_w]}", style=Color.WHITE))
                text = text[terminal_w:]

    def render(self, session: TuiSession) -> str:
        self.render_header("Chat", f"{session.api_base_url}  ·  {len(self.messages)} messages")
        divider()
        CONSOLE.print()

        self._render_messages()

        if self.streaming:
            CONSOLE.print()
            CONSOLE.print(Text("  [streaming...]", style=Color.MUTED))
            return ""

        CONSOLE.print()
        divider()
        if self.input_buffer:
            CONSOLE.print(Text(f"  Message: {self.input_buffer}", style=Color.HIGHLIGHT))
        else:
            CONSOLE.print(Text("  Type your message and press Enter", style=Color.MUTED))
        CONSOLE.print()
        CONSOLE.print(Text(self.binding_manager.format_footer(self.bindings), style=Color.MUTED))

        return self._handle_input(session)

    def _handle_input(self, session: TuiSession) -> str:
        import readchar

        while True:
            key = readchar.readkey()

            for b in self.binding_manager.global_bindings:
                if key in b.keys:
                    return b.action

            if key == readchar.key.ENTER:
                text = self.input_buffer.strip()
                if not text:
                    return "chat"

                self.messages.append(ChatMessage(role="user", content=text))
                self.input_buffer = ""
                self.streaming = True

                self.render_header("Chat", f"{session.api_base_url}  ·  {len(self.messages)} messages")
                divider()
                self._render_messages()
                CONSOLE.print()
                CONSOLE.print(Text("  Assistant", style=f"bold {Color.SUCCESS}"), end="")

                collected = ""
                for token in session.api_client.stream_chat(
                    session.api_base_url,
                    [{"role": m.role, "content": m.content} for m in self.messages],
                    temperature=0.8,
                    max_tokens=256,
                ):
                    collected += token
                    CONSOLE.print(Text(token, style=Color.WHITE), end="")
                CONSOLE.print()

                self.streaming = False
                self.messages.append(ChatMessage(role="assistant", content=collected))
                return "chat"

            elif key == "/":
                self.messages.clear()
                self.input_buffer = ""
                return "chat"

            elif key == readchar.key.BACKSPACE:
                self.input_buffer = self.input_buffer[:-1]
                return "chat"

            elif len(key) == 1 and key.isprintable():
                self.input_buffer += key
                return "chat"

            return "chat"
