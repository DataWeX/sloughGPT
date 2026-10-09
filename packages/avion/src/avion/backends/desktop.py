"""Desktop backend — mouse/keyboard/screen control for desktop apps.

Requires: pip install pyautogui (and Pillow for screenshots).
All third-party imports are lazy. Coordinates are screen pixels.
"""

from __future__ import annotations

from typing import Any


def _pyautogui():
    try:
        import pyautogui

        return pyautogui
    except ImportError as e:
        raise ImportError(
            "pyautogui is not installed. Install it with: pip install pyautogui"
        ) from e


class DesktopBackend:
    """OS-level automation: no DOM, coordinates only.

    Implements the InteractionBackend protocol (mouse_*, keyboard_*)
    so Mouse/Keyboard controllers and InteractionChain drive it directly.
    """

    name = "desktop"

    def __init__(self):
        self._started = False

    async def start(self) -> None:
        _pyautogui()
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def mouse_move(self, x: float, y: float) -> None:
        import asyncio

        pg = _pyautogui()
        await asyncio.to_thread(pg.moveTo, x, y)

    async def mouse_down(self, button: str = "left") -> None:
        import asyncio

        pg = _pyautogui()
        await asyncio.to_thread(pg.mouseDown, button=button)

    async def mouse_up(self, button: str = "left") -> None:
        import asyncio

        pg = _pyautogui()
        await asyncio.to_thread(pg.mouseUp, button=button)

    async def mouse_click(self, x: float, y: float, button: str = "left") -> None:
        import asyncio

        pg = _pyautogui()
        await asyncio.to_thread(pg.click, x, y, button=button)

    async def mouse_double_click(self, x: float, y: float) -> None:
        import asyncio

        pg = _pyautogui()
        await asyncio.to_thread(pg.doubleClick, x, y)

    async def mouse_scroll(self, x: float, y: float, delta_x: int = 0, delta_y: int = 0) -> None:
        import asyncio

        pg = _pyautogui()
        await asyncio.to_thread(pg.moveTo, x, y)
        if delta_y:
            await asyncio.to_thread(pg.scroll, delta_y)
        if delta_x:
            await asyncio.to_thread(pg.hscroll, delta_x)

    async def keyboard_down(self, key: str) -> None:
        import asyncio

        pg = _pyautogui()
        await asyncio.to_thread(pg.keyDown, key)

    async def keyboard_up(self, key: str) -> None:
        import asyncio

        pg = _pyautogui()
        await asyncio.to_thread(pg.keyUp, key)

    async def keyboard_press(self, key: str, hold_ms: float = 0) -> None:
        import asyncio

        pg = _pyautogui()
        if "+" in key:  # "Control+a" style combos
            parts = [p.strip() for p in key.split("+")]
            await asyncio.to_thread(pg.hotkey, *parts)
        else:
            await asyncio.to_thread(pg.press, key)

    async def keyboard_type(self, text: str, delay_ms: float = 0) -> None:
        import asyncio

        pg = _pyautogui()
        await asyncio.to_thread(pg.typewrite, text, interval=delay_ms / 1000 if delay_ms else 0)

    async def get_viewport_size(self) -> tuple[int, int]:
        import asyncio

        pg = _pyautogui()
        size = await asyncio.to_thread(pg.size)
        return (size.width, size.height)

    async def get_screenshot_as_bytes(self) -> bytes:
        import asyncio
        import io

        pg = _pyautogui()
        img = await asyncio.to_thread(pg.screenshot)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    async def screenshot(self, path: str | None = None) -> bytes:
        data = await self.get_screenshot_as_bytes()
        if path:
            with open(path, "wb") as f:
                f.write(data)
        return data

    async def wait_for_timeout(self, ms: float) -> None:
        import asyncio

        await asyncio.sleep(ms / 1000)

    async def locate_text(self, text: str) -> list[dict[str, Any]]:
        """Locate text via OCR when pytesseract is available, else []."""
        try:
            import pytesseract  # type: ignore[import]
        except ImportError:
            return []
        import asyncio
        import io

        from PIL import Image  # type: ignore[import]

        data = await self.get_screenshot_as_bytes()
        img = Image.open(io.BytesIO(data))
        words = await asyncio.to_thread(
            pytesseract.image_to_data, img, output_type=pytesseract.Output.DICT
        )
        out = []
        for i, word in enumerate(words.get("text", [])):
            if word and text.lower() in word.lower():
                out.append(
                    {
                        "text": word,
                        "x": words["left"][i],
                        "y": words["top"][i],
                        "width": words["width"][i],
                        "height": words["height"][i],
                    }
                )
        return out
