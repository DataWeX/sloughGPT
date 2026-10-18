"""navigator.webdriver tell — the cheapest bot signal, removed at the source.

Every browser driven through the automation protocol reports
``navigator.webdriver === true`` — readable by any page at document
creation, before a single action happens. ``CdpBackend.start()``
injects a ``Page.addScriptToEvaluateOnNewDocument`` script that shadows
the prototype getter, so page scripts read ``undefined`` instead.

Real browser, no mocks. Degrades with a stated skip when no Chromium
is available.
"""

from __future__ import annotations

import pytest
from avion.launch import Profile, find_chromium, launch_chromium

_LAUNCH_ARGS = ("--no-sandbox", "--disable-gpu")


def _chromium_or_skip() -> str:
    try:
        return find_chromium()
    except RuntimeError:
        pytest.skip("no Chromium available — set AVION_CHROMIUM or install chromium")


class TestWebDriverTell:
    async def test_navigator_webdriver_is_falsy_on_a_loaded_page(self, tmp_path):
        binary = _chromium_or_skip()
        launched = launch_chromium(
            Profile("wdtell", root=tmp_path / "profiles"),
            headless=True,
            binary=binary,
            extra_args=_LAUNCH_ARGS,
            timeout=30.0,
        )
        try:
            backend = launched.backend
            await backend.start()
            try:
                await backend.navigate("data:text/html,<title>tell</title><p>probe page</p>")
                value = await backend.evaluate("navigator.webdriver")
                assert value in (None, False), (
                    f"navigator.webdriver leaked to page scripts: {value!r}"
                )
            finally:
                await backend.stop()
        finally:
            launched.stop()
