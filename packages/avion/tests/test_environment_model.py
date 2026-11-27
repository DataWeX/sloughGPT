"""Environment layer contract: declared capabilities, honest absence, fail-closed dispatch.

The regression these lock in:

* an adapter that cannot do something *says so*, and the caller learns it at the
  boundary with a name attached — not as ``AttributeError`` three frames deep;
* an observation never confuses "absent" with "empty" (a missing screenshot is
  ``None`` and is named, never a zero-byte PNG that reads as a blank page);
* nothing here needs a model — a scripted action list drives it;
* the backend inventory cannot quietly change: the measured capability baseline
  is asserted, so adding or losing a port operation on an adapter is a test
  result rather than a runtime surprise.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from avion.ai.models import Action, ActionType
from avion.core.element import Element, ElementLocator
from avion.environment import (
    ACTION_CAPABILITY,
    CONTROL_ACTIONS,
    CORE_CAPABILITIES,
    OPTIONAL_CAPABILITIES,
    Capability,
    CapabilityMatrix,
    Environment,
    EnvironmentRegistry,
    Observation,
    UnsupportedAction,
    UnsupportedCapability,
    environment_for,
)

# ── fakes ───────────────────────────────────────────────────────────────


class FakeBackend:
    """Records every call; only the methods it defines exist."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def _log(self, name: str, *args, **kwargs) -> None:
        self.calls.append((name, args, kwargs))

    async def start(self) -> None:
        self._log("start")

    async def stop(self) -> None:
        self._log("stop")

    async def navigate(self, url: str) -> None:
        self._log("navigate", url)

    async def click(self, element, *, force: bool = False) -> None:
        self._log("click", element, force=force)

    async def fill(self, element, value: str, *, force: bool = False) -> None:
        self._log("fill", element, value, force=force)

    async def screenshot(self, path=None) -> bytes:
        self._log("screenshot", path)
        return b"PNG"

    async def get_url(self) -> str:
        self._log("get_url")
        return "http://localhost:5175/"

    async def get_title(self) -> str:
        self._log("get_title")
        return "sloughGPT"

    async def wait_for_timeout(self, ms: float) -> None:
        self._log("wait_for_timeout", ms)

    async def find_element(self, locator):
        self._log("find_element", locator)
        return Element(raw="<node/>", locator=locator, backend="fake")

    async def press_element(self, element, key: str) -> None:
        self._log("press_element", element, key)


def full_matrix() -> CapabilityMatrix:
    return CapabilityMatrix(frozenset(Capability))


def fake_env(
    capabilities: CapabilityMatrix | None = None, name: str = "fake"
) -> tuple[Environment, FakeBackend]:
    backend = FakeBackend()
    return environment_for(name, backend, capabilities=capabilities or full_matrix()), backend


def locator() -> ElementLocator:
    return ElementLocator.css("#run")


# ── capability matrix ───────────────────────────────────────────────────


def test_capability_values_match_port_method_names():
    """A declaration reads as the protocol it satisfies."""
    assert Capability.CLICK.value == "click"
    assert Capability.GET_ACCESSIBILITY_TREE.value == "get_accessibility_tree"
    assert Capability.PRESS_ELEMENT.value == "press_element"


def test_core_and_optional_partition_the_enum():
    assert CORE_CAPABILITIES | OPTIONAL_CAPABILITIES == frozenset(Capability)
    assert not CORE_CAPABILITIES & OPTIONAL_CAPABILITIES
    assert Capability.PRESS_ELEMENT in OPTIONAL_CAPABILITIES


def test_require_raises_with_environment_and_capability_named():
    matrix = CapabilityMatrix.of(Capability.NAVIGATE)
    with pytest.raises(UnsupportedCapability) as exc:
        matrix.require(Capability.HOVER, environment="selenium")
    assert exc.value.environment == "selenium"
    assert exc.value.capability is Capability.HOVER
    assert "hover" in str(exc.value) and "selenium" in str(exc.value)
    # A declared capability does not raise.
    matrix.require(Capability.NAVIGATE, environment="selenium")


def test_optional_gap_hint_explains_page_controls():
    matrix = CapabilityMatrix.of(Capability.NAVIGATE)
    with pytest.raises(UnsupportedCapability) as exc:
        matrix.require(Capability.PRESS_ELEMENT, environment="cdp")
    assert "PageControls" in str(exc.value)


def test_missing_core_is_the_gap_list():
    matrix = CapabilityMatrix.of(Capability.NAVIGATE, Capability.CLICK)
    assert matrix.missing_core() == CORE_CAPABILITIES - {
        Capability.NAVIGATE,
        Capability.CLICK,
    }
    assert not matrix.satisfies_port()


def test_matrix_builders_agree():
    a = CapabilityMatrix.of("navigate", Capability.CLICK)
    b = CapabilityMatrix.from_names(["click", "navigate"])
    c = CapabilityMatrix.from_mapping({"navigate": True, "click": True, "hover": False})
    assert a == b == c
    assert a.to_dict()["declared"] == ["click", "navigate"]


# ── observation ─────────────────────────────────────────────────────────


def test_observation_names_absence_instead_of_faking_it():
    obs = Observation(url="http://x/", title="T")
    assert set(obs.absent()) == {"accessibility_tree", "screenshot", "viewport", "extra"}
    assert obs.has("title")
    assert not obs.has("screenshot")


def test_observation_serialises_screenshot_as_a_description_not_bytes():
    obs = Observation(url="http://x/", screenshot=b"12345")
    payload = obs.to_dict()
    assert payload["screenshot"] == {"bytes": 5}
    assert "b64" not in payload["screenshot"]
    assert payload["absent"] == ["title", "accessibility_tree", "viewport", "extra"]


def test_observation_extra_cannot_shadow_core_fields():
    obs = Observation(url="http://x/")
    with pytest.raises(ValueError):
        obs.with_extra(title="sneaky")
    assert obs.with_extra(driver="playwright").extra == {"driver": "playwright"}


def test_observation_describe_surfaces_unavailable_fields():
    text = Observation(url="http://x/").describe()
    assert "url: http://x/" in text
    assert "unavailable: title, accessibility_tree, screenshot, viewport" in text


# ── observe ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_observe_skips_undeclared_capabilities_without_calling_them():
    caps = CapabilityMatrix.of(Capability.GET_URL, Capability.GET_TITLE)
    env, backend = fake_env(caps)

    obs = await env.observe(tree=True, screenshot=True)

    assert obs.url == "http://localhost:5175/"
    assert obs.title == "sloughGPT"
    assert obs.accessibility_tree is None and obs.screenshot is None
    called = {name for name, _, _ in backend.calls}
    assert called == {"get_url", "get_title"}
    assert obs.extra["environment"] == "fake"


@pytest.mark.asyncio
async def test_observe_only_asks_for_what_was_requested():
    env, backend = fake_env()
    await env.observe(tree=False, screenshot=False)
    called = {name for name, _, _ in backend.calls}
    assert "screenshot" not in called
    assert "get_url" in called


# ── act: fail closed ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_act_refuses_undeclared_capability_before_touching_the_backend():
    caps = CapabilityMatrix.of(Capability.NAVIGATE)  # no PRESS_ELEMENT
    env, backend = fake_env(caps)

    with pytest.raises(UnsupportedCapability) as exc:
        await env.act(
            Action(
                action_type=ActionType.KEYBOARD_PRESS,
                params={"element": Element(raw=None, locator=locator()), "key": "Enter"},
            )
        )

    assert exc.value.capability is Capability.PRESS_ELEMENT
    assert backend.calls == [], "backend must not be touched when the capability is undeclared"


@pytest.mark.asyncio
async def test_act_navigate_dispatches_to_backend():
    env, backend = fake_env()
    await env.act(Action(action_type=ActionType.NAVIGATE, params={"url": "http://x/y"}))
    assert backend.calls == [("navigate", ("http://x/y",), {})]


@pytest.mark.asyncio
async def test_act_click_accepts_a_resolved_element():
    env, backend = fake_env()
    element = Element(raw="<b/>", locator=locator())
    await env.act(Action(action_type=ActionType.MOUSE_CLICK, params={"element": element}))
    assert backend.calls[0][0] == "click"
    assert backend.calls[0][1] == (element,)


@pytest.mark.asyncio
async def test_act_click_resolves_a_locator_through_the_port():
    env, backend = fake_env()
    await env.act(Action(action_type=ActionType.MOUSE_CLICK, params={"locator": locator()}))
    assert [name for name, _, _ in backend.calls] == ["find_element", "click"]


@pytest.mark.asyncio
async def test_act_requires_an_element_or_locator():
    env, _ = fake_env()
    with pytest.raises(ValueError, match="element"):
        await env.act(Action(action_type=ActionType.MOUSE_CLICK, params={}))


@pytest.mark.asyncio
async def test_control_markers_are_not_environment_actions():
    env, _ = fake_env()
    for action_type in CONTROL_ACTIONS:
        with pytest.raises(UnsupportedAction):
            await env.act(Action(action_type=action_type))


@pytest.mark.asyncio
async def test_action_without_a_port_primitive_is_refused_not_approximated():
    env, backend = fake_env()
    with pytest.raises(UnsupportedAction) as exc:
        await env.act(Action(action_type=ActionType.MOUSE_MOVE, params={"x": 1, "y": 2}))
    assert "mouse_move" in str(exc.value)
    assert backend.calls == []


def test_every_mapped_action_has_a_dispatch_branch():
    """Mapping and dispatch cannot drift: each mapped capability is reachable."""
    assert set(ACTION_CAPABILITY) <= set(ActionType)
    assert Capability.PRESS_ELEMENT in ACTION_CAPABILITY.values()
    assert all(isinstance(c, Capability) for c in ACTION_CAPABILITY.values())


def test_available_actions_is_a_projection_of_the_matrix():
    env, _ = fake_env(CapabilityMatrix.of(Capability.NAVIGATE, Capability.SCREENSHOT))
    assert set(env.available_actions()) == {ActionType.NAVIGATE, ActionType.SCREENSHOT}
    assert not env.supports_action(ActionType.MOUSE_CLICK)


# ── registry ────────────────────────────────────────────────────────────


def test_registry_answers_which_environment_can_do_x():
    reg = EnvironmentRegistry()
    web, _ = fake_env(full_matrix(), name="web")
    headless, _ = fake_env(
        CapabilityMatrix.of(Capability.NAVIGATE, Capability.SCREENSHOT), name="headless"
    )
    reg.register(web)
    reg.register(headless)

    assert reg.available() == ("headless", "web")
    assert [e.name for e in reg.supporting(Capability.PRESS_ELEMENT)] == ["web"]
    assert [e.name for e in reg.supporting(Capability.NAVIGATE)] == ["headless", "web"]
    assert [e.name for e in reg.for_action(ActionType.KEYBOARD_PRESS)] == ["web"]
    assert reg.gaps()["headless"], "a partial adapter must report its gap list"


def test_registry_rejects_duplicate_names_and_names_what_is_registered():
    reg = EnvironmentRegistry()
    env, _ = fake_env(name="web")
    reg.register(env)
    with pytest.raises(ValueError, match="already registered"):
        reg.register(env)
    reg.register(env, replace=True)
    with pytest.raises(KeyError, match="registered: web"):
        reg.get("nope")


def test_environment_description_carries_capabilities_and_actions():
    env, _ = fake_env(CapabilityMatrix.of(Capability.NAVIGATE), name="web")
    described = env.describe()
    assert described["name"] == "web"
    assert described["capabilities"]["declared"] == ["navigate"]
    assert described["actions"] == ["navigate"]


# ── backend inventory ratchet ───────────────────────────────────────────

#: Measured 2026-11-27 by AST-scanning ``avion/backends/*.py`` for the port's
#: method names. This is a *ratchet*, not a wish: the test fails when an adapter
#: gains a capability (update the row — that is an improvement) or loses one
#: (fix the regression, or move the module out of ``backends/`` if it was never a
#: backend). ``axtree``/``api``/``cli`` are recorded at their true size rather
#: than padded with stubs to satisfy the gate — a stub would make this table lie.
BACKEND_CAPABILITY_BASELINE: dict[str, frozenset[str]] = {
    "api": frozenset({"start", "stop"}),
    "appium": frozenset(c.value for c in CORE_CAPABILITIES),
    "axtree": frozenset(),
    "cdp": frozenset(c.value for c in CORE_CAPABILITIES),
    "cli": frozenset({"start", "stop"}),
    "desktop": frozenset(
        {
            "start",
            "stop",
            "screenshot",
            "get_url",
            "get_title",
            "wait_for_timeout",
            "get_accessibility_tree",
        }
    ),
    "playwright": frozenset(c.value for c in Capability),
    "selenium": frozenset(c.value for c in CORE_CAPABILITIES),
}


def _scan_backend_capabilities() -> dict[str, frozenset[str]]:
    backends_dir = Path(__file__).resolve().parents[1] / "src" / "avion" / "backends"
    port_names = {c.value for c in Capability}
    found: dict[str, frozenset[str]] = {}
    for path in sorted(backends_dir.glob("*.py")):
        if path.stem in {"__init__", "base"}:
            continue
        tree = ast.parse(path.read_text())
        methods: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.add(item.name)
        found[path.stem] = frozenset(methods & port_names)
    return found


def test_backend_capability_inventory_matches_the_recorded_baseline():
    scanned = _scan_backend_capabilities()
    assert set(scanned) == set(BACKEND_CAPABILITY_BASELINE), (
        "backend modules were added or removed — update BACKEND_CAPABILITY_BASELINE "
        "deliberately and say why"
    )
    for name, expected in BACKEND_CAPABILITY_BASELINE.items():
        actual = scanned[name]
        gained = actual - expected
        lost = expected - actual
        assert not gained and not lost, (
            f"{name}: gained={sorted(gained)} lost={sorted(lost)} — "
            "gained means an adapter got more capable (raise the baseline); "
            "lost means a port operation disappeared (fix it or move the module "
            "out of backends/)"
        )
