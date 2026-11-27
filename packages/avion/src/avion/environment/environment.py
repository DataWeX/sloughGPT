"""An environment is a backend plus what it honestly declares it can do.

This is the object a driver above the port holds. It is deliberately dumb: it
holds no model, makes no decisions, and never probes. It answers two questions —

* :meth:`Environment.observe` — what does it look like right now?
* :meth:`Environment.act` — do this one thing.

— and refuses anything it did not declare, *before* touching the backend. That
refusal is the point: the failure lands at the boundary with a name and a
capability in the message, instead of surfacing three frames deep as
``AttributeError: 'SeleniumBackend' object has no attribute 'press_element'``
long after the caller believed the action had happened.

Nothing here needs a language model, vision, or network. A scripted action list
(``EchoModel``), a rule table (``RuleBasedModel``) or a human at a REPL can drive
it; a model is one *optional* node on top, not a precondition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from avion.ai.models import Action, ActionType
from avion.core.element import Backend, Element, ElementLocator

from .capability import Capability, CapabilityMatrix, UnsupportedCapability
from .observation import Observation


class UnsupportedAction(RuntimeError):
    """Raised when an ``ActionType`` has no environment primitive behind it.

    Distinct from :class:`UnsupportedCapability`: the *backend* may well be
    capable, but this action has no port operation that expresses it (a
    coordinate ``mouse_move``, a ``done`` control marker). Kept separate so a
    caller can tell "wrong environment" from "wrong action for any environment".
    """

    def __init__(self, action_type: ActionType, *, hint: str = "") -> None:
        self.action_type = action_type
        self.hint = hint
        message = f"no environment primitive for action {action_type.value!r}"
        if hint:
            message = f"{message} — {hint}"
        super().__init__(message)


#: ``ActionType`` -> the single port capability that executes it.
#:
#: Only one-to-one mappings belong here. An action that needs a capability the
#: port does not have is *absent*, not approximated: silently turning a
#: coordinate move into an element hover would make the recording claim an
#: interaction that never happened.
ACTION_CAPABILITY: Mapping[ActionType, Capability] = {
    ActionType.NAVIGATE: Capability.NAVIGATE,
    ActionType.SCREENSHOT: Capability.SCREENSHOT,
    ActionType.WAIT: Capability.WAIT_FOR_TIMEOUT,
    ActionType.MOUSE_CLICK: Capability.CLICK,
    ActionType.KEYBOARD_TYPE: Capability.FILL,
    ActionType.KEYBOARD_PRESS: Capability.PRESS_ELEMENT,
}

#: Actions that are control flow rather than environment interaction.
CONTROL_ACTIONS: frozenset[ActionType] = frozenset(
    {ActionType.DONE, ActionType.FAIL, ActionType.TOOL_CALL}
)


@dataclass(frozen=True)
class Environment:
    """A named backend with a declared capability matrix."""

    name: str
    backend: Backend
    capabilities: CapabilityMatrix

    # -- introspection ------------------------------------------------------
    def supports(self, capability: Capability) -> bool:
        return self.capabilities.supports(capability)

    def require(self, *capabilities: Capability) -> None:
        """Fail closed, naming *this* environment."""
        self.capabilities.require(*capabilities, environment=self.name)

    def supports_action(self, action_type: ActionType) -> bool:
        capability = ACTION_CAPABILITY.get(action_type)
        return capability is not None and self.supports(capability)

    def available_actions(self) -> tuple[ActionType, ...]:
        """The action vocabulary this environment can actually execute.

        This is what a model should be told, instead of the full enum — the
        difference between "the tool exists" and "this environment has it".
        """
        return tuple(a for a in ACTION_CAPABILITY if self.supports_action(a))

    def describe(self) -> dict[str, Any]:
        """Environment description for a report, a registry listing, or a prompt."""
        return {
            "name": self.name,
            "backend": getattr(self.backend, "name", type(self.backend).__name__),
            "capabilities": self.capabilities.to_dict(),
            "actions": [a.value for a in self.available_actions()],
        }

    # -- observation --------------------------------------------------------
    async def observe(
        self,
        *,
        tree: bool = True,
        screenshot: bool = False,
        viewport: bool = False,
    ) -> Observation:
        """Snapshot the environment, skipping (and naming) what it cannot give.

        Requesting something undeclared is not an error here — an observation is
        a report of the world, and "this backend has no accessibility tree" is
        part of that report. Acting is different: see :meth:`act`.
        """
        url = ""
        if self.supports(Capability.GET_URL):
            url = await self.backend.get_url()

        title: str | None = None
        if self.supports(Capability.GET_TITLE):
            title = await self.backend.get_title()

        accessibility_tree: Mapping[str, Any] | None = None
        if tree and self.supports(Capability.GET_ACCESSIBILITY_TREE):
            accessibility_tree = await self.backend.get_accessibility_tree()

        shot: bytes | None = None
        if screenshot and self.supports(Capability.SCREENSHOT):
            shot = await self.backend.screenshot()

        size: tuple[int, int] | None = None
        if viewport:
            # The port has no viewport *getter* — only the optional setter
            # (``PageControls.set_viewport_size``). So this is an adapter extra,
            # read if exposed and absent otherwise; gating it on the setter
            # capability would claim knowledge the port never promises.
            raw = getattr(self.backend, "viewport_size", None)
            if isinstance(raw, tuple) and len(raw) == 2:
                size = (int(raw[0]), int(raw[1]))

        return Observation(
            url=url,
            title=title,
            accessibility_tree=accessibility_tree,
            screenshot=shot,
            viewport=size,
            extra={"environment": self.name},
        )

    # -- action -------------------------------------------------------------
    async def act(self, action: Action) -> None:
        """Execute one action, or refuse it before the backend is touched."""
        if action.action_type in CONTROL_ACTIONS:
            raise UnsupportedAction(
                action.action_type,
                hint="control markers are handled by the driver, not the environment",
            )
        capability = ACTION_CAPABILITY.get(action.action_type)
        if capability is None:
            raise UnsupportedAction(
                action.action_type,
                hint="no port capability expresses this action; it is not approximated on purpose",
            )
        self.require(capability)

        params = action.params
        if capability is Capability.NAVIGATE:
            await self.backend.navigate(str(params["url"]))
        elif capability is Capability.SCREENSHOT:
            await self.backend.screenshot(params.get("path"))
        elif capability is Capability.WAIT_FOR_TIMEOUT:
            await self.backend.wait_for_timeout(float(params.get("ms", 0)))
        elif capability is Capability.CLICK:
            element = await self._element(params)
            await self.backend.click(element, force=bool(params.get("force", False)))
        elif capability is Capability.FILL:
            element = await self._element(params)
            await self.backend.fill(
                element, str(params["value"]), force=bool(params.get("force", False))
            )
        elif capability is Capability.PRESS_ELEMENT:
            element = await self._element(params)
            await self.backend.press_element(element, str(params["key"]))
        else:  # pragma: no cover - mapping and dispatch must stay in step
            raise UnsupportedAction(
                action.action_type,
                hint=f"capability {capability.value!r} has no dispatch branch",
            )

    async def _element(self, params: Mapping[str, Any]) -> Element:
        """Resolve the element an action targets, declaring what it needs."""
        element = params.get("element")
        if isinstance(element, Element):
            return element
        locator = params.get("locator")
        if isinstance(locator, ElementLocator):
            self.require(Capability.FIND_ELEMENT)
            found = await self.backend.find_element(locator)
            if found is None:
                raise LookupError(f"element not found for locator {locator!r}")
            return found
        raise ValueError(
            "action params must carry an 'element' (Element) or 'locator' "
            "(ElementLocator) for element-targeted actions"
        )


def environment_for(
    name: str,
    backend: Backend,
    *,
    capabilities: CapabilityMatrix | None = None,
) -> Environment:
    """Bind a backend to its declaration.

    ``capabilities`` is required in practice — passing ``None`` declares nothing,
    which makes the environment honest but useless (every action refused). It is
    allowed so a probe environment can be built to *measure* what a backend can
    do without pretending it can do it.
    """
    return Environment(
        name=name,
        backend=backend,
        capabilities=capabilities or CapabilityMatrix(),
    )


__all__ = [
    "ACTION_CAPABILITY",
    "CONTROL_ACTIONS",
    "Environment",
    "UnsupportedAction",
    "environment_for",
    "UnsupportedCapability",
]
