"""Capability contract for environment adapters.

The port — ``avion.core.element.Backend`` — says what a backend *may* be asked
to do. It says nothing about what a given adapter actually answers. Python's
Protocol is structural, so nothing checks it: measured on 2026-11-27,
``avion/backends/`` contained modules at 18/18 (playwright, appium, cdp,
selenium), 7/18 (desktop), 2/18 (api, cli) and 0/18 (axtree — not a backend at
all), and every caller discovered that at call time, one AttributeError at a
time.

So support is **declared, never probed**. A declaration can be reviewed and
tested; a probe is stateful and lies when a backend fails lazily (a hover that
throws only once a window is focused). Calling an undeclared capability raises
:class:`UnsupportedCapability` — fail closed, never a silent no-op.

The goal is honesty, not uniformity: ``appium`` has no DOM and ``cli`` has no
viewport, and both should say so rather than fake a method to satisfy a gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping


class Capability(str, Enum):
    """One operation an environment adapter can support.

    Values match the ``Backend`` protocol method names so a declaration reads
    as the port it satisfies (``Capability.CLICK.value == "click"``).
    """

    # --- Backend protocol (core) -------------------------------------------
    START = "start"
    STOP = "stop"
    NAVIGATE = "navigate"
    FIND_ELEMENT = "find_element"
    FIND_ELEMENTS = "find_elements"
    CLICK = "click"
    FILL = "fill"
    SELECT_OPTION = "select_option"
    CHECK = "check"
    UNCHECK = "uncheck"
    HOVER = "hover"
    SCREENSHOT = "screenshot"
    GET_URL = "get_url"
    GET_TITLE = "get_title"
    WAIT_FOR = "wait_for"
    WAIT_FOR_TIMEOUT = "wait_for_timeout"
    EVALUATE = "evaluate"
    GET_ACCESSIBILITY_TREE = "get_accessibility_tree"

    # --- PageControls protocol (optional, beyond core) ----------------------
    WAIT_FOR_FUNCTION = "wait_for_function"
    SET_VIEWPORT_SIZE = "set_viewport_size"
    FOCUS = "focus"
    PRESS_ELEMENT = "press_element"


CORE_CAPABILITIES: frozenset[Capability] = frozenset(
    c
    for c in Capability
    if c
    not in {
        Capability.WAIT_FOR_FUNCTION,
        Capability.SET_VIEWPORT_SIZE,
        Capability.FOCUS,
        Capability.PRESS_ELEMENT,
    }
)

OPTIONAL_CAPABILITIES: frozenset[Capability] = frozenset(Capability) - CORE_CAPABILITIES


class UnsupportedCapability(RuntimeError):
    """Raised when an adapter is asked for a capability it did not declare.

    Carries the environment and the capability so a caller (or a model reading
    the error) can choose a different environment or a different action instead
    of retrying blindly.
    """

    def __init__(
        self,
        environment: str,
        capability: Capability,
        *,
        declared: Iterable[Capability] = (),
        hint: str = "",
    ) -> None:
        self.environment = environment
        self.capability = capability
        self.declared = frozenset(declared)
        self.hint = hint
        message = f"environment {environment!r} does not declare capability {capability.value!r}"
        if hint:
            message = f"{message} — {hint}"
        super().__init__(message)


@dataclass(frozen=True)
class CapabilityMatrix:
    """What one environment adapter supports, declared up front.

    Frozen and hashable so a matrix can be compared, diffed in a report, or
    handed to a model as part of the environment description.
    """

    declared: frozenset[Capability] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "declared", frozenset(self.declared))

    # -- queries ------------------------------------------------------------
    def supports(self, capability: Capability) -> bool:
        return capability in self.declared

    def require(self, *capabilities: Capability, environment: str = "environment") -> None:
        """Raise :class:`UnsupportedCapability` for the first missing one."""
        for capability in capabilities:
            if capability not in self.declared:
                raise UnsupportedCapability(
                    environment,
                    capability,
                    declared=self.declared,
                    hint=self._hint_for(capability),
                )

    def missing(self) -> frozenset[Capability]:
        """Every capability this adapter cannot answer."""
        return frozenset(Capability) - self.declared

    def missing_core(self) -> frozenset[Capability]:
        """The port operations this adapter cannot answer (the honest gap list)."""
        return CORE_CAPABILITIES - self.declared

    def satisfies_port(self) -> bool:
        """True only when the whole core ``Backend`` protocol is declared."""
        return CORE_CAPABILITIES <= self.declared

    def satisfies_page_controls(self) -> bool:
        return OPTIONAL_CAPABILITIES <= self.declared

    # -- projections --------------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        return {
            "declared": sorted(c.value for c in self.declared),
            "missing_core": sorted(c.value for c in self.missing_core()),
            "satisfies_port": self.satisfies_port(),
            "satisfies_page_controls": self.satisfies_page_controls(),
        }

    @classmethod
    def of(cls, *capabilities: Capability | str) -> CapabilityMatrix:
        """Build from enum members or raw port method names."""
        return cls(frozenset(_coerce(c) for c in capabilities))

    @classmethod
    def from_names(cls, names: Iterable[str]) -> CapabilityMatrix:
        return cls(frozenset(_coerce(n) for n in names))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, bool]) -> CapabilityMatrix:
        """Declare from ``{port_method: supported}`` — e.g. an adapter's own table."""
        return cls(frozenset(_coerce(k) for k, ok in mapping.items() if ok))

    def _hint_for(self, capability: Capability) -> str:
        if capability in OPTIONAL_CAPABILITIES:
            return (
                "this is a PageControls capability; only adapters that declare "
                "it implement the optional page-control protocol"
            )
        return (
            "capability is declared per adapter — see "
            "avion.environment.CapabilityMatrix; do not probe the backend"
        )


def _coerce(value: Capability | str) -> Capability:
    if isinstance(value, Capability):
        return value
    return Capability(value)
