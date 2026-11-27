"""Environment layer — the namespace for what an adapter is and what it can do.

Three objects, one each for the three questions a driver has:

``CapabilityMatrix``  what may I ask this environment to do?  (declared, frozen)
``Observation``       what does it look like right now?        (absence named)
``Environment``       do this one thing.                       (fail closed)

plus ``EnvironmentRegistry`` for "which environment can do X".

Layering, bottom to top::

    backends/*      transport (playwright, appium, cdp, selenium, desktop, …)
    environment/    this package — declaration + observation + single-action dispatch
    ai/agent.py     the loop that decides what to do next (optional, model-driven)
    journeys/       named journeys, projected onto the above

Nothing in this package imports the agent, and nothing here needs a model:
``Environment.observe`` and ``Environment.act`` are enough for a scripted run.
A model is one optional node above, reached through ``avion.ai.models.VisionModel``.
"""

from __future__ import annotations

from .capability import (
    CORE_CAPABILITIES,
    OPTIONAL_CAPABILITIES,
    Capability,
    CapabilityMatrix,
    UnsupportedCapability,
)
from .environment import (
    ACTION_CAPABILITY,
    CONTROL_ACTIONS,
    Environment,
    UnsupportedAction,
    environment_for,
)
from .observation import Observation
from .registry import DEFAULT_REGISTRY, EnvironmentRegistry, register_environments

__all__ = [
    "ACTION_CAPABILITY",
    "CONTROL_ACTIONS",
    "CORE_CAPABILITIES",
    "Capability",
    "CapabilityMatrix",
    "DEFAULT_REGISTRY",
    "Environment",
    "EnvironmentRegistry",
    "OPTIONAL_CAPABILITIES",
    "Observation",
    "UnsupportedAction",
    "UnsupportedCapability",
    "environment_for",
    "register_environments",
]
