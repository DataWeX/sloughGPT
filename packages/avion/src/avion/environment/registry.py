"""Which environments exist, and which one can do a given thing.

The registry answers the gap question directly instead of leaving callers to
branch on names:

    registry.supporting(Capability.PRESS_ELEMENT)   # -> only playwright, today
    registry.for_action(ActionType.MOUSE_CLICK)     # -> everything that declares click

That is the difference between "handle all these gaps and unavailabilities" and
"discover them at call time". A caller asks; the registry answers from
declarations; nothing is probed.
"""

from __future__ import annotations

from typing import Iterable, Iterator

from avion.ai.models import ActionType

from .capability import Capability
from .environment import Environment


class EnvironmentRegistry:
    """Named environments, with capability queries over the whole set."""

    def __init__(self) -> None:
        self._environments: dict[str, Environment] = {}

    # -- registration -------------------------------------------------------
    def register(self, environment: Environment, *, replace: bool = False) -> Environment:
        """Add an environment. A duplicate name is an error unless replaced."""
        if environment.name in self._environments and not replace:
            raise ValueError(
                f"environment {environment.name!r} is already registered; "
                "pass replace=True to override deliberately"
            )
        self._environments[environment.name] = environment
        return environment

    def unregister(self, name: str) -> Environment:
        return self._environments.pop(name)

    # -- lookup -------------------------------------------------------------
    def get(self, name: str) -> Environment:
        try:
            return self._environments[name]
        except KeyError:
            available = ", ".join(sorted(self._environments)) or "(none)"
            raise KeyError(f"no environment {name!r}; registered: {available}") from None

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._environments))

    def supporting(self, *capabilities: Capability) -> tuple[Environment, ...]:
        """Environments declaring every one of ``capabilities``."""
        return tuple(
            env
            for _, env in sorted(self._environments.items())
            if all(env.supports(c) for c in capabilities)
        )

    def for_action(self, action_type: ActionType) -> tuple[Environment, ...]:
        """Environments that can actually execute ``action_type``."""
        return tuple(
            env for _, env in sorted(self._environments.items()) if env.supports_action(action_type)
        )

    def gaps(self) -> dict[str, list[str]]:
        """Per environment, the port operations it cannot answer.

        The report form of the honest gap list — the thing that was previously
        only discoverable by crashing.
        """
        return {
            name: sorted(c.value for c in env.capabilities.missing_core())
            for name, env in sorted(self._environments.items())
        }

    # -- dunder -------------------------------------------------------------
    def __contains__(self, name: object) -> bool:
        return name in self._environments

    def __len__(self) -> int:
        return len(self._environments)

    def __iter__(self) -> Iterator[Environment]:
        return iter(sorted(self._environments.values(), key=lambda e: e.name))


#: Process-wide registry. Import it, don't build a second one — two registries
#: is the same drift as two manifests.
DEFAULT_REGISTRY = EnvironmentRegistry()


def register_environments(environments: Iterable[Environment]) -> None:
    for environment in environments:
        DEFAULT_REGISTRY.register(environment)


__all__ = ["DEFAULT_REGISTRY", "EnvironmentRegistry", "register_environments"]
