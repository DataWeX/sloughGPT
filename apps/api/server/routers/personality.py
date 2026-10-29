"""Personality evolution history — a descriptor-projected capability crossing.

One capability, one descriptor (``SPEC``), one projection: verb, auth, request
validation, response envelope and OpenAPI metadata are all derived by
``create_router`` — see ``docs/TRANSPORT_PROJECTIONS.md`` and AGENTS.md
"Endpoint & Transport Rule". The handler moved out of ``ConsciousnessRouter``
so the crossing registers through the contract pipeline (manifest boot +
registry-vs-served verification); the wire path and envelope are unchanged.

Tier: external utility tooling — a site read route. It never enters the chat
tool registry, the ``domain.tools`` engine, or anything the model reasons over.

Mounted with an empty ``prefix`` and the full path in ``RouteSpec`` on purpose:
``main.py`` owns the pre-lifespan ``/consciousness`` router, and a second
manifest router under the same prefix would trip ``_phase6_routers``'s
already-registered skip.
"""

from __future__ import annotations

from typing import Any

from infrastructure.contract import RouteSpec, create_router

from domain.agents import ToolSpec


async def personality_evolution_history() -> dict[str, Any]:
    """Compute voice/traits trends from the episode log.

    Returns the raw payload — the response envelope is projected from the
    descriptor by ``create_router``, never hand-built here.
    """
    from domain.cognition import PersonalityProfile, get_consciousness

    episodes = get_consciousness().self_model.episodes

    if not episodes:
        return {"history": [], "labels": {}}

    defaults = PersonalityProfile()
    current = defaults.to_dict()
    history = []
    step = max(1, len(episodes) // 50)

    for i, ep in enumerate(episodes):
        growth = ep.growth_delta
        qualia = ep.qualia

        if growth > 0.05:
            current["voice"]["confidence"] = min(1.0, current["voice"]["confidence"] + 0.01)
        if qualia.get("novelty", 0) > 0.7:
            current["traits"]["openness"] = min(1.0, current["traits"]["openness"] + 0.005)
        if qualia.get("coherence", 1) < 0.3:
            current["voice"]["verbosity"] = max(0.0, current["voice"]["verbosity"] - 0.01)
        if qualia.get("valence", 0) > 0.5:
            current["traits"]["agreeableness"] = min(
                1.0, current["traits"]["agreeableness"] + 0.005
            )

        if i % step == 0 or i == len(episodes) - 1:
            history.append(
                {
                    "timestamp": ep.timestamp,
                    "step": i,
                    "voice": {k: round(v, 4) for k, v in current["voice"].items()},
                    "traits": {k: round(v, 4) for k, v in current["traits"].items()},
                }
            )

    return {
        "history": history,
        "labels": {
            "voice": list(defaults.voice.keys()),
            "traits": list(defaults.traits.keys()),
        },
    }


SPEC = ToolSpec(
    name="consciousness.personality_history",
    description="Personality evolution history derived from episodes (voice and trait trends).",
    parameters=[],
    execute=personality_evolution_history,
    auth_scope="authenticated",
    idempotent=True,
    result={
        "type": "object",
        "properties": {
            "history": {"type": "array", "items": {"type": "object"}},
            "labels": {"type": "object"},
        },
    },
    version="1",
)

router = create_router(
    SPEC,
    RouteSpec(
        path="/consciousness/personality/history",
        summary="Personality evolution history derived from episodes",
    ),
    personality_evolution_history,
    tags=("consciousness",),
)
