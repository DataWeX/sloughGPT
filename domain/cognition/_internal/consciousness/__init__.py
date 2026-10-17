"""consciousness — Self-awareness, qualia, meta-cognition, narrative.

Canonical implementation of the consciousness subsystem. Re-exported by
``domain.cognition`` (canonical facade).

HOST CONTRACT — the seam an embedding app speaks through
========================================================

This subsystem is a component, not an application. It owns the inner life
(records, beliefs, qualia, dreams); the host owns everything else. An app
that adopts a mind needs only this much:

    mind = create_mind("/absolute/path/to/store")   # your storage, your mind

    mind.process(input_text, response)   # IN:  the world reports an experience
    mind.maybe_dream()                   # TICK: sleep gate, host-owned cadence
    mind.self_report()                   # OUT: beliefs, doubts, timeline, last dream
    mind.locus()                         # OUT: awakening / dreaming / rationalizing
    mind.dream()                         # OUT: force one night (rarely needed)

Contract terms:

* **Storage is injected, never assumed.** ``create_mind`` requires an absolute
  path and refuses to guess; a component that picks its own ledger location is
  not stitchable.
* **Instances are independent.** Two minds in one process share nothing — not
  records, not beliefs, not storage. This is tested, not assumed.
* **The cadence is the host's.** The engine is synchronous and wall-clock-free;
  it never starts timers or loops of its own. Call it when you want a tick.
* **The model is optional.** Nothing here imports inference, numpy, or the API
  layer. A mind runs on the standard library alone.
* **``get_consciousness()`` is a convenience default**, not the only road in.
  It is the ONE mind a host creates for itself; ``create_mind`` is for hosts
  that own their own.

Everything not listed above — ``SelfModel``, ``QualiaEngine``,
``NarrativeGenerator`` and the rest — is the gateway's bookkeeping. Reach into
it and you own its upgrades.
"""

from __future__ import annotations

from pathlib import Path

from domain.cognition._internal.consciousness.config import ConsciousnessConfig
from domain.cognition._internal.consciousness.engine import (
    ConsciousnessEngine,
    get_consciousness,
    peek_consciousness,
    reset_consciousness,
)
from domain.cognition._internal.consciousness.evaluation import (
    ConsciousnessEvaluator,
    EvaluationReport,
)
from domain.cognition._internal.consciousness.meta_cognition import (
    MetaCognition,
    MetaCognitiveReport,
)
from domain.cognition._internal.consciousness.narrative import NarrativeGenerator
from domain.cognition._internal.consciousness.personality import (
    PersonalityManager,
    PersonalityProfile,
)
from domain.cognition._internal.consciousness.qualia import QualiaEngine, QualiaState
from domain.cognition._internal.consciousness.self_model import (
    Reflection,
    SelfEpisode,
    SelfIdentity,
    SelfModel,
)


def create_mind(store_path: str, level: int = 2) -> ConsciousnessEngine:
    """Construct an independent mind for an embedding app.

    Where ``get_consciousness()`` is the ONE mind this process happens to serve,
    ``create_mind`` is for an app that owns its own: separate storage, separate
    beliefs, separate dreams, and no cross-talk with any other mind in the
    process. Multiple minds coexist — that is tested, not assumed.

    ``store_path`` must be ABSOLUTE. A component that guesses its storage
    location writes into whatever directory the embedding app happens to be
    running in — which is how a mind ends up keeping its ledger in the
    merchant's basement. Refuse rather than guess.
    """
    path = Path(store_path)
    if not path.is_absolute():
        raise ValueError(
            f"create_mind() requires an absolute store_path, got {store_path!r}. "
            "A relative path would write into the embedding app's working directory."
        )
    return ConsciousnessEngine(ConsciousnessConfig(level=level, store_path=str(path)))


__all__ = [
    "ConsciousnessConfig",
    "ConsciousnessEngine",
    "ConsciousnessEvaluator",
    "EvaluationReport",
    "MetaCognition",
    "MetaCognitiveReport",
    "NarrativeGenerator",
    "PersonalityManager",
    "PersonalityProfile",
    "QualiaEngine",
    "QualiaState",
    "Reflection",
    "SelfEpisode",
    "SelfIdentity",
    "SelfModel",
    "get_consciousness",
    "peek_consciousness",
    "reset_consciousness",
    "create_mind",
]
