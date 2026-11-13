"""AppScope — the per-conversation configuration instance (the mini-app scope).

A conversation is the instance boundary of the app: model settings, context
toggles and conversation instructions that differ from the global defaults
live on the conversation, resolved once per turn instead of being carried on
every wire payload (the scattered-read this replaces: the frontend used to
ship temperature/max_tokens on each POST /chat, making the client the de
facto resolver).

Three layers of intent, most specific first — the invariant is the same one
``_apply_meta_weights``'s docstring already states ("an explicitly-set value
is user intent and outranks a derived one"), extended to three layers:

    turn (wire ``model_fields_set``)  >  conversation delta  >  global settings

Presence-based, never value-based: a field *absent* from a layer is unset;
``False``/``0`` in a delta are set values and win. ``None`` in the resolved
scope means "unset — leave the ``TurnRequest`` default alone".

Precedence is per field, not per layer: a turn may override only
temperature while the conversation supplies max_tokens and the global
settings supply the rest. Layers never write back — resolution flows
global → conversation → turn only, so changing a global default propagates
to every mini-app that has not overridden it, and nothing a conversation
does can move anyone else's defaults.

Scope-owned fields map to real turn mechanisms only (``TurnRequest`` /
``GenerationSettings``); fields with no consumer in the turn path are not
declared here — a contract field nothing reads is a lie waiting to happen.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from domain.inference.engine import TurnRequest

#: Transport fields: applied onto ``TurnRequest`` when the wire left them
#: unset. ``system_prompt``/``knowledge`` are additive (composed, not
#: transported) and handled at the seam, not by ``apply_scope``.
SCOPE_TRANSPORT_FIELDS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "max_tokens",
    "use_context_core",
    "use_rag",
)

#: Fields the global settings layer can supply. Only generation params have
#: a global store (``domain.settings.GenerationSettings``); context toggles
#: and conversation instructions are conversation-owned only.
_SCOPE_GLOBAL_FIELDS: frozenset[str] = frozenset(
    {"temperature", "top_p", "top_k", "repetition_penalty", "max_tokens"}
)

#: GenerationSettings spells this one differently than TurnRequest.
_GLOBAL_ATTR: dict[str, str] = {"max_tokens": "max_new_tokens"}


@dataclass(frozen=True, slots=True)
class AppScope:
    """Resolved configuration for one conversation — a frozen value object.

    ``None`` means "this layer did not set the field" (fall through to the
    ``TurnRequest`` default). Set fields are user intent: they bypass
    server-side derivation (the meta-weight nudge) exactly like a wire-
    explicit value does.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    max_tokens: int | None = None
    system_prompt: str | None = None
    knowledge: tuple[str, ...] | None = None
    use_context_core: bool | None = None
    use_rag: bool | None = None

    @property
    def set_fields(self) -> frozenset[str]:
        """Names of fields this scope actually sets (presence, not value)."""
        return frozenset(f.name for f in fields(self) if getattr(self, f.name) is not None)


def resolve_scope(
    *,
    conversation: Mapping[str, Any] | None = None,
    global_settings: Any = None,
) -> AppScope:
    """Resolve the below-turn layers: conversation delta over global settings.

    Pure — no I/O, no session reads (the seam loads the layers). Per field:
    a present, non-``None`` conversation value wins; else the global settings
    value for generation params; else unset. ``global_settings`` is a
    ``domain.settings.GenerationSettings`` (duck-typed to stay import-light).
    """
    conv = dict(conversation or {})
    values: dict[str, Any] = {}
    for f in fields(AppScope):
        name = f.name
        conv_value = conv.get(name)
        if conv_value is not None:
            # The wire twin stores knowledge as list[str]; coerce so the
            # frozen value object never holds a mutable field.
            values[name] = tuple(conv_value) if name == "knowledge" else conv_value
            continue
        if name in _SCOPE_GLOBAL_FIELDS and global_settings is not None:
            values[name] = getattr(global_settings, _GLOBAL_ATTR.get(name, name))
    return AppScope(**values)


def apply_scope(
    req: TurnRequest,
    scope: AppScope,
    *,
    conversation: Mapping[str, Any] | None = None,
) -> frozenset[str]:
    """Apply resolved scope onto ``req`` in place; return conversation-won fields.

    Transport fields are applied only where the wire left them unset
    (``model_fields_set``) — an explicit wire value is this turn's intent and
    outranks both lower layers (that is what keeps one-off overrides such as
    regenerate-with-temperature working). The returned set names the fields
    that won *from the conversation layer*: the seam passes it to
    ``_apply_meta_weights`` as ``explicit`` so a heuristic may not move a
    value the user deliberately scoped to this conversation.

    ``system_prompt``/``knowledge`` are additive and composed by the seam
    (they must merge with soul/agent composition), not transported here.
    """
    conv = conversation or {}
    applied: set[str] = set()
    for name in SCOPE_TRANSPORT_FIELDS:
        if name in req.model_fields_set:
            continue
        value = getattr(scope, name)
        if value is None:
            continue
        setattr(req, name, value)
        if conv.get(name) is not None:
            applied.add(name)
    return frozenset(applied)
