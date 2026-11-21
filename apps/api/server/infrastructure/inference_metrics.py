"""Single seam for tok/s recording at the API record sites (card 20261119_132133).

PR 1 (domain) gave ``ServerState.record_inference`` the decode-window contract
and the ``cached`` flag; this module is the apps/api half: every router record
site counts tokens and records through here, so the three previously
incompatible sites (word splits vs tokenizer ids, whole-turn clocks vs decode
windows, a cache hit that recorded nothing) cannot drift apart again.

Two functions, called at all eight record sites in ``routers/``:

- ``count_tokens(text, *, state=None, provider=None)`` — tokenizer-first
  priority chain: ``state.tokenizer.encode`` (the loaded model's tokenizer)
  -> ``provider.count_tokens`` (the provider's own tokenizer) -> a word split
  as the last resort. Never raises.
- ``record(tokens, elapsed_ms, model="", *, cached=False)`` — owns the lazy
  ``get_server_state`` import and the warn-not-raise around recording, so a
  metrics failure degrades to a log line instead of failing a response, and
  no site can forget the ``cached`` flag (PR 1 contract: cache hits stay in
  lifetime totals but are excluded from ``get_tokens_per_second``).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def count_tokens(text: str, *, state=None, provider=None) -> int:
    """Count tokens in ``text`` — tokenizer-first, word split last resort.

    Args:
        text: text to count (generated output or a prompt).
        state: object exposing a ``tokenizer`` with ``encode`` (the API
            ``state`` module or any server-state-like object).
        provider: inference provider exposing ``count_tokens``.

    Returns:
        A token count; never raises — degrades to ``len(text.split())``.
    """
    if state is not None:
        try:
            tokenizer = getattr(state, "tokenizer", None)
            if tokenizer is not None and hasattr(tokenizer, "encode"):
                return len(tokenizer.encode(text))
        except Exception as e:
            logger.debug("tokenizer encode failed, trying provider: %s", e)
    if provider is not None:
        try:
            counted = provider.count_tokens(text)
        except Exception as e:
            logger.debug("provider count_tokens failed, falling back to word split: %s", e)
        else:
            # isinstance, not int(): int() on a test double returns 1 silently,
            # which would fabricate a token count the tokenizer never produced.
            if isinstance(counted, int):
                return counted
    return len(text.split())


def record(tokens: int, elapsed_ms: float, model: str = "", *, cached: bool = False) -> None:
    """Record one inference at the shared tok/s seam. Never raises.

    ``elapsed_ms`` is the decode window measured at the call site (PR 1
    contract); ``cached=True`` marks a prompt-cache hit — kept in lifetime
    totals and model metrics, excluded from ``get_tokens_per_second``.
    """
    try:
        from domain.infrastructure.server_state import get_server_state

        get_server_state().record_inference(
            tokens=tokens, elapsed_ms=elapsed_ms, model=model, cached=cached
        )
    except Exception as e:
        logger.warning("Failed to record inference metrics: %s", e)
