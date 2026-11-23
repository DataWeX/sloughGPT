"""One tok/s recording seam for the API record sites (card 20261119_132133).

PR 2/2 of the tok/s unification: every ``record_inference`` call site in
``apps/api`` funnels through ``infrastructure.inference_metrics`` so token
counts and clocks cannot drift apart again.

- ``count_tokens(text, *, state=None, provider=None)`` — tokenizer-first
  counting: state tokenizer -> provider ``count_tokens`` -> word split as the
  last resort. Replaces the ``len(text.split())`` word counts at the chat,
  models, and llm_v1 record sites (words != tokens: a 3-word sentence can be
  7 tokenizer ids).
- ``record(tokens, elapsed_ms, model="", *, cached=False)`` — owns the lazy
  ``get_server_state`` import and the warn-not-raise around recording, so a
  metrics failure can never fail a response and no site can silently drop
  the ``cached`` flag (PR 1 contract: cache hits stay in lifetime totals but
  are excluded from ``get_tokens_per_second``).

The structural tests pin the record sites themselves: word-split counts, the
``elapsed_ms=0`` clock, and the silent coalescer cache-hit path must not
survive in the three routers.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from infrastructure.inference_metrics import count_tokens, record

ROUTERS = Path(__file__).resolve().parents[1] / "routers"


def _src(name: str) -> str:
    return (ROUTERS / name).read_text(encoding="utf-8")


# ── count_tokens: tokenizer-first priority chain ────────────────────────────


class _FakeTokenizer:
    """``encode()`` returns a fixed id list, so token count != word count."""

    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def encode(self, text: str) -> list[int]:
        return list(self._ids)


def test_count_tokens_prefers_state_tokenizer_over_words():
    # 3 words but 7 tokenizer ids — a word split would under-report by 4.
    state = SimpleNamespace(tokenizer=_FakeTokenizer([1, 2, 3, 4, 5, 6, 7]))
    assert count_tokens("one two three", state=state) == 7


def test_count_tokens_uses_provider_when_state_has_no_tokenizer():
    provider = SimpleNamespace(count_tokens=lambda text: 42)
    # state present but tokenizer-less (model not loaded) must fall through.
    assert count_tokens("hello world", state=SimpleNamespace(), provider=provider) == 42


def test_count_tokens_word_split_is_last_resort():
    assert count_tokens("a b c d") == 4  # no state, no provider


def test_count_tokens_survives_tokenizer_exception():
    # A broken tokenizer must degrade to the provider, never raise.

    class _Boom:
        def encode(self, text: str) -> list[int]:
            raise ValueError("tokenizer exploded")

    provider = SimpleNamespace(count_tokens=lambda text: 42)
    assert count_tokens("x", state=SimpleNamespace(tokenizer=_Boom()), provider=provider) == 42


# ── record: cached flag + never-raise seam ──────────────────────────────────


def test_record_cached_hit_excluded_from_tps_but_kept_in_lifetime():
    from domain.infrastructure.server_state import get_server_state, reset_server_state

    reset_server_state()
    try:
        record(100, 1000.0, model="m", cached=True)  # cache hit: no decode
        state = get_server_state()
        assert state.get_tokens_per_second() == 0.0  # excluded (PR 1 contract)
        assert state.inference_count == 1  # lifetime keeps it
        assert state.total_tokens == 100

        record(50, 1000.0, model="m")  # real decode
        # window = non-cached records only: 50 tok / 1.0 s
        assert state.get_tokens_per_second() == 50.0
    finally:
        reset_server_state()


def test_record_never_raises_when_state_unavailable(monkeypatch):
    import domain.infrastructure.server_state as ss

    def _boom():
        raise RuntimeError("state unavailable")

    # record() must lazy-import get_server_state at call time (else this
    # monkeypatch cannot bite) and swallow the failure with a warning.
    monkeypatch.setattr(ss, "get_server_state", _boom)
    record(1, 1.0, model="m")


# ── structural ratchets: the record sites stay wired through the helper ─────


def test_no_word_split_token_counts_in_routers():
    assert "len(full_response.split())" not in _src("inference.py")  # chat site
    assert "len(result.split())" not in _src("models.py")  # infer site


def test_llm_v1_records_through_helper_not_private_wrapper():
    src = _src("llm_v1.py")
    assert "_record_inference(" not in src  # wrapper deleted
    assert "inference_metrics import" in src  # helper imported instead


def test_no_zero_elapsed_clock_in_inference_router():
    # The WebSocket site recorded elapsed_ms=0 — an honest clock replaces it.
    assert "elapsed_ms=0" not in _src("inference.py")


def test_coalescer_cache_hit_records_with_cached_flag():
    src = _src("inference.py")
    assert "cached=True" in src  # silent cache hit now records (PR 1 contract)
    assert "len(cached.split())" not in src  # meta reports tokens, not words


def test_all_three_routers_import_the_helper():
    for name in ("inference.py", "models.py", "llm_v1.py"):
        assert "inference_metrics" in _src(name), name
