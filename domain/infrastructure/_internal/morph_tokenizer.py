"""Backward-compatibility shim."""
import domains.infrastructure.morph_tokenizer as _mod
globals().update(vars(_mod))
