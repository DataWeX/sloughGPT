"""
tokenizer.py - Inference tokenizer access for native backends.

Resolves the ``get_tokenizer()`` import used by the C-accelerated
``NativeEngine`` and ``CTransformProvider``.  Delegates to the shared
BPE/Unigram lifecycle in ``domains.training.tokenizer_manager`` so all
backends share one tokenizer instance.
"""

from typing import Any


def get_tokenizer() -> Any:
    """Return the shared tokenizer, creating a default BPE if none is loaded.

    Returns:
        The TokenizerManager's current tokenizer instance (SloBPE/SloUnigram).

    Side effects:
        - Creates a default ``SloBPE`` on first access if the manager has none.
    """
    from domains.training.tokenizer_manager import get_tokenizer_manager
    return get_tokenizer_manager().get_tokenizer()
