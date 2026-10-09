"""setup_providers — wire up default provider + processor pipeline."""

from __future__ import annotations

import logging
from typing import Any

from .processors.knowledge import KnowledgeProcessor
from .processors.personality import PersonalityProcessor
from .processors.style import StyleProcessor
from .processors.tool_use import ToolDef, ToolUseProcessor
from .processors.vision import VisionProcessor
from .protocols import ChatMessage, MessageProcessor, ModelCapabilities, ModelProvider
from .registry import (
    _providers,
    apply_processors,
    clear_providers,
    get_processor,
    get_provider,
    list_processors,
    list_providers,
    register_processor,
    register_provider,
)
from .router import ProviderRouter
from .slo_transformer import SloTransformerProvider

logger = logging.getLogger("slo.models.provider")

# =============================================================================
# setup_providers — wire up default provider + full processor pipeline
# =============================================================================


def _server_from_provider(provider: Any, process_guard: Any) -> Any:
    """Build a guard-backed ``SloNetServer`` from a provider's model/tokenizer.

    Delegates to ``SloNetChatProvider.to_server()`` when available. Returns
    ``None`` if the provider cannot build a server.

    Args:
        provider: Provider instance with ``to_server()`` (e.g. SloNetChatProvider)
        process_guard: ``ProcessGuard`` to attach for subprocess delegation

    Returns:
        SloNetServer bound to the provider, or None.
    """
    try:
        if not hasattr(provider, "to_server"):
            return None
        return provider.to_server(process_guard=process_guard)
    except Exception as e:
        logger.warning("Failed to build guard-backed server: %s", e, extra={"tag": "MODEL"})
        return None


def setup_providers(
    slonet_hf_id: str | None = None,
    slonet_provider=None,
    slonet_server=None,
    model_registry=None,
    process_guard=None,
    quantize: bool = False,
    quant_bits: int = 8,
    quant_mode: str = "symmetric",
    personality_traits: dict[str, float] | None = None,
    slonet_path: str | None = None,
    native_slnc_path: str | None = None,
) -> None:
    """Register providers and build the default processor pipeline.

    Registers:
    - ``"native-c"``: NativeTransformerProvider (C-accelerated, if feature flag enabled
      or ``native_slnc_path`` given)
    - ``"slonet-native"``: SloNetChatProvider (if slonet_hf_id or slonet_provider given)
    - ``"default"``: ProviderRouter with processor chain → text provider

    When a ``SloNetServer`` is given via ``slonet_server``, it is attached to
    the provider for concurrency control, circuit breaker, and warmup. When a
    ``ProcessGuard`` is given via ``process_guard`` and no server is provided,
    a guard-backed ``SloNetServer`` is built from the provider automatically.

    Processor chain (in order):
    1. VisionProcessor — caption images
    2. ToolUseProcessor — tool-call instructions
    3. PersonalityProcessor — soul trait injection
    4. StyleProcessor — formality/directness/verbosity

    KnowledgeProcessor is NOT in the router pipeline — knowledge is per-request
    and handled by the caller (inference.py) via ``apply_processors()``.

    Args:
        slonet_hf_id: HuggingFace model ID for SloNet model (used as the
            tokenizer model ID when ``slonet_path`` is given)
        slonet_provider: Pre-loaded provider (skips re-loading)
        slonet_server: ``SloNetServer`` instance for concurrency/circuit-breaker
        model_registry: Optional ModelRegistry for lifecycle management
        process_guard: Optional ``ProcessGuard`` — attached to the provider
            via a guard-backed ``SloNetServer`` when ``slonet_server`` is None
        quantize: Whether to quantize the model
        quant_bits: Quantization bit width
        quant_mode: Quantization mode
        personality_traits: Optional personality traits dict
        slonet_path: Direct path to a local .slnc file (e.g. a compiled
            fine-tuned model) to load via ``SloNetChatProvider.from_slnc``
        native_slnc_path: Direct path to a local .slnc file to load through
            the native C engine (``NativeEngine.from_slnc_file``). When given,
            ``native-c`` is registered and becomes the text provider.
    """
    text_provider_name = None

    # Try native C inference engine first (highest priority if enabled)
    try:
        from domain.shared._internal.feature_flags import is_enabled

        native_on = is_enabled("native_c_inference")
        if native_on or native_slnc_path:
            from domain.inference._internal.native.engine import (
                NativeTransformerProvider,
                get_engine,
            )

            engine = get_engine()
            if not engine.loaded and native_slnc_path:
                engine = engine.from_slnc_file(native_slnc_path)
            if engine.loaded:
                register_provider(
                    "native-c", NativeTransformerProvider(engine, model_id="native-c")
                )
                text_provider_name = "native-c"
                logger.info(
                    "Registered native-c provider (Apple Accelerate BLAS)", extra={"tag": "MODEL"}
                )
    except Exception as e:
        logger.debug("Native C inference not available: %s", e, extra={"tag": "MODEL"})

    if slonet_provider is not None:
        # Attach SloNetServer if provided, else build one from the guard
        if slonet_server is None and process_guard is not None:
            slonet_server = _server_from_provider(slonet_provider, process_guard)
        if slonet_server is not None and hasattr(slonet_provider, "set_server"):
            slonet_provider.set_server(slonet_server)
            logger.info(
                "Attached SloNetServer to provider: %s",
                getattr(slonet_provider, "_model_id", "?"),
                extra={"tag": "MODEL"},
            )
        register_provider("slonet-native", slonet_provider)
        text_provider_name = "slonet-native"
        logger.info(
            "Registered slonet-native provider: %s (pre-loaded%s)",
            getattr(slonet_provider, "_model_id", "?"),
            ", server-backed" if slonet_server else "",
            extra={"tag": "MODEL"},
        )
    elif slonet_path:
        try:
            from domain.inference._internal.slonet_provider import SloNetChatProvider

            slonet_provider = SloNetChatProvider.from_slnc(
                slonet_path,
                model_id=slonet_hf_id or "gpt2",
                quantize=quantize,
                quant_bits=quant_bits,
                quant_mode=quant_mode,
                free_quantized_originals=True,
            )
            # Attach SloNetServer if provided, else build one from the guard
            if slonet_server is None and process_guard is not None:
                slonet_server = _server_from_provider(slonet_provider, process_guard)
            if slonet_server is not None and hasattr(slonet_provider, "set_server"):
                slonet_provider.set_server(slonet_server)
            register_provider("slonet-native", slonet_provider)
            text_provider_name = "slonet-native"
            logger.info(
                "Registered slonet-native provider from local .slnc: %s (quant=%s%s)",
                slonet_path,
                f"int{quant_bits}" if quantize else "none",
                ", server-backed" if slonet_server else "",
                extra={"tag": "MODEL"},
            )
        except Exception as e:
            logger.warning(
                "Failed to load slonet-native provider from %s: %s",
                slonet_path,
                e,
                extra={"tag": "MODEL"},
            )
    elif slonet_hf_id:
        try:
            from domain.inference._internal.slonet_provider import SloNetChatProvider
            from domain.infrastructure._internal.model_resolver import (
                get_model_dir as _get_model_dir,
            )

            _cache_dir = _get_model_dir(slonet_hf_id)
            _slnc = _cache_dir / "model.slnc"
            if not _slnc.exists():
                raise FileNotFoundError(f"No .slnc file for {slonet_hf_id} at {_slnc}")
            slonet_provider = SloNetChatProvider.from_slnc(
                str(_slnc),
                model_id=slonet_hf_id,
                quantize=quantize,
                quant_bits=quant_bits,
                quant_mode=quant_mode,
                free_quantized_originals=True,
            )
            # Attach SloNetServer if provided, else build one from the guard
            if slonet_server is None and process_guard is not None:
                slonet_server = _server_from_provider(slonet_provider, process_guard)
            if slonet_server is not None and hasattr(slonet_provider, "set_server"):
                slonet_provider.set_server(slonet_server)
            register_provider("slonet-native", slonet_provider)
            text_provider_name = "slonet-native"
            logger.info(
                "Registered slonet-native provider: %s (quant=%s%s)",
                slonet_hf_id,
                f"int{quant_bits}" if quantize else "none",
                ", server-backed" if slonet_server else "",
                extra={"tag": "MODEL"},
            )
        except Exception as e:
            logger.warning(
                "Failed to load slonet-native provider %s: %s",
                slonet_hf_id,
                e,
                extra={"tag": "MODEL"},
            )
    else:
        # Standalone SloNet mode: auto-detect a cached .slnc model
        default_model = None
        try:
            from domain.infrastructure._internal.config import get_config
            from domain.infrastructure._internal.model_resolver import (
                get_model_dir as _get_model_dir,
            )

            cfg = get_config()
            default_model = cfg.model.name if cfg.model.autoload else None
            if default_model:
                _cache_dir = _get_model_dir(default_model)
                _slnc = _cache_dir / "model.slnc"
                if _slnc.exists():
                    from domain.inference._internal.slonet_provider import SloNetChatProvider

                    auto_provider = SloNetChatProvider.from_slnc(
                        str(_slnc),
                        model_id=default_model,
                        quantize=quantize,
                        quant_bits=quant_bits,
                        quant_mode=quant_mode,
                        free_quantized_originals=True,
                    )
                    if slonet_server is not None and hasattr(auto_provider, "set_server"):
                        auto_provider.set_server(slonet_server)
                    register_provider("slonet-native", auto_provider)
                    text_provider_name = "slonet-native"
                    logger.info(
                        "Auto-detected slonet-native provider: %s",
                        default_model,
                        extra={"tag": "MODEL"},
                    )
        except Exception as e:
            logger.warning(
                "Auto-detect slonet-native failed for %s: %s",
                default_model or "unknown",
                e,
                extra={"tag": "MODEL"},
            )

    # Build default ProviderRouter with full processor pipeline
    existing = _providers.get("default")
    _is_slonet = existing is not None and type(existing).__name__ in (
        "SloTransformerProvider",
        "SloNetChatProvider",
    )
    # Rebuild the default router only when a text provider was successfully
    # registered, or when no default router exists yet. If the requested model
    # failed to load (text_provider_name is None) and a working default router
    # already exists, keep it — otherwise the failed load would clobber the
    # active model's router with an empty one, breaking inference.
    if not _is_slonet and (text_provider_name or existing is None):
        router = ProviderRouter()
        vision_proc = VisionProcessor("multimodal")
        tool_proc = ToolUseProcessor()
        personality_proc = PersonalityProcessor(traits=personality_traits)
        style_proc = StyleProcessor()
        router.add_processor(vision_proc)
        router.add_processor(tool_proc)
        router.add_processor(personality_proc)
        router.add_processor(style_proc)
        if text_provider_name:
            router.set_text_provider(text_provider_name)
        register_provider("default", router)
        # Register processors in the registry for lookup by routers
        register_processor("vision", vision_proc)
        register_processor("tool_use", tool_proc)
        register_processor("personality", personality_proc)
        register_processor("style", style_proc)
        logger.info(
            "Registered default router (processors=%s, text=%s)",
            [type(p).__name__ for p in router._processors],
            text_provider_name,
            extra={"tag": "MODEL"},
        )
    else:
        logger.info(
            "SloNet provider active as default — skipping router override", extra={"tag": "MODEL"}
        )


def update_personality_traits(traits: dict[str, float]) -> None:
    """Update processors in the default router with new soul traits.

    PersonalityProcessor receives all traits.
    StyleProcessor receives formality/directness if present in the trait dict.

    Called when a soul is switched to keep the processor pipeline in sync.
    """
    router = _providers.get("default")
    if router is None or not isinstance(router, ProviderRouter):
        return
    for proc in router._processors:
        if isinstance(proc, PersonalityProcessor):
            proc.set_traits(traits)
            logger.info(
                "Updated personality traits: %s", list(traits.keys()), extra={"tag": "MODEL"}
            )
        elif isinstance(proc, StyleProcessor):
            formality = traits.get("formality", 0.5)
            directness = traits.get("directness", 0.5)
            proc.set_style(formality=formality, directness=directness)
            logger.info(
                "Updated style: formality=%.2f directness=%.2f",
                formality,
                directness,
                extra={"tag": "MODEL"},
            )


__all__ = [
    # Types
    "ChatMessage",
    "ModelCapabilities",
    # Protocols
    "ModelProvider",
    "MessageProcessor",
    # Registries
    "register_provider",
    "get_provider",
    "list_providers",
    "clear_providers",
    "register_processor",
    "get_processor",
    "list_processors",
    "apply_processors",
    # Providers
    "SloTransformerProvider",
    # Processors
    "VisionProcessor",
    "KnowledgeProcessor",
    "ToolUseProcessor",
    "ToolDef",
    "PersonalityProcessor",
    "StyleProcessor",
    # Router
    "ProviderRouter",
    # Setup
    "setup_providers",
    "update_personality_traits",
]
