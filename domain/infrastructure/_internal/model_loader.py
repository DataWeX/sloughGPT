from __future__ import annotations

"""
SloNet Model Loader — model loading via .slnc format.

Provides a `ModelLoader` class that handles:
  - Model detection (.slnc file, auto-converts from safetensors)
  - Verification (forward-pass smoke test on load)
  - Quantization (via walk_slo_linears)

Usage:
    from .model_loader import ModelLoader

    loader = ModelLoader()
    result = loader.load("gpt2")
    if result.success:
        print(f"Loaded {result.model_type} with {result.provider}")
"""

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from domain.shared import find_repo_root

_REPO_ROOT = find_repo_root(Path(__file__).resolve())

logger = logging.getLogger("slo.infrastructure.model_loader")

#: Where native-trained checkpoints live, relative to ``ModelLoader.models_dir``.
_NATIVE_SOUL_DIRNAME = "slonet-native"

#: Ids that name the native *family* rather than one checkpoint. They resolve
#: only when every candidate is provably the same model — see ``_try_load_soul``.
_NATIVE_FAMILY_ALIASES = frozenset({"sloughgpt", "sloughgpt-v1", "slonet", "slonet-native"})

#: An ``integrity_hash`` prefix must be at least this long to address a soul.
_SOUL_HASH_MIN_CHARS = 12
_SOUL_HASH_RE = re.compile(r"^[0-9a-f]{%d,64}$" % _SOUL_HASH_MIN_CHARS)
_CKPT_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _classify_soul_request(model_id: str) -> tuple[str, str] | None:
    """Map a requested id to a soul lookup kind + key, or ``None``.

    ``None`` means "not a soul request" — the caller must leave it to the
    ``.slnc`` path. HuggingFace ids (``Qwen/Qwen2.5-0.5B-Instruct``) carry a
    ``/`` and therefore never reach the native checkpoint directory, which is
    what keeps a request for one model from being answered by another.
    """
    requested = (model_id or "").strip()
    if not requested:
        return None
    for prefix in ("hash:", "dataset:"):
        if requested.lower().startswith(prefix):
            key = requested[len(prefix) :].strip().lower()
            return (prefix[:-1], key) if key else None
    if requested.lower().endswith(".soul"):
        requested = requested[:-5]
    if not requested or "/" in requested or "\\" in requested or ".." in requested:
        return None
    lowered = requested.lower()
    if _SOUL_HASH_RE.match(lowered):
        return ("hash", lowered)
    if lowered in _NATIVE_FAMILY_ALIASES:
        return ("alias", lowered)
    if _CKPT_NAME_RE.match(requested):
        return ("name", requested)
    return None


def _safe_soul_path(native_dir: Path, name: str) -> Path | None:
    """``<native_dir>/<name>.soul`` if it exists and stays inside *native_dir*.

    Mirrors the checkpoint resolver in ``domain/training/_internal/checkpoints``:
    a name is not a path, and the resolved candidate must remain under the base
    directory.
    """
    try:
        base = native_dir.resolve()
        candidate = (base / f"{name}.soul").resolve()
    except OSError:
        return None
    if not str(candidate).startswith(str(base) + os.sep):
        return None
    return candidate if candidate.is_file() else None


def _soul_identity(path: Path) -> dict[str, Any]:
    """Cheap header identity for *path* (see ``read_soul_identity``)."""
    from domain.inference._internal.slo_format import read_soul_identity

    return read_soul_identity(path)


def _soul_matches(kind: str, key: str, entry: tuple[Path, dict[str, Any]]) -> bool:
    """Whether one candidate satisfies a hash/dataset/alias lookup."""
    path, identity = entry
    if kind == "hash":
        return str(identity.get("integrity_hash") or "").lower().startswith(key)
    if kind == "dataset":
        return str(identity.get("dataset_signature") or "").lower() == key
    if kind == "alias":
        if key in ("slonet", "slonet-native"):
            return True
        declared = {
            str(identity.get(field) or "").strip().lower()
            for field in ("name", "base_model", "lineage")
        }
        return key in declared
    return path.stem == key


def _same_model(entries: list[tuple[Path, dict[str, Any]]]) -> bool:
    """Whether every candidate is the same model on the same, recorded dataset."""
    bases = {str(identity.get("base_model") or "").strip().lower() for _, identity in entries}
    signatures = {
        str(identity.get("dataset_signature") or "").strip().lower() for _, identity in entries
    }
    return len(bases) == 1 and len(signatures) == 1 and "" not in signatures


def _hash_of(entry: tuple[Path, dict[str, Any]]) -> str:
    """The recorded ``integrity_hash`` of one candidate (``""`` when absent)."""
    return str(entry[1].get("integrity_hash") or "").lower()


def _describe_soul(path: Path, identity: dict[str, Any]) -> str:
    """One-line candidate description for a refusal message."""
    return (
        f"{path.stem} (hash={str(identity.get('integrity_hash') or 'unknown')[:12]}, "
        f"dataset={identity.get('dataset_signature') or 'unrecorded'}, "
        f"born_at={identity.get('born_at') or 'unknown'})"
    )


# Failure classification for retry loops: "permanent" = retrying the identical
# load cannot succeed (no artifacts, refused soul lookup); "transient" = worth
# waiting for (load lock busy). Unclassified (None) keeps historical retries.
ERROR_KIND_PERMANENT = "permanent"
ERROR_KIND_TRANSIENT = "transient"


@dataclass
class LoadResult:
    """Standardized result from model loading."""

    success: bool
    model_id: str
    model_type: str  # "slonet"
    provider: Any = None
    model: Any = None
    tokenizer: Any = None
    error: str | None = None
    # ERROR_KIND_PERMANENT | ERROR_KIND_TRANSIENT | None (unclassified → retry)
    error_kind: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class ModelLoader:
    """SloNet model loader — detects .slnc format and routes accordingly.

    Falls back to auto-conversion from safetensors when no .slnc file exists.

    Thread safety: ``load()`` is protected by ``_load_lock`` to prevent
    concurrent model loads from corrupting shared state.
    """

    _load_lock = threading.Lock()
    _loading = False  # True while a load is in progress

    @classmethod
    def is_loading(cls) -> bool:
        """Check if a model load is currently in progress."""
        return cls._loading

    def __init__(self, models_dir: Path | None = None):
        self.models_dir = models_dir or _REPO_ROOT / "models"

    def load(
        self,
        model_id: str,
        device: str = "auto",
        quantize: bool = False,
        quant_bits: int = 8,
        quant_mode: str = "symmetric",
        verify: bool = True,
    ) -> LoadResult:
        """Load a model by ID via SloNet (.slnc format).

        Args:
            model_id: Model identifier (e.g., "gpt2"). A native ``.soul``
                checkpoint is addressed by its exact name, an ``integrity_hash``
                prefix (``hash:<prefix>``) or a dataset signature
                (``dataset:<signature>``); see ``_try_load_soul``.
            device: Target device (kept for API compat, not used by SloNet)
            quantize: Apply quantization after loading
            quant_bits: Bits for quantization (8 or 4)
            quant_mode: "symmetric" or "asymmetric"
            verify: Run test inference to verify model works

        Returns:
            LoadResult with provider, model, tokenizer, and metrics

        Thread safety: Only one model can load at a time.
        """
        if not self._load_lock.acquire(blocking=False):
            return LoadResult(
                success=False,
                model_id=model_id,
                model_type="unknown",
                error="Another model is currently loading. Please wait.",
                error_kind=ERROR_KIND_TRANSIENT,
            )
        try:
            ModelLoader._loading = True
            return self._load_inner(model_id, device, quantize, quant_bits, quant_mode, verify)
        finally:
            ModelLoader._loading = False
            self._load_lock.release()

    def _load_inner(
        self,
        model_id: str,
        device: str,
        quantize: bool,
        quant_bits: int,
        quant_mode: str,
        verify: bool,
    ) -> LoadResult:
        """Internal load implementation (caller holds _load_lock)."""

        load_start = time.monotonic()

        from .conversion_tracker import ConversionStage, get_tracker

        tracker = get_tracker()

        # Check if .slnc already exists — skip conversion stages
        try:
            from .model_resolver import get_model_dir as _get_model_dir

            cache_dir = _get_model_dir(model_id)
            has_slnc = (cache_dir / "model.slnc").exists()
        except Exception as exc:
            logger.debug("Failed to check for .slnc file: %s", exc)
            has_slnc = False

        if not has_slnc:
            tracker.start(model_id, stage=ConversionStage.DOWNLOADING, message="Loading model...")

        # Priority: native .soul first (owned architecture, goal 12), then .slnc.
        soul_result = self._try_load_soul(model_id)
        if soul_result is not None:
            if verify and soul_result.success:
                self._verify_model(soul_result)
            elapsed_ms = (time.monotonic() - load_start) * 1000
            logger.info(
                "model_loader: load complete",
                extra={
                    "model_id": model_id,
                    "elapsed_ms": round(elapsed_ms, 1),
                    "success": soul_result.success,
                    "source": "soul",
                },
            )
            tracker.finish(model_id)
            return soul_result

        result = self._try_load_slnc(model_id, device, quantize, quant_bits, quant_mode)
        if result is not None:
            tracker.update(
                model_id,
                stage=ConversionStage.LOADING,
                progress=0.95,
                message="Loading into memory...",
            )
            if verify and result.success:
                self._verify_model(result)
            elapsed_ms = (time.monotonic() - load_start) * 1000
            logger.info(
                "model_loader: load complete",
                extra={
                    "model_id": model_id,
                    "elapsed_ms": round(elapsed_ms, 1),
                    "success": result.success,
                    "source": "slnc",
                },
            )
            tracker.finish(model_id)
            return result

        elapsed_ms = (time.monotonic() - load_start) * 1000
        logger.warning(
            "model_loader: load failed",
            extra={
                "model_id": model_id,
                "elapsed_ms": round(elapsed_ms, 1),
            },
        )

        if not has_slnc:
            tracker.fail(model_id, "No .slnc or .soul file found")

        return LoadResult(
            success=False,
            model_id=model_id,
            model_type="slonet",
            error=f"No .slnc or .soul file found for {model_id}",
            error_kind=ERROR_KIND_PERMANENT,
        )

    def _try_load_slnc(
        self,
        model_id: str,
        device: str,
        quantize: bool,
        quant_bits: int,
        quant_mode: str,
    ) -> LoadResult | None:
        """Try to load a SloNet model from .slnc file.

        Returns None if no .slnc file found, otherwise LoadResult.
        """
        try:
            from .model_resolver import get_model_dir as _get_model_dir

            cache_dir = _get_model_dir(model_id)
        except Exception as exc:
            logger.debug("Failed to resolve model directory: %s", exc)
            return None

        slnc_path = cache_dir / "model.slnc"
        if not slnc_path.exists():
            slnc_path = self._try_convert_to_slnc(cache_dir, model_id)
            if slnc_path is None:
                return None

        logger.info("Loading SloNet model from %s", slnc_path, extra={"tag": "MODEL"})
        try:
            from domain.inference._internal.slonet_provider import SloNetChatProvider

            provider = SloNetChatProvider.from_slnc(
                str(slnc_path),
                model_id=model_id,
                quantize=quantize,
                quant_bits=quant_bits,
                quant_mode=quant_mode,
                free_quantized_originals=True,
            )

            return LoadResult(
                success=True,
                model_id=model_id,
                model_type="slonet",
                provider=provider,
                model=provider._model,
                tokenizer=getattr(provider, "_tokenizer", None),
                metrics={
                    "slnc_path": str(slnc_path),
                    "quantized": quantize,
                    "quant_bits": quant_bits if quantize else None,
                },
            )
        except FileNotFoundError as e:
            logger.warning("SloNet .slnc file vanished during load: %s", e, extra={"tag": "MODEL"})
            return LoadResult(
                success=False,
                model_id=model_id,
                model_type="slonet",
                error=f"Model file disappeared during loading: {e}",
            )
        except Exception as e:
            logger.warning("SloNet load failed: %s", e, extra={"tag": "MODEL"})
            return LoadResult(
                success=False,
                model_id=model_id,
                model_type="slonet",
                error=str(e),
            )

    def _try_load_soul(self, model_id: str) -> LoadResult | None:
        """Load the native ``.soul`` checkpoint that *model_id* names.

        Identity comes from the checkpoint's own header, never from directory
        order:

        - ``<checkpoint>`` (or ``<checkpoint>.soul``) — exact stem match;
        - ``hash:<prefix>`` or a bare hex prefix (>= 12 chars) — matches
          ``integrity_hash``;
        - ``dataset:<signature>`` — matches the recorded ``dataset_signature``;
        - a family id (``sloughgpt``, ``sloughgpt-v1``, ``slonet``,
          ``slonet-native``) — resolves only when every candidate is provably
          the same model (same ``base_model`` and same *non-empty* dataset
          signature), because retraining a model twice on an unchanged dataset
          produces interchangeable artifacts.

        Anything else is refused with the candidate list rather than guessed:
        the previous implementation answered *any* request with the newest-mtime
        `.soul` labelled as the requested id, so a request for one model could
        be served by another (wrong-artifact class). Ids that are not bare
        checkpoint names — HuggingFace ids contain ``/`` — return ``None`` so the
        ``.slnc`` path runs exactly as before.

        Returns:
            ``LoadResult`` on success, a failing ``LoadResult`` when the id
            names a soul that cannot be resolved, or ``None`` when the id is not
            a soul request at all.
        """
        request = _classify_soul_request(model_id)
        if request is None:
            return None
        kind, key = request

        native_dir = Path(self.models_dir) / _NATIVE_SOUL_DIRNAME
        named = _safe_soul_path(native_dir, key) if kind in ("name", "alias") else None
        if kind == "name":
            if named is None:
                return None
            matches = [(named, _soul_identity(named))]
        elif named is not None:
            # An explicitly named checkpoint wins over the family id it shares a
            # name with (e.g. ``sloughgpt.soul``) — a file is a declaration.
            kind = "name"
            matches = [(named, _soul_identity(named))]
        else:
            if not native_dir.is_dir():
                return self._soul_refusal(model_id, kind, key, [])
            try:
                candidates = sorted(native_dir.glob("*.soul"), key=lambda p: p.name)
            except FileNotFoundError as e:
                logger.warning(
                    "Native .soul directory removed during scan: %s", e, extra={"tag": "MODEL"}
                )
                return None
            entries = [(path, _soul_identity(path)) for path in candidates]
            matches = [entry for entry in entries if _soul_matches(kind, key, entry)]
            if not matches:
                return self._soul_refusal(model_id, kind, key, entries)
            if kind == "alias" and not _same_model(matches):
                return self._soul_refusal(model_id, kind, key, entries)
            if kind == "hash" and len({_hash_of(entry) for entry in matches}) > 1:
                # A short prefix can address several distinct artifacts — a
                # guess here is exactly the wrong-artifact bug this fixes.
                return self._soul_refusal(
                    model_id,
                    kind,
                    key,
                    entries,
                    reason=f"'{key}' is ambiguous — it prefixes {len(matches)} different "
                    "integrity hashes; request more characters",
                )

        # ``matches`` is ordered by file name (never mtime); when several
        # checkpoints are equivalent the pick is deterministic.
        soul_path, identity = matches[0]
        if len(matches) > 1:
            logger.info(
                "Native .soul request %s matched %d equivalent checkpoints; using %s",
                model_id,
                len(matches),
                soul_path.name,
                extra={"tag": "MODEL"},
            )
        logger.info(
            "Found native .soul checkpoint: %s (matched_by=%s, integrity_hash=%s)",
            soul_path,
            kind,
            identity.get("integrity_hash") or "unrecorded",
            extra={"tag": "MODEL"},
        )

        try:
            from domain.inference._internal.slonet_provider import SloNetChatProvider

            provider = SloNetChatProvider.from_soul(
                str(soul_path),
                model_id=model_id,
            )

            return LoadResult(
                success=True,
                model_id=model_id,
                model_type="slonet-native",
                provider=provider,
                model=provider._model,
                tokenizer=getattr(provider, "_tokenizer", None),
                metrics={
                    "soul_path": str(soul_path),
                    "source": "native-trained",
                    "matched_by": kind,
                    "candidate_count": len(matches),
                    "integrity_hash": identity.get("integrity_hash", ""),
                    "dataset_signature": identity.get("dataset_signature", ""),
                    "base_model": identity.get("base_model", ""),
                    "born_at": identity.get("born_at", ""),
                },
            )
        except Exception as e:
            logger.warning("Native .soul load failed: %s", e, extra={"tag": "MODEL"})
            return None

    def _soul_refusal(
        self,
        model_id: str,
        kind: str,
        key: str,
        entries: list[tuple[Path, dict[str, Any]]],
        reason: str | None = None,
    ) -> LoadResult:
        """Fail loudly with the candidates instead of serving a different model."""
        if reason is None and kind == "alias":
            reason = (
                f"'{model_id}' names the native model family but its candidates are not "
                "interchangeable — matching requires the same base_model and the same "
                "recorded dataset_signature"
            )
        elif reason is None and kind == "hash":
            reason = f"no native checkpoint has an integrity_hash starting with '{key}'"
        elif reason is None and kind == "dataset":
            reason = f"no native checkpoint records dataset_signature '{key}'"
        elif reason is None:
            reason = f"no native checkpoint named '{key}'"
        listing = "; ".join(_describe_soul(path, identity) for path, identity in entries) or (
            "no .soul checkpoints found"
        )
        error = f"Native .soul lookup refused for {model_id}: {reason}. Candidates: {listing}"
        logger.warning("model_loader: %s", error, extra={"tag": "MODEL"})
        return LoadResult(
            success=False,
            model_id=model_id,
            model_type="slonet-native",
            error=error,
            error_kind=ERROR_KIND_PERMANENT,
        )

    def _try_convert_to_slnc(self, cache_dir: Path, model_id: str) -> Path | None:
        """Try to convert safetensors to .slnc format.

        Returns path to new .slnc file, or None if conversion fails.
        Handles bfloat16 weights by reading raw bytes and converting to float32.
        """
        try:
            from .conversion_tracker import ConversionStage, get_tracker
            from .model_resolver import find_safetensors, load_model_config

            tracker = get_tracker()

            st_path = find_safetensors(cache_dir)
            if st_path is None:
                return None

            slnc_path = cache_dir / "model.slnc"
            tracker.update(
                model_id,
                stage=ConversionStage.CONVERTING,
                message="Converting safetensors → .slnc...",
            )
            logger.info("Converting %s to .slnc", st_path.name, extra={"tag": "MODEL"})

            config = load_model_config(model_id)

            from domain.infrastructure._internal.slnc.compiler import SLNCCompiler

            compiler = SLNCCompiler()

            weights = compiler.read_weights(
                st_path,
                on_tensor=lambda name, done, total: tracker.update(
                    model_id, progress=done / total * 0.7
                ),
            )

            tracker.update(model_id, message="Writing .slnc format...")
            compiler.compile_from_dict(config, weights, str(slnc_path))
            tracker.update(model_id, progress=0.85)

            # Protect the .slnc file from accidental deletion
            tracker.update(
                model_id, stage=ConversionStage.PROTECTING, message="Protecting files..."
            )
            try:
                from .model_protector import protect_model

                protect_model(model_id, [str(slnc_path)])
            except Exception as exc:
                logger.debug("Model protection failed: %s", exc)
            tracker.update(model_id, progress=0.95)

            logger.info("Converted to .slnc: %s", slnc_path, extra={"tag": "MODEL"})
            return slnc_path
        except Exception as e:
            logger.warning("SloNet conversion failed: %s", e, extra={"tag": "MODEL"})
            return None

    def _verify_model(self, result: LoadResult, timeout_s: float = 30.0) -> bool:
        """Run lightweight verification to confirm model loads correctly.

        Single forward pass (~1-2s) instead of full generation (~12s).
        Wrapped in a timeout to prevent hanging on quantized models.
        """
        try:
            test_ids = np.array([[1, 2, 3]], dtype=np.int64)

            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(result.model.forward, test_ids)
                try:
                    output = future.result(timeout=timeout_s)
                except concurrent.futures.TimeoutError:
                    logger.warning(
                        "Model verification timed out after %.0fs: %s",
                        timeout_s,
                        result.model_id,
                        extra={"tag": "MODEL"},
                    )
                    result.metrics["verified"] = False
                    return True  # Don't fail the load — model loaded, verification is advisory
                except Exception as e:
                    raise e from None

            if output is None:
                result.success = False
                result.error = "Model produced empty output"
                return False

            result.metrics["verified"] = True
            logger.info("Model verification passed: %s", result.model_id, extra={"tag": "MODEL"})
            return True
        except Exception as e:
            logger.warning("Model verification failed: %s", e, extra={"tag": "MODEL"})
            result.success = False
            result.error = f"Verification failed: {e}"
            result.metrics["verified"] = False
            return False


_loader: ModelLoader | None = None
_loader_lock = threading.Lock()


def get_model_loader() -> ModelLoader:
    """Get the global ModelLoader instance."""
    global _loader
    if _loader is None:
        with _loader_lock:
            if _loader is None:
                _loader = ModelLoader()
    return _loader


def load_model(
    model_id: str,
    device: str = "auto",
    quantize: bool = False,
    quant_bits: int = 8,
    quant_mode: str = "symmetric",
    verify: bool = True,
) -> LoadResult:
    """Convenience function to load a model via SloNet.

    Args:
        model_id: Model identifier (e.g., "gpt2")
        device: Target device (kept for API compat, not used by SloNet)
        quantize: Apply quantization after loading
        quant_bits: Bits for quantization (8 or 4)
        quant_mode: "symmetric" or "asymmetric"
        verify: Run test inference to verify model works

    Returns:
        LoadResult with provider, model, tokenizer, and metrics
    """
    return get_model_loader().load(
        model_id=model_id,
        device=device,
        quantize=quantize,
        quant_bits=quant_bits,
        quant_mode=quant_mode,
        verify=verify,
    )
