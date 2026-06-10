"""
VLM Router — Vision-Language Model inference + DPO training endpoints.

Provides:
- ``POST /vlm/generate`` — image-conditioned text generation
- ``POST /vlm/dpo`` — trigger DPO training on the active HF model
- ``GET /vlm/status`` — VLM and DPO status

DPO training uses feedback from the chat (thumbs up/down) to fine-tune
the active HuggingFace model (Qwen) with LoRA adapters.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Optional

import torch
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

logger = logging.getLogger("man.vlm_router")

router = APIRouter(prefix="/vlm", tags=["vlm"])

# ── Global state ───────────────────────────────────────────────────
_vlm_inference = None
_vlm_lock = Lock()

_dpo_state = {
    "last_run": None,
    "status": "idle",
    "result": None,
    "accepted_count": 0,
    "rejected_count": 0,
}
_dpo_lock = Lock()


# ── Schemas ────────────────────────────────────────────────────────

class VLMGenerateRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded JPEG/PNG image")
    prompt: str = Field("Describe this image in detail.", description="Text prompt")
    max_new_tokens: int = Field(128, ge=1, le=1024)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)


class VLMGenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    elapsed_ms: float


class DPOTriggerRequest(BaseModel):
    max_pairs: int = Field(6, ge=1, le=50)
    learning_rate: float = Field(5e-6, ge=1e-8, le=1e-3)


class DPOStatusResponse(BaseModel):
    status: str
    last_run: Optional[str] = None
    result: Optional[dict] = None
    accepted_count: int = 0
    rejected_count: int = 0


class VLMLoadRequest(BaseModel):
    model_dir: str = Field("models/vlm-finetuned", description="Path to VLM training output directory")


class VLMTrainRequest(BaseModel):
    data_path: str = Field(..., description="Path to JSONL with image-text pairs")
    stage1_epochs: int = Field(1, ge=1, le=10)
    stage2_epochs: int = Field(2, ge=0, le=10)
    batch_size: int = Field(4, ge=1, le=32)
    learning_rate: float = Field(1e-3, ge=1e-6, le=1.0)
    lora_rank: int = Field(8, ge=1, le=64)
    output_dir: str = Field("models/vlm-finetuned", description="Output directory")


class PDFAnalysisRequest(BaseModel):
    pdf_path: str = Field(..., description="Path to the PDF file on the server")
    question: str = Field("Summarize this document.", description="Question or instruction about the document")
    per_page: bool = Field(False, description="Analyze each page individually")
    max_new_tokens: int = Field(512, ge=64, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


_TRAINING_STATE = {
    "status": "idle",
    "job_id": None,
    "progress": None,
    "result": None,
    "error": None,
}
_training_lock = Lock()


# ── Helpers ────────────────────────────────────────────────────────

def _get_vlm_inference():
    """Get the VLM inference engine (lazy-init once)."""
    global _vlm_inference
    if _vlm_inference is not None:
        return _vlm_inference
    from domains.inference.vlm_inference import VLMInference
    with _vlm_lock:
        if _vlm_inference is not None:
            return _vlm_inference
        try:
            _vlm_inference = VLMInference()
        except FileNotFoundError:
            logger.info("VLM not loaded (no trained checkpoint found)")
        except Exception as e:
            logger.warning("Failed to load VLM inference: %s", e)
    return _vlm_inference


def _get_active_model_and_tokenizer():
    """Get the currently loaded HF model and tokenizer from server state."""
    try:
        import state as server_state
        return server_state.model, server_state.tokenizer
    except Exception:
        return None, None


# ── Endpoints ──────────────────────────────────────────────────────

@router.post("/generate")
async def vlm_generate(req: VLMGenerateRequest):
    """Generate text conditioned on an image using the trained VLM.

    Decodes the base64 image, runs it through SigLIP → connector → Qwen,
    and returns generated text. Falls back to 400 if VLM not loaded.
    """
    vlm = _get_vlm_inference()
    if vlm is None:
        raise HTTPException(status_code=400, detail="VLM not loaded. Train a VLM model first.")

    try:
        import base64
        image_data = base64.b64decode(req.image_base64)
        from PIL import Image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        t0 = time.time()
        text = vlm.generate(
            image,
            text=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        elapsed = (time.time() - t0) * 1000

        return VLMGenerateResponse(
            text=text,
            tokens_generated=len(text.split()),
            elapsed_ms=round(elapsed, 1),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("VLM generate failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dpo")
async def trigger_dpo(req: DPOTriggerRequest):
    """Trigger DPO training on the active HF model using feedback pairs.

    Loads (chosen, rejected, prompt) pairs from the feedback database,
    applies gradient descent on chosen + gradient ascent on rejected,
    and runs a quality guard (PPL benchmark → rollback if >5% degradation).

    The training targets LoRA parameters if LoRA is enabled on the model,
    or all trainable parameters otherwise.
    """
    global _dpo_state

    model, tokenizer = _get_active_model_and_tokenizer()
    if model is None or tokenizer is None:
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")

    with _dpo_lock:
        if _dpo_state["status"] == "running":
            raise HTTPException(status_code=409, detail="DPO training already in progress")
        _dpo_state["status"] = "running"
        _dpo_state["result"] = None

    try:
        from domains.feedback.hf_dpo import HFDPOTrainer

        trainer = HFDPOTrainer(
            model=model,
            tokenizer=tokenizer,
            learning_rate=req.learning_rate,
        )

        t0 = time.time()
        result = trainer.train(max_pairs=req.max_pairs)
        elapsed = time.time() - t0

        result["elapsed_seconds"] = round(elapsed, 1)

        with _dpo_lock:
            _dpo_state["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            _dpo_state["result"] = result
            _dpo_state["status"] = result["status"]
            if result["status"] == "accepted":
                _dpo_state["accepted_count"] += 1
            elif result["status"] == "rejected":
                _dpo_state["rejected_count"] += 1

        logger.info("DPO completed: %s (%ds)", result["status"], round(elapsed))

        return {
            "status": result["status"],
            "steps": result.get("steps", 0),
            "avg_loss": result.get("avg_loss"),
            "ppl_before": result.get("ppl_before"),
            "ppl_after": result.get("ppl_after"),
            "ppl_delta_pct": result.get("ppl_delta_pct"),
            "pairs_trained": result.get("pairs_trained", 0),
            "elapsed_seconds": round(elapsed, 1),
        }

    except Exception as e:
        with _dpo_lock:
            _dpo_state["status"] = "error"
            _dpo_state["result"] = {"error": str(e)}
        logger.error("DPO failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dpo/status")
async def dpo_status():
    """Get DPO training status and history."""
    with _dpo_lock:
        return DPOStatusResponse(
            status=_dpo_state["status"],
            last_run=_dpo_state["last_run"],
            result=_dpo_state["result"],
            accepted_count=_dpo_state["accepted_count"],
            rejected_count=_dpo_state["rejected_count"],
        )


@router.post("/load")
async def load_vlm(req: VLMLoadRequest):
    """Load a trained VLM checkpoint for inference and chat.

    Loads from the directory produced by ``VLMTrainer.train()``,
    containing ``vlm_config.json``, ``connector.pt``, and ``final/``.

    Registers the VLM both as:
    - ``_vlm_inference`` for ``/vlm/generate`` endpoint
    - ``VLMProvider`` for chat via the default provider router
    """
    global _vlm_inference
    try:
        from domains.inference.vlm_inference import VLMInference
        vlm = VLMInference(model_dir=req.model_dir)
        with _vlm_lock:
            _vlm_inference = vlm

        # Also register as a chat provider
        try:
            from domains.models.provider import load_vlm_provider, register_provider, get_provider
            provider = load_vlm_provider(req.model_dir, model_id_str="vlm")
            register_provider("vlm", provider)
            default_router = get_provider("default")
            if default_router is not None and hasattr(default_router, "set_text_provider"):
                default_router.set_text_provider("vlm")
            logger.info("VLM registered as chat provider from %s", req.model_dir)
        except Exception as pe:
            logger.warning("Failed to register VLM chat provider: %s", pe)

        logger.info("VLM loaded from %s", req.model_dir)
        return {"status": "loaded", "model_dir": req.model_dir}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pdf")
async def analyze_pdf(req: PDFAnalysisRequest):
    """Analyze a PDF document using VLM.

    Extracts text and renders page images from a PDF, then feeds
    both into the VLM engine for analysis.

    Args:
        req.pdf_path: Path to the PDF file on the server.
        req.question: Question about the document.
        req.per_page: If true, analyze each page individually.
        req.max_new_tokens: Max tokens for the response.
        req.temperature: Sampling temperature.

    Returns:
        Analysis text (or per-page results).
    """
    from pathlib import Path as _Path
    if not _Path(req.pdf_path).exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {req.pdf_path}")
    try:
        from domains.inference.pdf_vlm import PDFVLMProcessor
        p = PDFVLMProcessor(max_pages=10)
        if req.per_page:
            results = p.analyze_pages(
                req.pdf_path,
                question=req.question,
                max_new_tokens=min(req.max_new_tokens, 256),
                temperature=req.temperature,
            )
            return {"status": "ok", "pages": results}
        else:
            text = p.analyze(
                req.pdf_path,
                question=req.question,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
            )
            return {"status": "ok", "analysis": text}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="VLM checkpoint not found. Train a VLM model first via /vlm/train.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pdf/upload")
async def analyze_pdf_upload(
    file: UploadFile = File(..., description="PDF file to analyze"),
    question: str = Form("Summarize this document."),
    per_page: bool = Form(False),
    max_new_tokens: int = Form(512, ge=64, le=2048),
    temperature: float = Form(0.7, ge=0.0, le=2.0),
):
    """Upload a PDF file for VLM analysis.

    Accepts multipart file upload. Saves temporarily, analyzes, then removes.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    tmp_path = Path(f"/tmp/vlm_pdf_{int(time.time())}_{file.filename}")
    try:
        content = await file.read()
        tmp_path.write_bytes(content)

        from domains.inference.pdf_vlm import PDFVLMProcessor
        from pathlib import Path as _P
        if not _P(tmp_path).exists():
            raise HTTPException(status_code=400, detail="Failed to save uploaded PDF")
        p = PDFVLMProcessor(max_pages=10)
        if per_page:
            results = p.analyze_pages(
                str(tmp_path),
                question=question,
                max_new_tokens=min(max_new_tokens, 256),
                temperature=temperature,
            )
            return {"status": "ok", "pages": results}
        else:
            text = p.analyze(
                str(tmp_path),
                question=question,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            return {"status": "ok", "analysis": text}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="VLM checkpoint not found. Train a VLM model first via /vlm/train.")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@router.post("/train")
async def train_vlm(req: VLMTrainRequest):
    """Start VLM training in background. Returns job ID and status.

    Training runs in a daemon thread. Poll ``GET /vlm/train/status``
    for progress.

    The dataset at ``data_path`` must be a JSONL file with:
      ``{"image_path": "...", "caption": "..."}``
    or
      ``{"image_path": "...", "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}``
    """
    with _training_lock:
        if _TRAINING_STATE["status"] == "running":
            raise HTTPException(status_code=409, detail="Training already in progress")
        _TRAINING_STATE["status"] = "running"
        _TRAINING_STATE["error"] = None
        _TRAINING_STATE["result"] = None

    job_id = f"vlm_{int(time.time())}"

    def _run():
        try:
            from domains.training.multimodal import VLMTrainer, VLMConfig

            config = VLMConfig(
                stage1_epochs=req.stage1_epochs,
                stage2_epochs=req.stage2_epochs,
                batch_size=req.batch_size,
                stage1_lr=req.learning_rate,
                stage2_lr=req.learning_rate * 0.02,
                lora_rank=req.lora_rank,
                output_dir=req.output_dir,
            )
            trainer = VLMTrainer(config)
            result = trainer.train(data_path=req.data_path)
            with _training_lock:
                _TRAINING_STATE["status"] = "completed" if result.get("status") == "completed" else "error"
                _TRAINING_STATE["result"] = result
                _TRAINING_STATE["job_id"] = job_id
            logger.info("VLM training %s: %s", result.get("status"), req.data_path)
        except Exception as e:
            with _training_lock:
                _TRAINING_STATE["status"] = "error"
                _TRAINING_STATE["error"] = str(e)
            logger.error("VLM training failed: %s", e)

    import threading
    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return {
        "status": "started",
        "job_id": job_id,
        "data_path": req.data_path,
        "output_dir": req.output_dir,
    }


@router.get("/train/status")
async def train_status():
    """Get VLM training status."""
    with _training_lock:
        return {
            "status": _TRAINING_STATE["status"],
            "job_id": _TRAINING_STATE["job_id"],
            "progress": _TRAINING_STATE["progress"],
            "result": _TRAINING_STATE["result"],
            "error": _TRAINING_STATE["error"],
        }


@router.get("/status")
async def vlm_status():
    """Get VLM and DPO status (no model loading triggered)."""
    with _training_lock:
        train_state = {
            "status": _TRAINING_STATE["status"],
            "job_id": _TRAINING_STATE["job_id"],
            "result": _TRAINING_STATE["result"],
            "error": _TRAINING_STATE["error"],
        }
    return {
        "vlm_loaded": _vlm_inference is not None,
        "training": train_state,
        "dpo": DPOStatusResponse(
            status=_dpo_state["status"],
            last_run=_dpo_state["last_run"],
            result=_dpo_state["result"],
            accepted_count=_dpo_state["accepted_count"],
            rejected_count=_dpo_state["rejected_count"],
        ).model_dump(),
    }
