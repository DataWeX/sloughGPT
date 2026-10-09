"""Phoneme Router — dedicated endpoints for phoneme operations.

Phoneme encoding lives under voice (pronunciation is how the model speaks).
Uses domain.voice.PhonemeEngine as the single source of truth.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel, Field
from schemas.common import endpoint, raise_error, success_response

logger = logging.getLogger("slo.routers.phoneme")


class EncodeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to encode")
    language: str | None = Field(default=None, description="Language code hint")


class ScoreRequest(BaseModel):
    target: str = Field(..., description="Target pronunciation")
    spoken: str = Field(..., description="Spoken pronunciation to score")
    language: str | None = Field(default=None, description="Language code hint")


class BatchEncodeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100, description="Texts to encode")
    language: str | None = Field(default=None, description="Language code hint")


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")


class PhonemeRouter:
    """Dedicated phoneme API endpoints."""

    def __init__(self):
        self.router = APIRouter(prefix="/phoneme", tags=["phoneme"])
        self._engine = None
        self._register_routes()

    def _get_engine(self):
        if self._engine is None:
            from domain.voice import get_phoneme_engine

            self._engine = get_phoneme_engine()
        return self._engine

    def _register_routes(self):
        self.router.add_api_route("/encode", self.encode, methods=["POST"])
        self.router.add_api_route("/decode", self.decode, methods=["POST"])
        self.router.add_api_route("/visualize", self.visualize, methods=["POST"])
        self.router.add_api_route("/score", self.score, methods=["POST"])
        self.router.add_api_route("/batch-encode", self.batch_encode, methods=["POST"])
        self.router.add_api_route("/detect-language", self.detect_language, methods=["POST"])
        self.router.add_api_route("/synthesize", self.synthesize, methods=["POST"])
        self.router.add_api_route("/languages", self.languages, methods=["GET"])
        self.router.add_api_route("/status", self.status, methods=["GET"])

    @endpoint("phoneme.encode")
    async def encode(
        self, req: EncodeRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Encode text to phoneme IDs."""
        engine = self._get_engine()
        result = engine.encode(req.text, language=req.language)
        if not result.success:
            raise_error(result.error, "E_ENCODE_FAILED", status_code=500)
        return success_response(data=result.data)

    @endpoint("phoneme.decode")
    async def decode(
        self, ids: list[int], auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Decode phoneme IDs back to text."""
        engine = self._get_engine()
        result = engine.decode(ids)
        if not result.success:
            raise_error(result.error, "E_DECODE_FAILED", status_code=500)
        return success_response(data=result.data)

    @endpoint("phoneme.visualize")
    async def visualize(
        self, req: EncodeRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Get a visual representation of phoneme encoding."""
        engine = self._get_engine()
        result = engine.visualize(req.text, language=req.language)
        if not result.success:
            raise_error(result.error, "E_VISUALIZE_FAILED", status_code=500)
        return success_response(data=result.data)

    @endpoint("phoneme.score")
    async def score(
        self, req: ScoreRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Score pronunciation accuracy."""
        engine = self._get_engine()
        result = engine.score(req.target, req.spoken, language=req.language)
        if not result.success:
            raise_error(result.error, "E_SCORE_FAILED", status_code=500)
        return success_response(data=result.data)

    @endpoint("phoneme.batch_encode")
    async def batch_encode(
        self, req: BatchEncodeRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Encode multiple texts to phoneme IDs."""
        engine = self._get_engine()
        result = engine.batch_encode(req.texts, language=req.language)
        if not result.success:
            raise_error(result.error, "E_BATCH_ENCODE_FAILED", status_code=500)
        return success_response(data=result.data)

    @endpoint("phoneme.detect_language")
    async def detect_language(
        self, req: EncodeRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Detect the language of input text."""
        engine = self._get_engine()
        result = engine.detect_language(req.text)
        if not result.success:
            raise_error(result.error, "E_DETECT_FAILED", status_code=500)
        return success_response(data=result.data)

    @endpoint("phoneme.synthesize")
    async def synthesize(
        self, req: SynthesizeRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Synthesize text to speech via the voice engine."""
        engine = self._get_engine()
        result = engine.synthesize(req.text)
        if not result.success:
            raise_error(result.error, "E_SYNTHESIZE_FAILED", status_code=500)
        return success_response(data=result.data)

    @endpoint("phoneme.languages")
    async def languages(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """List supported languages."""
        engine = self._get_engine()
        result = engine.supported_languages()
        if not result.success:
            raise_error(result.error, "E_LANGUAGES_FAILED", status_code=500)
        return success_response(data=result.data)

    @endpoint("phoneme.status")
    async def status(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Get phoneme engine status."""
        engine = self._get_engine()
        return success_response(data=engine.status())


router = PhonemeRouter().router
