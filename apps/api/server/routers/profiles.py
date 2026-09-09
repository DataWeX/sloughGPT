"""
Profiles Router — serving profile management endpoints.

GET  /profiles              — list all profiles
GET  /profiles/{id}         — get a single profile
POST /profiles/apply        — apply a profile (body: {"profile_id": "..."})
GET  /profiles/active       — get the currently active profile
"""

from __future__ import annotations

import logging

from domains.infrastructure.serving_profiles import (
    apply_profile,
    detect_recommended_profile,
    get_active_profile_id,
    get_profile,
    list_profiles,
)
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from schemas.common import endpoint, success_response

logger = logging.getLogger("slo.routers.profiles")


class ApplyProfileRequest(BaseModel):
    profile_id: str = Field(..., min_length=1, max_length=50)


class ProfilesRouter:
    def __init__(self) -> None:
        self.router = APIRouter(prefix="/profiles", tags=["profiles"])
        self._register_routes()

    def _register_routes(self) -> None:
        # Fixed routes FIRST — before parameterized /{profile_id}
        self.router.add_api_route("/apply", self.apply_profile, methods=["POST"])
        self.router.add_api_route("/active", self.get_active_profile, methods=["GET"])
        self.router.add_api_route("/recommend", self.recommend_profile, methods=["GET"])
        self.router.add_api_route("", self.list_profiles, methods=["GET"])
        self.router.add_api_route("/{profile_id}", self.get_profile, methods=["GET"])

    @endpoint("profiles.list")
    async def list_profiles(self) -> dict:
        """List all available serving profiles."""
        return success_response(data=list_profiles())

    @endpoint("profiles.get")
    async def get_profile(self, profile_id: str) -> dict:
        """Get a single profile by ID."""
        p = get_profile(profile_id)
        if p is None:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id!r} not found")
        return success_response(data=p.to_dict())

    @endpoint("profiles.apply")
    async def apply_profile(self, req: ApplyProfileRequest) -> dict:
        """Apply a serving profile to the running process."""
        result = apply_profile(req.profile_id)
        return success_response(data=result)

    @endpoint("profiles.active")
    async def get_active_profile(self) -> dict:
        """Get the currently active profile ID."""
        profile_id = get_active_profile_id()
        p = get_profile(profile_id)
        return success_response(data={
            "active_profile_id": profile_id,
            "profile": p.to_dict() if p else None,
        })

    @endpoint("profiles.recommend")
    async def recommend_profile(self) -> dict:
        """Auto-detect and recommend the best profile for this hardware."""
        import multiprocessing

        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            ram_gb = multiprocessing.cpu_count() * 4  # rough estimate

        has_gpu = False
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            has_gpu = bool(result.stdout.strip())
        except Exception:
            logger.debug("GPU detection failed (nvidia-smi not available)")

        recommended = detect_recommended_profile(ram_gb, has_gpu)
        return success_response(data={
            "recommended_profile_id": recommended,
            "detected_ram_gb": round(ram_gb, 1),
            "has_gpu": has_gpu,
        })


router = ProfilesRouter().router
