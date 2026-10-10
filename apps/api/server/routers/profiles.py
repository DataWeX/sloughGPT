"""
Profiles Router — serving profile management endpoints.

GET  /profiles              — list all profiles
GET  /profiles/{id}         — get a single profile
POST /profiles/apply        — apply a profile (body: {"profile_id": "..."})
GET  /profiles/active       — get the currently active profile
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from schemas.common import endpoint, success_response

from domain.infrastructure.serving_profiles import (
    apply_profile,
    detect_recommended_profile,
    get_active_profile_id,
    get_profile,
    list_profiles,
)

logger = logging.getLogger("slo.routers.profiles")

# GPU presence is a hardware constant for the life of the process, but
# /profiles/recommend probed it on every request. nvidia-smi costs a
# subprocess spawn (~50-300ms) even when it answers instantly, so memoise the
# answer. A hot-plugged GPU needs a restart to be seen, which matches how the
# rest of the stack enumerates hardware at import.
#
# None means "not probed yet". Only a definitive answer is stored — see the
# call site: a timeout must not pin False for the process lifetime.
_GPU_PRESENT: bool | None = None


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
        try:
            result = apply_profile(req.profile_id)
            return success_response(data=result)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @endpoint("profiles.active")
    async def get_active_profile(self) -> dict:
        """Get the currently active profile ID."""
        profile_id = get_active_profile_id()
        p = get_profile(profile_id)
        return success_response(
            data={
                "active_profile_id": profile_id,
                "profile": p.to_dict() if p else None,
            }
        )

    @endpoint("profiles.recommend")
    async def recommend_profile(self) -> dict:
        """Auto-detect and recommend the best profile for this hardware."""
        import multiprocessing

        try:
            import psutil

            ram_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            ram_gb = multiprocessing.cpu_count() * 4  # rough estimate

        global _GPU_PRESENT

        has_gpu = _GPU_PRESENT
        if has_gpu is None:
            try:
                import subprocess

                # Blocking subprocess — thread it so a 5s nvidia-smi timeout
                # cannot stall the single uvicorn event loop.
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                has_gpu = bool(result.stdout.strip())
                # Memoise only the definitive answer. A timeout is transient
                # and must not pin "no GPU" for the life of the process, so
                # the exception path deliberately leaves _GPU_PRESENT as None.
                _GPU_PRESENT = has_gpu
            except Exception:
                logger.debug("GPU detection failed (nvidia-smi not available)")
                has_gpu = False

        recommended = detect_recommended_profile(ram_gb, has_gpu)
        return success_response(
            data={
                "recommended_profile_id": recommended,
                "detected_ram_gb": round(ram_gb, 1),
                "has_gpu": has_gpu,
            }
        )


router = ProfilesRouter().router
