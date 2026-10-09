"""
Cloud Training Router — endpoints for cloud training management.

Delegates to TrainingEngine; does not import domain.training._internal.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from infrastructure.auth import require_auth_if_enabled
from schemas.common import classify_and_raise, endpoint, safe_audit_log, success_response

logger = logging.getLogger("slo.routers.cloud_training")


class CloudTrainingRouter:
    """Router for cloud training management."""

    def __init__(self):
        self.router = APIRouter(prefix="/cloud-training", tags=["cloud-training"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/jobs", self.list_jobs, methods=["GET"])
        self.router.add_api_route("/submit", self.submit_job, methods=["POST"])
        self.router.add_api_route("/{job_id}/status", self.job_status, methods=["GET"])
        self.router.add_api_route("/{job_id}/cancel", self.cancel_job, methods=["POST"])

    @staticmethod
    def _engine():
        from domain.training.engine import get_training_engine

        return get_training_engine()

    @endpoint("cloud_training.list_jobs")
    async def list_jobs(self, limit: int = 10) -> dict:
        """List recent cloud training jobs."""
        try:
            result = self._engine().list_cloud_jobs(limit=limit)
            if not result.success:
                raise RuntimeError(result.error or "list failed")
            return success_response(data=result.data)
        except Exception as e:
            classify_and_raise(e, source="cloud_training.list_jobs")

    @endpoint("cloud_training.submit")
    async def submit_job(
        self,
        provider: str = "local",
        dataset_id: str = "",
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Submit a cloud training job."""
        try:
            result = self._engine().submit_cloud_job(provider=provider, dataset_id=dataset_id)
            if not result.success:
                raise RuntimeError(result.error or "submit failed")
            safe_audit_log("cloud_training.submit", resource=dataset_id, detail=provider)
            return success_response(data=result.data)
        except Exception as e:
            classify_and_raise(e, source="cloud_training.submit")

    @endpoint("cloud_training.job_status")
    async def job_status(self, job_id: str) -> dict:
        """Get status of a cloud training job."""
        try:
            result = self._engine().cloud_job_status(job_id)
            if not result.success:
                raise RuntimeError(result.error or "status failed")
            return success_response(data=result.data)
        except Exception as e:
            classify_and_raise(e, source="cloud_training.job_status")

    @endpoint("cloud_training.cancel")
    async def cancel_job(
        self,
        job_id: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Cancel a cloud training job."""
        try:
            result = self._engine().cancel_cloud_job(job_id)
            if not result.success:
                raise RuntimeError(result.error or "cancel failed")
            safe_audit_log("cloud_training.cancel", resource=job_id)
            return success_response(data=result.data)
        except Exception as e:
            classify_and_raise(e, source="cloud_training.cancel")


router = CloudTrainingRouter().router
