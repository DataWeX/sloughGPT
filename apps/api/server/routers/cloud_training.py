"""
Cloud Training Router — endpoints for cloud training management.
"""

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
        self.router.add_api_route(
            "/jobs", self.list_jobs, methods=["GET"]
        )
        self.router.add_api_route(
            "/submit", self.submit_job, methods=["POST"]
        )
        self.router.add_api_route(
            "/{job_id}/status", self.job_status, methods=["GET"]
        )
        self.router.add_api_route(
            "/{job_id}/cancel", self.cancel_job, methods=["POST"]
        )

    @endpoint("cloud_training.list_jobs")
    async def list_jobs(self, limit: int = 10) -> dict:
        """List recent cloud training jobs."""
        try:
            from domains.training.cloud import get_provider
            provider = get_provider("local")
            jobs = provider.list_jobs(limit)
            return success_response(data={
                "jobs": [
                    {
                        "job_id": j.job_id,
                        "provider": j.provider,
                        "status": j.status,
                        "progress": j.progress,
                        "error": j.error,
                    }
                    for j in jobs
                ]
            })
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
            from domains.training.cloud import get_provider, CloudTrainingConfig
            p = get_provider(provider)
            config = CloudTrainingConfig(provider=provider)
            job_id = p.submit_job(config, dataset_id, "train.py", {})
            safe_audit_log("cloud_training.submit", resource=dataset_id, detail=provider)
            return success_response(data={
                "job_id": job_id,
                "provider": provider,
                "status": "submitted",
            })
        except Exception as e:
            classify_and_raise(e, source="cloud_training.submit")

    @endpoint("cloud_training.job_status")
    async def job_status(self, job_id: str) -> dict:
        """Get status of a cloud training job."""
        try:
            from domains.training.cloud import get_provider
            provider = get_provider("local")
            status = provider.get_status(job_id)
            return success_response(data={
                "job_id": status.job_id,
                "provider": status.provider,
                "status": status.status,
                "progress": status.progress,
                "error": status.error,
            })
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
            from domains.training.cloud import get_provider
            provider = get_provider("local")
            cancelled = provider.cancel_job(job_id)
            safe_audit_log("cloud_training.cancel", resource=job_id)
            return success_response(data={
                "job_id": job_id,
                "cancelled": cancelled,
            })
        except Exception as e:
            classify_and_raise(e, source="cloud_training.cancel")


router = CloudTrainingRouter().router
