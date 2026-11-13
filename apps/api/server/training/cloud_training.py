"""
Cloud Training Router — endpoints for cloud training management.

Delegates to TrainingEngine; does not import domain.training._internal.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from fastapi import APIRouter, Depends
from infrastructure.auth import require_auth_if_enabled
from infrastructure.contract import RouteSpec, create_router
from schemas.common import classify_and_raise, endpoint, safe_audit_log, success_response

from domain.agents import ToolParam, ToolSpec

logger = logging.getLogger("slo.training.cloud_training")


class CloudTrainingRouter:
    """Router for cloud training management."""

    # One projection per process (contract registry: same identity
    # re-registering = double projection). Later instances — test fixtures —
    # share the router the first instance registered.
    _shared_router: ClassVar[APIRouter | None] = None

    def __init__(self):
        if CloudTrainingRouter._shared_router is not None:
            self.router = CloudTrainingRouter._shared_router
            return
        # Descriptor projection (card 20261101_154058): full paths in the
        # RouteSpecs — no prefix, byte-identical paths on direct mount.
        self.router = APIRouter()
        self._register_routes()
        CloudTrainingRouter._shared_router = self.router

    def _register_routes(self):
        # envelope=False: handlers keep their success_response wraps; submit is
        # query-carried (the hand-built twin's plain defaults + tests' ?params).
        self.router.include_router(
            create_router(
                ToolSpec(
                    name="training.cloud_training.list_jobs",
                    description="List recent cloud training jobs.",
                    parameters=[ToolParam("limit", "integer", "", False)],
                    execute=self.list_jobs,
                    auth_scope="public",
                    idempotent=True,
                    version="1",
                ),
                RouteSpec(
                    path="/cloud-training/jobs",
                    envelope=False,
                    summary="List Cloud Training Jobs",
                    tags=["cloud-training"],
                ),
                self.list_jobs,
            )
        )
        self.router.include_router(
            create_router(
                ToolSpec(
                    name="training.cloud_training.submit",
                    description="Submit a cloud training job.",
                    parameters=[
                        ToolParam("provider", "string", "", False),
                        ToolParam("dataset_id", "string", "", False),
                    ],
                    execute=self.submit_job,
                    auth_scope="authenticated",
                    idempotent=False,
                    version="1",
                ),
                RouteSpec(
                    path="/cloud-training/submit",
                    input="query",
                    envelope=False,
                    summary="Submit Cloud Training Job",
                    tags=["cloud-training"],
                ),
                self.submit_job,
            )
        )
        self.router.include_router(
            create_router(
                ToolSpec(
                    name="training.cloud_training.job_status",
                    description="Get status of a cloud training job.",
                    parameters=[ToolParam("job_id", "string", "", True)],
                    execute=self.job_status,
                    auth_scope="public",
                    idempotent=True,
                    version="1",
                ),
                RouteSpec(
                    path="/cloud-training/{job_id}/status",
                    envelope=False,
                    summary="Get Cloud Training Job Status",
                    tags=["cloud-training"],
                ),
                self.job_status,
            )
        )
        self.router.include_router(
            create_router(
                ToolSpec(
                    name="training.cloud_training.cancel",
                    description="Cancel a cloud training job.",
                    parameters=[ToolParam("job_id", "string", "", True)],
                    execute=self.cancel_job,
                    auth_scope="authenticated",
                    idempotent=False,
                    version="1",
                ),
                RouteSpec(
                    path="/cloud-training/{job_id}/cancel",
                    envelope=False,
                    summary="Cancel Cloud Training Job",
                    tags=["cloud-training"],
                ),
                self.cancel_job,
            )
        )

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
