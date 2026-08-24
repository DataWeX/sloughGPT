"""
Status Router - Overall service health and info
"""
from fastapi import APIRouter
from datetime import datetime, timezone

from schemas.common import success_response, classify_and_raise


class StatusRouter:
    def __init__(self):
        self._start_time = datetime.now()
        self.router = APIRouter(tags=["status"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/status", self.get_status, methods=["GET"])
        self.router.add_api_route("/ready", self.ready, methods=["GET"])
        self.router.add_api_route("/live", self.live, methods=["GET"])

    async def get_status(self) -> dict:
        """Return overall service health status with uptime and timestamp."""
        try:
            uptime = (datetime.now() - self._start_time).total_seconds()
            return success_response(data={
                "status": "healthy",
                "uptime_seconds": uptime,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            classify_and_raise(e, source="status.get")

    async def ready(self) -> dict:
        """Kubernetes-style readiness probe.

        Returns ready=True when the service can accept traffic.

        Returns:
            Success envelope with ready: True.
        """
        return success_response(data={"ready": True})

    async def live(self) -> dict:
        """Kubernetes-style liveness probe.

        Returns alive=True when the process is running and responsive.

        Returns:
            Success envelope with alive: True.
        """
        return success_response(data={"alive": True})


router = StatusRouter().router
