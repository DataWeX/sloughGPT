"""
Status Router - Overall service health and info
"""
from fastapi import APIRouter
from datetime import datetime, timezone

from schemas.common import success_response


class StatusRouter:
    def __init__(self):
        self._start_time = datetime.now()
        self.router = APIRouter(tags=["status"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/status", self.get_status, methods=["GET"])
        self.router.add_api_route("/ready", self.ready, methods=["GET"])
        self.router.add_api_route("/live", self.live, methods=["GET"])

    async def get_status(self):
        """Get overall service status"""
        import psutil

        uptime = (datetime.now() - self._start_time).total_seconds()

        return success_response(data={
            "status": "healthy",
            "version": self._app_version(),
            "uptime_seconds": uptime,
            "timestamp": datetime.now(timezone.utc).timestamp(),
        })

    @staticmethod
    def _app_version() -> str:
        """Resolve the installed package version (with fallback)."""
        try:
            from importlib.metadata import version as _pkg_version
            return _pkg_version("sloughgpt")
        except Exception:
            return "unknown"

    async def ready(self):
        """Readiness check"""
        return success_response(data={"ready": True})

    async def live(self):
        """Liveness check"""
        return success_response(data={"alive": True})


router = StatusRouter().router
