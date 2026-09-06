"""
Status Router - Overall service health and info
"""

from datetime import datetime, timezone

from fastapi import APIRouter
from schemas.common import classify_and_raise, success_response


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
            return success_response(
                data={
                    "status": "healthy",
                    "uptime_seconds": uptime,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        except Exception as e:
            classify_and_raise(e, source="status.get")

    async def ready(self) -> dict:
        try:
            """Kubernetes-style readiness probe.

            Returns ready=True when the service can accept traffic.
            Checks that critical subsystems are initialized.
            """
            checks = {}
            # Check database connectivity
            try:
                from domains.feedback.database import get_feedback_db

                db = get_feedback_db()
                checks["database"] = db is not None
            except Exception:
                checks["database"] = False

            # Check inference engine
            try:
                from domains.inference.native.engine import get_engine

                engine = get_engine()
                checks["inference"] = engine is not None
            except Exception:
                checks["inference"] = False

            ready = all(checks.values()) if checks else True
            return success_response(data={"ready": ready, "checks": checks})

        except Exception as e:
            classify_and_raise(e, source="status.ready")

    async def live(self) -> dict:
        try:
            """Kubernetes-style liveness probe.

            Returns alive=True when the process is running and responsive.
            """
            return success_response(data={"alive": True})

        except Exception as e:
            classify_and_raise(e, source="status.live")


router = StatusRouter().router
