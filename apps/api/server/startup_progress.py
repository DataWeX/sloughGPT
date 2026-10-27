"""
Startup progress tracking — shared between main.py (writes phases during
lifespan) and health.py (reads phases for the /health/startup-progress endpoint).
"""

STARTUP_PHASE: dict = {"phase": "initializing", "step": 0, "total": 8, "message": "Starting..."}


def set_model_error(reason: "str | None") -> None:
    """Record (or clear) the permanent model-load failure verdict.

    The load thread is the single writer: it calls this with the real reason
    at every permanent failure exit of autoload, and with None when a load
    succeeds. Readiness surfaces (``_get_model_status``, health) read it so a
    failed load answers "no model — <reason>" instead of "still loading"
    forever. In-flight loads leave it None.
    """
    STARTUP_PHASE["model_error"] = reason
