"""
Startup progress tracking — shared between main.py (writes phases during
lifespan) and health.py (reads phases for the /health/startup-progress endpoint).
"""

STARTUP_PHASE: dict = {"phase": "initializing", "step": 0, "total": 6, "message": "Starting..."}
