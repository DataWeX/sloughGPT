"""
Startup progress tracking — shared between main.py (writes phases during
lifespan) and health.py (reads phases for the /health/startup-progress endpoint).
"""

STARTUP_PHASE: dict = {"phase": "initializing", "step": 0, "total": 8, "message": "Starting..."}

# Terminal outcome of the background autoload: pending -> loading -> ready | failed,
# plus the reason when it ends failed. infrastructure/startup.py writes it,
# routers/inference._get_model_status() reads it — without this a load that ends
# in failure falls through to phase "ready" and every readiness check keeps
# answering "Model still loading" for a load that stopped minutes ago.
MODEL_LOAD_STATE: dict = {"status": "pending", "error": None}
