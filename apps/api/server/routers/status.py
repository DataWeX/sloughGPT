"""
Status Router - Overall service health and info
"""
from fastapi import APIRouter
from datetime import datetime

router = APIRouter(tags=["status"])

_start_time = datetime.now()


@router.get("/status")
async def get_status():
    """Get overall service status"""
    import psutil
    
    uptime = (datetime.now() - _start_time).total_seconds()
    
    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/ready")
async def ready():
    """Readiness check"""
    return {"ready": True}


@router.get("/live")
async def live():
    """Liveness check"""
    return {"alive": True}