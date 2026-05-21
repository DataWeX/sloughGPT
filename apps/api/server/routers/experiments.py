"""
Experiments Router - ML experiment tracking
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime
import json
from pathlib import Path

router = APIRouter(prefix="/experiments", tags=["experiments"])

REPO_ROOT = Path(__file__).parent.parent.parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "data" / "experiments"


class ExperimentCreate(BaseModel):
    name: str
    config: Optional[dict] = None


@router.post("")
async def create_experiment(req: ExperimentCreate):
    """Create a new experiment"""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    exp_id = f"{req.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = EXPERIMENTS_DIR / exp_id
    exp_dir.mkdir(exist_ok=True)
    return {"id": exp_id, "name": req.name, "created": True}


@router.get("")
async def list_experiments():
    """List all experiments"""
    if not EXPERIMENTS_DIR.exists():
        return {"experiments": [], "count": 0}
    exps = [d.name for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()]
    return {"experiments": exps, "count": len(exps)}


@router.get("/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get experiment details"""
    path = EXPERIMENTS_DIR / experiment_id
    if not path.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"id": experiment_id, "path": str(path)}


@router.get("/{experiment_id}/runs")
async def get_experiment_runs(experiment_id: str):
    """Get runs for an experiment"""
    path = EXPERIMENTS_DIR / experiment_id
    if not path.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")
    runs = list(path.glob("*.json"))
    return {"runs": len(runs)}


@router.post("/{experiment_id}/complete")
async def complete_experiment(experiment_id: str):
    """Mark experiment as complete"""
    return {"id": experiment_id, "status": "completed"}


@router.post("/{experiment_id}/log_metric")
async def log_metric(experiment_id: str, metric_name: str, value: float, step: int = 0):
    """Log a metric for an experiment."""
    import json, os, datetime
    log_dir = os.path.join(os.path.dirname(__file__), "..", "data", "experiments")
    os.makedirs(log_dir, exist_ok=True)
    entry = {"experiment_id": experiment_id, "metric": metric_name, "value": value, "step": step, "timestamp": datetime.datetime.utcnow().isoformat()}
    with open(os.path.join(log_dir, f"{experiment_id}_metrics.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"status": "logged", "experiment_id": experiment_id, "metric": metric_name}


@router.post("/{experiment_id}/log_param")
async def log_param(experiment_id: str, param_name: str, value: Any):
    """Log a parameter for an experiment."""
    import json, os, datetime
    log_dir = os.path.join(os.path.dirname(__file__), "..", "data", "experiments")
    os.makedirs(log_dir, exist_ok=True)
    entry = {"experiment_id": experiment_id, "param": param_name, "value": value, "timestamp": datetime.datetime.utcnow().isoformat()}
    with open(os.path.join(log_dir, f"{experiment_id}_params.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"status": "logged", "experiment_id": experiment_id, "param": param_name}