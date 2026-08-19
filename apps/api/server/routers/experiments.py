"""
Experiments Router - ML experiment tracking
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime, timezone
import json
import re
from pathlib import Path

from schemas.common import raise_error, success_response, classify_and_raise, safe_audit_log


class ExperimentCreate(BaseModel):
    name: str
    config: Optional[dict] = None


class ExperimentsRouter:
    """Router for ML experiment creation, tracking, and metric logging."""

    def __init__(self):
        self.router = APIRouter(prefix="/experiments", tags=["experiments"])
        self.REPO_ROOT = Path(__file__).parent.parent.parent.parent
        self.EXPERIMENTS_DIR = self.REPO_ROOT / "data" / "experiments"
        self._VALID_EXP_ID = re.compile(r'^[a-zA-Z0-9_\-]+$')
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("", self.create_experiment, methods=["POST"])
        self.router.add_api_route("", self.list_experiments, methods=["GET"])
        self.router.add_api_route("/{experiment_id}", self.get_experiment, methods=["GET"])
        self.router.add_api_route("/{experiment_id}", self.delete_experiment, methods=["DELETE"])
        self.router.add_api_route("/{experiment_id}/runs", self.get_experiment_runs, methods=["GET"])
        self.router.add_api_route("/{experiment_id}/data", self.get_experiment_data, methods=["GET"])
        self.router.add_api_route("/{experiment_id}/complete", self.complete_experiment, methods=["POST"])
        self.router.add_api_route("/{experiment_id}/log_metric", self.log_metric, methods=["POST"])
        self.router.add_api_route("/{experiment_id}/log_param", self.log_param, methods=["POST"])

    async def create_experiment(self, req: ExperimentCreate) -> dict:
        """Create a new ML experiment with a timestamped directory.

        Generates a unique experiment ID from the name and current timestamp,
        creates the experiment directory under data/experiments/, and returns
        the experiment metadata.

        Args:
            req: ExperimentCreate with name (required) and optional config dict.

        Returns:
            Success envelope with id, name, and created flag.

        Side effects:
            - Creates a directory under data/experiments/ for the experiment.
            - Writes an audit log entry for the creation.
        """
        self.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        exp_id = f"{req.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_dir = self.EXPERIMENTS_DIR / exp_id
        exp_dir.mkdir(exist_ok=True)
        safe_audit_log("experiment.create", resource=exp_id, detail=req.name)
        return success_response(data={"id": exp_id, "name": req.name, "created": True})

    async def list_experiments(self) -> dict:
        """List all ML experiments stored on disk.

        Scans the data/experiments/ directory for subdirectories, each
        representing an experiment, and returns their names.

        Returns:
            Success envelope with experiments array and count.
        """
        if not self.EXPERIMENTS_DIR.exists():
            return success_response(data={"experiments": [], "count": 0})
        exps = [d.name for d in self.EXPERIMENTS_DIR.iterdir() if d.is_dir()]
        return success_response(data={"experiments": exps, "count": len(exps)})

    async def get_experiment(self, experiment_id: str) -> dict:
        """Retrieve metadata for a single experiment by its ID.

        Validates the experiment ID format and checks that the directory
        exists under data/experiments/.

        Args:
            experiment_id: The unique experiment identifier.

        Returns:
            Success envelope with id and filesystem path.

        Raises:
            400 if the experiment ID is invalid.
            404 if the experiment directory is not found.
        """
        if not self._VALID_EXP_ID.match(experiment_id) or '..' in experiment_id:
            raise_error("Invalid experiment ID", "E_BAD_REQUEST", status_code=400)
        path = (self.EXPERIMENTS_DIR / experiment_id).resolve()
        if not path.exists() or not str(path).startswith(str(self.EXPERIMENTS_DIR.resolve())):
            raise_error("Experiment not found", "E_NOT_FOUND", status_code=404)
        return success_response(data={"id": experiment_id, "path": str(path)})

    async def delete_experiment(self, experiment_id: str) -> dict:
        """Delete an experiment and all its data."""
        import shutil
        if not self._VALID_EXP_ID.match(experiment_id) or '..' in experiment_id:
            raise_error("Invalid experiment ID", "E_BAD_REQUEST", status_code=400)
        path = (self.EXPERIMENTS_DIR / experiment_id).resolve()
        if not path.exists() or not str(path).startswith(str(self.EXPERIMENTS_DIR.resolve())):
            raise_error("Experiment not found", "E_NOT_FOUND", status_code=404)
        shutil.rmtree(path)
        safe_audit_log("experiment.delete", resource=experiment_id)
        return success_response(data={"id": experiment_id, "deleted": True})

    async def get_experiment_runs(self, experiment_id: str) -> dict:
        """Get runs for an experiment"""
        if not self._VALID_EXP_ID.match(experiment_id) or '..' in experiment_id:
            raise_error("Invalid experiment ID", "E_BAD_REQUEST", status_code=400)
        path = (self.EXPERIMENTS_DIR / experiment_id).resolve()
        if not path.exists() or not str(path).startswith(str(self.EXPERIMENTS_DIR.resolve())):
            raise_error("Experiment not found", "E_NOT_FOUND", status_code=404)
        runs = list(path.glob("*.json"))
        return success_response(data={"runs": len(runs)})

    async def get_experiment_data(self, experiment_id: str) -> dict:
        """Get logged metrics and params for an experiment."""
        e_id = experiment_id
        if not self._VALID_EXP_ID.match(e_id) or '..' in e_id:
            raise_error("Invalid experiment ID", "E_BAD_REQUEST", status_code=400)
        metrics_file = self.EXPERIMENTS_DIR / f"{e_id}_metrics.jsonl"
        params_file = self.EXPERIMENTS_DIR / f"{e_id}_params.jsonl"
        status_file = self.EXPERIMENTS_DIR / f"{e_id}_status.json"
        metrics = []
        params = []
        status = None
        if metrics_file.exists():
            with open(metrics_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            metrics.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        if params_file.exists():
            with open(params_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            params.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        if status_file.exists():
            with open(status_file) as f:
                try:
                    status = json.load(f)
                except json.JSONDecodeError:
                    pass
        return success_response(data={"id": e_id, "metrics": metrics, "params": params, "status": status})

    async def complete_experiment(self, experiment_id: str) -> dict:
        """Mark experiment as complete and persist status to disk."""
        e_id = experiment_id
        if not self._VALID_EXP_ID.match(e_id) or '..' in e_id:
            raise_error("Invalid experiment ID", "E_BAD_REQUEST", status_code=400)
        self.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        status_file = self.EXPERIMENTS_DIR / f"{e_id}_status.json"
        status_data = {
            "experiment_id": e_id,
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(status_file, "w") as f:
            json.dump(status_data, f)
        return success_response(data={"id": e_id, "status": "completed"})

    async def log_metric(self, experiment_id: str, metric_name: str, value: float, step: int = 0) -> dict:
        """Log a metric for an experiment."""
        e_id = experiment_id
        self.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        if not self._VALID_EXP_ID.match(e_id) or '..' in e_id:
            raise_error("Invalid experiment ID", "E_BAD_REQUEST", status_code=400)
        entry = {"experiment_id": e_id, "metric": metric_name, "value": value, "step": step, "timestamp": datetime.now(timezone.utc).isoformat()}
        with open(self.EXPERIMENTS_DIR / f"{e_id}_metrics.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        return success_response(data={"status": "logged", "experiment_id": e_id, "metric": metric_name})

    async def log_param(self, experiment_id: str, param_name: str, value: Any) -> dict:
        """Log a parameter for an experiment."""
        e_id = experiment_id
        self.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        if not self._VALID_EXP_ID.match(e_id) or '..' in e_id:
            raise_error("Invalid experiment ID", "E_BAD_REQUEST", status_code=400)
        entry = {"experiment_id": e_id, "param": param_name, "value": value, "timestamp": datetime.now(timezone.utc).isoformat()}
        with open(self.EXPERIMENTS_DIR / f"{e_id}_params.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        return success_response(data={"status": "logged", "experiment_id": e_id, "param": param_name})


router = ExperimentsRouter().router
