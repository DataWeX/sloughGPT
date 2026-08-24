"""
Experiments Router - ML experiment tracking
"""
import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from schemas.common import raise_error, success_response, classify_and_raise, safe_audit_log

logger = logging.getLogger("slo.routers.experiments")


class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, pattern=r'^[a-zA-Z0-9_\- ]+$')
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
        """Create a new ML experiment with a timestamped directory."""
        try:
            exp_id = f"{req.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            def _create():
                self.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
                exp_dir = self.EXPERIMENTS_DIR / exp_id
                exp_dir.mkdir(exist_ok=True)

            await asyncio.to_thread(_create)
            safe_audit_log("experiment.create", resource=exp_id, detail=req.name)
            return success_response(data={"id": exp_id, "name": req.name, "created": True})
        except Exception as e:
            classify_and_raise(e, source="create_experiment")

    async def list_experiments(self) -> dict:
        try:
            """List all ML experiments stored on disk.

            Scans the data/experiments/ directory for subdirectories, each
            representing an experiment, and returns their names.

            Returns:
                Success envelope with experiments array and count.
            """
            def _scan():
                if not self.EXPERIMENTS_DIR.exists():
                    return []
                return [d.name for d in self.EXPERIMENTS_DIR.iterdir() if d.is_dir()]
            exps = await asyncio.to_thread(_scan)
            return success_response(data={"experiments": exps, "count": len(exps)})

        except Exception as e:
            classify_and_raise(e, source="experiments.list_experiments")
    async def get_experiment(self, experiment_id: str) -> dict:
        try:
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
            def _check():
                p = (self.EXPERIMENTS_DIR / experiment_id).resolve()
                return p.exists() and str(p).startswith(str(self.EXPERIMENTS_DIR.resolve())), str(p)
            exists, path_str = await asyncio.to_thread(_check)
            if not exists:
                raise_error("Experiment not found", "E_NOT_FOUND", status_code=404)
            return success_response(data={"id": experiment_id, "path": path_str})

        except Exception as e:
            classify_and_raise(e, source="experiments.get_experiment")
    async def delete_experiment(self, experiment_id: str) -> dict:
        try:
            """Delete an experiment and all its data."""
            import shutil
            if not self._VALID_EXP_ID.match(experiment_id) or '..' in experiment_id:
                raise_error("Invalid experiment ID", "E_BAD_REQUEST", status_code=400)
            def _check_delete():
                p = (self.EXPERIMENTS_DIR / experiment_id).resolve()
                if not p.exists() or not str(p).startswith(str(self.EXPERIMENTS_DIR.resolve())):
                    return None
                shutil.rmtree(p)
                return True
            result = await asyncio.to_thread(_check_delete)
            if result is None:
                raise_error("Experiment not found", "E_NOT_FOUND", status_code=404)
            safe_audit_log("experiment.delete", resource=experiment_id)
            return success_response(data={"id": experiment_id, "deleted": True})

        except Exception as e:
            classify_and_raise(e, source="experiments.delete_experiment")
    async def get_experiment_runs(self, experiment_id: str) -> dict:
        try:
            """Get runs for an experiment"""
            if not self._VALID_EXP_ID.match(experiment_id) or '..' in experiment_id:
                raise_error("Invalid experiment ID", "E_BAD_REQUEST", status_code=400)
            def _check_runs():
                p = (self.EXPERIMENTS_DIR / experiment_id).resolve()
                if not p.exists() or not str(p).startswith(str(self.EXPERIMENTS_DIR.resolve())):
                    return None
                return list(p.glob("*.json"))
            runs = await asyncio.to_thread(_check_runs)
            if runs is None:
                raise_error("Experiment not found", "E_NOT_FOUND", status_code=404)
            return success_response(data={"runs": len(runs)})

        except Exception as e:
            classify_and_raise(e, source="experiments.get_experiment_runs")
    async def get_experiment_data(self, experiment_id: str) -> dict:
        """Get logged metrics and params for an experiment."""
        e_id = experiment_id
        if not self._VALID_EXP_ID.match(e_id) or '..' in e_id:
            raise_error("Invalid experiment ID", "E_BAD_REQUEST", status_code=400)
        metrics_file = self.EXPERIMENTS_DIR / f"{e_id}_metrics.jsonl"
        params_file = self.EXPERIMENTS_DIR / f"{e_id}_params.jsonl"
        status_file = self.EXPERIMENTS_DIR / f"{e_id}_status.json"

        def _read_data():
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
                            except json.JSONDecodeError as e:
                                logger.warning("Corrupt metric line in %s: %s", e_id, e)
            if params_file.exists():
                with open(params_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                params.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                logger.warning("Corrupt param line in %s: %s", e_id, e)
            if status_file.exists():
                with open(status_file) as f:
                    try:
                        status = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.warning("Corrupt status file %s: %s", status_file.name, e)
            return metrics, params, status

        try:
            metrics, params, status = await asyncio.to_thread(_read_data)
            return success_response(data={"id": e_id, "metrics": metrics, "params": params, "status": status})
        except Exception as e:
            classify_and_raise(e, source="get_experiment_data")

    async def complete_experiment(self, experiment_id: str) -> dict:
        """Mark experiment as complete and persist status to disk."""
        e_id = experiment_id
        if not self._VALID_EXP_ID.match(e_id) or '..' in e_id:
            raise_error("Invalid experiment ID", "E_BAD_REQUEST", status_code=400)
        try:
            def _write_status():
                self.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
                status_file = self.EXPERIMENTS_DIR / f"{e_id}_status.json"
                status_data = {
                    "experiment_id": e_id,
                    "status": "completed",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                with open(status_file, "w") as f:
                    json.dump(status_data, f)
            await asyncio.to_thread(_write_status)
            safe_audit_log("experiment.complete", resource=e_id)
            return success_response(data={"id": e_id, "status": "completed"})
        except Exception as e:
            classify_and_raise(e, source="complete_experiment")

    async def log_metric(self, experiment_id: str, metric_name: str, value: float, step: int = 0) -> dict:
        """Log a metric for an experiment."""
        e_id = experiment_id
        if not self._VALID_EXP_ID.match(e_id) or '..' in e_id:
            raise_error("Invalid experiment ID", "E_BAD_REQUEST", status_code=400)
        try:
            def _write_metric():
                self.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
                entry = {"experiment_id": e_id, "metric": metric_name, "value": value, "step": step, "timestamp": datetime.now(timezone.utc).isoformat()}
                with open(self.EXPERIMENTS_DIR / f"{e_id}_metrics.jsonl", "a") as f:
                    f.write(json.dumps(entry) + "\n")
            await asyncio.to_thread(_write_metric)
            safe_audit_log("experiment.log_metric", resource=e_id, detail=f"metric={metric_name} value={value}")
            return success_response(data={"status": "logged", "experiment_id": e_id, "metric": metric_name})
        except Exception as e:
            classify_and_raise(e, source="log_metric")

    async def log_param(self, experiment_id: str, param_name: str, value: Any) -> dict:
        """Log a parameter for an experiment."""
        e_id = experiment_id
        if not self._VALID_EXP_ID.match(e_id) or '..' in e_id:
            raise_error("Invalid experiment ID", "E_BAD_REQUEST", status_code=400)
        try:
            def _write_param():
                self.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
                entry = {"experiment_id": e_id, "param": param_name, "value": value, "timestamp": datetime.now(timezone.utc).isoformat()}
                with open(self.EXPERIMENTS_DIR / f"{e_id}_params.jsonl", "a") as f:
                    f.write(json.dumps(entry) + "\n")
            await asyncio.to_thread(_write_param)
            safe_audit_log("experiment.log_param", resource=e_id, detail=f"param={param_name}")
            return success_response(data={"status": "logged", "experiment_id": e_id, "param": param_name})
        except Exception as e:
            classify_and_raise(e, source="log_param")


router = ExperimentsRouter().router
