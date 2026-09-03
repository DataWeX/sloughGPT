"""
Experiments Router - ML experiment tracking

Uses MogDB as the storage engine with automatic JSON sync.
Each experiment, metric, param, and status is a MogDB document.
The JSON files are written to data/experiments_json/ for human readability.
"""
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from schemas.common import raise_error, success_response, classify_and_raise, safe_audit_log
from infrastructure.auth import require_auth_if_enabled

logger = logging.getLogger("slo.routers.experiments")


class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, pattern=r'^[a-zA-Z0-9_\- ]+$')
    config: Optional[dict] = None


def _get_db():
    from mogdb import MogDB
    import os
    repo_root = Path(__file__).parent.parent.parent.parent
    db_path = os.path.join(repo_root, "data", "experiments_mogdb")
    sync_path = os.path.join(repo_root, "data", "experiments_json")
    return MogDB(db_path, sync_dir=sync_path)


class ExperimentsRouter:
    """Router for ML experiment creation, tracking, and metric logging."""

    def __init__(self):
        self.router = APIRouter(prefix="/experiments", tags=["experiments"])
        self._VALID_EXP_ID = re.compile(r'^[a-zA-Z0-9_\-]+$')
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("", self.create_experiment, methods=["POST"])
        self.router.add_api_route("", self.list_experiments, methods=["GET"])
        self.router.add_api_route("/compare", self.compare_experiments, methods=["GET"])
        self.router.add_api_route("/{experiment_id}", self.get_experiment, methods=["GET"])
        self.router.add_api_route("/{experiment_id}", self.delete_experiment, methods=["DELETE"])
        self.router.add_api_route("/{experiment_id}/runs", self.get_experiment_runs, methods=["GET"])
        self.router.add_api_route("/{experiment_id}/data", self.get_experiment_data, methods=["GET"])
        self.router.add_api_route("/{experiment_id}/complete", self.complete_experiment, methods=["POST"])
        self.router.add_api_route("/{experiment_id}/log_metric", self.log_metric, methods=["POST"])
        self.router.add_api_route("/{experiment_id}/log_param", self.log_param, methods=["POST"])

    async def create_experiment(self, req: ExperimentCreate, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Create a new ML experiment."""
        try:
            exp_id = f"{req.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            db = _get_db()
            col = db.collection("experiments")
            col.insert_one({
                "experiment_id": exp_id,
                "name": req.name,
                "config": req.config or {},
                "status": "created",
            })
            safe_audit_log("experiment.create", resource=exp_id, detail=req.name)
            return success_response(data={"id": exp_id, "name": req.name, "created": True})
        except Exception as e:
            classify_and_raise(e, source="create_experiment")

    async def list_experiments(self) -> dict:
        """List all ML experiments."""
        try:
            db = _get_db()
            col = db.collection("experiments")
            docs = col.find()
            exp_ids = sorted(set(d.get("experiment_id", "") for d in docs))
            return success_response(data={"experiments": exp_ids, "count": len(exp_ids)})
        except Exception as e:
            classify_and_raise(e, source="experiments.list_experiments")

    async def get_experiment(self, experiment_id: str) -> dict:
        """Retrieve metadata for a single experiment."""
        try:
            if not self._VALID_EXP_ID.match(experiment_id) or '..' in experiment_id:
                raise_error("Invalid experiment ID", "E_BAD_REQUEST", status_code=400)
            db = _get_db()
            col = db.collection("experiments")
            doc = col.find_one({"experiment_id": experiment_id})
            if not doc:
                raise_error("Experiment not found", "E_NOT_FOUND", status_code=404)
            return success_response(data={"id": experiment_id, "name": doc.get("name"), "config": doc.get("config", {})})
        except Exception as e:
            classify_and_raise(e, source="experiments.get_experiment")

    async def delete_experiment(self, experiment_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Delete an experiment and all its data."""
        try:
            if not self._VALID_EXP_ID.match(experiment_id) or '..' in experiment_id:
                raise_error("Invalid experiment ID", "E_BAD_REQUEST", status_code=400)
            db = _get_db()
            # Delete experiment metadata
            exp_col = db.collection("experiments")
            deleted = exp_col.delete_many({"experiment_id": experiment_id})
            # Delete associated metrics
            metrics_col = db.collection("metrics")
            metrics_col.delete_many({"experiment_id": experiment_id})
            # Delete associated params
            params_col = db.collection("params")
            params_col.delete_many({"experiment_id": experiment_id})
            # Delete associated status
            status_col = db.collection("status")
            status_col.delete_many({"experiment_id": experiment_id})
            if not deleted:
                raise_error("Experiment not found", "E_NOT_FOUND", status_code=404)
            safe_audit_log("experiment.delete", resource=experiment_id)
            return success_response(data={"id": experiment_id, "deleted": True})
        except Exception as e:
            classify_and_raise(e, source="experiments.delete_experiment")

    async def get_experiment_runs(self, experiment_id: str) -> dict:
        """Get run count for an experiment."""
        try:
            if not self._VALID_EXP_ID.match(experiment_id) or '..' in experiment_id:
                raise_error("Invalid experiment ID", "E_BAD_REQUEST", status_code=400)
            db = _get_db()
            col = db.collection("experiments")
            doc = col.find_one({"experiment_id": experiment_id})
            if not doc:
                raise_error("Experiment not found", "E_NOT_FOUND", status_code=404)
            # Count metrics as a proxy for runs
            metrics_col = db.collection("metrics")
            count = metrics_col.count({"experiment_id": experiment_id})
            return success_response(data={"runs": count})
        except Exception as e:
            classify_and_raise(e, source="experiments.get_experiment_runs")

    async def get_experiment_data(self, experiment_id: str) -> dict:
        """Get logged metrics and params for an experiment."""
        e_id = experiment_id
        if not self._VALID_EXP_ID.match(e_id) or '..' in e_id:
            raise_error("Invalid experiment ID", "E_BAD_REQUEST", status_code=400)
        try:
            db = _get_db()
            metrics_col = db.collection("metrics")
            params_col = db.collection("params")
            status_col = db.collection("status")

            metrics = metrics_col.find({"experiment_id": e_id})
            params = params_col.find({"experiment_id": e_id})
            status_doc = status_col.find_one({"experiment_id": e_id})

            return success_response(data={
                "id": e_id,
                "metrics": metrics,
                "params": params,
                "status": status_doc,
            })
        except Exception as e:
            classify_and_raise(e, source="get_experiment_data")

    async def complete_experiment(self, experiment_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Mark experiment as complete."""
        e_id = experiment_id
        if not self._VALID_EXP_ID.match(e_id) or '..' in e_id:
            raise_error("Invalid experiment ID", "E_BAD_REQUEST", status_code=400)
        try:
            db = _get_db()
            status_col = db.collection("status")
            existing = status_col.find_one({"experiment_id": e_id})
            status_data = {
                "experiment_id": e_id,
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            if existing:
                status_col.update_one(
                    {"experiment_id": e_id},
                    {"$set": {"status": "completed", "completed_at": status_data["completed_at"]}},
                )
            else:
                status_col.insert_one(status_data)
            safe_audit_log("experiment.complete", resource=e_id)
            return success_response(data={"id": e_id, "status": "completed"})
        except Exception as e:
            classify_and_raise(e, source="complete_experiment")

    async def compare_experiments(self, ids: str = Query(..., description="Comma-separated experiment IDs")) -> dict:
        """Compare metrics across multiple experiments."""
        try:
            exp_ids = [eid.strip() for eid in ids.split(",") if eid.strip()]
            if len(exp_ids) < 2:
                raise_error("Provide at least 2 experiment IDs", "E_BAD_REQUEST", status_code=400)
            if len(exp_ids) > 10:
                raise_error("Maximum 10 experiments to compare", "E_BAD_REQUEST", status_code=400)
            for eid in exp_ids:
                if not self._VALID_EXP_ID.match(eid) or '..' in eid:
                    raise_error(f"Invalid experiment ID: {eid}", "E_BAD_REQUEST", status_code=400)

            db = _get_db()
            metrics_col = db.collection("metrics")
            params_col = db.collection("params")

            results = {}
            for eid in exp_ids:
                # Aggregate: last value per metric name, all params merged
                metric_docs = metrics_col.find({"experiment_id": eid})
                param_docs = params_col.find({"experiment_id": eid})
                metric_summary = {}
                for m in metric_docs:
                    metric_summary[m.get("metric", "")] = m.get("value")
                param_dict = {}
                for p in param_docs:
                    param_dict[p.get("param", "")] = p.get("value")
                results[eid] = {"metrics": metric_summary, "params": param_dict}

            all_metric_keys = sorted(set().union(*(r["metrics"].keys() for r in results.values())))
            all_param_keys = sorted(set().union(*(r["params"].keys() for r in results.values())))

            comparison = {
                "experiments": exp_ids,
                "metrics": {eid: results[eid]["metrics"] for eid in exp_ids},
                "params": {eid: results[eid]["params"] for eid in exp_ids},
                "metric_keys": all_metric_keys,
                "param_keys": all_param_keys,
            }
            return success_response(data=comparison)
        except Exception as e:
            classify_and_raise(e, source="compare_experiments")

    async def log_metric(self, experiment_id: str, metric_name: str, value: float, step: int = 0, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Log a metric for an experiment."""
        e_id = experiment_id
        if not self._VALID_EXP_ID.match(e_id) or '..' in e_id:
            raise_error("Invalid experiment ID", "E_BAD_REQUEST", status_code=400)
        try:
            db = _get_db()
            col = db.collection("metrics")
            col.insert_one({
                "experiment_id": e_id,
                "metric": metric_name,
                "value": value,
                "step": step,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            safe_audit_log("experiment.log_metric", resource=e_id, detail=f"metric={metric_name} value={value}")
            return success_response(data={"status": "logged", "experiment_id": e_id, "metric": metric_name})
        except Exception as e:
            classify_and_raise(e, source="log_metric")

    async def log_param(self, experiment_id: str, param_name: str = Query(..., min_length=1, max_length=200), value: Any = None, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Log a parameter for an experiment."""
        e_id = experiment_id
        if not self._VALID_EXP_ID.match(e_id) or '..' in e_id:
            raise_error("Invalid experiment ID", "E_BAD_REQUEST", status_code=400)
        try:
            db = _get_db()
            col = db.collection("params")
            col.insert_one({
                "experiment_id": e_id,
                "param": param_name,
                "value": value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            safe_audit_log("experiment.log_param", resource=e_id, detail=f"param={param_name}")
            return success_response(data={"status": "logged", "experiment_id": e_id, "param": param_name})
        except Exception as e:
            classify_and_raise(e, source="log_param")


router = ExperimentsRouter().router
