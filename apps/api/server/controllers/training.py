"""
Training Controller - Business logic for training operations
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import json


class TrainingController:
    """Controller for training business logic"""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.checkpoints_dir = repo_root / "models" / "checkpoints"
        self.training_dir = repo_root / "data" / "training"

    def create_job(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new training job"""
        import uuid
        job_id = str(uuid.uuid4())[:8]

        job = {
            "id": job_id,
            "name": name,
            "config": config,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
        }

        jobs_file = self.training_dir / "jobs.json"
        jobs = self._load_jobs()
        jobs[job_id] = job
        self._save_jobs(jobs)

        return job

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID"""
        jobs = self._load_jobs()
        return jobs.get(job_id)

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs"""
        return list(self._load_jobs().values())

    def update_job_status(self, job_id: str, status: str, **kwargs) -> Dict[str, Any]:
        """Update job status"""
        jobs = self._load_jobs()
        if job_id in jobs:
            jobs[job_id]["status"] = status
            jobs[job_id].update(kwargs)
            self._save_jobs(jobs)
        return jobs.get(job_id, {})

    def start_job(self, job_id: str) -> Dict[str, Any]:
        """Start a job"""
        return self.update_job_status(
            job_id, "running",
            started_at=datetime.now().isoformat()
        )

    def stop_job(self, job_id: str) -> Dict[str, Any]:
        """Stop a job"""
        return self.update_job_status(
            job_id, "stopped",
            stopped_at=datetime.now().isoformat()
        )

    def get_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints"""
        if not self.checkpoints_dir.exists():
            return []

        checkpoints = []
        for ckpt in self.checkpoints_dir.glob("*.pt"):
            checkpoints.append({
                "name": ckpt.stem,
                "path": str(ckpt),
                "size_mb": ckpt.stat().st_size / (1024 * 1024),
                "created_at": datetime.fromtimestamp(ckpt.stat().st_ctime).isoformat(),
            })
        return checkpoints

    def get_training_datasets(self) -> List[Dict[str, Any]]:
        """List training datasets"""
        data_dir = self.repo_root / "data" / "features"
        if not data_dir.exists():
            return []

        datasets = []
        for d in data_dir.iterdir():
            if d.is_dir():
                files = list(d.glob("*.jsonl"))
                datasets.append({
                    "name": d.name,
                    "path": str(d),
                    "file_count": len(files),
                })
        return datasets

    def _load_jobs(self) -> Dict[str, Any]:
        """Load jobs from file"""
        jobs_file = self.training_dir / "jobs.json"
        if not jobs_file.exists():
            return {}
        try:
            return json.loads(jobs_file.read_text())
        except Exception:
            return {}

    def _save_jobs(self, jobs: Dict[str, Any]) -> None:
        """Save jobs to file"""
        self.training_dir.mkdir(parents=True, exist_ok=True)
        jobs_file = self.training_dir / "jobs.json"
        jobs_file.write_text(json.dumps(jobs, indent=2))


def get_training_controller() -> TrainingController:
    """Get training controller instance"""
    from pathlib import Path
    repo_root = Path(__file__).parent.parent.parent.parent
    return TrainingController(repo_root)
