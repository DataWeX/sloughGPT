"""
Batch Processor - Ported from recovered batch_processor.py
"""

import time
import threading
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue


class BatchProcessor:
    """Process multiple datasets in parallel."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.results: List[Dict] = []
        self.progress_queue: queue.Queue = queue.Queue()
        self._running = False

    def process_datasets(
        self, datasets: List[Dict], operation: str, callback: Optional[Callable] = None
    ) -> List[Dict]:
        """Process multiple datasets with specified operation."""
        print(f"🔄 Processing {len(datasets)} datasets with operation: {operation}")

        self.results = []
        self._running = True

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_dataset = {
                executor.submit(self._process_single_dataset, dataset, operation): dataset
                for dataset in datasets
            }

            completed = 0
            for future in as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                try:
                    result = future.result()
                    result["dataset"] = dataset.get("name", "unknown")
                    self.results.append(result)
                    completed += 1
                    print(
                        f"✅ Completed {completed}/{len(datasets)}: {dataset.get('name', 'unknown')}"
                    )

                    if callback:
                        callback(result)

                except Exception as e:
                    error_result = {
                        "dataset": dataset.get("name", "unknown"),
                        "status": "error",
                        "error": str(e),
                    }
                    self.results.append(error_result)
                    print(f"❌ Failed {dataset.get('name', 'unknown')}: {e}")

        self._running = False
        return self.results

    def _process_single_dataset(self, dataset: Dict, operation: str) -> Dict:
        """Process a single dataset."""
        return {
            "status": "success",
            "operation": operation,
            "dataset": dataset.get("name", "unknown"),
            "timestamp": time.time(),
        }

    def cancel(self) -> None:
        """Cancel processing."""
        self._running = False

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress."""
        return {
            "running": self._running,
            "completed": len(self.results),
        }


class JobScheduler:
    """Schedule and manage training jobs."""

    def __init__(self):
        self.jobs: List[Dict] = []
        self._lock = threading.Lock()

    def add_job(
        self, name: str, command: str, schedule: Optional[str] = None, priority: int = 0
    ) -> str:
        """Add a job to the scheduler."""
        job_id = f"job_{len(self.jobs)}"

        with self._lock:
            self.jobs.append(
                {
                    "id": job_id,
                    "name": name,
                    "command": command,
                    "schedule": schedule,
                    "priority": priority,
                    "status": "pending",
                    "created_at": time.time(),
                }
            )

        return job_id

    def get_pending_jobs(self) -> List[Dict]:
        """Get pending jobs sorted by priority."""
        with self._lock:
            pending = [j for j in self.jobs if j["status"] == "pending"]
            return sorted(pending, key=lambda x: x["priority"], reverse=True)

    def mark_complete(self, job_id: str) -> None:
        """Mark a job as complete."""
        with self._lock:
            for job in self.jobs:
                if job["id"] == job_id:
                    job["status"] = "complete"
                    job["completed_at"] = time.time()


__all__ = ["BatchProcessor", "JobScheduler"]
