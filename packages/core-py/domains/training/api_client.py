"""
API Client for SloughGPT Training Pipeline

Unified client for communicating with SloughGPT API endpoints.
Supports progress reporting, status updates, and checkpoint management.
"""

import json
from typing import Optional
from dataclasses import dataclass
from enum import Enum

import httpx


class TrainingStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingJob:
    """Training job information."""

    id: str
    name: str
    model: str
    dataset: str
    user_id: str
    status: TrainingStatus = TrainingStatus.PENDING
    progress: int = 0
    epochs: int = 0
    current_epoch: int = 0
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    loss: float = 0.0


@dataclass
class APIConfig:
    """Configuration for API client."""

    base_url: str = "http://localhost:8000"
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    api_key: Optional[str] = None
    auth_token: Optional[str] = None


class APIClient:
    """
    Client for SloughGPT API communication.

    Handles:
    - Progress reporting
    - Status updates
    - Checkpoint management
    - Error handling and retries
    """

    def __init__(self, config: APIConfig):
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url, timeout=config.timeout, headers=self._get_default_headers()
        )
        self._job_id: Optional[str] = None
        self._job: Optional[TrainingJob] = None

    def _get_default_headers(self) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        if self.config.auth_token:
            headers["Authorization"] = f"Bearer {self.config.auth_token}"
        return headers

    async def start_training_job(
        self,
        name: str,
        model: str,
        dataset: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        user_id: str = "local",
    ) -> TrainingJob:
        """
        Start a new training job on the API.

        Args:
            name: Name of the training job
            model: Model ID to train
            dataset: Dataset path or identifier
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            user_id: User ID (default: "local")

        Returns:
            TrainingJob object with job information
        """
        url = f"{self.config.base_url}/training/start"

        payload = {
            "name": name,
            "model": model,
            "dataset": dataset,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "user_id": user_id,
        }

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            job = TrainingJob(
                id=data["job_id"],
                name=name,
                model=model,
                dataset=dataset,
                user_id=user_id,
                status=TrainingStatus.PENDING,
                epochs=epochs,
                experiment_id=data.get("experiment_id"),
                run_id=data.get("run_id"),
            )

            self._job_id = job.id
            self._job = job
            return job

        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Failed to start training job: {e.response.text}") from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Network error starting training job: {e}") from e

    async def update_progress(
        self, epoch: int, total_epochs: int, step: int = 0, total_steps: int = 0, loss: float = 0.0
    ) -> None:
        """
        Update training progress on the API.

        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            step: Current step within epoch
            total_steps: Total steps per epoch
            loss: Current loss value
        """
        if not self._job_id:
            raise RuntimeError("Cannot update progress - no job started")

        if total_steps > 0:
            # Calculate progress based on both epoch and step
            epoch_progress = (epoch / total_epochs) * 100
            step_progress = (step / total_steps) * (100 / total_epochs)
            progress = int(epoch_progress + step_progress)
        else:
            # Fallback to epoch-based progress
            progress = int((epoch / total_epochs) * 100)

        url = f"{self.config.base_url}/training/jobs/{self._job_id}"

        payload = {"status": "running", "progress": progress, "current_epoch": epoch, "loss": loss}

        try:
            response = await self._client.patch(url, json=payload)
            response.raise_for_status()

            # Update local job state
            if self._job:
                self._job.status = TrainingStatus.RUNNING
                self._job.progress = progress
                self._job.current_epoch = epoch
                self._job.loss = loss

        except httpx.HTTPStatusError as e:
            print(f"Warning: Failed to update progress: {e.response.text}")
        except httpx.RequestError as e:
            print(f"Warning: Network error updating progress: {e}")

    async def complete_training(self, loss: float = 0.0) -> None:
        """
        Mark training as completed.

        Args:
            loss: Final loss value
        """
        if not self._job_id:
            raise RuntimeError("Cannot complete training - no job started")

        url = f"{self.config.base_url}/training/jobs/{self._job_id}"

        payload = {"status": "completed", "progress": 100, "loss": loss}

        try:
            response = await self._client.patch(url, json=payload)
            response.raise_for_status()

            # Update local job state
            if self._job:
                self._job.status = TrainingStatus.COMPLETED
                self._job.progress = 100
                self._job.loss = loss

        except httpx.HTTPStatusError as e:
            print(f"Warning: Failed to complete training: {e.response.text}")
        except httpx.RequestError as e:
            print(f"Warning: Network error completing training: {e}")

    async def fail_training(self, error: Exception) -> None:
        """
        Mark training as failed.

        Args:
            error: Exception that caused failure
        """
        if not self._job_id:
            raise RuntimeError("Cannot fail training - no job started")

        url = f"{self.config.base_url}/training/jobs/{self._job_id}"

        payload = {"status": "failed", "error": str(error)}

        try:
            response = await self._client.patch(url, json=payload)
            response.raise_for_status()

            # Update local job state
            if self._job:
                self._job.status = TrainingStatus.FAILED

        except httpx.HTTPStatusError as e:
            print(f"Warning: Failed to mark training as failed: {e.response.text}")
        except httpx.RequestError as e:
            print(f"Warning: Network error failing training: {e}")

    async def save_checkpoint(self, path: str, epoch: int, loss: float) -> None:
        """
        Save model checkpoint to API.

        Args:
            path: Local path to checkpoint file
            epoch: Epoch number for checkpoint
            loss: Loss value at checkpoint
        """
        if not self._job_id:
            raise RuntimeError("Cannot save checkpoint - no job started")

        url = f"{self.config.base_url}/checkpoints"

        try:
            # Read checkpoint file
            with open(path, "rb") as f:
                checkpoint_data = f.read()

            files = {
                "file": (os.path.basename(path), checkpoint_data, "application/octet-stream"),
                "metadata": (
                    None,
                    json.dumps(
                        {
                            "job_id": self._job_id,
                            "epoch": epoch,
                            "loss": loss,
                            "filename": os.path.basename(path),
                        }
                    ),
                    "application/json",
                ),
            }

            response = await self._client.post(url, files=files)
            response.raise_for_status()

            print(f"Checkpoint saved: {path} (epoch {epoch}, loss {loss:.4f})")

        except httpx.HTTPStatusError as e:
            print(f"Warning: Failed to save checkpoint: {e.response.text}")
        except httpx.RequestError as e:
            print(f"Warning: Network error saving checkpoint: {e}")
        except FileNotFoundError:
            print(f"Warning: Checkpoint file not found: {path}")

    async def get_job_status(self) -> TrainingJob:
        """
        Get current job status from API.

        Returns:
            TrainingJob object with latest status
        """
        if not self._job_id:
            raise RuntimeError("No job started")

        url = f"{self.config.base_url}/training/jobs/{self._job_id}"

        try:
            response = await self._client.get(url)
            response.raise_for_status()
            data = response.json()

            job = TrainingJob(
                id=data["id"],
                name=data["name"],
                model=data["model"],
                dataset=data["dataset"],
                user_id=data["user_id"],
                status=TrainingStatus(data["status"]),
                progress=data.get("progress", 0),
                epochs=data.get("epochs", 0),
                current_epoch=data.get("current_epoch", 0),
                experiment_id=data.get("experiment_id"),
                run_id=data.get("run_id"),
                loss=data.get("loss", 0.0),
            )

            self._job = job
            return job

        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Failed to get job status: {e.response.text}") from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Network error getting job status: {e}") from e

    async def stream_progress(self, callback) -> None:
        """
        Stream progress updates from API.

        Args:
            callback: Function to call with progress updates
        """
        if not self._job_id:
            raise RuntimeError("No job started")

        url = f"{self.config.base_url}/training/jobs/{self._job_id}/stream"

        try:
            async with self._client.stream("GET", url) as response:
                response.raise_for_status()
                async for chunk in response.aiter_text():
                    # Chunk format: data: {...}\n\n
                    if chunk.startswith("data:"):
                        data_str = chunk[len("data:") :].strip()
                        if data_str:
                            try:
                                data = json.loads(data_str)
                                callback(data)
                            except json.JSONDecodeError:
                                continue
        except httpx.HTTPStatusError as e:
            print(f"Warning: Failed to stream progress: {e.response.text}")
        except httpx.RequestError as e:
            print(f"Warning: Network error streaming progress: {e}")

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


# Convenience functions


async def report_progress(
    api_client: APIClient,
    epoch: int,
    total_epochs: int,
    step: int = 0,
    total_steps: int = 0,
    loss: float = 0.0,
) -> None:
    """Convenience function for progress reporting."""
    try:
        await api_client.update_progress(epoch, total_epochs, step, total_steps, loss)
    except Exception as e:
        print(f"Progress reporting error: {e}")


async def save_checkpoint(api_client: APIClient, path: str, epoch: int, loss: float) -> None:
    """Convenience function for checkpoint saving."""
    try:
        await api_client.save_checkpoint(path, epoch, loss)
    except Exception as e:
        print(f"Checkpoint saving error: {e}")
