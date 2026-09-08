"""Persistent training job registry backed by MogDB.

Completed jobs may expose a ``checkpoint`` path; native trainer ``*.soul`` embeds
``stoi`` / ``itos`` / ``chars`` — see ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
"""

from __future__ import annotations

from typing import Any

from .job_store import PersistentTrainingJobs

training_jobs: PersistentTrainingJobs = PersistentTrainingJobs()
