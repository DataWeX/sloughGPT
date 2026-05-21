"""In-memory training job registry (replace with DB/queue for production).

Completed jobs may expose a ``checkpoint`` path; native ``step_*.pt`` embeds
``stoi`` / ``itos`` / ``chars`` — see ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
"""

from __future__ import annotations

from typing import Any

training_jobs: dict[str, dict[str, Any]] = {}
