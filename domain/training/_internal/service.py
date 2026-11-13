"""Training Service — backward-compatible re-export facade.

All logic has been split into focused modules:
- state.py: TrainingState, state management, getters/setters
- helpers.py: pure helpers (experiment logging, parsing, soul utils)
- checkpoints.py: checkpoint operations (load, list, scan, describe)
- turbo.py: turbo training (start, worker, status)
- sessions.py: from-sessions training
- stream.py: SSE stream business logic

This module re-exports everything for backward compatibility.
"""

from __future__ import annotations

# Re-export checkpoints
from .checkpoints import (  # noqa: F401
    checkpoint_info,
    delete_checkpoint,
    download_checkpoint_path,
    export_all_metrics,
    export_checkpoint_mobile,
    find_checkpoint,
    get_all_checkpoint_data,
    list_checkpoints,
    load_checkpoint,
    load_lora_soul,
    load_soul,
)

# Re-export helpers
from .helpers import (  # noqa: F401
    _VALID_DATASET_ID,
    _finite_payload,
    build_soul_prompt,
    cross_entropy_loss,
    describe_checkpoint,
    get_soul_name,
    get_soul_traits,
    log_experiment_metric,
    log_experiment_param,
    parse_subtitle_text,
    read_slo_json_header,
    resolve_dataset_path,
)

# Re-export sessions
from .sessions import start_from_sessions_training  # noqa: F401

# Re-export state
from .state import (  # noqa: F401
    CHECKPOINTS_DIR,
    LORA_DIR,
    MAX_CHECKPOINT_DISK_MB,
    REPO_ROOT,
    SOU_MAGIC,
    TURBO_DIR,
    VALID_CKPT_NAME,
    TrainingState,
    _state,
    get_cancel_event,
    get_pause_event,
    get_pgq,
    get_state,
    get_turbo_cancel_event,
    get_turbo_lock,
    get_turbo_pause_event,
    get_turbo_state,
    set_cancel_event,
    set_pause_event,
)

# Re-export stream
from .stream import (  # noqa: F401
    cleanup_stream_state,
    process_training_completion,
)

# Re-export turbo
from .turbo import (  # noqa: F401
    get_turbo_status,
    run_turbo_worker,
    start_turbo_training,
)


async def get_log() -> list[str]:
    """Read recent training log lines."""
    from pathlib import Path

    log_file = Path(REPO_ROOT) / "logs" / "training.log"
    if not log_file.exists():
        return []
    try:
        lines = log_file.read_text().strip().splitlines()
        return lines[-100:]
    except Exception:
        return []
