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

# Re-export state
from .state import (  # noqa: F401
    REPO_ROOT,
    CHECKPOINTS_DIR,
    LORA_DIR,
    TURBO_DIR,
    MAX_CHECKPOINT_DISK_MB,
    VALID_CKPT_NAME,
    SOU_MAGIC,
    TrainingState,
    get_state,
    get_turbo_state,
    get_turbo_lock,
    get_turbo_pause_event,
    get_turbo_cancel_event,
    get_cancel_event,
    set_cancel_event,
    get_pause_event,
    set_pause_event,
    get_pgq,
)

# Re-export helpers
from .helpers import (  # noqa: F401
    _finite_payload,
    _VALID_DATASET_ID,
    log_experiment_metric,
    log_experiment_param,
    parse_subtitle_text,
    resolve_dataset_path,
    build_soul_prompt,
    get_soul_name,
    get_soul_traits,
    read_slo_json_header,
    describe_checkpoint,
)

# Re-export checkpoints
from .checkpoints import (  # noqa: F401
    find_checkpoint,
    load_soul,
    load_lora_soul,
    list_checkpoints,
    delete_checkpoint,
    load_checkpoint,
    download_checkpoint_path,
    checkpoint_info,
    get_all_checkpoint_data,
    export_all_metrics,
    export_checkpoint_mobile,
)

# Re-export turbo
from .turbo import (  # noqa: F401
    get_turbo_status,
    start_turbo_training,
    run_turbo_worker,
)

# Re-export sessions
from .sessions import start_from_sessions_training  # noqa: F401

# Re-export stream
from .stream import (  # noqa: F401
    process_training_completion,
    cleanup_stream_state,
)
