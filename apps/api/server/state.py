"""
Shared server state - allows routers to access global variables without circular imports.
"""
from typing import Any, Optional, TYPE_CHECKING

model: Optional[Any] = None
tokenizer: Optional[Any] = None
model_type: Optional[str] = None
checkpoint: Optional[Any] = None
soul_engine: Optional[Any] = None
current_soul: Optional[Any] = None
gen_config: Optional[Any] = None
_self_train_proc: Optional[Any] = None
_torch_available = False
model_request_logger: Optional[Any] = None
