"""NPUDevice — compat shim for the unified NPU driver.

There is one NPU implementation: :class:`domain.shell._internal.kernel_npu.NPUDevice`.
This module keeps the historical standalone import path
(``from .npu_device import NPUDevice``) resolving to that same class, so the
/dev seam, the VM seam, and older callers all share one device, one model
registry, and one ioctl vocabulary.

``NPUModel`` is re-exported for callers that imported it from here.
"""

from __future__ import annotations

from .kernel_npu import NPUDevice, NPUModel

__all__ = ["NPUDevice", "NPUModel"]
