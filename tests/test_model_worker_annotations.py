"""``from __future__ import annotations`` must be a real module-level import.

In model_worker.py the future import sat INSIDE the module docstring
(between ``Architecture::`` and the ASCII diagram), so it was text, not an
import. Annotations stayed eager and ``mp.Queue | None`` raised TypeError —
``mp.Queue`` is a bound method, not a type. The crash points are the nested
``_stream`` defs (:385/:659): their signature annotations evaluate when the
enclosing worker-main function executes, i.e. at worker spawn. (The
``self._*_q: mp.Queue | None`` attribute annotations at :806-808 do NOT
evaluate at runtime — Python skips annotation evaluation for complex targets
in function bodies.)

The gate checks the ``CO_FUTURE_ANNOTATIONS`` flag the future import installs
on the module's own compile (``dont_inherit=True`` — without it, a test file
that itself imports the future would mask the defect) and replays the exact
nested-def pattern that killed worker spawn under those flags.
"""

from __future__ import annotations
import __future__

from pathlib import Path

import domain.infrastructure._internal.model_worker as mw

_SRC = Path(mw.__file__).read_text(encoding="utf-8")
_FUTURE_FLAG = __future__.annotations.compiler_flag


class TestWorkerAnnotationsAreLazy:
    def test_module_compiles_with_future_annotations_flag(self):
        code = compile(_SRC, mw.__file__, "exec", flags=0, dont_inherit=True)
        assert code.co_flags & _FUTURE_FLAG, (
            "model_worker.py must carry a real module-level "
            "'from __future__ import annotations' (a docstring copy does not count)"
        )

    def test_mp_queue_or_none_nested_def_survives_under_module_flags(self):
        # Exact pattern from the nested _stream defs (:385/:659). Under the
        # module's compile flags the def must execute without evaluating
        # annotations — before the fix this TypeError'd at worker spawn.
        module_code = compile(_SRC, mw.__file__, "exec", flags=0, dont_inherit=True)
        snippet = (
            "import multiprocessing as mp\n"
            "def _outer():\n"
            "    def _stream(hb_q: mp.Queue | None = None):\n"
            "        pass\n"
        )
        code = compile(
            snippet,
            "<mp-queue-probe>",
            "exec",
            flags=module_code.co_flags & _FUTURE_FLAG,
            dont_inherit=True,
        )
        namespace: dict = {}
        exec(code, namespace)  # TypeError here before the fix
        namespace["_outer"]()  # nested def executes inside — the spawn path
