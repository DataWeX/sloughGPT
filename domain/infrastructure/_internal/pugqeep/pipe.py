"""
Process queue and Pipe — the single admission door for the execution stack.

PGQ is a library, so this module is engine-agnostic: it knows nothing about
graphs, pools, routing or dispatch policy. It answers one question — how many
processes are admitted and still in flight, and does admitting one more fit?

Two queues, two concerns, deliberately separate:

* :class:`ProcessQueue` manages **processes** — spawned, working, running,
  still in flight. Its ``maxsize`` is the execution stack's diameter.
* ``TaskQueue`` (``task_queue.py``) settles the **tasks it has already taken
  on** — submit / next / complete / fail, persistence, pause / resume. It is
  left exactly as it is; this module does not read it.

Swirls are the tips of the tree — the terminal points where a traversal ends,
e.g. where model loading stops. They are data-side topology and are
deliberately absent here: the execution stack bounds capacity, not shape.
"""

from __future__ import annotations

import threading
import time
from typing import Any

#: Default execution-stack diameter. Matches ``EngineConfig.queue_size``.
DEFAULT_CAPACITY = 128

#: How long a blocked producer sleeps between capacity re-checks.
#:
#: Retiring finished processes is *lazy, on read* — so no completion callback
#: exists to forget, and no notification can be missed: a producer already
#: parked on a full queue re-sweeps on its own clock instead of waiting to be
#: told. Only blocked producers pay this; admission that fits never waits.
_RETRY_INTERVAL = 0.05


def _finished(process: Any) -> bool:
    try:
        return bool(getattr(process, "is_done", False))
    except Exception:
        # An unreadable process must not be able to wedge the queue; treat it
        # as finished so its slot comes back rather than leaking capacity.
        return True


class ProcessQueue:
    """A bounded registry of admitted, not-yet-finished processes.

    The **only** management layer in PGQ. Spawning, pooling and execution
    all report here instead of tracking liveness themselves, so one object
    answers every question about the stack: how large it may become, how
    large it is, and how much room is left.

    ``maxsize`` is the execution stack's diameter. Nothing else in PGQ
    declares it — :meth:`Pipe.diameter` reads it straight from here, so the
    bound has exactly one source of truth and is reported lazily on read.
    """

    __slots__ = ("maxsize", "_items", "_cond", "_closed", "_peak", "_waits")

    def __init__(self, maxsize: int = DEFAULT_CAPACITY) -> None:
        if maxsize <= 0:
            raise ValueError(f"ProcessQueue maxsize must be positive, got {maxsize}")
        self.maxsize = int(maxsize)
        #: insertion-ordered proc.id -> Process, for O(1) removal by id
        self._items: dict[str, Any] = {}
        self._cond = threading.Condition()
        self._closed = False
        self._peak = 0
        self._waits = 0

    # ── the bound, derived on read ───────────────────────────────────────

    def diameter(self) -> int:
        """Max depth the stack may reach — read from ``maxsize``, never re-stated."""
        return self.maxsize

    def usage(self) -> int:
        """Admitted processes that are still in flight."""
        with self._cond:
            return self._usage_locked()

    def headroom(self) -> int:
        """Room left before admission starts blocking."""
        with self._cond:
            return self.maxsize - self._usage_locked()

    @property
    def peak_usage(self) -> int:
        """Deepest the stack has ever been — observed, not configured."""
        with self._cond:
            return self._peak

    @property
    def blocked_waits(self) -> int:
        """How many times admission had to wait: the backpressure signal."""
        with self._cond:
            return self._waits

    def __len__(self) -> int:
        return self.usage()

    def __contains__(self, proc_id: object) -> bool:
        with self._cond:
            return proc_id in self._items

    def __iter__(self):
        with self._cond:
            return iter(list(self._items.values()))

    def __repr__(self) -> str:
        return f"<ProcessQueue {self.usage()}/{self.maxsize}>"

    # ── admission ────────────────────────────────────────────────────────

    def put(self, proc: Any, timeout: float | None = None) -> bool:
        """Admit one process, blocking while the stack is full.

        Full is **backpressure**, not an error: the caller waits for room
        rather than being turned away. ``False`` means the wait timed out or
        the queue was closed.

        Admitting a process that is already admitted is a no-op. Two doors
        (``Engine.spawn`` and ``Pool.branch``) sit in front of this queue and
        a process routinely passes both, so double-counting them would make
        ``usage()`` disagree with reality.
        """
        pid = getattr(proc, "id", None)
        if pid is None:
            raise TypeError(f"cannot admit {type(proc).__name__}: missing id")
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._cond:
            while True:
                if pid in self._items:
                    return True
                if self._closed:
                    return False
                if len(self._items) < self.maxsize:
                    self._items[pid] = proc
                    if len(self._items) > self._peak:
                        self._peak = len(self._items)
                    self._cond.notify_all()
                    return True

                # At the ceiling, and nowhere else. The retire scan costs
                # O(stack depth), so paying it on every admission would make
                # admitting N processes O(N * diameter) -- measured at 9.6 ms
                # per spawn against a 100k queue. Keeping it off the
                # uncontended path leaves that path a plain dict insert;
                # usage()/headroom() still retire on read, and a parked
                # producer re-scans on its own clock.
                if self._retire_locked():
                    continue

                # Capacity hit -> block. This is the only reason admission
                # ever waits; nothing here rejects.
                self._waits += 1
                wait_for = _RETRY_INTERVAL
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    wait_for = min(wait_for, remaining)
                self._cond.wait(wait_for)

    def release(self, proc_or_id: Any) -> bool:
        """Drop a process eagerly.

        Optional. :meth:`usage` retires finished processes anyway, so
        correctness never depends on a caller remembering to call this — but
        calling it frees headroom without waiting for the next read.
        """
        pid = getattr(proc_or_id, "id", proc_or_id)
        with self._cond:
            gone = self._items.pop(pid, None) is not None
            if gone:
                self._cond.notify_all()
            return gone

    def clear(self) -> None:
        """Empty the stack without closing it (Engine teardown)."""
        with self._cond:
            self._items.clear()
            self._cond.notify_all()

    # ── lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Wake every blocked producer and refuse further admission."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def open(self) -> None:
        """Re-arm after :meth:`close` so a stopped Engine can run again."""
        with self._cond:
            self._closed = False
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    def stats(self) -> dict:
        with self._cond:
            self._retire_locked()
            return {
                "diameter": self.maxsize,
                "usage": len(self._items),
                "headroom": self.maxsize - len(self._items),
                "peak": self._peak,
                "waits": self._waits,
                "closed": self._closed,
            }

    # ── internals (caller holds _cond) ───────────────────────────────────

    def _usage_locked(self) -> int:
        self._retire_locked()
        return len(self._items)

    def _retire_locked(self) -> int:
        """Drop everything already finished. Lazy, on read, no callback."""
        done = [pid for pid, p in self._items.items() if _finished(p)]
        for pid in done:
            del self._items[pid]
        if done:
            self._cond.notify_all()
        return len(done)


class Pipe:
    """The single door in front of the execution stack.

    Owns exactly one :class:`ProcessQueue`. Every entry point capable of
    putting a process on the stack goes through here, which is what makes
    ``diameter() / usage() / headroom()`` derivable rather than declared:
    they are three views of one queue, read lazily.
    """

    __slots__ = ("name", "queue")

    def __init__(self, name: str = "pipe", capacity: int = DEFAULT_CAPACITY) -> None:
        self.name = name
        self.queue = ProcessQueue(maxsize=capacity)

    # ── the contract ─────────────────────────────────────────────────────

    def admit(self, proc: Any, timeout: float | None = None) -> bool:
        """Put ``proc`` on the stack, blocking while it is full.

        Returns ``False`` only if the wait timed out or the pipe was closed.
        """
        return self.queue.put(proc, timeout=timeout)

    def release(self, proc_or_id: Any) -> bool:
        """Take ``proc`` off the stack, freeing headroom."""
        return self.queue.release(proc_or_id)

    def owns(self, proc: Any) -> bool:
        """True if ``proc`` is currently admitted here."""
        return getattr(proc, "id", None) in self.queue

    # ── the bound, three derived views of one queue ──────────────────────

    def diameter(self) -> int:
        """Maximum depth of the execution stack."""
        return self.queue.diameter()

    def usage(self) -> int:
        """Depth right now: admitted and still in flight."""
        return self.queue.usage()

    def headroom(self) -> int:
        """Room before admission blocks."""
        return self.queue.headroom()

    def stats(self) -> dict:
        stats = self.queue.stats()
        stats["name"] = self.name
        return stats

    # ── lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        self.queue.close()

    def open(self) -> None:
        self.queue.open()

    @property
    def is_closed(self) -> bool:
        return self.queue.closed

    @property
    def peak_usage(self) -> int:
        return self.queue.peak_usage

    @property
    def blocked_waits(self) -> int:
        return self.queue.blocked_waits

    def __len__(self) -> int:
        return self.queue.usage()

    def __contains__(self, proc: Any) -> bool:
        return self.owns(proc)

    def __repr__(self) -> str:
        return f"<Pipe {self.name} {self.usage()}/{self.diameter()}>"


class PipeClosed(RuntimeError):
    """Admission was refused because the pipe has been shut down."""
