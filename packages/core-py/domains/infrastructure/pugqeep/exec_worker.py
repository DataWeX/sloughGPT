"""Exec-mode child entry point for ``SubprocessProcess`` (``start_method="exec"``).

Launched by the parent as ``python -m domains.infrastructure.pugqeep.exec_worker
<channel_fd>``. The job ``(fn, args, kwargs, capture, config)`` arrives pickled
on stdin; the framed PGQ channel arrives as an inherited fd.

Why exec and not fork: a fork child inherits every lock its parent's threads
hold — one BLAS/OpenMP futex held at fork time and the child deadlocks before
``fn`` runs its first line. ``exec`` replaces the image, so nothing is
inherited but the channels PGQ explicitly passes (card df5affc7).

Owned by PGQ: no ``multiprocessing`` anywhere in this path — the channel is
the same ``FrameHandler`` the fork child uses, and the job rides stdin so
channel frames never mix with job bytes.
"""

from __future__ import annotations

import pickle
import socket
import sys


def main() -> None:
    fd = int(sys.argv[1])
    sock = socket.socket(fileno=fd)
    fn, args, kwargs, capture, config = pickle.loads(sys.stdin.buffer.read())
    sys.stdin.close()

    from .engine import FrameHandler, _child_main

    handler = FrameHandler(sock)
    # parent_sock=None: the exec child holds only its own channel end.
    _child_main(
        None,
        handler,
        fn=fn,
        args=args,
        kwargs=kwargs,
        capture=capture,
        config=config,
    )


if __name__ == "__main__":
    main()
