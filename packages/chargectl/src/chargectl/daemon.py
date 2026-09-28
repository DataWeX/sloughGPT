"""daemon — enforce the charge policy on a fixed interval.

The threshold is state that decays: reboot, AC re-plug, suspend and vendor drivers all
reset it, and a laptop that sleeps through a threshold change never gets re-capped. So a
daemon re-asserts the policy on a timer and records what it did in a state file that the
API and the UI read.

This module is deliberately the *only* place with a loop. :mod:`chargectl.policy` stays
pure (rules, no I/O), :mod:`chargectl.control` stays one-shot (one write, no loop), and
the CLI's ``--once`` runs exactly one tick so the whole thing is testable without
sleeping.

A machine with no writable threshold (VM, container) ticks in *dry-run*: it reports what
it would have written and exits 0 — a dev box must not look broken.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import Any

from .control import clear_limit, probe, set_band, set_floor, set_limit
from .policy import (
    CLEAR,
    SET_CEILING,
    SET_FLOOR,
    Policy,
    decide,
    default_state_path,
    explain,
    load_policy,
)
from .status import DEFAULT_SYS_BASE, BatteryReader

logger = logging.getLogger("chargectl.daemon")


def read_state(path: str | Path | None = None) -> dict[str, Any]:
    """Read the daemon state file. Missing or corrupt file → ``{}``."""
    target = Path(path) if path is not None else default_state_path()
    try:
        if not target.is_file():
            return {}
        raw = json.loads(target.read_text() or "{}")
        return raw if isinstance(raw, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def write_state(state: dict[str, Any], path: str | Path | None = None) -> Path:
    """Persist daemon state atomically (tmp file + ``os.replace``). Never raises."""
    target = Path(path) if path is not None else default_state_path()
    payload = json.dumps(state, indent=2, sort_keys=True) + "\n"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".chargectl-", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, target)
    except OSError as exc:  # pragma: no cover - read-only filesystem
        logger.debug("state write failed: %s", exc)
    return target


def apply_directive(
    directive: Any, policy: Policy, sys_base: str | Path = DEFAULT_SYS_BASE
) -> dict[str, Any]:
    """Turn one :class:`~chargectl.policy.Directive` into a threshold write. Never raises."""
    action = directive.action
    if action == CLEAR:
        return clear_limit(sys_base=sys_base).as_dict()
    if action == SET_FLOOR:
        return set_floor(int(directive.value), sys_base=sys_base).as_dict()
    if action == SET_CEILING:
        ceiling = int(directive.value)
        if policy.controls_floor:
            capability = probe(sys_base)
            if capability.start_supported:
                return set_band(policy.floor, ceiling, sys_base=sys_base).as_dict()
        return set_limit(ceiling, sys_base=sys_base).as_dict()
    return {
        "applied": False,
        "supported": False,
        "limit": None,
        "reason": f"unknown action {action!r}",
        "path": None,
    }


def tick(
    policy: Policy,
    *,
    sys_base: str | Path = DEFAULT_SYS_BASE,
    reader: BatteryReader | None = None,
    state_path: str | Path | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """One enforcement pass: read → decide → write → record. Never raises."""
    ts = time.time() if now is None else now
    reader = reader if reader is not None else BatteryReader(sys_base=sys_base)
    state = read_state(state_path)
    owned = bool(state.get("owned"))

    try:
        status = reader.read(now=ts)
        capability = probe(sys_base)
    except Exception as exc:  # pragma: no cover - defensive: a tick must never die
        logger.warning("tick read failed: %s", exc)
        return {**state, "updated_at": ts, "error": f"{type(exc).__name__}: {exc}"}

    directive = decide(
        status,
        capability,
        policy,
        current_limit=capability.current_limit,
        current_floor=capability.current_floor,
        owned=owned,
    )

    result: dict[str, Any] | None = None
    if directive is None:
        action: dict[str, Any] | None = None
    else:
        action = directive.as_dict()
        try:
            result = apply_directive(directive, policy, sys_base=sys_base)
        except Exception as exc:  # pragma: no cover - control never raises, belt and braces
            result = {"applied": False, "reason": f"{type(exc).__name__}: {exc}"}
        if result.get("applied"):
            owned = directive.action != CLEAR

    new_state: dict[str, Any] = {
        "pid": os.getpid(),
        "updated_at": ts,
        "enabled": bool(policy.enabled),
        "policy": policy.as_dict(),
        "explain": explain(policy, capability),
        "status": status.as_dict(),
        "capability": {
            "supported": capability.supported,
            "writable": capability.writable,
            "start_supported": capability.start_supported,
            "current_limit": capability.current_limit,
            "current_floor": capability.current_floor,
            "incumbent": capability.incumbent,
            "reason": capability.reason,
        },
        "action": action,
        "result": result,
        "owned": owned,
        "dry_run": not capability.supported,
        "error": None,
    }
    write_state(new_state, state_path)
    return new_state


def run_daemon(
    *,
    interval: float | None = None,
    once: bool = False,
    policy_path: str | Path | None = None,
    state_path: str | Path | None = None,
    sys_base: str | Path = DEFAULT_SYS_BASE,
) -> int:
    """Run ticks until stopped. ``once`` runs a single tick and returns.

    Returns a process exit code: ``0`` always — an unsupported machine is a valid
    machine, and systemd should not restart-loop it.
    """
    logging.basicConfig(
        level=os.environ.get("CHARGECTL_LOG", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    stopping = {"flag": False}

    def _stop(signum: int, _frame: Any) -> None:
        stopping["flag"] = True
        logger.info("signal %s received — finishing tick", signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _stop)
        except (ValueError, OSError):  # pragma: no cover - not the main thread
            pass

    logger.info("chargectl daemon starting (once=%s)", once)
    while True:
        policy, error = load_policy(policy_path)
        if error:
            logger.warning("policy unreadable (%s) — using defaults", error)
        try:
            state = tick(policy, sys_base=sys_base, state_path=state_path)
        except Exception as exc:  # pragma: no cover - tick already guards itself
            logger.warning("tick failed: %s", exc)
            state = {}
        _log_tick(policy, state)

        if once:
            return 0
        wait = float(interval if interval is not None else policy.interval_seconds)
        waited = 0.0
        while waited < wait and not stopping["flag"]:
            time.sleep(1.0)
            waited += 1.0
            if waited % 30 == 0:  # pick up policy edits without waiting a full interval
                break
        if stopping["flag"]:
            break
    logger.info("chargectl daemon stopped")
    return 0


def _log_tick(policy: Policy, state: dict[str, Any]) -> None:
    action = state.get("action")
    result = state.get("result") or {}
    if action is None:
        logger.debug("idle: %s", state.get("explain", ""))
        return
    applied = result.get("applied")
    level = logging.INFO if applied else logging.WARNING
    logger.log(
        level,
        "%s → %s%s | %s",
        action.get("action"),
        action.get("value"),
        "" if applied is None else (" (applied)" if applied else " (NOT applied)"),
        result.get("reason") or action.get("reason"),
    )


__all__ = ["apply_directive", "read_state", "run_daemon", "tick", "write_state"]
