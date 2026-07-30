"""
Shell permissions manager -- gate destructive operations.

Every command that can modify state, destroy data, or affect the system
is classified by risk level.  The permissions manager decides what runs
and what requires explicit grant.

Risk levels:
  SAFE      -- read-only, no side effects (help, health, status, ...)
  ELEVATED  -- modifies shell state (alias, set, source, py, ...)
  DANGEROUS -- modifies filesystem or external state (protect, ...)
  CRITICAL  -- affects the whole system (shutdown, boot, svc, load, ...)

Usage:
    perms = ShellPermissions()
    perms.check("rm", args)        # raises PermissionError if not granted
    perms.grant("rm")             # allow for this session
    perms.grant("rm", persist=True)  # allow and save to disk
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class Risk:
    SAFE = "safe"
    ELEVATED = "elevated"
    DANGEROUS = "dangerous"
    CRITICAL = "critical"


# ── Command → risk mapping ──────────────────────────────────────────

# Read-only / no side effects
_SAFE = frozenset({
    "help", "exit", "which",
    "type", "history", "fc", "whoami",
    "tutorial", "status", "metrics", "health", "tokenizer",
    "devices", "lsdev", "asm", "wm", "win", "uptime",
    "pwd", "echo", "cat", "head", "tail", "wc", "less",
    "sort", "uniq", "find", "grep", "ls", "tee", "xargs", "diff", "stat", "du",
    "cut", "tr", "seq", "nl", "fold", "tac", "env", "printenv", "yes", "realpath",
    "dirname", "basename", "nproc", "hostname", "uname", "shuf", "rev", "paste", "comm",
})

# Modifies shell state only (aliases, env, history, variables, jobs)
_ELEVATED = frozenset({
    "alias", "unalias", "set", "export", "read", "source", ".",
    "py", "ai", "cd",
    "bg", "fg", "watch",
})

# Modifies filesystem or external resources
_DANGEROUS = frozenset({
    "protect", "unprotect",
    "rm", "chmod", "chown", "mv", "cp", "dd", "touch",
    "mkfs", "fsck", "fdisk", "mount", "umount",
    "mkdir", "rmdir",
})

# Affects system, models, training, processes, services
_CRITICAL = frozenset({
    "boot", "shutdown", "svc", "load", "unload", "switch",
    "train", "kill", "gen", "chat", "agents", "remember",
    "recall", "note", "api", "vmrun", "vmperms",
    "permit", "deny", "permissions", "confirm",
})

# Command → risk classification
_RISK_MAP: dict[str, str] = {}
for _cmd in _SAFE:
    _RISK_MAP[_cmd] = Risk.SAFE
for _cmd in _ELEVATED:
    _RISK_MAP[_cmd] = Risk.ELEVATED
for _cmd in _DANGEROUS:
    _RISK_MAP[_cmd] = Risk.DANGEROUS
for _cmd in _CRITICAL:
    _RISK_MAP[_cmd] = Risk.CRITICAL

# rm -rf is always critical regardless of command name
# Force patterns — specific args escalate risk regardless of base command
_FORCE_PATTERNS = {
    "rm": {"-rf", "-fr", "--recursive", "-r", "-R"},
    "chmod": {"777", "000", "a+rwx", "a-rwx", "a+rw", "a-rw"},
}


class ShellPermissions:
    """Gate destructive shell operations by risk level.

    Default policy:
      SAFE      → always allowed
      ELEVATED  → allowed (shell-local state)
      DANGEROUS → blocked until granted
      CRITICAL  → blocked until granted

    Grants persist to ``~/.config/shell_permissions.json`` when
    ``persist=True``.
    """

    _config_path = Path.home() / ".config" / "shell_permissions.json"

    def __init__(self) -> None:
        self._granted: set[str] = set()
        self._denied: set[str] = set()
        self._policy: dict[str, str] = {
            Risk.SAFE: "allow",
            Risk.ELEVATED: "allow",
            Risk.DANGEROUS: "deny",
            Risk.CRITICAL: "deny",
        }
        self._load_persistent()

    # ── Core API ─────────────────────────────────────────────────

    def check(self, cmd: str, args: str = "") -> None:
        """Check if a command is allowed.  Raises PermissionError if denied."""
        risk = self.classify(cmd, args)
        action = self._policy.get(risk, "deny")

        if action == "allow":
            return

        if cmd in self._granted or f"{cmd} {args}" in self._granted:
            return

        if cmd in self._denied:
            raise PermissionError(
                f"Permission denied: {cmd} (risk={risk}). "
                f"Use `permit {cmd}` to grant."
            )

        raise PermissionError(
            f"Permission denied: {cmd} (risk={risk}). "
            f"Use `permit {cmd}` to grant, or `permit --all-{risk}` to allow all {risk} commands."
        )

    def grant(self, cmd: str, persist: bool = False) -> None:
        """Grant permission for a command."""
        self._granted.add(cmd)
        self._denied.discard(cmd)
        if persist:
            self._save_persistent()

    def revoke(self, cmd: str, persist: bool = False) -> None:
        """Revoke a previously granted permission."""
        self._granted.discard(cmd)
        if persist:
            self._save_persistent()

    def set_policy(self, risk: str, action: str) -> None:
        """Set policy for a risk level: 'allow' or 'deny'."""
        if action not in ("allow", "deny"):
            raise ValueError(f"action must be 'allow' or 'deny', got {action!r}")
        self._policy[risk] = action

    def is_granted(self, cmd: str) -> bool:
        return cmd in self._granted

    def classify(self, cmd: str, args: str = "") -> str:
        """Classify a command's risk level."""
        base = cmd.lstrip("-").split()[0] if cmd.startswith("-") else cmd
        # Check force patterns (rm -rf → critical even though rm is dangerous)
        patterns = _FORCE_PATTERNS.get(base, set())
        if patterns and args:
            arg_parts = set(args.split())
            if patterns & arg_parts:
                return Risk.CRITICAL
        return _RISK_MAP.get(base, Risk.ELEVATED)

    def list_granted(self) -> list[str]:
        return sorted(self._granted)

    def list_dangerous(self) -> list[str]:
        return sorted(_DANGEROUS | _CRITICAL)

    # ── Persistence ──────────────────────────────────────────────

    def _load_persistent(self) -> None:
        if self._config_path.exists():
            try:
                data = json.loads(self._config_path.read_text())
                self._granted = set(data.get("granted", []))
                self._policy.update(data.get("policy", {}))
            except Exception:
                pass

    def _save_persistent(self) -> None:
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "granted": sorted(self._granted),
                "policy": self._policy,
            }
            self._config_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass
