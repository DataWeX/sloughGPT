"use strict"

import logging
import re
import shlex

logger = logging.getLogger("slo.infra.shell_sandbox")

# Commands that are never allowed in the shell endpoint
_BLOCKED_COMMANDS = frozenset(
    {
        "rm",
        "rmdir",
        "dd",
        "mkfs",
        "format",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "init",
        "su",
        "sudo",
        "passwd",
        "chown",
        "chmod",
        "chroot",
        "mount",
        "umount",
        "fdisk",
        "parted",
        "iptables",
        "ip6tables",
        "nft",
        "ufw",
        "curl",
        "wget",
        "nc",
        "ncat",
        "netcat",
        "socat",
        "python",
        "python3",
        "perl",
        "ruby",
        "node",
        "php",
        "pip",
        "pip3",
        "npm",
        "npx",
        "yarn",
        "eval",
        "exec",
        "source",
        "fork",
        "nohup",
        "disown",
        "crontab",
        "at",
        "batch",
        "docker",
        "podman",
        "kubectl",
        "helm",
        "ssh",
        "scp",
        "rsync",
        "telnet",
        "ftp",
        "sftp",
    }
)

# Patterns that indicate dangerous operations
_DANGEROUS_PATTERNS = [
    r">\s*/",  # Redirect to root paths
    r">\s*/etc",  # Write to /etc
    r">\s*/var",  # Write to /var
    r">\s*/usr",  # Write to /usr
    r">\s*/tmp",  # Write to /tmp (less dangerous but still suspicious)
    r"\|\s*sh\b",  # Pipe to shell
    r"\|\s*bash\b",  # Pipe to bash
    r"`.*`",  # Backtick execution
    r"\$\(.*\)",  # Subshell execution
    r"169\.254\.",  # AWS metadata
    r"metadata\.google",  # GCP metadata
]

_DANGEROUS_RE = [re.compile(p) for p in _DANGEROUS_PATTERNS]


class ShellSecurityError(Exception):
    """Raised when a command fails security checks."""

    pass


def validate_command(command: str) -> None:
    """Validate a shell command against security policies.

    Raises ShellSecurityError if the command is not allowed.
    """
    command = command.strip()
    if not command:
        raise ShellSecurityError("Empty command")

    # Check dangerous patterns
    for pattern in _DANGEROUS_RE:
        if pattern.search(command):
            logger.warning("Blocked dangerous shell command pattern: %s", command[:80])
            raise ShellSecurityError(f"Command contains blocked pattern: {pattern.pattern}")

    # Parse the command to extract the base command
    try:
        tokens = shlex.split(command)
    except ValueError:
        # shlex.split failed (e.g., unclosed quotes) - allow it, the shell will handle
        return

    if not tokens:
        return

    # Get the base command (first token, strip path)
    base_cmd = tokens[0].rsplit("/", 1)[-1]

    if base_cmd in _BLOCKED_COMMANDS:
        logger.warning("Blocked shell command: %s (from: %s)", base_cmd, command[:80])
        raise ShellSecurityError(f"Command '{base_cmd}' is not allowed")


def get_command_summary(command: str) -> str:
    """Get a safe summary of the command for logging."""
    try:
        tokens = shlex.split(command)
        if len(tokens) <= 3:
            return command
        return f"{tokens[0]} ... ({len(tokens)} args)"
    except ValueError:
        return command[:80]
