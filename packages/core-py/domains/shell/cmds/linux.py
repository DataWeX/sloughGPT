"""
LinuxCommandsMixin — 57 built-in Linux/Unix commands extracted from ShellREPL.

Provides: cd, pwd, echo, ls, cat, mkdir, rm, touch, cp, mv, head, tail, wc,
grep, sort, uniq, find, tee, xargs, time, chmod, du, diff, stat, cut, tr,
seq, nl, fold, tac, env, yes, realpath, dirname, basename, nproc, hostname,
uname, shuf, rev, paste, comm, test, printf, expand, unexpand, id, logname,
mktemp, who, od, join, clear, sleep, date, cal, ln.

Each method accesses ``self._print``, ``self._last_exit_code``, and optionally
``self._piped_input``, ``self._env``, ``self.os.vfs``, ``self._format_size``,
``self._execute_single``, ``self._check_permission`` — all provided by the
host ``ShellREPL`` class.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# ── ANSI helpers (imported from the parent module at class level) ──────
# We import them lazily inside methods that need color to avoid circular
# imports.  The constants are module-level in repl.py and stable.


class LinuxCommandsMixin:
    """Mixin providing 57 built-in Linux/Unix commands."""

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _format_size(size: int, human: bool = False) -> str:
        """Format byte count for human-readable display."""
        if not human:
            return f"{size:>8}"
        for unit in ("B", "K", "M", "G", "T"):
            if size < 1024:
                return f"{size:>4.1f}{unit}"
            size /= 1024
        return f"{size:>4.1f}P"

    # ── file system ─────────────────────────────────────────────────

    def _cmd_cd(self, args: str = "") -> None:
        """Change the working directory."""
        target = args.strip()
        if not target or target == "~":
            target = self._env.get("HOME", str(Path.home()))
        elif target == "-":
            target = self._env.get("OLDPWD", os.getcwd())
        try:
            old_cwd = os.getcwd()
            os.chdir(os.path.expanduser(target))
            self._env["OLDPWD"] = old_cwd
            self._last_exit_code = 0
        except FileNotFoundError:
            self._print(f"  cd: no such file or directory: {target}")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  cd: permission denied: {target}")
            self._last_exit_code = 1
        except NotADirectoryError:
            self._print(f"  cd: not a directory: {target}")
            self._last_exit_code = 1

    def _cmd_pwd(self, args: str = "") -> None:
        """Print the working directory."""
        self._print(os.getcwd())
        self._last_exit_code = 0

    def _cmd_echo(self, args: str = "") -> None:
        """Echo arguments to stdout."""
        self._print(args)
        self._last_exit_code = 0

    def _cmd_ls(self, args: str = "") -> None:
        """List directory contents (VFS-aware).

        Flags:
          -1    One entry per line
          -a    Show hidden entries (starting with .)
          -l    Long listing format
          -S    Sort by file size (largest first)
          -r    Reverse sort order
        """
        parts = args.strip().split() if args else []
        one_per_line = False
        show_hidden = False
        long_format = False
        sort_by_size = False
        reverse_sort = False
        targets = []
        for p in parts:
            if p.startswith("-") and len(p) > 1:
                flags = p[1:]
                if "1" in flags:
                    one_per_line = True
                if "a" in flags:
                    show_hidden = True
                if "l" in flags:
                    long_format = True
                if "S" in flags:
                    sort_by_size = True
                if "r" in flags:
                    reverse_sort = True
            else:
                targets.append(p)
        if not targets:
            targets = ["."]
        for target in targets:
            try:
                vfs = self.os.vfs
                if vfs and (target.startswith("/dev") or target.startswith("/proc")):
                    entries = vfs.listdir(target)
                else:
                    entries = os.listdir(os.path.expanduser(target))
                if entries is None:
                    self._print(f"  ls: cannot access '{target}': No such file or directory")
                    self._last_exit_code = 1
                    continue
                if not show_hidden:
                    entries = [e for e in entries if not e.startswith(".")]
                if sort_by_size:
                    def _size_key(e):
                        p2 = os.path.join(target, e) if target != "." else e
                        try:
                            return os.path.getsize(os.path.expanduser(p2))
                        except OSError:
                            return 0
                    entries.sort(key=_size_key, reverse=True)
                elif long_format:
                    pass
                else:
                    entries.sort()
                if reverse_sort and not sort_by_size:
                    entries.reverse()
                if len(targets) > 1:
                    self._print(f"\n{target}:")
                if long_format:
                    for e in entries:
                        path = os.path.join(target, e) if target != "." else e
                        if vfs:
                            is_dir = vfs.isdir(path)
                        else:
                            is_dir = os.path.isdir(os.path.expanduser(path))
                        prefix = "d" if is_dir else "-"
                        size = 0
                        if not vfs:
                            try:
                                size = os.path.getsize(os.path.expanduser(path))
                            except OSError:
                                pass
                        self._print(f"  {prefix}rwxr-xr-x  1 user  user  {size:>8}  {e}{'/' if is_dir else ''}")
                elif one_per_line:
                    for e in entries:
                        self._print(f"  {e}")
                else:
                    parts_list = []
                    for e in entries:
                        path = os.path.join(target, e) if target != "." else e
                        if vfs:
                            is_dir = vfs.isdir(path)
                        else:
                            is_dir = os.path.isdir(os.path.expanduser(path))
                        parts_list.append(e + ("/" if is_dir else ""))
                    self._print("  " + "  ".join(parts_list))
                self._last_exit_code = 0
            except FileNotFoundError:
                self._print(f"  ls: cannot access '{target}': No such file or directory")
                self._last_exit_code = 1
            except PermissionError:
                self._print(f"  ls: permission denied: {target}")
                self._last_exit_code = 1
            except NotADirectoryError:
                self._print(f"  ls: not a directory: {target}")
                self._last_exit_code = 1

    def _cmd_cat(self, args: str = "") -> None:
        """Concatenate and print files."""
        if not args:
            if self._piped_input:
                self._print(self._piped_input.rstrip("\n"))
                self._last_exit_code = 0
                return
            self._print("  Usage: cat <file>")
            self._last_exit_code = 1
            return
        target = args.strip()
        try:
            vfs = self.os.vfs
            if vfs and (target.startswith("/dev") or target.startswith("/proc")):
                content = vfs.read(target)
            else:
                content = Path(os.path.expanduser(target)).read_text()
            if content is None:
                self._print(f"  cat: {target}: No such file or directory")
                self._last_exit_code = 1
                return
            self._print(content.rstrip("\n"))
            self._last_exit_code = 0
        except FileNotFoundError:
            self._print(f"  cat: {target}: No such file or directory")
            self._last_exit_code = 1
        except IsADirectoryError:
            self._print(f"  cat: {target}: Is a directory")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  cat: permission denied: {target}")
            self._last_exit_code = 1

    def _cmd_mkdir(self, args: str = "") -> None:
        """Create directories (VFS-aware).

        Flags:
          -p    Create parent directories as needed
          -v    Verbose
        """
        if not args:
            self._print("  Usage: mkdir [-pv] <dir> [...]")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        parents = False
        verbose = False
        paths = []
        for p in parts:
            if p.startswith("-") and len(p) > 1:
                flags = p[1:]
                if "p" in flags:
                    parents = True
                if "v" in flags:
                    verbose = True
            else:
                paths.append(p)
        if not paths:
            self._print("  Usage: mkdir [-pv] <dir> [...]")
            self._last_exit_code = 1
            return
        for p in paths:
            target = os.path.expanduser(p)
            try:
                os.makedirs(target, exist_ok=False)
                if verbose:
                    self._print(f"  mkdir: created directory '{target}'")
                self._last_exit_code = 0
            except FileExistsError:
                self._print(f"  mkdir: cannot create directory '{target}': File exists")
                self._last_exit_code = 1
            except PermissionError:
                self._print(f"  mkdir: permission denied: {target}")
                self._last_exit_code = 1
            except FileNotFoundError:
                if parents:
                    try:
                        os.makedirs(target, exist_ok=True)
                        if verbose:
                            self._print(f"  mkdir: created directory '{target}'")
                        self._last_exit_code = 0
                    except PermissionError:
                        self._print(f"  mkdir: permission denied: {target}")
                        self._last_exit_code = 1
                else:
                    self._print(f"  mkdir: cannot create directory '{target}': No such file or directory")
                    self._last_exit_code = 1

    def _cmd_rm(self, args: str = "") -> None:
        """Remove files or directories."""
        if not args:
            self._print("  Usage: rm [-rf] <path>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        flags = [p for p in parts if p.startswith("-")]
        paths = [p for p in parts if not p.startswith("-")]
        recursive = any(f in ("-r", "-rf", "-fr", "-R") for f in flags)
        force = any(f in ("-f", "-rf", "-fr") for f in flags)
        if not paths:
            self._print("  Usage: rm [-rf] <path>")
            self._last_exit_code = 1
            return
        for p in paths:
            target = os.path.expanduser(p)
            try:
                if os.path.isdir(target) and recursive:
                    import shutil as _shutil
                    _shutil.rmtree(target)
                elif os.path.isdir(target):
                    self._print(f"  rm: cannot remove '{p}': Is a directory")
                    self._last_exit_code = 1
                    continue
                else:
                    os.remove(target)
                self._last_exit_code = 0
            except FileNotFoundError:
                if not force:
                    self._print(f"  rm: cannot remove '{p}': No such file or directory")
                    self._last_exit_code = 1
            except PermissionError:
                self._print(f"  rm: permission denied: {p}")
                self._last_exit_code = 1

    def _cmd_touch(self, args: str = "") -> None:
        """Create empty files or update timestamps (VFS-aware).

        Flags:
          -c    Do not create files (only update timestamps)
          -t    Use specified timestamp (ignored, for compat)
        """
        if not args:
            self._print("  Usage: touch [-c] <file> [file...]")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        no_create = False
        paths = []
        for p in parts:
            if p == "-c":
                no_create = True
            elif p.startswith("-") and len(p) > 1:
                pass
            else:
                paths.append(p)
        if not paths:
            self._print("  Usage: touch [-c] <file> [file...]")
            self._last_exit_code = 1
            return
        for p in paths:
            target = os.path.expanduser(p)
            try:
                if os.path.exists(target):
                    os.utime(target, None)
                elif not no_create:
                    Path(target).write_text("")
                self._last_exit_code = 0
            except PermissionError:
                self._print(f"  touch: permission denied: {p}")
                self._last_exit_code = 1

    def _cmd_cp(self, args: str = "") -> None:
        """Copy files (VFS-aware).

        Flags:
          -r    Recursive copy directories
          -f    Force (no prompt)
          -v    Verbose
          -p    Preserve permissions
          -i    Interactive (prompt before overwrite)
        """
        if not args:
            self._print("  Usage: cp [-rfvpi] <src> <dst>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        recursive = False
        force = False
        verbose = False
        preserve = False
        interactive = False
        paths = []
        for p in parts:
            if p.startswith("-") and len(p) > 1:
                flags = p[1:]
                if "r" in flags:
                    recursive = True
                if "f" in flags:
                    force = True
                if "v" in flags:
                    verbose = True
                if "p" in flags:
                    preserve = True
                if "i" in flags:
                    interactive = True
            else:
                paths.append(p)
        if len(paths) < 2:
            self._print("  Usage: cp [-rfvpi] <src> <dst>")
            self._last_exit_code = 1
            return
        src, dst = os.path.expanduser(paths[0]), os.path.expanduser(paths[1])
        try:
            import shutil as _shutil
            if os.path.isdir(src):
                _shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                if interactive and os.path.exists(dst):
                    if not self.console.confirm_overwrite(paths[1]):
                        self._last_exit_code = 1
                        return
                _shutil.copy2(src, dst) if preserve else _shutil.copy(src, dst)
            if verbose:
                self._print(f"  '{paths[0]}' -> '{paths[1]}'")
            self._last_exit_code = 0
        except FileNotFoundError:
            self._print(f"  cp: cannot stat '{paths[0]}': No such file or directory")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  cp: permission denied")
            self._last_exit_code = 1

    def _cmd_mv(self, args: str = "") -> None:
        """Move or rename files (VFS-aware).

        Flags:
          -f    Force (no prompt)
          -v    Verbose
          -i    Interactive (prompt before overwrite)
        """
        if not args:
            self._print("  Usage: mv [-fvi] <src> <dst>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        force = False
        verbose = False
        interactive = False
        paths = []
        for p in parts:
            if p.startswith("-") and len(p) > 1:
                flags = p[1:]
                if "f" in flags:
                    force = True
                if "v" in flags:
                    verbose = True
                if "i" in flags:
                    interactive = True
            else:
                paths.append(p)
        if len(paths) < 2:
            self._print("  Usage: mv [-fvi] <src> <dst>")
            self._last_exit_code = 1
            return
        src, dst = os.path.expanduser(paths[0]), os.path.expanduser(paths[1])
        try:
            if interactive and os.path.exists(dst):
                if not self.console.confirm_overwrite(paths[1]):
                    self._last_exit_code = 1
                    return
            os.rename(src, dst)
            if verbose:
                self._print(f"  '{paths[0]}' -> '{paths[1]}'")
            self._last_exit_code = 0
        except FileNotFoundError:
            self._print(f"  mv: cannot stat '{parts[0]}': No such file or directory")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  mv: permission denied")
            self._last_exit_code = 1

    def _cmd_chmod(self, args: str = "") -> None:
        """Change file permissions (chmod)."""
        if not args:
            self._print("  Usage: chmod <mode> <file>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        if len(parts) < 2:
            self._print("  Usage: chmod <mode> <file>")
            self._last_exit_code = 1
            return
        mode, target = parts[0], os.path.expanduser(parts[1])
        try:
            if mode.isdigit():
                os.chmod(target, int(mode, 8))
            else:
                self._print(f"  chmod: symbolic modes not supported (use octal, e.g. 644)")
                self._last_exit_code = 1
                return
            self._last_exit_code = 0
        except FileNotFoundError:
            self._print(f"  chmod: cannot access '{parts[1]}': No such file or directory")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  chmod: changing permissions of '{parts[1]}': Operation not permitted")
            self._last_exit_code = 1

    def _cmd_ln(self, args: str = "") -> None:
        """Create links: ln [-s] <target> <link_name>"""
        import shlex as _shlex
        argv = _shlex.split(args)
        symlink = False
        target = None
        link_name = None
        i = 0
        while i < len(argv):
            a = argv[i]
            if a in ("-s", "--symbolic"):
                symlink = True
                i += 1
            elif target is None:
                target = a
                i += 1
            elif link_name is None:
                link_name = a
                i += 1
            else:
                i += 1
        if not target or not link_name:
            self._print("  Usage: ln [-s] <target> <link_name>")
            self._last_exit_code = 1
            return
        try:
            if symlink:
                os.symlink(target, link_name)
            else:
                os.link(target, link_name)
        except OSError as ex:
            self._print(f"  ln: {ex}")
            self._last_exit_code = 1

    def _cmd_du(self, args: str = "") -> None:
        """Estimate disk usage of files/directories."""
        parts = args.strip().split() if args else []
        human = any(p == "-h" for p in parts)
        targets = [os.path.expanduser(p) for p in parts if p != "-h"]
        if not targets:
            targets = ["."]
        total = 0
        for target in targets:
            try:
                if os.path.isfile(target):
                    size = os.path.getsize(target)
                    total += size
                    label = self._format_size(size, human)
                    self._print(f"  {label}\t{target}")
                elif os.path.isdir(target):
                    sz = 0
                    for root, dirs, files in os.walk(target):
                        for f in files:
                            try:
                                sz += os.path.getsize(os.path.join(root, f))
                            except OSError:
                                pass
                    total += sz
                    label = self._format_size(sz, human)
                    self._print(f"  {label}\t{target}")
                else:
                    self._print(f"  du: cannot access '{target}': No such file or directory")
            except FileNotFoundError:
                self._print(f"  du: cannot access '{target}': No such file or directory")
        if len(targets) > 1:
            self._print(f"  {self._format_size(total, human)}\ttotal")
        self._last_exit_code = 0

    def _cmd_stat(self, args: str = "") -> None:
        """Display file or directory metadata.

        Flags:
          -c FMT  Custom format: %n name, %s size, %f mode (octal),
                  %F type, %A perms (rwx), %m mtime, %x atime
                  %u uid, %g gid, %h links
        """
        parts = args.strip().split() if args else []
        fmt = None
        target = None
        i = 0
        while i < len(parts):
            p = parts[i]
            if p == "-c" and i + 1 < len(parts):
                fmt = parts[i + 1]
                i += 2
            elif not p.startswith("-") and target is None:
                target = p
                i += 1
            else:
                i += 1
        if not target:
            self._print("  Usage: stat [-c FMT] <path>")
            self._last_exit_code = 1
            return
        target = os.path.expanduser(target)
        try:
            st = os.stat(target)
            import stat as _stat, time as _time
            if fmt:
                type_map = {
                    "directory": "directory",
                    "file": "regular file",
                    "other": "other",
                }
                kind = "directory" if os.path.isdir(target) else "file" if os.path.isfile(target) else "other"
                mtime = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(st.st_mtime))
                atime = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(st.st_atime))
                replacements = {
                    "%n": target,
                    "%s": str(st.st_size),
                    "%f": oct(_stat.S_IMODE(st.st_mode)),
                    "%F": type_map.get(kind, kind),
                    "%A": _stat.filemode(st.st_mode),
                    "%m": mtime,
                    "%x": atime,
                    "%u": str(st.st_uid),
                    "%g": str(st.st_gid),
                    "%h": str(st.st_nlink),
                }
                result = fmt
                for k, v in replacements.items():
                    result = result.replace(k, v)
                self._print(result)
            else:
                mode_str = _stat.filemode(st.st_mode)
                size = st.st_size
                mtime = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(st.st_mtime))
                atime = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(st.st_atime))
                kind = "directory" if os.path.isdir(target) else "file" if os.path.isfile(target) else "other"
                self._print(f"  File: {target}")
                self._print(f"  Size: {size:,} bytes  {self._format_size(size, human=True).strip()}")
                self._print(f"  Type: {kind}")
                self._print(f"  Mode: {mode_str} ({oct(_stat.S_IMODE(st.st_mode))})")
                self._print(f"  Modified: {mtime}")
                self._print(f"  Accessed: {atime}")
            self._last_exit_code = 0
        except FileNotFoundError:
            self._print(f"  stat: cannot stat '{target}': No such file or directory")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  stat: cannot stat '{target}': Permission denied")
            self._last_exit_code = 1

    # ── text processing (VFS-aware, piped-input aware) ──────────────

    def _cmd_head(self, args: str = "") -> None:
        """Output the first part of files (VFS-aware).

        Flags:
          -n N    Show first N lines (default 10)
          -c N    Show first N bytes
          -q      Suppress headers for multiple files
          -v      Always show headers
        """
        parts = args.strip().split() if args else []
        n = 10
        byte_mode = False
        quiet = False
        verbose = False
        targets = []
        i = 0
        while i < len(parts):
            p = parts[i]
            if p.startswith("-") and p[1:].isdigit():
                n = int(p[1:])
                i += 1
            elif p == "-n" and i + 1 < len(parts):
                n = int(parts[i + 1])
                i += 2
            elif p == "-c" and i + 1 < len(parts):
                n = int(parts[i + 1])
                byte_mode = True
                i += 2
            elif p == "-q":
                quiet = True
                i += 1
            elif p == "-v":
                verbose = True
                i += 1
            else:
                targets.append(p)
                i += 1
        if not targets:
            if self._piped_input:
                if byte_mode:
                    data = self._piped_input.encode("utf-8", errors="replace")
                    self._print(data[:n].decode("utf-8", errors="replace"), end="")
                else:
                    lines = self._piped_input.splitlines()
                    self._print("\n".join(lines[:n]))
                self._last_exit_code = 0
                return
            self._print("  Usage: head [-n N] [-c N] [-q] <file>")
            self._last_exit_code = 1
            return
        for path in targets:
            target = os.path.expanduser(path)
            try:
                vfs = self.os.vfs
                if vfs and (target.startswith("/dev") or target.startswith("/proc")):
                    content = vfs.read(target)
                else:
                    content = Path(target).read_text()
                if content is None:
                    self._print(f"  head: {path}: No such file or directory")
                    self._last_exit_code = 1
                    continue
                if byte_mode:
                    data = content.encode("utf-8", errors="replace")
                    result = data[:n].decode("utf-8", errors="replace")
                else:
                    lines = content.splitlines()
                    result = "\n".join(lines[:n])
                if len(targets) > 1 and not quiet:
                    self._print(f"==> {path} <==")
                self._print(result)
                self._last_exit_code = 0
            except FileNotFoundError:
                self._print(f"  head: {path}: No such file or directory")
                self._last_exit_code = 1

    def _cmd_tail(self, args: str = "") -> None:
        """Output the last part of files (VFS-aware).

        Flags:
          -n N  Show last N lines (default 10)
          -q    Suppress headers for multiple files
          -c N  Show last N bytes
        """
        parts = args.strip().split() if args else []
        n = 10
        byte_mode = False
        quiet = False
        targets = []
        i = 0
        while i < len(parts):
            p = parts[i]
            if p.startswith("-") and p[1:].isdigit():
                n = int(p[1:])
                i += 1
            elif p == "-n" and i + 1 < len(parts):
                n = int(parts[i + 1])
                i += 2
            elif p == "-c" and i + 1 < len(parts):
                n = int(parts[i + 1])
                byte_mode = True
                i += 2
            elif p == "-q":
                quiet = True
                i += 1
            else:
                targets.append(p)
                i += 1
        if not targets:
            if self._piped_input:
                if byte_mode:
                    data = self._piped_input.encode("utf-8", errors="replace")
                    self._print(data[-n:].decode("utf-8", errors="replace"), end="")
                else:
                    lines = self._piped_input.splitlines()
                    self._print("\n".join(lines[-n:]))
                self._last_exit_code = 0
                return
            self._print("  Usage: tail [-n N] [-c N] [-q] <file>")
            self._last_exit_code = 1
            return
        for path in targets:
            target = os.path.expanduser(path)
            try:
                vfs = self.os.vfs
                if vfs and (target.startswith("/dev") or target.startswith("/proc")):
                    content = vfs.read(target)
                else:
                    content = Path(target).read_text()
                if content is None:
                    self._print(f"  tail: {path}: No such file or directory")
                    self._last_exit_code = 1
                    continue
                if byte_mode:
                    data = content.encode("utf-8", errors="replace")
                    result = data[-n:].decode("utf-8", errors="replace")
                else:
                    lines = content.splitlines()
                    result = "\n".join(lines[-n:])
                if len(targets) > 1 and not quiet:
                    self._print(f"==> {path} <==")
                self._print(result)
                self._last_exit_code = 0
            except FileNotFoundError:
                self._print(f"  tail: {path}: No such file or directory")
                self._last_exit_code = 1

    def _cmd_wc(self, args: str = "") -> None:
        """Count lines, words, and characters (VFS-aware).

        Flags:
          -l  Show line count
          -w  Show word count
          -c  Show byte count
          -m  Show character count
          -L  Show length of longest line
        """
        parts = args.strip().split() if args else []
        show_lines = show_words = show_chars = show_maxlen = False
        targets = []
        for p in parts:
            if p == "-l":
                show_lines = True
            elif p == "-w":
                show_words = True
            elif p == "-c":
                show_chars = True
            elif p == "-m":
                show_chars = True
            elif p == "-L":
                show_maxlen = True
            elif not p.startswith("-"):
                targets.append(p)
        if not show_lines and not show_words and not show_chars and not show_maxlen:
            show_lines = show_words = show_chars = True
        def _count(content):
            lines = len(content.splitlines())
            words = len(content.split())
            chars = len(content)
            maxlen = max((len(l) for l in content.splitlines()), default=0)
            return lines, words, chars, maxlen
        if not targets:
            if self._piped_input:
                lines, words, chars, maxlen = _count(self._piped_input)
                parts_list = []
                if show_lines:
                    parts_list.append(f"{lines:4}")
                if show_words:
                    parts_list.append(f"{words:4}")
                if show_chars:
                    parts_list.append(f"{chars:4}")
                if show_maxlen:
                    parts_list.append(f"{maxlen:4}")
                self._print(" ".join(parts_list))
                self._last_exit_code = 0
                return
            self._print("  Usage: wc [-l] [-w] [-c] [-m] [-L] <file>")
            self._last_exit_code = 1
            return
        total_l = total_w = total_c = total_ml = 0
        for target in targets:
            target = os.path.expanduser(target)
            try:
                vfs = self.os.vfs
                if vfs and (target.startswith("/dev") or target.startswith("/proc")):
                    content = vfs.read(target)
                else:
                    content = Path(target).read_text()
                if content is None:
                    self._print(f"  wc: {target}: No such file or directory")
                    self._last_exit_code = 1
                    return
                lines, words, chars, maxlen = _count(content)
                total_l += lines
                total_w += words
                total_c += chars
                total_ml = max(total_ml, maxlen)
                parts_list = []
                if show_lines:
                    parts_list.append(f"{lines:4}")
                if show_words:
                    parts_list.append(f"{words:4}")
                if show_chars:
                    parts_list.append(f"{chars:4}")
                if show_maxlen:
                    parts_list.append(f"{maxlen:4}")
                if len(targets) > 1:
                    parts_list.append(target)
                self._print(" ".join(parts_list))
            except FileNotFoundError:
                self._print(f"  wc: {target}: No such file or directory")
                self._last_exit_code = 1
                return
        if len(targets) > 1:
            parts_list = []
            if show_lines:
                parts_list.append(f"{total_l:4}")
            if show_words:
                parts_list.append(f"{total_w:4}")
            if show_chars:
                parts_list.append(f"{total_c:4}")
            if show_maxlen:
                parts_list.append(f"{total_ml:4}")
            parts_list.append("total")
            self._print(" ".join(parts_list))
        self._last_exit_code = 0

    def _cmd_grep(self, args: str = "") -> None:
        """Search for patterns in files or piped input (VFS-aware).

        Flags:
            -i  Ignore case
            -v  Invert match
            -c  Count matching lines only
            -l  Print only filenames with matches
            -n  Prefix each output line with line number
            -w  Match whole words only
            -o  Print only the matched part of each line
            -m N  Stop after N matches
            -e PATTERN  Use PATTERN as the pattern (allows patterns starting with -)
            -E  Extended regular expressions
            -A N  Print N lines after each match
            -B N  Print N lines before each match
            -C N  Print N lines of context (both sides)
        """
        if not args and not self._piped_input:
            self._print("  Usage: grep [flags] <pattern> [file...]")
            self._last_exit_code = 1
            return
        import re as _re
        parts = args.strip().split()

        # Parse -e patterns (collected separately)
        e_patterns: list[str] = []
        i = 0
        clean_parts = []
        while i < len(parts):
            if parts[i] == "-e" and i + 1 < len(parts):
                e_patterns.append(parts[i + 1])
                i += 2
            elif parts[i] in ("-m", "-A", "-B", "-C") and i + 1 < len(parts):
                clean_parts.append(parts[i])
                i += 2
            else:
                clean_parts.append(parts[i])
                i += 1

        flags = [p for p in clean_parts if p.startswith("-")]
        non_flags = [p for p in clean_parts if not p.startswith("-")]

        # Parse simple flags
        ignore_case = any(f in ("-i", "-vi") for f in flags)
        invert = any(f in ("-v", "-vi") for f in flags)
        count_only = "-c" in flags
        files_only = "-l" in flags
        line_numbers = "-n" in flags
        word_boundary = "-w" in flags
        only_matching = "-o" in flags
        extended_regex = "-E" in flags
        max_count = 0
        for idx_p, p in enumerate(parts):
            if p.startswith("-m") and len(p) > 2:
                try:
                    max_count = int(p[2:])
                except ValueError:
                    pass
            elif p == "-m" and idx_p + 1 < len(parts):
                try:
                    max_count = int(parts[idx_p + 1])
                except ValueError:
                    pass

        # Parse -A/-B/-C with numeric args from the original parts list
        after_context = 0
        before_context = 0
        context = 0
        j = 0
        while j < len(parts):
            if parts[j] in ("-A", "-B", "-C") and j + 1 < len(parts):
                try:
                    n = int(parts[j + 1])
                    if parts[j] == "-A":
                        after_context = n
                    elif parts[j] == "-B":
                        before_context = n
                    else:
                        context = n
                    j += 2
                    continue
                except ValueError:
                    pass
            j += 1

        # Apply -C as both -A and -B
        if context > 0:
            after_context = max(after_context, context)
            before_context = max(before_context, context)

        # Combine -e patterns with positional pattern
        pattern = "|".join(e_patterns) if e_patterns else (non_flags[0] if non_flags else "")
        targets = non_flags[1:] if len(e_patterns) == 0 and len(non_flags) > 1 else (non_flags if e_patterns and len(non_flags) > 0 else [])
        if not e_patterns:
            targets = non_flags[1:] if len(non_flags) > 1 else []
        recursive = any(f in ("-r", "-R") for f in flags)

        if not pattern:
            self._print("  Usage: grep [flags] <pattern> [file...]")
            self._last_exit_code = 1
            return

        try:
            # Build regex
            kwargs = {"flags": _re.IGNORECASE} if ignore_case else {}
            if extended_regex:
                pat = _re.compile(pattern, _re.VERBOSE | (_re.IGNORECASE if ignore_case else 0))
            elif word_boundary:
                pat = _re.compile(r"\b" + pattern + r"\b", **kwargs) if kwargs else _re.compile(r"\b" + pattern + r"\b")
            else:
                pat = _re.compile(pattern, **kwargs) if kwargs else _re.compile(pattern)
        except _re.error as e:
            self._print(f"  grep: invalid pattern: {e}")
            self._last_exit_code = 2
            return

        matched_any = False
        try:
            if targets:
                # Collect (label, content) pairs from all targets
                file_pairs: list[tuple[str, str]] = []
                for t in targets:
                    t_path = os.path.expanduser(t)
                    if recursive and os.path.isdir(t_path):
                        for root, _dirs, fnames in os.walk(t_path):
                            for fn in sorted(fnames):
                                fp = os.path.join(root, fn)
                                try:
                                    c = Path(fp).read_text()
                                    file_pairs.append((fp, c))
                                except (OSError, UnicodeDecodeError):
                                    pass
                    elif os.path.isdir(t_path):
                        self._print(f"  grep: {t}: Is a directory")
                        self._last_exit_code = 2
                        return
                    else:
                        vfs = self.os.vfs
                        if vfs and (t_path.startswith("/dev") or t_path.startswith("/proc")):
                            content = vfs.read(t_path)
                        else:
                            content = Path(t_path).read_text()
                        if content is None:
                            self._print(f"  grep: {t}: No such file or directory")
                            self._last_exit_code = 1
                            return
                        file_pairs.append((t, content))
                for label, content in file_pairs:
                    lines = content.splitlines()
                    found = self._grep_search(lines, pat, invert, count_only, files_only,
                                              line_numbers, after_context, before_context, label,
                                              only_matching=only_matching, max_count=max_count)
                    if found:
                        matched_any = True
            else:
                lines = self._piped_input.splitlines()
                matched_any = self._grep_search(lines, pat, invert, count_only, files_only,
                                                line_numbers, after_context, before_context, None,
                                                only_matching=only_matching, max_count=max_count)
            self._last_exit_code = 0 if matched_any else 1
        except FileNotFoundError:
            self._print(f"  grep: {targets[0]}: No such file or directory")
            self._last_exit_code = 1

    def _grep_search(self, lines, pat, invert, count_only, files_only,
                     line_numbers, after_context, before_context, label,
                     only_matching=False, max_count=0):
        """Core grep logic on a list of lines. Returns True if any match."""
        matched_indices = set()
        match_count = 0
        for idx, line in enumerate(lines):
            found = bool(pat.search(line))
            if invert:
                found = not found
            if found:
                match_count += 1
                if max_count > 0 and match_count > max_count:
                    break
                matched_indices.add(idx)

        if count_only:
            c = min(len(matched_indices), max_count) if max_count > 0 else len(matched_indices)
            if label and len(lines) > 0:
                self._print(f"  {label}:{c}")
            else:
                self._print(f"  {c}")
            return bool(matched_indices)

        if files_only:
            if matched_indices:
                self._print(f"  {label or '<stdin>'}")
            return bool(matched_indices)

        # Build output with context
        output_indices = set()
        for idx in matched_indices:
            for b in range(max(0, idx - before_context), idx):
                output_indices.add(b)
            for a in range(idx + 1, min(len(lines), idx + 1 + after_context)):
                output_indices.add(a)
            output_indices.add(idx)

        matched = 0
        prev_printed = -2
        for idx in sorted(output_indices):
            if max_count > 0 and matched >= max_count:
                break
            if before_context > 0 or after_context > 0:
                if idx not in matched_indices and (prev_printed < idx - 1):
                    if prev_printed >= 0:
                        self._print("  --")
                    for b in range(max(0, idx - before_context), idx):
                        if b in output_indices and b not in matched_indices:
                            prefix = f"{b + 1}:" if line_numbers else ""
                            self._print(f"  {prefix}{lines[b]}")
            if idx in matched_indices and only_matching:
                for m in pat.finditer(lines[idx]):
                    prefix = f"{idx + 1}:" if line_numbers else ""
                    self._print(f"  {prefix}{m.group()}")
            else:
                prefix = f"{idx + 1}:" if line_numbers else ""
                self._print(f"  {prefix}{lines[idx]}")
            if idx in matched_indices:
                matched += 1
            prev_printed = idx
        return matched > 0

    def _cmd_sort(self, args: str = "") -> None:
        """Sort lines of text (from file or piped input).

        Flags:
          -r  Reverse sort
          -n  Numeric sort (by first field)
          -f  Case-insensitive sort
          -u  Unique lines only
          -R  Random shuffle
          -k N  Sort by field N (1-based)
          -t C  Field separator character
        """
        parts = args.strip().split() if args else []
        flags = [p for p in parts if p.startswith("-")]
        reverse = any(f in ("-r",) for f in flags)
        random_shuffle = any(f in ("-R",) for f in flags)
        numeric = any(f in ("-n", "-g") for f in flags)
        ignore_case = any(f in ("-f",) for f in flags)
        unique = any(f in ("-u",) for f in flags)
        sort_key = 1
        field_sep = None
        targets = []
        i = 0
        while i < len(parts):
            p = parts[i]
            if p == "-k" and i + 1 < len(parts):
                try:
                    sort_key = int(parts[i + 1])
                except ValueError:
                    pass
                i += 2
            elif p == "-t" and i + 1 < len(parts):
                field_sep = parts[i + 1]
                i += 2
            elif p.startswith("-k") and len(p) > 2:
                try:
                    sort_key = int(p[2:])
                except ValueError:
                    pass
                i += 1
            elif p.startswith("-t") and len(p) > 2:
                field_sep = p[2:]
                i += 1
            elif not p.startswith("-"):
                targets.append(p)
                i += 1
            else:
                i += 1
        if targets:
            target = os.path.expanduser(targets[0])
            try:
                lines = Path(target).read_text().splitlines()
            except FileNotFoundError:
                self._print(f"  sort: {targets[0]}: No such file or directory")
                self._last_exit_code = 1
                return
        elif self._piped_input:
            lines = self._piped_input.splitlines()
        else:
            self._print("  Usage: sort [-r] [-n] [-u] [-k N] [-t C] [file]")
            self._last_exit_code = 1
            return

        def _sort_key(line):
            if field_sep is not None:
                fields = line.split(field_sep)
            else:
                fields = line.split()
            idx = min(sort_key - 1, len(fields) - 1) if fields else 0
            val = fields[idx] if fields else ""
            if numeric:
                try:
                    return float(val)
                except ValueError:
                    return 0.0
            if ignore_case:
                return val.lower()
            return val

        if random_shuffle:
            import random as _random
            _random.shuffle(lines)
        else:
            lines.sort(key=_sort_key, reverse=reverse)
        if unique:
            seen = set()
            deduped = []
            for l in lines:
                if l not in seen:
                    seen.add(l)
                    deduped.append(l)
            lines = deduped
        self._print("\n".join(lines))
        self._last_exit_code = 0

    def _cmd_uniq(self, args: str = "") -> None:
        """Remove adjacent duplicate lines. Flags: -c count, -i case-insensitive, -d duplicates-only, -u unique-only, -f N skip fields, -s N skip chars."""
        parts = args.strip().split() if args else []
        count_mode = False
        ignore_case = False
        dup_only = False
        uniq_only = False
        skip_fields = 0
        skip_chars = 0
        target = None
        i = 0
        while i < len(parts):
            p = parts[i]
            if p == "-c":
                count_mode = True
                i += 1
            elif p == "-i":
                ignore_case = True
                i += 1
            elif p == "-d":
                dup_only = True
                i += 1
            elif p == "-u":
                uniq_only = True
                i += 1
            elif p == "-f" and i + 1 < len(parts):
                try:
                    skip_fields = int(parts[i + 1])
                except ValueError:
                    pass
                i += 2
            elif p.startswith("-f") and len(p) > 2:
                try:
                    skip_fields = int(p[2:])
                except ValueError:
                    pass
                i += 1
            elif p == "-s" and i + 1 < len(parts):
                try:
                    skip_chars = int(parts[i + 1])
                except ValueError:
                    pass
                i += 2
            elif p.startswith("-s") and len(p) > 2:
                try:
                    skip_chars = int(p[2:])
                except ValueError:
                    pass
                i += 1
            elif not p.startswith("-"):
                target = p
                i += 1
            else:
                i += 1
        if target:
            try:
                lines = Path(os.path.expanduser(target)).read_text().splitlines()
            except FileNotFoundError:
                self._print(f"  uniq: {target}: No such file or directory")
                self._last_exit_code = 1
                return
        elif self._piped_input:
            lines = self._piped_input.splitlines()
        else:
            self._print("  Usage: uniq [-c] [-i] [-d] [-u] [-f N] [-s N] [file]")
            self._last_exit_code = 1
            return
        def _key(line):
            k = line
            if skip_fields > 0:
                fields = k.split(None, skip_fields)
                k = fields[-1] if fields else ""
            if skip_chars > 0:
                k = k[skip_chars:]
            if ignore_case:
                k = k.lower()
            return k
        out = []
        groups = []
        if lines:
            cur_key = _key(lines[0])
            cur_group = [lines[0]]
            for l in lines[1:]:
                k = _key(l)
                if k == cur_key:
                    cur_group.append(l)
                else:
                    groups.append((cur_key, cur_group))
                    cur_key = k
                    cur_group = [l]
            groups.append((cur_key, cur_group))
        for key, group in groups:
            show = True
            if dup_only and len(group) == 1:
                show = False
            if uniq_only and len(group) > 1:
                show = False
            if not show:
                continue
            if count_mode:
                out.append(f"{len(group):>7} {group[0]}")
            else:
                out.append(group[0])
        self._print("\n".join(out))
        self._last_exit_code = 0

    def _cmd_find(self, args: str = "") -> None:
        """Search for files by name pattern (VFS-aware).

        Flags:
          -name PATTERN   Match filename against pattern
          -iname PATTERN  Case-insensitive name match
          -type f|d       Filter by type (file or directory)
          -maxdepth N     Limit recursion depth
        """
        if not args:
            self._print("  Usage: find [dir] [-name pattern] [-type f|d] [-maxdepth N]")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        search_dir = "."
        pattern = None
        file_type = None
        max_depth = None
        i = 0
        while i < len(parts):
            p = parts[i]
            if p in ("-name", "-iname") and i + 1 < len(parts):
                import fnmatch as _fnmatch
                pat_val = parts[i + 1]
                if p == "-iname":
                    _match_fn = lambda name, _p=pat_val.lower(): _fnmatch.fnmatch(name.lower(), _p)
                else:
                    _match_fn = lambda name, _p=pat_val: _fnmatch.fnmatch(name, _p)
                pattern = pat_val
                i += 2
            elif p == "-type" and i + 1 < len(parts):
                file_type = parts[i + 1]
                i += 2
            elif p == "-maxdepth" and i + 1 < len(parts):
                try:
                    max_depth = int(parts[i + 1])
                except ValueError:
                    pass
                i += 2
            elif not p.startswith("-"):
                search_dir = p
                i += 1
            else:
                i += 1
        if pattern is None and file_type is None:
            self._print("  Usage: find [dir] [-name pattern] [-type f|d] [-maxdepth N]")
            self._last_exit_code = 1
            return
        search_path = os.path.expanduser(search_dir)
        try:
            matches = []
            for root, dirs, files in os.walk(search_path):
                depth = root.replace(search_path, "").count(os.sep)
                if max_depth is not None and depth >= max_depth:
                    dirs.clear()
                    continue
                names = []
                if file_type == "f":
                    names = files
                elif file_type == "d":
                    names = dirs
                else:
                    names = files + dirs
                for name in names:
                    if pattern is None or _match_fn(name):
                        matches.append(os.path.join(root, name))
            if matches:
                self._print("\n".join(matches))
            self._last_exit_code = 0
        except FileNotFoundError:
            self._print(f"  find: '{search_dir}': No such file or directory")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  find: '{search_dir}': Permission denied")
            self._last_exit_code = 1

    def _cmd_tee(self, args: str = "") -> None:
        """Read stdin and write to both stdout and file(s)."""
        if not self._piped_input:
            self._print("  Usage: <command> | tee [-a] <file>")
            self._last_exit_code = 1
            return
        parts = args.strip().split() if args else []
        append = any(p == "-a" for p in parts)
        files = [p for p in parts if p != "-a"]
        mode = "a" if append else "w"
        for fname in files:
            try:
                with open(os.path.expanduser(fname), mode) as f:
                    f.write(self._piped_input)
                    if not self._piped_input.endswith("\n"):
                        f.write("\n")
            except (OSError, PermissionError) as e:
                self._print(f"  tee: {fname}: {e}")
                self._last_exit_code = 1
                return
        self._print(self._piped_input.rstrip("\n"))
        self._last_exit_code = 0

    @staticmethod
    def _expand_posix_class(s: str) -> str:
        """Expand POSIX character class notation like [:alpha:] to character list."""
        s = s.strip("'\"")
        if not (s.startswith("[:") and s.endswith(":]")):
            return s
        cls = s[2:-2]
        classes = {
            "alpha": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "digit": "0123456789",
            "alnum": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "lower": "abcdefghijklmnopqrstuvwxyz",
            "upper": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "space": " \t\n\r\f\v",
            "blank": " \t",
            "punct": '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
            "xdigit": "0123456789abcdefABCDEF",
        }
        return classes.get(cls, s)

    def _cmd_cut(self, args: str = "") -> None:
        """Cut fields, characters, or bytes from lines of text (file or piped input)."""
        if not args and not self._piped_input:
            self._print("  Usage: cut -f<N>[,...] [-d<delim>] [-s] [file]  or  cut -c<N>[,...] [file]  or  cut -b<N>[,...] [file]")
            self._last_exit_code = 1
            return
        parts = args.strip().split() if args else []
        delim = "\t"
        fields = []
        char_ranges = []
        byte_ranges = []
        suppress_no_delim = False
        target = None
        for p in parts:
            if p.startswith("-d") and len(p) > 2:
                delim = self._expand_posix_class(p[2:])
            elif p == "-s":
                suppress_no_delim = True
            elif p.startswith("-f") and len(p) > 2:
                for part in p[2:].split(","):
                    if "-" in part:
                        a, b = part.split("-", 1)
                        fields.extend(range(int(a) if a else 1, (int(b) if b else 9999) + 1))
                    else:
                        fields.append(int(part))
            elif p.startswith("-b") and len(p) > 2:
                for part in p[2:].split(","):
                    if "-" in part:
                        a, b = part.split("-", 1)
                        byte_ranges.append((int(a) if a else 1, int(b) if b else 9999))
                    else:
                        n = int(part)
                        byte_ranges.append((n, n))
            elif p.startswith("-c") and len(p) > 2:
                for part in p[2:].split(","):
                    if "-" in part:
                        a, b = part.split("-", 1)
                        char_ranges.append((int(a) if a else 1, int(b) if b else 9999))
                    else:
                        n = int(part)
                        char_ranges.append((n, n))
            elif not p.startswith("-"):
                target = p
        mode_count = (1 if fields else 0) + (1 if char_ranges else 0) + (1 if byte_ranges else 0)
        if mode_count == 0:
            self._print("  cut: you must specify fields (-f), characters (-c), or bytes (-b)")
            self._last_exit_code = 1
            return
        if mode_count > 1:
            self._print("  cut: you cannot combine -f, -c, and -b")
            self._last_exit_code = 1
            return
        try:
            if target:
                content = Path(os.path.expanduser(target)).read_text()
            elif self._piped_input:
                content = self._piped_input
            else:
                self._print("  cut: no input")
                self._last_exit_code = 1
                return
        except FileNotFoundError:
            self._print(f"  cut: {target}: No such file or directory")
            self._last_exit_code = 1
            return
        out_lines = []
        for line in content.splitlines():
            if byte_ranges:
                raw = line.encode("utf-8", errors="replace")
                chosen = []
                for lo, hi in byte_ranges:
                    for i in range(lo, hi + 1):
                        if i <= len(raw):
                            chosen.append(raw[i - 1:i])
                out_lines.append(b"".join(chosen).decode("utf-8", errors="replace"))
            elif char_ranges:
                chars = list(line)
                chosen = []
                for lo, hi in char_ranges:
                    for i in range(lo, hi + 1):
                        if i <= len(chars):
                            chosen.append(chars[i - 1])
                out_lines.append("".join(chosen))
            else:
                if len(delim) > 1:
                    has_delim = re.search(f"[{re.escape(delim)}]", line) is not None
                    cols = re.split(f"[{re.escape(delim)}]", line)
                    join_delim = delim[0]
                else:
                    has_delim = delim in line
                    cols = line.split(delim)
                    join_delim = delim
                if suppress_no_delim and not has_delim:
                    continue
                chosen = []
                for f in fields:
                    if f <= len(cols):
                        chosen.append(cols[f - 1])
                out_lines.append(join_delim.join(chosen))
        self._print("\n".join(out_lines))
        self._last_exit_code = 0

    def _cmd_tr(self, args: str = "") -> None:
        """Translate, delete, or squeeze characters (piped input only).

        Flags:
          -d   Delete characters in SET1
          -s   Squeeze repeated characters to single occurrence
          -c   Complement SET1 (operate on all chars NOT in SET1)
          -t   Truncate SET2 to length of SET1
        """
        if not self._piped_input:
            self._print("  Usage: <command> | tr [-d] [-s] [-c] [-t] <set1> <set2>")
            self._last_exit_code = 1
            return
        import shlex as _shlex
        parts = _shlex.split(args) if args else []
        delete = False
        squeeze = False
        complement = False
        truncate = False
        non_flags = []
        for p in parts:
            if p == "-d":
                delete = True
            elif p == "-s":
                squeeze = True
            elif p == "-c" or p == "-C":
                complement = True
            elif p == "-t":
                truncate = True
            elif not p.startswith("-"):
                non_flags.append(p)
        if not non_flags or (not delete and not squeeze and len(non_flags) < 2):
            self._print("  Usage: <command> | tr [-d] [-s] [-c] [-t] <set1> [set2]")
            self._last_exit_code = 1
            return
        set1 = non_flags[0]
        set2 = non_flags[1] if len(non_flags) > 1 else ""

        def _expand(s: str) -> str:
            result = []
            i = 0
            while i < len(s):
                # POSIX character classes: [:alpha:], [:digit:], etc.
                if s[i:i+2] == "[:" and ":]" in s[i:]:
                    end = s.index(":]", i)
                    cls = s[i+2:end]
                    if cls == "alpha":
                        result.extend("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
                    elif cls == "digit":
                        result.extend("0123456789")
                    elif cls == "alnum":
                        result.extend("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                    elif cls == "lower":
                        result.extend("abcdefghijklmnopqrstuvwxyz")
                    elif cls == "upper":
                        result.extend("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                    elif cls == "space":
                        result.extend(" \t\n\r\f\v")
                    elif cls == "blank":
                        result.extend(" \t")
                    elif cls == "punct":
                        result.extend('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
                    elif cls == "cntrl":
                        result.extend(chr(c) for c in range(32))
                    elif cls == "graph":
                        result.extend(chr(c) for c in range(33, 127))
                    elif cls == "print":
                        result.extend(chr(c) for c in range(32, 127))
                    elif cls == "xdigit":
                        result.extend("0123456789abcdefABCDEF")
                    else:
                        # Unknown class: treat as literal
                        result.extend(s[i:end+2])
                    i = end + 2
                elif i + 2 < len(s) and s[i + 1] == "-" and ord(s[i]) < ord(s[i + 2]):
                    result.extend(chr(c) for c in range(ord(s[i]), ord(s[i + 2]) + 1))
                    i += 3
                else:
                    result.append(s[i])
                    i += 1
            return "".join(result)

        expanded1 = _expand(set1)
        expanded2 = _expand(set2)
        if complement:
            all_chars = "".join(chr(c) for c in range(256))
            expanded1 = "".join(c for c in all_chars if c not in expanded1)
        if delete:
            result = self._piped_input.translate(str.maketrans("", "", expanded1))
        elif squeeze:
            import re as _re
            if delete:
                result = self._piped_input.translate(str.maketrans("", "", expanded1))
                pattern = "|".join(_re.escape(c) for c in expanded1) if expanded1 else None
                if pattern:
                    result = _re.sub(rf"({pattern})+", lambda m: m.group(0)[0], result)
            else:
                pattern = "|".join(_re.escape(c) for c in expanded1) if expanded1 else None
                if pattern:
                    result = _re.sub(rf"({pattern})+", lambda m: m.group(0)[0], self._piped_input)
                else:
                    result = self._piped_input
        else:
            trans = str.maketrans(expanded1, expanded2[:len(expanded1)].ljust(len(expanded1), expanded2[-1] if expanded2 else ""))
            result = self._piped_input.translate(trans)
        self._print(result.rstrip("\n"))
        self._last_exit_code = 0

    def _cmd_nl(self, args: str = "") -> None:
        """Number lines of a file or piped input.

        Flags:
          -w N  Number width (zero-padded, default auto)
          -s C  Separator string (default \\t)
          -b a  Number all lines (default: body only, skip blanks)
        """
        parts = args.strip().split() if args else []
        width = 0
        sep = "\t"
        body_all = False
        target = None
        i = 0
        while i < len(parts):
            p = parts[i]
            if p == "-w" and i + 1 < len(parts):
                try:
                    width = int(parts[i + 1])
                except ValueError:
                    pass
                i += 2
            elif p.startswith("-w") and len(p) > 2:
                try:
                    width = int(p[2:])
                except ValueError:
                    pass
                i += 1
            elif p == "-s" and i + 1 < len(parts):
                sep = parts[i + 1]
                i += 2
            elif p.startswith("-s") and len(p) > 2:
                sep = p[2:]
                i += 1
            elif p == "-b" and i + 1 < len(parts):
                if parts[i + 1] == "a":
                    body_all = True
                i += 2
            elif not p.startswith("-") and target is None:
                target = p
                i += 1
            else:
                i += 1
        try:
            if target:
                content = Path(os.path.expanduser(target)).read_text()
            elif self._piped_input:
                content = self._piped_input
            else:
                self._print("  Usage: nl [-w N] [-s SEP] [-b a] [file]")
                self._last_exit_code = 1
                return
        except FileNotFoundError:
            self._print(f"  nl: {target}: No such file or directory")
            self._last_exit_code = 1
            return
        lines = content.splitlines()
        if not body_all:
            numbered = [(i + 1, l) for i, l in enumerate(lines) if l.strip()]
        else:
            numbered = [(i + 1, l) for i, l in enumerate(lines)]
        if width > 0:
            out = "\n".join(f"{n:0{width}d}{sep}{l}" for n, l in numbered)
        else:
            out = "\n".join(f"{n}{sep}{l}" for n, l in numbered)
        self._print(out)
        self._last_exit_code = 0

    def _cmd_fold(self, args: str = "") -> None:
        """Wrap long lines. Supports -w width (default 80) and -s (break at spaces)."""
        if not args and not self._piped_input:
            self._print("  Usage: fold [-w width] [-s] [file]")
            self._last_exit_code = 1
            return
        parts = args.strip().split() if args else []
        width = 80
        break_spaces = False
        target = None
        i = 0
        while i < len(parts):
            p = parts[i]
            if p == "-w" and i + 1 < len(parts):
                width = int(parts[i + 1])
                i += 2
            elif p.startswith("-w") and len(p) > 2:
                width = int(p[2:])
                i += 1
            elif p == "-s":
                break_spaces = True
                i += 1
            elif p.startswith("-"):
                i += 1
            else:
                target = p
                i += 1
        try:
            if target:
                content = Path(os.path.expanduser(target)).read_text()
            elif self._piped_input:
                content = self._piped_input
            else:
                self._print("  fold: no input")
                self._last_exit_code = 1
                return
        except FileNotFoundError:
            self._print(f"  fold: {target}: No such file or directory")
            self._last_exit_code = 1
            return
        out_lines = []
        for line in content.splitlines():
            if not break_spaces:
                for i in range(0, len(line), width):
                    out_lines.append(line[i:i + width])
            else:
                while len(line) > width:
                    bp = line.rfind(" ", 0, width + 1)
                    if bp <= 0:
                        bp = width
                    out_lines.append(line[:bp])
                    line = line[bp:].lstrip()
                out_lines.append(line)
        self._print("\n".join(out_lines))
        self._last_exit_code = 0

    def _cmd_tac(self, args: str = "") -> None:
        """Reverse lines of a file or piped input (cat backwards)."""
        if not args and not self._piped_input:
            self._print("  Usage: tac [file]")
            self._last_exit_code = 1
            return
        try:
            if args:
                content = Path(os.path.expanduser(args.strip())).read_text()
            else:
                content = self._piped_input
        except FileNotFoundError:
            self._print(f"  tac: {args.strip()}: No such file or directory")
            self._last_exit_code = 1
            return
        lines = content.splitlines()
        self._print("\n".join(reversed(lines)))
        self._last_exit_code = 0

    def _cmd_shuf(self, args: str = "") -> None:
        """Shuffle lines of a file or piped input."""
        if not args and not self._piped_input:
            self._print("  Usage: shuf [file]")
            self._last_exit_code = 1
            return
        try:
            if args:
                content = Path(os.path.expanduser(args.strip())).read_text()
            else:
                content = self._piped_input
        except FileNotFoundError:
            self._print(f"  shuf: {args.strip()}: No such file or directory")
            self._last_exit_code = 1
            return
        import random as _random
        lines = content.splitlines()
        _random.shuffle(lines)
        self._print("\n".join(lines))
        self._last_exit_code = 0

    def _cmd_rev(self, args: str = "") -> None:
        """Reverse characters in each line of a file or piped input."""
        if not args and not self._piped_input:
            self._print("  Usage: rev [file]")
            self._last_exit_code = 1
            return
        try:
            if args:
                content = Path(os.path.expanduser(args.strip())).read_text()
            else:
                content = self._piped_input
        except FileNotFoundError:
            self._print(f"  rev: {args.strip()}: No such file or directory")
            self._last_exit_code = 1
            return
        for line in content.splitlines():
            self._print(line[::-1])
        self._last_exit_code = 0

    def _cmd_paste(self, args: str = "") -> None:
        """Merge lines of files side by side.

        Flags:
          -d DELIMS  Comma-separated delimiters (cycled per column)
          -s         Serialize (join lines of one file)
        """
        parts = args.strip().split() if args else []
        delim = "\t"
        delims = None
        serialize = False
        files = []
        i = 0
        while i < len(parts):
            if parts[i] == "-d" and i + 1 < len(parts):
                raw = parts[i + 1]
                if len(raw) > 1:
                    delims = list(raw)
                else:
                    delim = raw
                i += 2
            elif parts[i].startswith("-d") and len(parts[i]) > 2:
                raw = parts[i][2:]
                if len(raw) > 1:
                    delims = list(raw)
                else:
                    delim = raw
                i += 1
            elif parts[i] == "-s":
                serialize = True
                i += 1
            else:
                files.append(parts[i])
                i += 1
        if not files and not self._piped_input:
            self._print("  Usage: paste [-d DELIMS] [-s] <file1> [file2 ...]")
            self._last_exit_code = 1
            return
        try:
            if files:
                readers = [Path(os.path.expanduser(f)).read_text().splitlines() for f in files]
            else:
                readers = [self._piped_input.splitlines()]
        except FileNotFoundError as e:
            self._print(f"  paste: {e.filename}: No such file or directory")
            self._last_exit_code = 1
            return
        import itertools as _itertools
        if serialize:
            for reader in readers:
                if delims:
                    d = delims[0]
                else:
                    d = delim
                self._print(d.join(reader))
        else:
            for row in _itertools.zip_longest(*readers, fillvalue=""):
                cells = list(row)
                if delims:
                    result = cells[0]
                    for ci in range(1, len(cells)):
                        d = delims[(ci - 1) % len(delims)]
                        result += d + cells[ci]
                    self._print(result)
                else:
                    self._print(delim.join(cells))
        self._last_exit_code = 0

    def _cmd_comm(self, args: str = "") -> None:
        """Compare two sorted files line by line.

        Flags:
          -1  Suppress column 1 (unique to file1)
          -2  Suppress column 2 (unique to file2)
          -3  Suppress column 3 (shared lines)
        """
        parts = args.strip().split() if args else []
        suppress1 = False
        suppress2 = False
        suppress3 = False
        targets = []
        for p in parts:
            if p == "-1":
                suppress1 = True
            elif p == "-2":
                suppress2 = True
            elif p == "-3":
                suppress3 = True
            elif not p.startswith("-"):
                targets.append(p)
        if len(targets) < 2:
            self._print("  Usage: comm [-1] [-2] [-3] <file1> <file2>")
            self._last_exit_code = 1
            return
        f1, f2 = os.path.expanduser(targets[0]), os.path.expanduser(targets[1])
        try:
            lines1 = Path(f1).read_text().splitlines()
            lines2 = Path(f2).read_text().splitlines()
        except FileNotFoundError as e:
            self._print(f"  comm: {e.filename}: No such file or directory")
            self._last_exit_code = 1
            return
        i = j = 0
        while i < len(lines1) and j < len(lines2):
            if lines1[i] < lines2[j]:
                if not suppress1:
                    self._print(f"\t\t{lines1[i]}")
                i += 1
            elif lines1[i] > lines2[j]:
                if not suppress2:
                    self._print(f"\t{lines2[j]}")
                j += 1
            else:
                if not suppress3:
                    self._print(lines1[i])
                i += 1
                j += 1
        while i < len(lines1):
            if not suppress1:
                self._print(f"\t\t{lines1[i]}")
            i += 1
        while j < len(lines2):
            if not suppress2:
                self._print(f"\t{lines2[j]}")
            j += 1
        self._last_exit_code = 0

    def _cmd_column(self, args: str = "") -> None:
        """Display input in columns. Supports -t (table) and -s SEP (separator)."""
        parts = args.strip().split() if args else []
        table_mode = False
        sep = "\t"
        target = None
        i = 0
        while i < len(parts):
            p = parts[i]
            if p == "-t":
                table_mode = True
                i += 1
            elif p == "-s" and i + 1 < len(parts):
                sep = parts[i + 1]
                i += 2
            elif not p.startswith("-") and target is None:
                target = p
                i += 1
            else:
                i += 1
        try:
            if target:
                content = Path(os.path.expanduser(target)).read_text()
            elif self._piped_input:
                content = self._piped_input
            else:
                self._print("  Usage: column [-t] [-s SEP] [file]")
                self._last_exit_code = 1
                return
        except FileNotFoundError:
            self._print(f"  column: {target}: No such file or directory")
            self._last_exit_code = 1
            return
        lines_raw = content.splitlines()
        rows = [l.split(sep) for l in lines_raw]
        if not rows:
            self._last_exit_code = 0
            return
        ncols = max(len(r) for r in rows)
        col_widths = [0] * ncols
        for r in rows:
            for ci, cell in enumerate(r):
                if ci < ncols:
                    col_widths[ci] = max(col_widths[ci], len(cell))
        out_lines = []
        for r in rows:
            cells = []
            for ci in range(ncols):
                cell = r[ci] if ci < len(r) else ""
                if table_mode and ci < ncols - 1:
                    cells.append(cell.ljust(col_widths[ci]))
                else:
                    cells.append(cell)
            out_lines.append("  ".join(cells) if not table_mode else "  ".join(cells))
        self._print("\n".join(out_lines))
        self._last_exit_code = 0

    def _cmd_expand(self, args: str = "") -> None:
        """Convert tabs to spaces. Supports -t TABSIZE (default 8)."""
        parts = args.strip().split() if args else []
        tabsize = 8
        target = None
        i = 0
        while i < len(parts):
            p = parts[i]
            if p == "-t" and i + 1 < len(parts):
                try:
                    tabsize = int(parts[i + 1])
                except ValueError:
                    pass
                i += 2
            elif not p.startswith("-") and target is None:
                target = p
                i += 1
            else:
                i += 1
        if not target and not self._piped_input:
            self._print("  Usage: expand [-t TABSIZE] [file]")
            self._last_exit_code = 1
            return
        try:
            if target:
                content = Path(os.path.expanduser(target)).read_text()
            else:
                content = self._piped_input
        except FileNotFoundError:
            self._print(f"  expand: {target}: No such file or directory")
            self._last_exit_code = 1
            return
        self._print(content.expandtabs(tabsize).rstrip("\n"))
        self._last_exit_code = 0

    def _cmd_unexpand(self, args: str = "") -> None:
        """Convert spaces to tabs. Supports -t TABSIZE (default 8)."""
        parts = args.strip().split() if args else []
        tabsize = 8
        target = None
        i = 0
        while i < len(parts):
            p = parts[i]
            if p == "-t" and i + 1 < len(parts):
                try:
                    tabsize = int(parts[i + 1])
                except ValueError:
                    pass
                i += 2
            elif not p.startswith("-") and target is None:
                target = p
                i += 1
            else:
                i += 1
        if not target and not self._piped_input:
            self._print("  Usage: unexpand [-t TABSIZE] [file]")
            self._last_exit_code = 1
            return
        try:
            if target:
                content = Path(os.path.expanduser(target)).read_text()
            else:
                content = self._piped_input
        except FileNotFoundError:
            self._print(f"  unexpand: {target}: No such file or directory")
            self._last_exit_code = 1
            return
        lines = content.splitlines()
        out = []
        for line in lines:
            spaces = 0
            for ch in line:
                if ch == " ":
                    spaces += 1
                else:
                    break
            tabs, rem = divmod(spaces, tabsize)
            out.append("\t" * tabs + " " * rem + line[spaces:])
        self._print("\n".join(out))
        self._last_exit_code = 0

    # ── generators / data ───────────────────────────────────────────

    def _cmd_seq(self, args: str = "") -> None:
        """Generate a sequence of numbers."""
        if not args:
            self._print("  Usage: seq [first [increment]] last")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        try:
            if len(parts) == 1:
                first, inc, last = 1, 1, float(parts[0])
            elif len(parts) == 2:
                first, inc, last = float(parts[0]), 1, float(parts[1])
            elif len(parts) == 3:
                first, inc, last = float(parts[0]), float(parts[1]), float(parts[2])
            else:
                self._print("  seq: too many arguments")
                self._last_exit_code = 1
                return
        except ValueError:
            self._print(f"  seq: invalid number")
            self._last_exit_code = 1
            return

        # Check if any argument had a decimal point (preserves user intent)
        has_decimal = any("." in p for p in parts)

        if not has_decimal and first == int(first) and inc == int(inc) and last == int(last):
            nums = range(int(first), int(last) + 1, int(inc))
            self._print("\n".join(str(n) for n in nums))
        else:
            nums = []
            cur = first
            while cur <= last if inc > 0 else cur >= last:
                # Use :g format to avoid floating-point noise (e.g. 0.30000000000000004)
                nums.append(f"{cur:g}")
                cur += inc
            self._print("\n".join(nums))
        self._last_exit_code = 0

    def _cmd_yes(self, args: str = "") -> None:
        """Repeatedly output a line (default: 'y')."""
        s = args.strip() or "y"
        for _ in range(100):
            self._print(s)
        self._last_exit_code = 0

    def _cmd_printf(self, args: str = "") -> None:
        """Format and print data (supports %s, %d, %f, \\n, \\t)."""
        if not args:
            self._last_exit_code = 1
            return
        parts = args.strip().split(maxsplit=1)
        fmt = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        fmt = fmt.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
        arg_parts = rest.split() if rest else []
        arg_idx = 0
        out = []
        i = 0
        while i < len(fmt):
            if fmt[i] == "%" and i + 1 < len(fmt):
                spec = fmt[i + 1]
                if spec == "%":
                    out.append("%")
                    i += 2
                elif spec == "s":
                    val = arg_parts[arg_idx] if arg_idx < len(arg_parts) else ""
                    arg_idx += 1
                    out.append(val)
                    i += 2
                elif spec == "d":
                    val = arg_parts[arg_idx] if arg_idx < len(arg_parts) else "0"
                    arg_idx += 1
                    try:
                        out.append(str(int(val)))
                    except ValueError:
                        out.append("0")
                    i += 2
                elif spec == "f":
                    val = arg_parts[arg_idx] if arg_idx < len(arg_parts) else "0.0"
                    arg_idx += 1
                    try:
                        out.append(f"{float(val):f}")
                    except ValueError:
                        out.append("0.000000")
                    i += 2
                else:
                    out.append(fmt[i])
                    i += 1
            else:
                out.append(fmt[i])
                i += 1
        self._print("".join(out).rstrip("\n"))
        self._last_exit_code = 0

    # ── comparison / conditionals ───────────────────────────────────

    def _cmd_diff(self, args: str = "") -> None:
        """Compare two files line by line. Supports -u, -w, -q."""
        parts = args.strip().split() if args else []
        unified = False
        ignore_ws = False
        quiet = False
        targets = []
        for p in parts:
            if p == "-u":
                unified = True
            elif p == "-w":
                ignore_ws = True
            elif p == "-q":
                quiet = True
            elif not p.startswith("-"):
                targets.append(p)
        if len(targets) < 2:
            self._print("  Usage: diff [-u] [-w] [-q] <file1> <file2>")
            self._last_exit_code = 1
            return
        f1, f2 = os.path.expanduser(targets[0]), os.path.expanduser(targets[1])
        try:
            lines1 = Path(f1).read_text().splitlines()
            lines2 = Path(f2).read_text().splitlines()
        except FileNotFoundError as e:
            self._print(f"  diff: {e.filename}: No such file or directory")
            self._last_exit_code = 1
            return
        import difflib as _difflib
        if ignore_ws:
            norm = lambda s: " ".join(s.split())
            lines1 = [norm(l) for l in lines1]
            lines2 = [norm(l) for l in lines2]
        if unified:
            diff_gen = _difflib.unified_diff(
                lines1, lines2,
                fromfile=targets[0], tofile=targets[1],
            )
            from ..repl import _C_GREEN, _C_RED, _C_DIM, _C_RESET
            has_output = False
            for l in diff_gen:
                has_output = True
                if not quiet:
                    if l.startswith("+++") or l.startswith("---"):
                        self._print(f"  {_C_DIM}{l.rstrip()}{_C_RESET}")
                    elif l.startswith("+"):
                        self._print(f"  {_C_GREEN}{l.rstrip()}{_C_RESET}")
                    elif l.startswith("-"):
                        self._print(f"  {_C_RED}{l.rstrip()}{_C_RESET}")
                    elif l.startswith("@"):
                        self._print(f"  {_C_DIM}{l.rstrip()}{_C_RESET}")
                    else:
                        self._print(f"  {l.rstrip()}")
            if not has_output:
                self._last_exit_code = 0
            else:
                if quiet:
                    self._print(f"  Files {targets[0]} and {targets[1]} differ")
                self._last_exit_code = 1
        else:
            differ = _difflib.Differ()
            diffs = list(differ.compare(lines1, lines2))
            changes = [l for l in diffs if l.startswith(("+ ", "- ", "? "))]
            if not changes:
                self._last_exit_code = 0
                return
            if quiet:
                self._print(f"  Files {targets[0]} and {targets[1]} differ")
                self._last_exit_code = 1
                return
            from ..repl import _C_GREEN, _C_RED, _C_DIM, _C_RESET
            for l in diffs:
                if l.startswith("+ "):
                    self._print(f"  {_C_GREEN}{l}{_C_RESET}")
                elif l.startswith("- "):
                    self._print(f"  {_C_RED}{l}{_C_RESET}")
                elif l.startswith("? "):
                    self._print(f"  {_C_DIM}{l}{_C_RESET}")
            if quiet:
                self._print(f"  Files {targets[0]} and {targets[1]} differ")
            self._last_exit_code = 1

    def _cmd_test(self, args: str = "") -> None:
        """Evaluate conditional expression. Sets exit code 0=true, 1=false."""
        if not args:
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        if args.startswith("[ ") and args.endswith(" ]"):
            parts = args[2:-2].strip().split()
        if len(parts) == 2 and parts[0] == "-f":
            self._last_exit_code = 0 if Path(os.path.expanduser(parts[1])).is_file() else 1
        elif len(parts) == 2 and parts[0] == "-d":
            self._last_exit_code = 0 if Path(os.path.expanduser(parts[1])).is_dir() else 1
        elif len(parts) == 2 and parts[0] == "-e":
            p = Path(os.path.expanduser(parts[1]))
            self._last_exit_code = 0 if p.exists() else 1
        elif len(parts) == 2 and parts[0] == "-z":
            self._last_exit_code = 0 if len(parts[1]) == 0 else 1
        elif len(parts) == 2 and parts[0] == "-n":
            self._last_exit_code = 0 if len(parts[1]) > 0 else 1
        elif len(parts) == 3 and parts[1] == "=":
            self._last_exit_code = 0 if parts[0] == parts[2] else 1
        elif len(parts) == 3 and parts[1] == "!=":
            self._last_exit_code = 0 if parts[0] != parts[2] else 1
        elif len(parts) == 3 and parts[1] == "-eq":
            self._last_exit_code = 0 if int(parts[0]) == int(parts[2]) else 1
        elif len(parts) == 3 and parts[1] == "-ne":
            self._last_exit_code = 0 if int(parts[0]) != int(parts[2]) else 1
        elif len(parts) == 3 and parts[1] == "-lt":
            self._last_exit_code = 0 if int(parts[0]) < int(parts[2]) else 1
        elif len(parts) == 3 and parts[1] == "-le":
            self._last_exit_code = 0 if int(parts[0]) <= int(parts[2]) else 1
        elif len(parts) == 3 and parts[1] == "-gt":
            self._last_exit_code = 0 if int(parts[0]) > int(parts[2]) else 1
        elif len(parts) == 3 and parts[1] == "-ge":
            self._last_exit_code = 0 if int(parts[0]) >= int(parts[2]) else 1
        else:
            self._last_exit_code = 1

    # ── dispatch / execution ────────────────────────────────────────

    def _cmd_xargs(self, args: str = "") -> None:
        """Build and execute command from stdin."""
        parts = args.strip().split() if args.strip() else []
        n = None
        placeholder = null_terminated = no_run_if_empty = False
        cmd_parts = []
        i = 0
        while i < len(parts):
            if parts[i] == "-n" and i + 1 < len(parts):
                n = int(parts[i + 1])
                i += 2
            elif parts[i].startswith("-I") and len(parts[i]) > 2:
                placeholder = parts[i][2:]
                i += 1
            elif parts[i] == "-I" and i + 1 < len(parts):
                placeholder = parts[i + 1]
                i += 2
            elif parts[i] == "-0":
                null_terminated = True
                i += 1
            elif parts[i] == "-r":
                no_run_if_empty = True
                i += 1
            else:
                cmd_parts.append(parts[i])
                i += 1
        if not self._piped_input and not no_run_if_empty:
            self._print("  Usage: <command> | xargs [-n N] [-0] [-r] [-I{}] <cmd> [args...]")
            self._last_exit_code = 1
            return
        if null_terminated:
            items = [x for x in (self._piped_input or "").split("\0") if x]
        else:
            items = (self._piped_input or "").split()
        if no_run_if_empty and not items:
            self._last_exit_code = 0
            return
        if not cmd_parts:
            for item in items:
                self._print(item)
            self._last_exit_code = 0
            return
        if placeholder:
            for item in items:
                substituted = [part.replace(placeholder, item) for part in cmd_parts]
                if self._check_permission(substituted[0], " ".join(substituted[1:]) if len(substituted) > 1 else ""):
                    result = self._execute_single(" ".join(substituted))
                    if result:
                        self._print(result.rstrip("\n"))
        elif n:
            chunks = [items[i:i + n] for i in range(0, len(items), n)]
            for chunk in chunks:
                full_cmd = cmd_parts + chunk
                if self._check_permission(full_cmd[0], " ".join(full_cmd[1:]) if len(full_cmd) > 1 else ""):
                    result = self._execute_single(" ".join(full_cmd))
                    if result:
                        self._print(result.rstrip("\n"))
        else:
            full_cmd = cmd_parts + items
            if self._check_permission(full_cmd[0], " ".join(full_cmd[1:]) if len(full_cmd) > 1 else ""):
                result = self._execute_single(" ".join(full_cmd))
                if result:
                    self._print(result.rstrip("\n"))
        self._last_exit_code = 0

    def _cmd_time(self, args: str = "") -> None:
        """Time a command execution."""
        if not args:
            self._print("  Usage: time <command>")
            self._last_exit_code = 1
            return
        import time as _time
        from ..repl import _C_DIM, _C_RESET
        start = _time.perf_counter()
        self._execute_single(args)
        elapsed = _time.perf_counter() - start
        self._print(f"  {_C_DIM}real  {elapsed:.3f}s{_C_RESET}")
        self._last_exit_code = 0

    # ── system info ─────────────────────────────────────────────────

    def _cmd_env(self, args: str = "") -> None:
        """Print environment variables.

        Flags:
          -i    Ignore environment (start with empty env)
          -u NAME   Unset variable NAME
        """
        parts = args.strip().split() if args else []
        ignore = False
        unset_vars = []
        i = 0
        while i < len(parts):
            if parts[i] == "-i":
                ignore = True
                i += 1
            elif parts[i] == "-u" and i + 1 < len(parts):
                unset_vars.append(parts[i + 1])
                i += 2
            elif parts[i] == "--":
                i += 1
            else:
                break
        for name in unset_vars:
            self._env.pop(name, None)
        if ignore:
            for k in list(self._env.keys()):
                if k not in ("PATH", "HOME", "SHELL", "USER"):
                    del self._env[k]
        for k, v in sorted(self._env.items()):
            self._print(f"  {k}={v}")
        self._last_exit_code = 0

    def _cmd_realpath(self, args: str = "") -> None:
        """Resolve path to absolute."""
        if not args:
            self._print("  Usage: realpath <path>")
            self._last_exit_code = 1
            return
        p = os.path.expanduser(args.strip())
        try:
            self._print(os.path.realpath(p))
            self._last_exit_code = 0
        except OSError as e:
            self._print(f"  realpath: {e}")
            self._last_exit_code = 1

    def _cmd_dirname(self, args: str = "") -> None:
        """Strip last component from file path."""
        if not args:
            self._print("  Usage: dirname <path>")
            self._last_exit_code = 1
            return
        self._print(os.path.dirname(os.path.expanduser(args.strip())))
        self._last_exit_code = 0

    def _cmd_basename(self, args: str = "") -> None:
        """Strip directory from file path."""
        if not args:
            self._print("  Usage: basename <path> [suffix]")
            self._last_exit_code = 1
            return
        parts = args.strip().split(None, 1)
        name = os.path.basename(os.path.expanduser(parts[0]))
        if len(parts) > 1 and name.endswith(parts[1]):
            name = name[:-len(parts[1])]
        self._print(name)
        self._last_exit_code = 0

    def _cmd_nproc(self, args: str = "") -> None:
        """Print number of CPUs."""
        import os as _os
        self._print(str(_os.cpu_count() or 1))
        self._last_exit_code = 0

    def _cmd_hostname(self, args: str = "") -> None:
        """Print system hostname."""
        import socket as _socket
        self._print(_socket.gethostname())
        self._last_exit_code = 0

    def _cmd_uname(self, args: str = "") -> None:
        """Print system information."""
        import platform as _platform
        flags = args.strip().split() if args else []
        if not flags or "-a" in flags:
            self._print(f"  {_platform.system()} {_platform.release()} {_platform.machine()}")
        else:
            parts = []
            for f in flags:
                if "s" in f:
                    parts.append(_platform.system())
                if "r" in f:
                    parts.append(_platform.release())
                if "m" in f:
                    parts.append(_platform.machine())
            self._print(" ".join(parts))
        self._last_exit_code = 0

    def _cmd_id(self, args: str = "") -> None:
        """Print user identity."""
        import getpass as _gp, os as _os
        user = _gp.getuser()
        uid = _os.getuid() if hasattr(_os, "getuid") else "?"
        gid = _os.getgid() if hasattr(_os, "getgid") else "?"
        self._print(f"  uid={uid}({user}) gid={gid}({user})")
        self._last_exit_code = 0

    def _cmd_logname(self, args: str = "") -> None:
        """Print login name."""
        import getpass as _gp
        self._print(_gp.getuser())
        self._last_exit_code = 0

    def _cmd_mktemp(self, args: str = "") -> None:
        """Create a temporary file or directory."""
        import tempfile as _tf
        parts = args.strip().split()
        is_dir = any(p == "-d" for p in parts)
        try:
            if is_dir:
                path = _tf.mkdtemp()
            else:
                path = _tf.mkstemp()[1]
            self._print(path)
            self._last_exit_code = 0
        except OSError as e:
            self._print(f"  mktemp: {e}")
            self._last_exit_code = 1

    def _cmd_who(self, args: str = "") -> None:
        """Show who is logged on."""
        import os as _os, time as _time
        import getpass as _gp
        user = _gp.getuser()
        self._print(f"  {user}    console  {_time.strftime('%Y-%m-%d %H:%M')}")
        self._last_exit_code = 0

    def _cmd_od(self, args: str = "") -> None:
        """Dump file in various formats.

        Flags:
          -A ADDR  Address base: o (octal), x (hex), d (dec), n (none)
          -t TYPE  Data format: o (octal), x (hex), d (decimal), c (chars), u (unsigned)
          -j N     Skip N bytes
          -N N     Read only N bytes
          -v       Verbose (don't collapse identical lines)
        """
        parts = args.strip().split() if args else []
        target = None
        addr_base = "o"
        data_type = "o"
        skip = 0
        limit = 0
        verbose = False
        i = 0
        while i < len(parts):
            p = parts[i]
            if p == "-A" and i + 1 < len(parts):
                addr_base = parts[i + 1]
                i += 2
            elif p.startswith("-A") and len(p) > 2:
                addr_base = p[2:]
                i += 1
            elif p == "-t" and i + 1 < len(parts):
                data_type = parts[i + 1]
                i += 2
            elif p.startswith("-t") and len(p) > 2:
                data_type = p[2:]
                i += 1
            elif p == "-j" and i + 1 < len(parts):
                try:
                    skip = int(parts[i + 1])
                except ValueError:
                    pass
                i += 2
            elif p == "-N" and i + 1 < len(parts):
                try:
                    limit = int(parts[i + 1])
                except ValueError:
                    pass
                i += 2
            elif p == "-v":
                verbose = True
                i += 1
            elif p == "-x":
                data_type = "x"
                i += 1
            elif p == "-o":
                data_type = "o"
                i += 1
            elif p == "-d":
                data_type = "d"
                i += 1
            elif not p.startswith("-") and target is None:
                target = p
                i += 1
            else:
                i += 1
        if not target and not self._piped_input:
            self._print("  Usage: od [-A base] [-t type] [-j N] [-N N] <file>")
            self._last_exit_code = 1
            return
        try:
            if target:
                data = Path(os.path.expanduser(target)).read_bytes()
            else:
                data = self._piped_input.encode("utf-8", errors="replace")
        except FileNotFoundError:
            self._print(f"  od: {target}: No such file or directory")
            self._last_exit_code = 1
            return
        if skip:
            data = data[skip:]
        if limit:
            data = data[:limit]
        if not verbose:
            collapsed = []
            prev_chunk = None
            skip_next = False
            for i in range(0, len(data), 16):
                chunk = data[i:i + 16]
                if chunk == prev_chunk:
                    if not skip_next:
                        collapsed.append(("*", None))
                        skip_next = True
                else:
                    collapsed.append((i, chunk))
                    skip_next = False
                prev_chunk = chunk
            items = collapsed
        else:
            items = [(i, data[i:i + 16]) for i in range(0, len(data), 16)]
        for addr_or_star, chunk in items:
            if chunk is None:
                self._print("  *")
                continue
            if addr_base == "n":
                addr_str = ""
            elif addr_base == "d":
                addr_str = f"{addr_or_star:07d}"
            elif addr_base == "x":
                addr_str = f"{addr_or_star:07x}"
            else:
                addr_str = f"{addr_or_star:07o}"
            if data_type == "x":
                vals = " ".join(f"{b:02x}" for b in chunk)
            elif data_type == "d":
                vals = " ".join(f"{b:3d}" for b in chunk)
            elif data_type == "c":
                vals = " ".join(f"{b:3o} " + (chr(b) if 32 <= b < 127 else f"\\{b:03o}") for b in chunk)
            elif data_type == "u":
                vals = " ".join(f"{b:3d}" for b in chunk)
            else:
                vals = " ".join(f"{b:03o}" for b in chunk)
            ascii_repr = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            if addr_base == "n":
                self._print(f"  {vals:<48} {ascii_repr}")
            else:
                self._print(f"  {addr_str} {vals:<48} {ascii_repr}")
        self._last_exit_code = 0

    def _cmd_join(self, args: str = "") -> None:
        """Join lines of two files on a common field.

        Flags:
          -1 F  Join field in file1 (default 1)
          -2 F  Join field in file2 (default 1)
          -t C  Field separator (default whitespace)
          -a F  Print unpairable lines from file F (1 or 2)
          -e STR  Replace missing fields with STR
          -i     Case-insensitive join
        """
        parts = args.strip().split() if args else []
        field1 = 1
        field2 = 1
        sep = None
        orphans = set()
        empty_str = ""
        case_insensitive = False
        targets = []
        i = 0
        while i < len(parts):
            p = parts[i]
            if p == "-1" and i + 1 < len(parts):
                try:
                    field1 = int(parts[i + 1])
                except ValueError:
                    pass
                i += 2
            elif p == "-2" and i + 1 < len(parts):
                try:
                    field2 = int(parts[i + 1])
                except ValueError:
                    pass
                i += 2
            elif p == "-t" and i + 1 < len(parts):
                sep = parts[i + 1]
                i += 2
            elif p == "-a" and i + 1 < len(parts):
                orphans.add(int(parts[i + 1]))
                i += 2
            elif p == "-e" and i + 1 < len(parts):
                empty_str = parts[i + 1]
                i += 2
            elif p == "-i":
                case_insensitive = True
                i += 1
            elif not p.startswith("-"):
                targets.append(p)
                i += 1
            else:
                i += 1
        if len(targets) < 2:
            self._print("  Usage: join [-1 F] [-2 F] [-t C] [-a F] [-e STR] [-i] <file1> <file2>")
            self._last_exit_code = 1
            return
        f1, f2 = os.path.expanduser(targets[0]), os.path.expanduser(targets[1])
        try:
            raw1 = Path(f1).read_text().splitlines()
            raw2 = Path(f2).read_text().splitlines()
        except FileNotFoundError as e:
            self._print(f"  join: {e.filename}: No such file or directory")
            self._last_exit_code = 1
            return
        def _split(line):
            if sep:
                return line.split(sep)
            return line.split()
        def _get_field(parts, f):
            idx = f - 1
            return parts[idx] if 0 <= idx < len(parts) else ""
        d1 = {}
        for line in raw1:
            fields = _split(line)
            k = _get_field(fields, field1)
            if case_insensitive:
                k = k.lower()
            d1[k] = (fields, line)
        d2 = {}
        for line in raw2:
            fields = _split(line)
            k = _get_field(fields, field2)
            if case_insensitive:
                k = k.lower()
            d2[k] = (fields, line)
        matched_keys = set()
        for k in sorted(set(d1) | set(d2)):
            if k in d1 and k in d2:
                matched_keys.add(k)
                f1_fields, _ = d1[k]
                f2_fields, _ = d2[k]
                j1 = [f for fi, f in enumerate(f1_fields) if fi != field1 - 1]
                j2 = [f for fi, f in enumerate(f2_fields) if fi != field2 - 1]
                result = [k] + j1 + j2
                self._print(sep.join(result) if sep else " ".join(result))
            elif 1 in orphans and k in d1:
                f1_fields, _ = d1[k]
                result = f1_fields[:]
                while len(result) < 2:
                    result.append(empty_str)
                self._print(sep.join(result) if sep else " ".join(result))
            elif 2 in orphans and k in d2:
                f2_fields, _ = d2[k]
                result = [empty_str] + f2_fields
                self._print(sep.join(result) if sep else " ".join(result))
        self._last_exit_code = 0

    # ── misc ────────────────────────────────────────────────────────

    def _cmd_clear(self, args: str = "") -> None:
        """Clear the terminal screen."""
        self._print("\033[2J\033[H", end="")

    def _cmd_sleep(self, args: str = "") -> None:
        """Sleep for a specified time: sleep [SUFFIX]

        Suffixes: s (seconds, default), m (minutes), h (hours), d (days)
        """
        import time as _time
        parts = args.strip().split() if args else []
        if not parts:
            return
        total = 0.0
        for part in parts:
            mult = 1.0
            raw = part
            if raw.endswith("s"):
                raw = raw[:-1]
                mult = 1.0
            elif raw.endswith("m"):
                raw = raw[:-1]
                mult = 60.0
            elif raw.endswith("h"):
                raw = raw[:-1]
                mult = 3600.0
            elif raw.endswith("d"):
                raw = raw[:-1]
                mult = 86400.0
            try:
                total += float(raw) * mult
            except ValueError:
                pass
        if total > 0:
            _time.sleep(total)

    def _cmd_date(self, args: str = "") -> None:
        """Show current date and time: date [-u] [+format]"""
        from datetime import datetime as _dt, timezone as _tz
        argv = args.split()
        utc = False
        fmt = "%a %b %d %H:%M:%S %Z %Y"
        i = 0
        while i < len(argv):
            if argv[i] == "-u":
                utc = True
                i += 1
            elif argv[i].startswith("+"):
                fmt = argv[i][1:]
                i += 1
            else:
                i += 1
        now = _dt.now(_tz.utc if utc else None)
        self._print(now.strftime(fmt))

    def _cmd_cal(self, args: str = "") -> None:
        """Show a calendar: cal [[month] year]"""
        from datetime import datetime as _dt
        import calendar as _cal
        from ..repl import _C_BOLD, _C_RESET
        argv = args.split()
        now = _dt.now()
        if len(argv) == 0:
            year, month = now.year, now.month
        elif len(argv) == 1:
            year = int(argv[0])
            month = now.month if year == now.year else 1
        else:
            month, year = int(argv[0]), int(argv[1])
        if month < 1 or month > 12 or year < 1 or year > 9999:
            self._print(f"  cal: invalid date")
            return
        header = f"{_cal.month_name[month]} {year}".center(20)
        self._print(f"  {_C_BOLD}{header}{_C_RESET}")
        self._print(f"  Mo Tu We Th Fr Sa Su")
        first_dow = _cal.weekday(year, month, 1)
        days = _cal.monthrange(year, month)[1]
        line = "   " * first_dow
        for d in range(1, days + 1):
            line += f"{d:>2d} "
            if (first_dow + d) % 7 == 0:
                self._print(f"  {line}")
                line = ""
        if line.strip():
            self._print(f"  {line}")

    def _cmd_sed(self, args: str = "") -> None:
        """Stream editor: sed 's/pattern/replacement/[g]', '/pattern/d', 'd', 'Np', 'N,Mp', 'N,Md'."""
        import re as _re
        if not args:
            self._print("  Usage: sed [-n] 's/pattern/replacement/[g]' [file]")
            self._print("         sed [-n] '/pattern/d' [file]")
            self._print("         sed [-n] 'd' [file]")
            self._print("         sed [-n] 'Np' [file]")
            self._print("         sed [-n] 'N,Mp' [file]")
            self._print("         sed [-n] 'N,Md' [file]")
            self._last_exit_code = 1
            return
        quiet = False
        script = ""
        target = None
        raw_parts = args.strip().split()
        i = 0
        while i < len(raw_parts):
            p = raw_parts[i]
            if p == "-n":
                quiet = True
                i += 1
            elif p.startswith("s/") or p.startswith("s#"):
                # Extract full script from original args (may contain spaces)
                stripped = args.strip()
                idx = stripped.index(p)
                script = stripped[idx:]
                # If script ends with a file target, strip it
                # Find the closing delimiter
                sep = p[1]
                # Count unescaped delimiters
                depth = 0
                for ci, ch in enumerate(script):
                    if ch == sep and (ci == 0 or script[ci-1] != '\\'):
                        depth += 1
                if depth >= 3:
                    # Full s/// with flags — trim trailing non-script tokens
                    pass
                i = len(raw_parts)
            elif p.startswith("/") and p.endswith("/d"):
                script = p
                i += 1
            elif p.startswith("/") and p.endswith("/p"):
                script = p
                i += 1
            elif _re.match(r'^\d+,?\d*p$', p):
                script = p
                i += 1
            elif _re.match(r'^\d+,?\d*d$', p):
                script = p
                i += 1
            elif p.strip() == "d":
                script = p
                i += 1
            elif _re.match(r'^\d+[aic]\\', p) or _re.match(r'^\d+[aic]\\\\', p):
                # a\, i\, c\ commands: text after backslash can contain spaces
                # Reconstruct from original args to preserve spaces
                script = args.strip()[args.strip().index(p):]
                i = len(raw_parts)
            elif p.startswith("/") and (_re.search(r'/[aic]\\', p) or _re.search(r'/[aic]\\\\', p)):
                script = args.strip()[args.strip().index(p):]
                i = len(raw_parts)
            elif not p.startswith("-") and not script:
                script = p
                i += 1
            elif not p.startswith("-"):
                target = p
                i += 1
            else:
                i += 1
        if not script:
            self._print("  Usage: sed [-n] 's/pattern/replacement/[g]' [file]")
            self._last_exit_code = 1
            return
        try:
            if target:
                content = Path(os.path.expanduser(target)).read_text()
            elif self._piped_input:
                content = self._piped_input
            else:
                self._print("  sed: no input")
                self._last_exit_code = 1
                return
        except FileNotFoundError:
            self._print(f"  sed: {target}: No such file or directory")
            self._last_exit_code = 1
            return
        out_lines = []
        NR = 0
        for line in content.splitlines(keepends=True):
            NR += 1
            matched = False
            if script.startswith("s/") or script.startswith("s#"):
                sep = script[1]
                rest = script[2:]
                parts_r = rest.split(sep, 2)
                if len(parts_r) >= 3:
                    pat, repl, flags = parts_r[0], parts_r[1], parts_r[2]
                    try:
                        compiled = _re.compile(pat)
                    except _re.error:
                        self._print(f"  sed: invalid regex '{pat}'")
                        self._last_exit_code = 1
                        return
                    def _unescape(s):
                        s = s.replace("\\t", "\t")
                        s = s.replace("\\n", "\n")
                        s = s.replace("\\\\", "\\")
                        return s
                    new_line = compiled.sub(_unescape(repl), line, count=0 if "g" in flags else 1)
                    if quiet:
                        if new_line != line:
                            out_lines.append(new_line)
                    else:
                        out_lines.append(new_line)
                else:
                    if not quiet:
                        out_lines.append(line)
            elif script.endswith("/d"):
                pat_d = script[1:-2]
                try:
                    matched = _re.search(pat_d, line) is not None
                except _re.error:
                    self._print(f"  sed: invalid regex '{pat_d}'")
                    self._last_exit_code = 1
                    return
                if not matched:
                    out_lines.append(line)
            elif script.endswith("/p"):
                pat_p = script[1:-2]
                try:
                    matched = _re.search(pat_p, line) is not None
                except _re.error:
                    self._print(f"  sed: invalid regex '{pat_p}'")
                    self._last_exit_code = 1
                    return
                if matched:
                    out_lines.append(line)
            elif script.strip() == "d":
                continue
            elif _re.match(r'^\d+,\d+d$', script):
                m = _re.match(r'^(\d+),(\d+)d$', script)
                start = int(m.group(1))
                end = int(m.group(2))
                if start <= NR <= end:
                    continue
                else:
                    out_lines.append(line)
            elif _re.match(r'^(\d+)(,(\d+))?p$', script):
                m = _re.match(r'^(\d+)(?:,(\d+))?p$', script)
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else start
                if start <= NR <= end:
                    out_lines.append(line)
            elif _re.match(r'^(\d+)([aic])\\(.*)$', script):
                m = _re.match(r'^(\d+)([aic])\\(.*)$', script)
                line_num = int(m.group(1))
                cmd_type = m.group(2)
                text = m.group(3)
                if cmd_type == "c" and NR == line_num:
                    out_lines.append(text + "\n")
                else:
                    out_lines.append(line)
                    if cmd_type == "a" and NR == line_num:
                        out_lines.append(text + "\n")
                    elif cmd_type == "i" and NR == line_num:
                        out_lines.insert(-1, text + "\n")
            elif _re.match(r'^/([^/]+)/([aic])\\(.*)$', script):
                m = _re.match(r'^/([^/]+)/([aic])\\(.*)$', script)
                pat = m.group(1)
                cmd_type = m.group(2)
                text = m.group(3)
                try:
                    matched = _re.search(pat, line) is not None
                except _re.error:
                    self._print(f"  sed: invalid regex '{pat}'")
                    self._last_exit_code = 1
                    return
                if cmd_type == "c":
                    if matched:
                        out_lines.append(text + "\n")
                    else:
                        out_lines.append(line)
                elif cmd_type == "a":
                    out_lines.append(line)
                    if matched:
                        out_lines.append(text + "\n")
                elif cmd_type == "i":
                    if matched:
                        out_lines.append(text + "\n")
                    out_lines.append(line)
            else:
                if not quiet:
                    out_lines.append(line)
        self._print("".join(out_lines), end="")

    def _cmd_awk(self, args: str = "") -> None:
        """AWK-like processor: awk '-F<sep>' '{print $1,$2}' [file]"""
        import re as _re
        if not args:
            self._print("  Usage: awk '-F<sep>' '{print $1}' [file]")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        fs = None
        target = None
        script = None
        i = 0
        while i < len(parts):
            p = parts[i]
            if p.startswith("-F") and len(p) > 2:
                fs = p[2:]
                i += 1
            elif p.startswith("-F"):
                if i + 1 < len(parts):
                    fs = parts[i + 1]
                    i += 2
                else:
                    i += 1
            elif p.startswith("{") or (p.startswith("'{") and not script) or (p.startswith('"{') and not script):
                # Collect tokens until one ends with } or '} or "}
                script_tokens = []
                start_p = p.lstrip("'\"")
                first_end = start_p.endswith("}") or start_p.endswith("}'") or start_p.endswith('"}')
                script_tokens.append(start_p[1:] if first_end else start_p)
                while i < len(parts) and not (parts[i].endswith("}") or parts[i].endswith("}'") or parts[i].endswith('"}')):
                    i += 1
                    if i < len(parts):
                        end_p = parts[i]
                        is_end = end_p.endswith("}") or end_p.endswith("}'") or end_p.endswith('"}')
                        if is_end:
                            end_p_clean = end_p.rstrip("'\"")
                            if end_p_clean.endswith("}"):
                                end_p_clean = end_p_clean[:-1]
                            script_tokens.append(end_p_clean)
                        else:
                            script_tokens.append(end_p)
                raw = " ".join(script_tokens).strip("{}").strip("'\"").strip()
                script = raw
                i += 1
            elif not p.startswith("-") and not script:
                script = p
                i += 1
            elif not p.startswith("-"):
                target = p
                i += 1
            else:
                i += 1
        if not script:
            self._print("  Usage: awk '-F<sep>' '{print $1}' [file]")
            self._last_exit_code = 1
            return
        try:
            if target:
                content = Path(os.path.expanduser(target)).read_text()
            elif self._piped_input:
                content = self._piped_input
            else:
                self._print("  awk: no input")
                self._last_exit_code = 1
                return
        except FileNotFoundError:
            self._print(f"  awk: {target}: No such file or directory")
            self._last_exit_code = 1
            return
        out_lines = []
        NR = 0
        for raw_line in content.splitlines():
            NR += 1
            line = raw_line.rstrip("\n").rstrip("\r")
            if fs is not None:
                fields = line.split(fs) if fs else [line]
            else:
                fields = line.split()
            NF = len(fields)
            result = self._awk_resolve(script, line, fields, NF, NR)
            out_lines.append(result)
        self._print("\n".join(out_lines))

    def _awk_resolve(self, expr, line, fields, NF, NR):
        """Resolve an awk expression against a line's fields."""
        import re as _re
        result = expr
        tokens = {}
        token_idx = [0]
        def _save_field(m):
            key = f"\x00T{token_idx[0]}\x00"
            token_idx[0] += 1
            idx = int(m.group(1))
            tokens[key] = fields[idx - 1] if 1 <= idx <= NF else ""
            return key
        # Protect $0 from digit matching
        result = result.replace("$0", "\x00DOLLAR0\x00")
        result = _re.sub(r'\$(\d+)', _save_field, result)
        result = result.replace("\x00DOLLAR0\x00", line)
        result = _re.sub(r'\$NF', fields[-1] if fields else "", result)
        for key, val in tokens.items():
            result = result.replace(key, val)
        result = result.replace("NR", str(NR))
        result = result.replace("NF", str(NF))
        # Handle comma-separated print args
        if result.startswith("print "):
            inner = result[6:]
            parts = [p.strip() for p in inner.split(",")]
            if len(parts) > 1:
                quoted = []
                for p in parts:
                    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
                        quoted.append(p[1:-1])
                    else:
                        quoted.append(p)
                result = " ".join(quoted)
            elif inner.startswith('"') and inner.endswith('"'):
                result = inner[1:-1]
            elif inner.startswith("'") and inner.endswith("'"):
                result = inner[1:-1]
            else:
                result = inner
        return result

    def _cmd_tsort(self, args: str = "") -> None:
        """Topological sort. Reads pairs from stdin or file."""
        lines = None
        if args:
            try:
                lines = Path(os.path.expanduser(args.strip())).read_text().splitlines()
            except FileNotFoundError:
                self._print(f"  tsort: {args.strip()}: No such file or directory")
                self._last_exit_code = 1
                return
        elif self._piped_input:
            lines = self._piped_input.splitlines()
        else:
            self._print("  Usage: tsort [file]")
            self._last_exit_code = 1
            return
        graph = {}
        for line in lines:
            parts = line.split()
            if len(parts) == 2:
                a, b = parts
                graph.setdefault(a, [])
                graph.setdefault(b, [])
                graph[a].append(b)
            elif len(parts) == 1:
                graph.setdefault(parts[0], [])
        visited = set()
        temp = set()
        order = []

        def _visit(node):
            if node in temp:
                return
            if node in visited:
                return
            temp.add(node)
            for dep in graph.get(node, []):
                _visit(dep)
            temp.discard(node)
            visited.add(node)
            order.append(node)

        for node in graph:
            _visit(node)
        if order:
            self._print("\n".join(order))
        self._last_exit_code = 0

    def _cmd_strings(self, args: str = "") -> None:
        """Print printable character sequences from a binary file.

        Flags:
          -n MINLEN  Minimum string length (default 4)
        """
        parts = args.strip().split() if args else []
        min_len = 4
        target = None
        i = 0
        while i < len(parts):
            p = parts[i]
            if p == "-n" and i + 1 < len(parts):
                try:
                    min_len = int(parts[i + 1])
                except ValueError:
                    pass
                i += 2
            elif not p.startswith("-") and target is None:
                target = p
                i += 1
            else:
                i += 1
        if not target and not self._piped_input:
            self._print("  Usage: strings [-n MINLEN] <file>")
            self._last_exit_code = 1
            return
        try:
            if target:
                data = Path(os.path.expanduser(target)).read_bytes()
            else:
                data = self._piped_input.encode("utf-8", errors="replace")
        except FileNotFoundError:
            self._print(f"  strings: {target}: No such file or directory")
            self._last_exit_code = 1
            return
        import re as _re
        strings_found = _re.findall(rb'[\x20-\x7e]{%d,}' % min_len, data)
        if strings_found:
            self._print("\n".join(s.decode("ascii", errors="replace") for s in strings_found))
        self._last_exit_code = 0

    def _cmd_base64(self, args: str = "") -> None:
        """Base64 encode/decode. Supports -d (decode)."""
        parts = args.strip().split() if args else []
        decode = False
        target = None
        i = 0
        while i < len(parts):
            p = parts[i]
            if p == "-d" or p == "-D":
                decode = True
                i += 1
            elif not p.startswith("-") and target is None:
                target = p
                i += 1
            else:
                i += 1
        try:
            if target:
                data = Path(os.path.expanduser(target)).read_bytes()
            elif self._piped_input:
                data = self._piped_input.encode("utf-8") if not decode else self._piped_input.encode("utf-8")
            else:
                self._print("  Usage: base64 [-d] [file]")
                self._last_exit_code = 1
                return
        except FileNotFoundError:
            self._print(f"  base64: {target}: No such file or directory")
            self._last_exit_code = 1
            return
        import base64 as _b64
        if decode:
            try:
                import io as _io
                decoded = _b64.b64decode(data)
                self._print(decoded.decode("utf-8", errors="replace"))
            except Exception:
                self._print("  base64: invalid input")
                self._last_exit_code = 1
                return
        else:
            encoded = _b64.b64encode(data).decode("ascii")
            self._print(encoded)
        self._last_exit_code = 0

    def _cmd_cksum(self, args: str = "") -> None:
        """Compute CRC checksum of files or piped input."""
        targets = []
        parts = args.strip().split() if args else []
        for p in parts:
            if not p.startswith("-"):
                targets.append(p)
        if not targets and not self._piped_input:
            self._print("  Usage: cksum <file>...")
            self._last_exit_code = 1
            return
        if self._piped_input:
            data = self._piped_input.encode("utf-8", errors="replace")
            crc = 0
            for b in data:
                crc ^= b << 24
                for _ in range(8):
                    if crc & 0x80000000:
                        crc = (crc << 1) ^ 0x04C11DB7
                    else:
                        crc <<= 1
                    crc &= 0xFFFFFFFF
            crc = (~crc) & 0xFFFFFFFF
            self._print(f"{crc} {len(data)}")
            self._last_exit_code = 0
            return
        rc = 0
        for t in targets:
            tp = os.path.expanduser(t)
            try:
                data = Path(tp).read_bytes()
            except FileNotFoundError:
                self._print(f"  cksum: {t}: No such file or directory")
                rc = 1
                continue
            crc = 0
            for b in data:
                crc ^= b << 24
                for _ in range(8):
                    if crc & 0x80000000:
                        crc = (crc << 1) ^ 0x04C11DB7
                    else:
                        crc <<= 1
                    crc &= 0xFFFFFFFF
            crc = (~crc) & 0xFFFFFFFF
            self._print(f"{crc} {len(data)} {t}")
        self._last_exit_code = rc

    def _cmd_split(self, args: str = "") -> None:
        """Split a file into pieces.

        Flags:
          -l N    Split by lines (default 1000)
          -b N    Split by bytes (k/M/G suffixes)
          -d      Use numeric suffixes (default aa,ab,...)
          PREFIX  Output filename prefix (default 'x')
        """
        parts = args.strip().split() if args else []
        line_count = 0
        byte_count = 0
        numeric = False
        prefix = "x"
        targets = []
        i = 0
        while i < len(parts):
            p = parts[i]
            if p == "-l" and i + 1 < len(parts):
                try:
                    line_count = int(parts[i + 1])
                except ValueError:
                    pass
                i += 2
            elif p == "-b" and i + 1 < len(parts):
                raw = parts[i + 1]
                mult = 1
                if raw.endswith("k") or raw.endswith("K"):
                    mult = 1024
                    raw = raw[:-1]
                elif raw.endswith("m") or raw.endswith("M"):
                    mult = 1024 * 1024
                    raw = raw[:-1]
                elif raw.endswith("g") or raw.endswith("G"):
                    mult = 1024 * 1024 * 1024
                    raw = raw[:-1]
                try:
                    byte_count = int(raw) * mult
                except ValueError:
                    pass
                i += 2
            elif p == "-d":
                numeric = True
                i += 1
            elif not p.startswith("-"):
                targets.append(p)
                i += 1
            else:
                i += 1
        if targets:
            prefix = targets[0]
        if not line_count and not byte_count:
            line_count = 1000
        data = None
        if self._piped_input:
            data = self._piped_input
        else:
            self._print("  Usage: split [-l N] [-b N] [-d] [input] [prefix]")
            self._last_exit_code = 1
            return
        idx = 0
        if byte_count > 0:
            raw_bytes = data.encode("utf-8", errors="replace")
            while raw_bytes:
                chunk = raw_bytes[:byte_count]
                raw_bytes = raw_bytes[byte_count:]
                suffix = f"{idx:02d}" if numeric else chr(ord("a") + idx % 26) + chr(ord("a") + idx // 26 % 26)
                fname = f"{prefix}{suffix}"
                try:
                    Path(fname).write_bytes(chunk)
                except OSError:
                    self._print(f"  split: cannot write to {fname}")
                    self._last_exit_code = 1
                    return
                self._print(f"  {fname}")
                idx += 1
        else:
            lines = data.splitlines(keepends=True)
            while lines:
                chunk = lines[:line_count]
                lines = lines[line_count:]
                suffix = f"{idx:02d}" if numeric else chr(ord("a") + idx % 26) + chr(ord("a") + idx // 26 % 26)
                fname = f"{prefix}{suffix}"
                try:
                    Path(fname).write_text("".join(chunk))
                except OSError:
                    self._print(f"  split: cannot write to {fname}")
                    self._last_exit_code = 1
                    return
                self._print(f"  {fname}")
                idx += 1
        self._last_exit_code = 0

    # ── df ────────────────────────────────────────────────────────

    def _cmd_df(self, args: str = "") -> None:
        """Report disk space usage.

        Flags:
          -h    Human-readable sizes
          -B N  Block size
        """
        parts = args.strip().split() if args else []
        human = False
        block_size = 1024
        targets = []
        i = 0
        while i < len(parts):
            if parts[i] == "-h":
                human = True
                i += 1
            elif parts[i] == "-H":
                human = True
                i += 1
            elif parts[i] == "-B" and i + 1 < len(parts):
                block_size = int(parts[i + 1])
                i += 2
            elif parts[i].startswith("-") and parts[i][1:].isdigit():
                block_size = int(parts[i][1:])
                i += 1
            else:
                targets.append(parts[i])
                i += 1

        def _fmt(size_bytes):
            if human:
                for unit in ("", "K", "M", "G", "T"):
                    if abs(size_bytes) < 1024:
                        return f"{size_bytes:>6.1f}{unit}"
                    size_bytes /= 1024
                return f"{size_bytes:>6.1f}P"
            blocks = size_bytes // block_size
            return f"{blocks:>8}"

        self._print("Filesystem     1K-blocks      Used Available Use% Mounted on")
        if targets:
            for path in targets:
                target = os.path.expanduser(path)
                try:
                    st = os.statvfs(target)
                    total = st.f_blocks * st.f_frsize
                    free = st.f_bfree * st.f_frsize
                    avail = st.f_bavail * st.f_frsize
                    used = total - free
                    pct = int(used / total * 100) if total else 0
                    self._print(f"virtual-fs    {_fmt(total)}  {_fmt(used)}  {_fmt(avail)}  {pct:>2}% {target}")
                except (OSError, FileNotFoundError):
                    self._print(f"df: {path}: No such file or directory")
                    self._last_exit_code = 1
                    return
        else:
            cwd = os.getcwd()
            try:
                st = os.statvfs(cwd)
                total = st.f_blocks * st.f_frsize
                free = st.f_bfree * st.f_frsize
                avail = st.f_bavail * st.f_frsize
                used = total - free
                pct = int(used / total * 100) if total else 0
                self._print(f"virtual-fs    {_fmt(total)}  {_fmt(used)}  {_fmt(avail)}  {pct:>2}% {cwd}")
            except OSError:
                self._print(f"df: {cwd}: No such file or directory")
                self._last_exit_code = 1
        self._last_exit_code = 0

    # ── readlink ──────────────────────────────────────────────────

    def _cmd_readlink(self, args: str = "") -> None:
        """Resolve a symbolic link.

        Flags:
          -f    Canonicalize (resolve all components)
          -n    Do not add trailing newline
        """
        parts = args.strip().split() if args else []
        canonical = False
        target = None
        for p in parts:
            if p == "-f":
                canonical = True
            elif p == "-n":
                pass
            elif not p.startswith("-"):
                target = p
        if not target:
            self._print("  Usage: readlink [-f] <path>")
            self._last_exit_code = 1
            return
        path = os.path.expanduser(target)
        if canonical:
            try:
                resolved = os.path.realpath(path)
                self._print(resolved)
                self._last_exit_code = 0
            except (OSError, ValueError):
                self._last_exit_code = 1
        else:
            try:
                link = os.readlink(path)
                self._print(link)
                self._last_exit_code = 0
            except OSError:
                self._last_exit_code = 1

    # ── file ──────────────────────────────────────────────────────

    def _cmd_file(self, args: str = "") -> None:
        """Determine file type.

        Flags:
          -b    Brief (no filename prefix)
          -i    MIME type only
        """
        parts = args.strip().split() if args else []
        brief = False
        mime_only = False
        targets = []
        i = 0
        while i < len(parts):
            if parts[i] == "-b":
                brief = True
                i += 1
            elif parts[i] == "-i":
                mime_only = True
                i += 1
            else:
                targets.append(parts[i])
                i += 1
        if not targets:
            self._print("  Usage: file [-b] [-i] <file1> [...]")
            self._last_exit_code = 1
            return
        for path in targets:
            target = os.path.expanduser(path)
            try:
                vfs = self.os.vfs
                content = None
                if vfs and (target.startswith("/dev") or target.startswith("/proc")):
                    content = vfs.read(target)
                elif Path(target).exists():
                    content = Path(target).read_text(errors="replace")
                if content is None:
                    self._print(f"  file: {path}: cannot open")
                    self._last_exit_code = 1
                    continue
                if len(content) == 0:
                    desc = "empty"
                    mime = "application/x-empty"
                elif content.startswith("#!"):
                    shebang = content.split("\n")[0].strip()
                    if "python" in shebang:
                        desc = "Python script, ASCII text executable"
                        mime = "text/x-python"
                    elif "bash" in shebang or "sh" in shebang:
                        desc = "Bourne-Again shell script, ASCII text executable"
                        mime = "text/x-shellscript"
                    elif "node" in shebang or "nodejs" in shebang:
                        desc = "Node.js script, ASCII text executable"
                        mime = "application/javascript"
                    else:
                        desc = "script, ASCII text executable"
                        mime = "text/x-script"
                elif content.strip().startswith("{") or content.strip().startswith("["):
                    try:
                        import json as _json
                        _json.loads(content)
                        desc = "JSON data, ASCII text"
                        mime = "application/json"
                    except Exception:
                        desc = "ASCII text"
                        mime = "text/plain"
                elif content.strip().startswith("<"):
                    desc = "XML document, ASCII text"
                    mime = "application/xml"
                elif all(ord(c) < 128 for c in content):
                    desc = "ASCII text"
                    mime = "text/plain"
                else:
                    desc = "data"
                    mime = "application/octet-stream"
                if mime_only:
                    self._print(mime)
                elif brief:
                    self._print(desc)
                else:
                    self._print(f"{path}: {desc}")
                self._last_exit_code = 0
            except (OSError, PermissionError) as e:
                self._print(f"  file: {path}: {e}")
                self._last_exit_code = 1

    # ── timeout ───────────────────────────────────────────────────

    def _cmd_timeout(self, args: str = "") -> None:
        """Run a command with a time limit.

        Flags:
          -s SIG    Signal to send on timeout (default SIGTERM)
          -k SEC    Kill after SEC seconds if still running
          SEC       Timeout in seconds
          COMMAND   Command to run
        """
        parts = args.strip().split() if args else []
        if not parts:
            self._print("  Usage: timeout [-s SIG] [-k SEC] SEC COMMAND [...]")
            self._last_exit_code = 1
            return
        sig = "SIGTERM"
        kill_after = None
        i = 0
        while i < len(parts):
            if parts[i] == "-s" and i + 1 < len(parts):
                sig = parts[i + 1].upper()
                if not sig.startswith("SIG"):
                    sig = "SIG" + sig
                i += 2
            elif parts[i] == "-k" and i + 1 < len(parts):
                kill_after = float(parts[i + 1])
                i += 2
            elif parts[i] == "--":
                i += 1
            else:
                break
        if i >= len(parts):
            self._print("  Usage: timeout [-s SIG] [-k SEC] SEC COMMAND [...]")
            self._last_exit_code = 1
            return
        try:
            timeout_sec = float(parts[i])
            i += 1
        except ValueError:
            self._print(f"  timeout: invalid time interval '{parts[i]}'")
            self._last_exit_code = 1
            return
        cmd_parts = parts[i:]
        if not cmd_parts:
            self._print("  Usage: timeout [-s SIG] [-k SEC] SEC COMMAND [...]")
            self._last_exit_code = 1
            return
        cmd_str = " ".join(cmd_parts)
        import subprocess as _sp
        import signal as _signal
        import time as _time
        try:
            proc = _sp.Popen(cmd_str, shell=True, stdout=_sp.PIPE, stderr=_sp.PIPE)
            try:
                stdout, stderr = proc.communicate(timeout=timeout_sec)
                if stdout:
                    self._print(stdout.decode(errors="replace").rstrip())
                if stderr:
                    self._print(stderr.decode(errors="replace").rstrip(), file=sys.stderr)
                self._last_exit_code = proc.returncode
            except _sp.TimeoutExpired:
                sig_num = getattr(_signal, sig, _signal.SIGTERM)
                proc.send_signal(sig_num)
                try:
                    proc.wait(timeout=kill_after or 10)
                except _sp.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                self._print(f"  timeout: command timed out after {timeout_sec}s")
                self._last_exit_code = 124
        except Exception as e:
            self._print(f"  timeout: {e}")
            self._last_exit_code = 1

    # ── watch ─────────────────────────────────────────────────────

    def _cmd_watch(self, args: str = "") -> None:
        """Repeatedly run a command and display output.

        Flags:
          -n SEC    Interval between runs (default 2)
          -c        Clear screen between runs

        Legacy syntax: watch SEC COMMAND
        """
        parts = args.strip().split() if args else []
        if not parts:
            self._print("  Usage: watch [-n SEC] [-c] COMMAND [...]")
            self._last_exit_code = 1
            return
        interval = 2.0
        clear = False
        i = 0
        has_flags = False
        while i < len(parts):
            if parts[i] == "-n" and i + 1 < len(parts):
                try:
                    interval = float(parts[i + 1])
                except ValueError:
                    self._print(f"  Invalid interval: {parts[i + 1]}")
                    self._last_exit_code = 1
                    return
                i += 2
                has_flags = True
            elif parts[i] == "-c":
                clear = True
                i += 1
                has_flags = True
            elif parts[i] == "--":
                i += 1
                has_flags = True
            else:
                break
        cmd_parts = parts[i:]
        if not has_flags:
            try:
                interval = float(parts[0])
                cmd_parts = parts[1:]
            except ValueError:
                self._print(f"  Invalid interval: {parts[0]}")
                self._last_exit_code = 1
                return
        if not cmd_parts:
            self._print("  Usage: watch [-n SEC] [-c] COMMAND [...]")
            self._last_exit_code = 1
            return
        cmd_str = " ".join(cmd_parts)
        import subprocess as _sp
        import time as _time
        iterations = 0
        max_iterations = 5
        try:
            while iterations < max_iterations:
                if clear:
                    self._print("\033[2J\033[H", end="")
                from datetime import datetime as _dt
                now = _dt.now().strftime("%H:%M:%S")
                self._print(f"Every {interval}s: {cmd_str}  {now}")
                self._print("─" * 60)
                try:
                    result = _sp.run(
                        cmd_str, shell=True, capture_output=True, text=True, timeout=interval
                    )
                    output = result.stdout.rstrip()
                    if output:
                        self._print(output)
                    if result.stderr:
                        self._print(result.stderr.rstrip(), file=sys.stderr)
                except _sp.TimeoutExpired:
                    self._print(f"  watch: command timed out")
                iterations += 1
                if iterations < max_iterations:
                    _time.sleep(interval)
            self._last_exit_code = 0
        except KeyboardInterrupt:
            self._print("")
            self._last_exit_code = 130

    # ── yes ───────────────────────────────────────────────────────

    def _cmd_yes(self, args: str = "") -> None:
        """Repeatedly output a string: yes [STRING]"""
        string = args.strip() if args.strip() else "y"
        count = 0
        max_count = 100
        try:
            while count < max_count:
                self._print(string)
                count += 1
            self._last_exit_code = 0
        except KeyboardInterrupt:
            self._print("")
            self._last_exit_code = 130

    # ── pushd / popd / dirs ───────────────────────────────────────

    def _cmd_pushd(self, args: str = "") -> None:
        """Push directory onto stack: pushd [DIR]"""
        parts = args.strip().split() if args else []
        target = parts[0] if parts else None
        if not hasattr(self, '_dir_stack'):
            self._dir_stack = []
        old_cwd = os.getcwd()
        if target:
            try:
                os.chdir(os.path.expanduser(target))
            except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
                self._print(f"  pushd: {e}")
                self._last_exit_code = 1
                return
        elif self._dir_stack:
            os.chdir(self._dir_stack[-1])
        else:
            self._print("  pushd: no directory stack")
            self._last_exit_code = 1
            return
        self._dir_stack.append(old_cwd)
        self._print("  " + " ".join(self._dir_stack[::-1]))
        self._last_exit_code = 0

    def _cmd_popd(self, args: str = "") -> None:
        """Pop directory from stack: popd"""
        if not hasattr(self, '_dir_stack') or not self._dir_stack:
            self._print("  popd: directory stack empty")
            self._last_exit_code = 1
            return
        target = self._dir_stack.pop()
        try:
            os.chdir(target)
        except (FileNotFoundError, PermissionError) as e:
            self._print(f"  popd: {e}")
            self._last_exit_code = 1
            return
        self._print("  " + " ".join(self._dir_stack[::-1]))
        self._last_exit_code = 0

    def _cmd_dirs(self, args: str = "") -> None:
        """Display directory stack: dirs [-v]"""
        if not hasattr(self, '_dir_stack'):
            self._dir_stack = []
        stack = self._dir_stack[::-1] + [os.getcwd()]
        if "-v" in args:
            for i, d in enumerate(stack):
                self._print(f"  {i}  {d}")
        else:
            self._print("  " + " ".join(stack))
        self._last_exit_code = 0

    # ── tar ───────────────────────────────────────────────────────

    def _cmd_tar(self, args: str = "") -> None:
        """Archive files (tar-like).

        Flags:
          -c    Create archive
          -x    Extract archive
          -t    List archive contents
          -v    Verbose
          -f FILE    Archive file name
          -z    gzip compression
        """
        import tarfile as _tarfile
        parts = args.strip().split() if args else []
        if not parts:
            self._print("  Usage: tar [-ctxvf] [file] [path ...]")
            self._last_exit_code = 1
            return
        create = False
        extract = False
        list_mode = False
        verbose = False
        archive_file = None
        compress = False
        paths = []
        i = 0
        while i < len(parts):
            p = parts[i]
            if p.startswith("-") and len(p) > 1:
                flags = p[1:]
                if "c" in flags:
                    create = True
                if "x" in flags:
                    extract = True
                if "t" in flags:
                    list_mode = True
                if "v" in flags:
                    verbose = True
                if "z" in flags:
                    compress = True
                if "f" in flags:
                    if "f" == flags[-1] and i + 1 < len(parts):
                        archive_file = parts[i + 1]
                        i += 2
                    else:
                        idx = flags.index("f")
                        rest = flags[idx + 1:]
                        if rest:
                            archive_file = rest
                            i += 1
                        elif i + 1 < len(parts):
                            archive_file = parts[i + 1]
                            i += 2
                        else:
                            i += 1
                else:
                    i += 1
            else:
                paths.append(p)
                i += 1
        if not archive_file:
            self._print("  tar: archive file name required (-f FILE)")
            self._last_exit_code = 1
            return
        if create:
            try:
                mode = "w:gz" if compress else "w"
                with _tarfile.open(archive_file, mode) as tar:
                    for path in paths:
                        target = os.path.expanduser(path)
                        tar.add(target, arcname=path)
                        if verbose:
                            self._print(f"  {path}")
                self._last_exit_code = 0
            except Exception as e:
                self._print(f"  tar: {e}")
                self._last_exit_code = 1
        elif extract:
            try:
                mode = "r:gz" if compress else "r"
                with _tarfile.open(archive_file, mode) as tar:
                    tar.extractall(".")
                    if verbose:
                        for m in tar.getmembers():
                            self._print(f"  {m.name}")
                self._last_exit_code = 0
            except Exception as e:
                self._print(f"  tar: {e}")
                self._last_exit_code = 1
        elif list_mode:
            try:
                mode = "r:gz" if compress else "r"
                with _tarfile.open(archive_file, mode) as tar:
                    for m in tar.getmembers():
                        self._print(f"  {m.name}")
                self._last_exit_code = 0
            except Exception as e:
                self._print(f"  tar: {e}")
                self._last_exit_code = 1
        else:
            self._print("  Usage: tar [-ctxvf] [file] [path ...]")
            self._last_exit_code = 1

    # ── gzip / gunzip ─────────────────────────────────────────────

    def _cmd_gzip(self, args: str = "") -> None:
        """Compress files with gzip.

        Flags:
          -k    Keep original file
          -d    Decompress (same as gunzip)
          -v    Verbose
        """
        import gzip as _gzip
        parts = args.strip().split() if args else []
        keep = "-k" in parts
        decompress = "-d" in parts
        verbose = "-v" in parts
        paths = [p for p in parts if not p.startswith("-")]
        if not paths:
            self._print("  Usage: gzip [-kdv] <file> [...]")
            self._last_exit_code = 1
            return
        for path in paths:
            target = os.path.expanduser(path)
            if decompress:
                out_name = target[:-3] if target.endswith(".gz") else target + ".uncompressed"
                try:
                    with _gzip.open(target, "rb") as f_in:
                        data = f_in.read()
                    with open(out_name, "wb") as f_out:
                        f_out.write(data)
                    if not keep:
                        os.unlink(target)
                    if verbose:
                        self._print(f"  {path}")
                    self._last_exit_code = 0
                except Exception as e:
                    self._print(f"  gzip: {e}")
                    self._last_exit_code = 1
            else:
                out_name = target + ".gz"
                try:
                    with open(target, "rb") as f_in:
                        data = f_in.read()
                    with _gzip.open(out_name, "wb") as f_out:
                        f_out.write(data)
                    if not keep:
                        os.unlink(target)
                    if verbose:
                        self._print(f"  {path}")
                    self._last_exit_code = 0
                except Exception as e:
                    self._print(f"  gzip: {e}")
                    self._last_exit_code = 1

    def _cmd_gunzip(self, args: str = "") -> None:
        """Decompress gzip files."""
        self._cmd_gzip(f"-d {args}")

    # ── which ─────────────────────────────────────────────────────

    def _cmd_which(self, args: str = "") -> None:
        """Locate a command: which COMMAND [...]"""
        parts = args.strip().split() if args else []
        if not parts:
            self._print("  Usage: which <command> [...]")
            self._last_exit_code = 1
            return
        import shutil as _shutil
        for cmd in parts:
            if hasattr(self, f"_cmd_{cmd}"):
                self._print(f"  {cmd}: shell built-in")
            elif cmd in self.COMMANDS:
                self._print(f"  {cmd}: shell built-in")
            else:
                found = _shutil.which(cmd)
                if found:
                    self._print(f"  {found}")
                else:
                    self._print(f"  {cmd} not found")
        self._last_exit_code = 0

    # ── expr ──────────────────────────────────────────────────────

    def _cmd_expr(self, args: str = "") -> None:
        """Evaluate an expression: expr EXPRESSION

        Supports: +, -, *, /, %, length, substr, index, match
        """
        import re as _re
        expr = args.strip()
        if not expr:
            self._print("  Usage: expr EXPRESSION")
            self._last_exit_code = 1
            return
        try:
            # Handle length operator
            m = _re.match(r'^length\s+"(.*)"$', expr)
            if m:
                self._print(str(len(m.group(1))))
                self._last_exit_code = 0
                return
            m = _re.match(r"^length\s+'(.*)'$", expr)
            if m:
                self._print(str(len(m.group(1))))
                self._last_exit_code = 0
                return
            m = _re.match(r'^length\s+(\S+)$', expr)
            if m:
                self._print(str(len(m.group(1))))
                self._last_exit_code = 0
                return
            # Handle substr: substr STRING POS LENGTH
            m = _re.match(r'^substr\s+"(.*)"\s+(\d+)\s+(\d+)$', expr)
            if m:
                s, pos, length = m.group(1), int(m.group(2)), int(m.group(3))
                self._print(s[pos - 1:pos - 1 + length])
                self._last_exit_code = 0
                return
            # Handle index: index STRING CHARS
            m = _re.match(r'^index\s+"(.*)"\s+"(.*)"$', expr)
            if m:
                s, chars = m.group(1), m.group(2)
                idx = 0
                for i, c in enumerate(s):
                    if c in chars:
                        idx = i + 1
                        break
                self._print(str(idx))
                self._last_exit_code = 0
                return
            # Handle arithmetic: simplify expr args
            safe = expr.replace("*", " * ").replace("/", " / ").replace("%", " % ")
            safe = safe.replace("+", " + ").replace("-", " - ")
            tokens = safe.split()
            result = 0
            op = "+"
            for tok in tokens:
                if tok in ("+", "-", "*", "/", "%"):
                    op = tok
                else:
                    val = int(tok)
                    if op == "+":
                        result += val
                    elif op == "-":
                        result -= val
                    elif op == "*":
                        result *= val
                    elif op == "/":
                        result = result // val if val else 0
                    elif op == "%":
                        result = result % val if val else 0
            self._print(str(result))
            self._last_exit_code = 0
        except (ValueError, ZeroDivisionError):
            self._print("  expr: invalid expression")
            self._last_exit_code = 1

    # ── eval ──────────────────────────────────────────────────────

    def _cmd_eval(self, args: str = "") -> None:
        """Evaluate arguments as a command: eval COMMAND ..."""
        if not args:
            self._last_exit_code = 0
            return
        self._execute_single(args, "")
        self._last_exit_code = 0

    # ── wait ──────────────────────────────────────────────────────

    def _cmd_wait(self, args: str = "") -> None:
        """Wait for background processes: wait [PID]"""
        if not hasattr(self, '_bg_threads'):
            self._bg_threads = {}
        if not self._bg_threads:
            self._last_exit_code = 0
            return
        for name, t in list(self._bg_threads.items()):
            if t.is_alive():
                t.join(timeout=5)
        self._last_exit_code = 0

    # ── trap ──────────────────────────────────────────────────────

    def _cmd_trap(self, args: str = "") -> None:
        """Set a signal handler: trap COMMAND SIGNAL

        Simplified: trap '' SIGNAL to ignore, trap - SIGNAL to reset.
        """
        parts = args.strip().split() if args else []
        if len(parts) < 2:
            if not parts:
                self._print("  trap -- listing signal handlers")
            self._last_exit_code = 0
            return
        if not hasattr(self, '_trap_handlers'):
            self._trap_handlers = {}
        command = parts[0]
        for sig in parts[1:]:
            if command == "-":
                self._trap_handlers.pop(sig, None)
            elif command == "":
                self._trap_handlers[sig] = None
            else:
                self._trap_handlers[sig] = command
        self._last_exit_code = 0

    # ── local ─────────────────────────────────────────────────────

    def _cmd_local(self, args: str = "") -> None:
        """Declare local variables (no-op in non-function context): local VAR=value"""
        parts = args.strip().split() if args else []
        for part in parts:
            if "=" in part:
                name, value = part.split("=", 1)
                self._env[name] = value
            else:
                self._env[part] = ""
        self._last_exit_code = 0

    # ── exec ──────────────────────────────────────────────────────

    def _cmd_exec(self, args: str = "") -> None:
        """Execute a command (replaces current process in shell semantics): exec COMMAND"""
        if not args:
            self._last_exit_code = 0
            return
        self._execute_single(args, "")
        self._last_exit_code = 0

    # ── test / [ ──────────────────────────────────────────────────

    def _cmd_test(self, args: str = "") -> None:
        """Evaluate a conditional expression: test EXPRESSION"""
        import os as _os
        expr = args.strip()
        if not expr:
            self._last_exit_code = 1
            return
        # Strip [ and ] wrappers
        if expr.startswith("[") and expr.endswith("]"):
            expr = expr[1:-1].strip()
        parts = expr.split()
        if len(parts) == 1:
            self._last_exit_code = 0 if _os.path.exists(parts[0]) else 1
            return
        if len(parts) == 2:
            op, arg = parts
            if op == "-e":
                self._last_exit_code = 0 if _os.path.exists(arg) else 1
            elif op == "-f":
                self._last_exit_code = 0 if _os.path.isfile(arg) else 1
            elif op == "-d":
                self._last_exit_code = 0 if _os.path.isdir(arg) else 1
            elif op == "-r":
                self._last_exit_code = 0 if _os.path.exists(arg) and _os.access(arg, _os.R_OK) else 1
            elif op == "-w":
                self._last_exit_code = 0 if _os.path.exists(arg) and _os.access(arg, _os.W_OK) else 1
            elif op == "-x":
                self._last_exit_code = 0 if _os.path.exists(arg) and _os.access(arg, _os.X_OK) else 1
            elif op == "-s":
                self._last_exit_code = 0 if _os.path.exists(arg) and _os.path.getsize(arg) > 0 else 1
            elif op == "-z":
                self._last_exit_code = 0 if len(arg) == 0 else 1
            elif op == "-n":
                self._last_exit_code = 0 if len(arg) > 0 else 1
            elif op == "!":
                self._last_exit_code = 0
            else:
                self._last_exit_code = 1
            return
        if len(parts) == 3:
            left, op, right = parts
            if op == "=" or op == "==":
                self._last_exit_code = 0 if left == right else 1
            elif op == "!=":
                self._last_exit_code = 0 if left != right else 1
            elif op == "-eq":
                self._last_exit_code = 0 if int(left) == int(right) else 1
            elif op == "-ne":
                self._last_exit_code = 0 if int(left) != int(right) else 1
            elif op == "-lt":
                self._last_exit_code = 0 if int(left) < int(right) else 1
            elif op == "-le":
                self._last_exit_code = 0 if int(left) <= int(right) else 1
            elif op == "-gt":
                self._last_exit_code = 0 if int(left) > int(right) else 1
            elif op == "-ge":
                self._last_exit_code = 0 if int(left) >= int(right) else 1
            else:
                self._last_exit_code = 1
            return
        self._last_exit_code = 1
