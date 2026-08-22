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
        """List directory contents."""
        target = args.strip() or "."
        try:
            vfs = self.os.vfs
            if vfs and (target.startswith("/dev") or target.startswith("/proc")):
                entries = vfs.listdir(target)
            else:
                entries = os.listdir(os.path.expanduser(target))
            if entries is None:
                self._print(f"  ls: cannot access '{target}': No such file or directory")
                self._last_exit_code = 1
                return
            entries.sort()
            parts = []
            for e in entries:
                path = os.path.join(target, e) if target != "." else e
                if vfs:
                    is_dir = vfs.isdir(path)
                else:
                    is_dir = os.path.isdir(os.path.expanduser(path))
                suffix = "/" if is_dir else ""
                parts.append(e + suffix)
            if parts:
                self._print("  " + "  ".join(parts))
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
        """Create directories."""
        if not args:
            self._print("  Usage: mkdir <dir>")
            self._last_exit_code = 1
            return
        target = os.path.expanduser(args.strip())
        try:
            os.makedirs(target, exist_ok=False)
            self._last_exit_code = 0
        except FileExistsError:
            self._print(f"  mkdir: cannot create directory '{target}': File exists")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  mkdir: permission denied: {target}")
            self._last_exit_code = 1
        except FileNotFoundError:
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
        """Create empty files or update timestamps."""
        if not args:
            self._print("  Usage: touch <file> [file...]")
            self._last_exit_code = 1
            return
        for p in args.strip().split():
            target = os.path.expanduser(p)
            try:
                if os.path.exists(target):
                    os.utime(target, None)
                else:
                    Path(target).write_text("")
                self._last_exit_code = 0
            except PermissionError:
                self._print(f"  touch: permission denied: {p}")
                self._last_exit_code = 1

    def _cmd_cp(self, args: str = "") -> None:
        """Copy files."""
        if not args:
            self._print("  Usage: cp <src> <dst>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        if len(parts) < 2:
            self._print("  cp: missing destination")
            self._last_exit_code = 1
            return
        src, dst = os.path.expanduser(parts[0]), os.path.expanduser(parts[1])
        try:
            import shutil as _shutil
            if os.path.isdir(src):
                _shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                _shutil.copy2(src, dst)
            self._last_exit_code = 0
        except FileNotFoundError:
            self._print(f"  cp: cannot stat '{parts[0]}': No such file or directory")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  cp: permission denied")
            self._last_exit_code = 1

    def _cmd_mv(self, args: str = "") -> None:
        """Move or rename files."""
        if not args:
            self._print("  Usage: mv <src> <dst>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        if len(parts) < 2:
            self._print("  mv: missing destination")
            self._last_exit_code = 1
            return
        src, dst = os.path.expanduser(parts[0]), os.path.expanduser(parts[1])
        try:
            os.rename(src, dst)
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
        """Display file or directory metadata."""
        if not args:
            self._print("  Usage: stat <path>")
            self._last_exit_code = 1
            return
        target = os.path.expanduser(args.strip())
        try:
            st = os.stat(target)
            import stat as _stat, time as _time
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
            self._print(f"  stat: cannot stat '{args.strip()}': No such file or directory")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  stat: cannot stat '{args.strip()}': Permission denied")
            self._last_exit_code = 1

    # ── text processing (VFS-aware, piped-input aware) ──────────────

    def _cmd_head(self, args: str = "") -> None:
        """Output the first part of files (VFS-aware)."""
        parts = args.strip().split() if args else []
        n = 10
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
            else:
                targets.append(p)
                i += 1
        if not targets:
            if self._piped_input:
                lines = self._piped_input.splitlines()
                self._print("\n".join(lines[:n]))
                self._last_exit_code = 0
                return
            self._print("  Usage: head [-N | -n N] <file>")
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
                lines = content.splitlines()
                out = "\n".join(lines[:n])
                if len(targets) > 1:
                    self._print(f"==> {path} <==")
                self._print(out)
                self._last_exit_code = 0
            except FileNotFoundError:
                self._print(f"  head: {path}: No such file or directory")
                self._last_exit_code = 1

    def _cmd_tail(self, args: str = "") -> None:
        """Output the last part of files (VFS-aware)."""
        parts = args.strip().split() if args else []
        n = 10
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
            else:
                targets.append(p)
                i += 1
        if not targets:
            if self._piped_input:
                lines = self._piped_input.splitlines()
                self._print("\n".join(lines[-n:]))
                self._last_exit_code = 0
                return
            self._print("  Usage: tail [-N | -n N] <file>")
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
                lines = content.splitlines()
                out = "\n".join(lines[-n:])
                if len(targets) > 1:
                    self._print(f"==> {path} <==")
                self._print(out)
                self._last_exit_code = 0
            except FileNotFoundError:
                self._print(f"  tail: {path}: No such file or directory")
                self._last_exit_code = 1

    def _cmd_wc(self, args: str = "") -> None:
        """Count lines, words, and characters (VFS-aware)."""
        parts = args.strip().split() if args else []
        show_lines = show_words = show_chars = False
        targets = []
        for p in parts:
            if p == "-l":
                show_lines = True
            elif p == "-w":
                show_words = True
            elif p == "-c" or p == "-m":
                show_chars = True
            elif not p.startswith("-"):
                targets.append(p)
        if not show_lines and not show_words and not show_chars:
            show_lines = show_words = show_chars = True
        if not targets:
            if self._piped_input:
                lines = len(self._piped_input.splitlines())
                words = len(self._piped_input.split())
                chars = len(self._piped_input)
                parts_list = []
                if show_lines:
                    parts_list.append(f"{lines:4}")
                if show_words:
                    parts_list.append(f"{words:4}")
                if show_chars:
                    parts_list.append(f"{chars:4}")
                self._print(" ".join(parts_list))
                self._last_exit_code = 0
                return
            self._print("  Usage: wc [-l] [-w] [-c] <file>")
            self._last_exit_code = 1
            return
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
                lines = len(content.splitlines())
                words = len(content.split())
                chars = len(content)
                parts_list = []
                if show_lines:
                    parts_list.append(f"{lines:4}")
                if show_words:
                    parts_list.append(f"{words:4}")
                if show_chars:
                    parts_list.append(f"{chars:4}")
                if len(targets) > 1:
                    parts_list.append(target)
                self._print(" ".join(parts_list))
                self._last_exit_code = 0
            except FileNotFoundError:
                self._print(f"  wc: {target}: No such file or directory")
                self._last_exit_code = 1

    def _cmd_grep(self, args: str = "") -> None:
        """Search for patterns in files or piped input (VFS-aware).

        Flags:
            -i  Ignore case
            -v  Invert match
            -c  Count matching lines only
            -l  Print only filenames with matches
            -n  Prefix each output line with line number
            -w  Match whole words only
            -A N  Print N lines after each match
            -B N  Print N lines before each match
            -C N  Print N lines of context (both sides)
        """
        if not args and not self._piped_input:
            self._print("  Usage: grep [-i] [-v] [-c] [-l] [-n] [-w] [-A N] [-B N] [-C N] <pattern> [file]")
            self._last_exit_code = 1
            return
        import re as _re
        parts = args.strip().split()
        flags = [p for p in parts if p.startswith("-")]
        non_flags = [p for p in parts if not p.startswith("-")]

        # Parse simple flags
        ignore_case = any(f in ("-i", "-vi") for f in flags)
        invert = any(f in ("-v", "-vi") for f in flags)
        count_only = "-c" in flags
        files_only = "-l" in flags
        line_numbers = "-n" in flags
        word_boundary = "-w" in flags

        # Parse -A/-B/-C with numeric args
        after_context = 0
        before_context = 0
        context = 0
        i = 0
        clean_parts = []
        while i < len(parts):
            if parts[i] in ("-A", "-B", "-C") and i + 1 < len(parts):
                try:
                    n = int(parts[i + 1])
                    if parts[i] == "-A":
                        after_context = n
                    elif parts[i] == "-B":
                        before_context = n
                    else:
                        context = n
                    i += 2
                    continue
                except ValueError:
                    pass
            clean_parts.append(parts[i])
            i += 1

        # Re-parse non-flags from clean_parts
        non_flags = [p for p in clean_parts if not p.startswith("-")]
        pattern = non_flags[0] if non_flags else ""
        targets = non_flags[1:] if len(non_flags) > 1 else []
        recursive = any(f in ("-r", "-R") for f in flags)

        if not pattern:
            self._print("  Usage: grep <pattern> [file...]")
            self._last_exit_code = 1
            return

        # Apply -C as both -A and -B
        if context > 0:
            after_context = max(after_context, context)
            before_context = max(before_context, context)

        try:
            # Build regex
            kwargs = {"flags": _re.IGNORECASE} if ignore_case else {}
            if word_boundary:
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
                                              line_numbers, after_context, before_context, label)
                    if found:
                        matched_any = True
            else:
                lines = self._piped_input.splitlines()
                matched_any = self._grep_search(lines, pat, invert, count_only, files_only,
                                                line_numbers, after_context, before_context, None)
            self._last_exit_code = 0 if matched_any else 1
        except FileNotFoundError:
            self._print(f"  grep: {targets[0]}: No such file or directory")
            self._last_exit_code = 1

    def _grep_search(self, lines, pat, invert, count_only, files_only,
                     line_numbers, after_context, before_context, label):
        """Core grep logic on a list of lines. Returns True if any match."""
        matched_indices = set()
        for idx, line in enumerate(lines):
            found = bool(pat.search(line))
            if invert:
                found = not found
            if found:
                matched_indices.add(idx)

        if count_only:
            if label and len(lines) > 0:
                self._print(f"  {label}:{len(matched_indices)}")
            else:
                self._print(f"  {len(matched_indices)}")
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
            if before_context > 0 or after_context > 0:
                if idx not in matched_indices and (prev_printed < idx - 1):
                    if prev_printed >= 0:
                        self._print("  --")
                    for b in range(max(0, idx - before_context), idx):
                        if b in output_indices and b not in matched_indices:
                            prefix = f"{b + 1}:" if line_numbers else ""
                            self._print(f"  {prefix}{lines[b]}")
            prefix = f"{idx + 1}:" if line_numbers else ""
            self._print(f"  {prefix}{lines[idx]}")
            if idx in matched_indices:
                matched += 1
            prev_printed = idx
        return matched > 0

    def _cmd_sort(self, args: str = "") -> None:
        """Sort lines of text (from file or piped input)."""
        parts = args.strip().split() if args else []
        flags = [p for p in parts if p.startswith("-")]
        targets = [p for p in parts if not p.startswith("-")]
        reverse = any(f in ("-r", "-R") for f in flags)
        numeric = any(f in ("-n", "-g") for f in flags)
        unique = any(f in ("-u",) for f in flags)
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
            self._print("  Usage: sort [-r] [-n] [-u] [file]")
            self._last_exit_code = 1
            return
        if numeric:
            lines.sort(key=lambda x: float(x.split()[0]) if x.split() else 0, reverse=reverse)
        else:
            lines.sort(reverse=reverse)
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
        """Remove adjacent duplicate lines (from file or piped input)."""
        parts = args.strip().split() if args else []
        count_mode = False
        ignore_case = False
        target = None
        for p in parts:
            if p == "-c":
                count_mode = True
            elif p == "-i":
                ignore_case = True
            elif not p.startswith("-"):
                target = p
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
            self._print("  Usage: uniq [-c] [-i] [file]")
            self._last_exit_code = 1
            return
        out = []
        prev = None
        count = 0
        for l in lines:
            key = l.lower() if ignore_case else l
            prev_key = prev.lower() if ignore_case and prev is not None else prev
            if key == prev_key:
                count += 1
            else:
                if prev is not None:
                    if count_mode:
                        out.append(f"{count:>7} {prev}")
                    else:
                        out.append(prev)
                prev = l
                count = 1
        if prev is not None:
            if count_mode:
                out.append(f"{count:>7} {prev}")
            else:
                out.append(prev)
        self._print("\n".join(out))
        self._last_exit_code = 0

    def _cmd_find(self, args: str = "") -> None:
        """Search for files by name pattern (VFS-aware)."""
        if not args:
            self._print("  Usage: find [dir] -name <pattern>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        search_dir = "."
        pattern = None
        i = 0
        while i < len(parts):
            if parts[i] in ("-name", "-iname") and i + 1 < len(parts):
                import fnmatch as _fnmatch
                pattern = parts[i + 1]
                if parts[i] == "-iname":
                    pattern = pattern.lower()
                    def _match_fn(name, pat=pattern):
                        return _fnmatch.fnmatch(name.lower(), pat)
                else:
                    _match_fn = lambda name, p=pattern: _fnmatch.fnmatch(name, p)
                i += 2
            elif not parts[i].startswith("-"):
                search_dir = parts[i]
                i += 1
            else:
                i += 1
        if pattern is None:
            self._print("  Usage: find [dir] -name <pattern>")
            self._last_exit_code = 1
            return
        search_path = os.path.expanduser(search_dir)
        try:
            matches = []
            for root, dirs, files in os.walk(search_path):
                for name in files + dirs:
                    if _match_fn(name):
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
        """Translate or delete characters (piped input only)."""
        if not self._piped_input:
            self._print("  Usage: <command> | tr <set1> <set2>")
            self._last_exit_code = 1
            return
        parts = args.strip().split() if args else []
        delete = any(p == "-d" for p in parts)
        squeeze = any(p == "-s" for p in parts)
        sets = [p for p in parts if not p.startswith("-")]
        if len(sets) < 1 or (not delete and not squeeze and len(sets) < 2):
            self._print("  Usage: <command> | tr <set1> <set2>")
            self._last_exit_code = 1
            return
        set1 = sets[0]
        set2 = sets[1] if len(sets) > 1 else ""

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
        if delete:
            result = self._piped_input.translate(str.maketrans("", "", expanded1))
        elif squeeze:
            import re as _re
            result = _re.sub(rf"[{_re.escape(expanded1)}]+", lambda m: m.group(0)[0], self._piped_input)
        else:
            trans = str.maketrans(expanded1, expanded2[:len(expanded1)].ljust(len(expanded1), expanded2[-1] if expanded2 else ""))
            result = self._piped_input.translate(trans)
        self._print(result.rstrip("\n"))
        self._last_exit_code = 0

    def _cmd_nl(self, args: str = "") -> None:
        """Number lines of a file or piped input."""
        if not args and not self._piped_input:
            self._print("  Usage: nl [file]")
            self._last_exit_code = 1
            return
        try:
            if args:
                content = Path(os.path.expanduser(args.strip())).read_text()
            else:
                content = self._piped_input
        except FileNotFoundError:
            self._print(f"  nl: {args.strip()}: No such file or directory")
            self._last_exit_code = 1
            return
        lines = content.splitlines()
        out = "\n".join(f"{i + 1}\t{line}" for i, line in enumerate(lines))
        self._print(out)
        self._last_exit_code = 0

    def _cmd_fold(self, args: str = "") -> None:
        """Wrap long lines at a specified width (default 80)."""
        if not args and not self._piped_input:
            self._print("  Usage: fold [-w width] [file]")
            self._last_exit_code = 1
            return
        parts = args.strip().split() if args else []
        width = 80
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
            for i in range(0, len(line), width):
                out_lines.append(line[i:i + width])
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
        """Merge lines of files side by side."""
        parts = args.strip().split() if args else []
        delim = "\t"
        files = []
        i = 0
        while i < len(parts):
            if parts[i] == "-d" and i + 1 < len(parts):
                delim = parts[i + 1]
                i += 2
            elif parts[i].startswith("-d") and len(parts[i]) > 2:
                delim = parts[i][2:]
                i += 1
            else:
                files.append(parts[i])
                i += 1
        if not files and not self._piped_input:
            self._print("  Usage: paste [-d DELIM] <file1> [file2 ...]")
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
        for row in _itertools.zip_longest(*readers, fillvalue=""):
            self._print(delim.join(row))
        self._last_exit_code = 0

    def _cmd_comm(self, args: str = "") -> None:
        """Compare two sorted files line by line."""
        if not args:
            self._print("  Usage: comm <file1> <file2>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        if len(parts) < 2:
            self._print("  Usage: comm <file1> <file2>")
            self._last_exit_code = 1
            return
        f1, f2 = os.path.expanduser(parts[0]), os.path.expanduser(parts[1])
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
                self._print(f"\t\t{lines1[i]}")
                i += 1
            elif lines1[i] > lines2[j]:
                self._print(f"\t{lines2[j]}")
                j += 1
            else:
                self._print(lines1[i])
                i += 1
                j += 1
        while i < len(lines1):
            self._print(f"\t\t{lines1[i]}")
            i += 1
        while j < len(lines2):
            self._print(f"\t{lines2[j]}")
            j += 1
        self._last_exit_code = 0

    def _cmd_expand(self, args: str = "") -> None:
        """Convert tabs to spaces (piped input or file)."""
        if not args and not self._piped_input:
            self._print("  Usage: expand [file]")
            self._last_exit_code = 1
            return
        try:
            if args:
                content = Path(os.path.expanduser(args.strip())).read_text()
            else:
                content = self._piped_input
        except FileNotFoundError:
            self._print(f"  expand: {args.strip()}: No such file or directory")
            self._last_exit_code = 1
            return
        self._print(content.expandtabs(8).rstrip("\n"))
        self._last_exit_code = 0

    def _cmd_unexpand(self, args: str = "") -> None:
        """Convert spaces to tabs (piped input or file)."""
        if not args and not self._piped_input:
            self._print("  Usage: unexpand [file]")
            self._last_exit_code = 1
            return
        try:
            if args:
                content = Path(os.path.expanduser(args.strip())).read_text()
            else:
                content = self._piped_input
        except FileNotFoundError:
            self._print(f"  unexpand: {args.strip()}: No such file or directory")
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
            tabs, rem = divmod(spaces, 8)
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
        """Compare two files line by line. Supports -u (unified) and -w (ignore whitespace)."""
        parts = args.strip().split() if args else []
        unified = False
        ignore_ws = False
        targets = []
        for p in parts:
            if p == "-u":
                unified = True
            elif p == "-w":
                ignore_ws = True
            elif not p.startswith("-"):
                targets.append(p)
        if len(targets) < 2:
            self._print("  Usage: diff [-u] [-w] <file1> <file2>")
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
                self._last_exit_code = 1
        else:
            differ = _difflib.Differ()
            diffs = list(differ.compare(lines1, lines2))
            changes = [l for l in diffs if l.startswith(("+ ", "- ", "? "))]
            if not changes:
                self._last_exit_code = 0
                return
            from ..repl import _C_GREEN, _C_RED, _C_DIM, _C_RESET
            for l in diffs:
                if l.startswith("+ "):
                    self._print(f"  {_C_GREEN}{l}{_C_RESET}")
                elif l.startswith("- "):
                    self._print(f"  {_C_RED}{l}{_C_RESET}")
                elif l.startswith("? "):
                    self._print(f"  {_C_DIM}{l}{_C_RESET}")
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
        if not self._piped_input:
            self._print("  Usage: <command> | xargs [-n N] [-0] [-r] [-I{}] <cmd> [args...]")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
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
        if null_terminated:
            items = [x for x in self._piped_input.split("\0") if x]
        else:
            items = self._piped_input.split()
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
        """Print environment variables."""
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
        """Dump file in octal/hex format."""
        if not args:
            self._print("  Usage: od <file>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        target = None
        base = "o"
        for p in parts:
            if p == "-x":
                base = "x"
            elif p == "-o":
                base = "o"
            elif p == "-d":
                base = "d"
            elif not p.startswith("-"):
                target = p
        if not target:
            self._print("  od: no file specified")
            self._last_exit_code = 1
            return
        try:
            data = Path(os.path.expanduser(target)).read_bytes()
        except FileNotFoundError:
            self._print(f"  od: {target}: No such file or directory")
            self._last_exit_code = 1
            return
        for i in range(0, len(data), 16):
            chunk = data[i:i + 16]
            addr = f"{i:07o}" if base == "o" else f"{i:07x}"
            if base == "o":
                vals = " ".join(f"{b:03o}" for b in chunk)
            elif base == "x":
                vals = " ".join(f"{b:02x}" for b in chunk)
            else:
                vals = " ".join(f"{b:3d}" for b in chunk)
            ascii_repr = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            self._print(f"  {addr} {vals:<48} {ascii_repr}")
        self._last_exit_code = 0

    def _cmd_join(self, args: str = "") -> None:
        """Join lines of two files on a common field."""
        if not args:
            self._print("  Usage: join <file1> <file2>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        if len(parts) < 2:
            self._print("  Usage: join <file1> <file2>")
            self._last_exit_code = 1
            return
        f1, f2 = os.path.expanduser(parts[0]), os.path.expanduser(parts[1])
        try:
            lines1 = [l.split(None, 1) for l in Path(f1).read_text().splitlines()]
            lines2 = [l.split(None, 1) for l in Path(f2).read_text().splitlines()]
        except FileNotFoundError as e:
            self._print(f"  join: {e.filename}: No such file or directory")
            self._last_exit_code = 1
            return
        d1 = {l[0]: l[1] if len(l) > 1 else "" for l in lines1}
        d2 = {l[0]: l[1] if len(l) > 1 else "" for l in lines2}
        for key in sorted(set(d1) & set(d2)):
            self._print(f"{key} {d1[key]} {d2[key]}")
        self._last_exit_code = 0

    # ── misc ────────────────────────────────────────────────────────

    def _cmd_clear(self, args: str = "") -> None:
        """Clear the terminal screen."""
        self._print("\033[2J\033[H", end="")

    def _cmd_sleep(self, args: str = "") -> None:
        """Sleep for N seconds: sleep <seconds>"""
        try:
            secs = float(args.strip())
        except ValueError:
            secs = 1.0
        import time as _time
        _time.sleep(secs)

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
                script = p
                i += 1
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
            elif p.startswith("{") and p.endswith("}"):
                script = p[1:-1]
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
        return result
