"""
AI Networking Processor — Core VM module.

Constants, exceptions, data structures, Memory, DeviceBus, CPU,
Assembler, VMRunner, opcode handlers, and dispatch table.

This is the single core VM file. Device drivers live in vm_devices.py,
assembly programs and self-test live in vm_programs.py.

Layers 4-6 (drivers, libraries, applications) sit on top of this machine.
The VM never imports domain classes. It only knows registers, tensors,
and the generic DEV_OPEN/DEV_CALL/DEV_CLOSE protocol.
"""

from __future__ import annotations

import re
import struct
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Callable

import numpy as np

logger = logging.getLogger("slo.vm")


# ── Constants ────────────────────────────────────────────────────────────────

NUM_REGS = 16
MAX_INSTRUCTIONS = 100_000
MAX_CALL_DEPTH = 256
MEM_SIZE = 65536
STACK_BASE = 0xFFFF
F_ZERO = 1 << 0
F_NEG = 1 << 2


# ── Exceptions ───────────────────────────────────────────────────────────────

class VMFault(Exception):
    """Base VM fault."""


class InsFault(VMFault):
    """Invalid instruction or operand."""


class Halt(VMFault):
    """Program halted (normal termination)."""


class MemFault(VMFault):
    """Memory access violation."""


class SysFault(VMFault):
    """Syscall error."""


class DeviceFault(VMFault):
    """Device error."""


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Instruction:
    """Decoded instruction ready for execution."""
    opcode: str
    operands: list
    line_num: int = 0
    raw: str = ""


@dataclass
class TraceEntry:
    """Snapshot of machine state at one execution cycle."""
    cycle: int
    pc: int
    instruction: str
    registers: dict
    heap_keys: list


# ── ISA Definition ───────────────────────────────────────────────────────────

OPCODES = {
    "LOAD_CONST": "Rd, value          Rd = constant",
    "LOAD_SHAPE": "Rd, rows, cols     Rd = zeros(rows,cols)",
    "MOV":        "Rd, Rs             Rd = Rs",
    "STORE":      "Rs, key            heap[key] = Rs",
    "LOAD":       "Rd, key            Rd = heap[key]",
    "FREE":       "key                delete heap[key]",
    "PRINT":      "Rs                 output Rs",
    "NOP":        "                   no operation",
    "IADD":       "Rd, Ra, Rb         Rd = int(Ra) + int(Rb)",
    "ISUB":       "Rd, Ra, Rb         Rd = int(Ra) - int(Rb)",
    "IMUL":       "Rd, Ra, Rb         Rd = int(Ra) * int(Rb)",
    "IDIV":       "Rd, Ra, Rb         Rd = int(Ra) // int(Rb)",
    "IAND":       "Rd, Ra, Rb         Rd = int(Ra) & int(Rb)",
    "IOR":        "Rd, Ra, Rb         Rd = int(Ra) | int(Rb)",
    "IXOR":       "Rd, Ra, Rb         Rd = int(Ra) ^ int(Rb)",
    "ISHL":       "Rd, Ra, Rb         Rd = int(Ra) << int(Rb)",
    "ISHR":       "Rd, Ra, Rb         Rd = int(Ra) >> int(Rb)",
    "INEG":       "Rd, Ra             Rd = -int(Ra)",
    "INC":        "Rd                 Rd += 1",
    "DEC":        "Rd                 Rd -= 1",
    "ICMP":       "Ra, Rb             set CMP_FLAG",
    "ADD":        "Rd, Ra, Rb         Rd = Ra + Rb",
    "SUB":        "Rd, Ra, Rb         Rd = Ra - Rb",
    "MUL":        "Rd, Ra, Rb         Rd = Ra * Rb",
    "DIV":        "Rd, Ra, Rb         Rd = Ra / Rb",
    "NEG":        "Rd, Ra             Rd = -Ra",
    "ABS":        "Rd, Ra             Rd = |Ra|",
    "MATMUL":     "Rd, Ra, Rb         Rd = Ra @ Rb",
    "TRANSPOSE":  "Rd, Ra             Rd = Ra.T",
    "DOT":        "Rd, Ra, Rb         Rd = Ra . Rb",
    "NORM":       "Rd, Ra             Rd = ||Ra||",
    "SUM":        "Rd, Ra             Rd = sum(Ra)",
    "MEAN":       "Rd, Ra             Rd = mean(Ra)",
    "MAX":        "Rd, Ra             Rd = max(Ra)",
    "ARGMAX":     "Rd, Ra             Rd = argmax(Ra)",
    "RESHAPE":    "Rd, Ra, R, C       Rd = reshape(Ra, (R,C))",
    "SHAPE":      "Rd, Ra             Rd = list(Ra.shape)",
    "SIZE":       "Rd, Ra             Rd = element count",
    "RELU":       "Rd, Ra             Rd = max(0, Ra)",
    "GELU":       "Rd, Ra             Rd = gelu(Ra)",
    "SIGMOID":    "Rd, Ra             Rd = sigmoid(Ra)",
    "TANH":       "Rd, Ra             Rd = tanh(Ra)",
    "SOFTMAX":    "Rd, Ra             Rd = softmax(Ra)",
    "LAYERNORM":  "Rd, Ra             Rd = layer_norm(Ra)",
    "RMSNORM":    "Rd, Ra             Rd = rms_norm(Ra)",
    "RANDN":      "Rd, rows, cols     Rd = randn(rows,cols)",
    "RANDUNIF":   "Rd, rows, cols, lo, hi  Rd = uniform(lo,hi)",
    "CMP":        "Ra, Rb             set CMP_FLAG (-1/0/+1)",
    "TEST":       "Ra                 set CMP_FLAG (0 or 1)",
    "JMP":        "label              PC = label",
    "JZ":         "label              if CMP_FLAG == 0: PC = label",
    "JNZ":        "label              if CMP_FLAG != 0: PC = label",
    "JGT":        "label              if CMP_FLAG > 0: PC = label",
    "JGE":        "label              if CMP_FLAG >= 0: PC = label",
    "JLT":        "label              if CMP_FLAG < 0: PC = label",
    "JLE":        "label              if CMP_FLAG <= 0: PC = label",
    "CALL":       "label              push PC+1, PC = label",
    "RET":        "                   pop call stack",
    "LOOP":       "Rd, label          Rd -= 1; if Rd != 0: PC = label",
    "HALT":       "                   stop execution",
    "SYSCALL":    "                   R0=handler(R7, [R0..R5])",
    "DEV_OPEN":   "Rd, name           Rd = device handle",
    "DEV_CALL":   "Rd, H, method, args...  Rd = device.method(*args)",
    "DEV_CLOSE":  "H                  release device handle",
    "DEV_INFO":   "Rd, H              Rd = device.info()",
    "PUSH":       "Rs                 stack.push(Rs); sp -= 1",
    "POP":        "Rd                 sp += 1; Rd = stack[sp]",
    "FADD":       "Rd, Ra, Rb         Rd = float(Ra) + float(Rb)",
    "FSUB":       "Rd, Ra, Rb         Rd = float(Ra) - float(Rb)",
    "FMUL":       "Rd, Ra, Rb         Rd = float(Ra) * float(Rb)",
    "FDIV":       "Rd, Ra, Rb         Rd = float(Ra) / float(Rb)",
    "FCMP":       "Ra, Rb             set CMP_FLAG for floats",
    "ALLOC":      "Rd, size           Rd = heap.alloc(size)",
    "MEMINFO":    "Rd                 Rd = heap.usage()",
    "IN":         "Rd, port           Rd = bus.read_io(port)",
    "OUT":        "port, Rs           bus.write_io(port, Rs)",
}


# ── Memory Subsystem ─────────────────────────────────────────────────────────

class Memory:
    """Named tensor heap with LRU access tracking."""

    def __init__(self):
        self._heap = {}
        self._lru = []
        self._alloc_sizes = {}

    def store(self, key, value):
        self._heap[key] = value
        self._touch(key)
        self._alloc_sizes[key] = value.nbytes if isinstance(value, np.ndarray) else 0

    def load(self, key):
        if key not in self._heap:
            raise InsFault(f"heap key not found: {key}")
        self._touch(key)
        return self._heap[key]

    def free(self, key):
        self._heap.pop(key, None)
        self._alloc_sizes.pop(key, None)
        if key in self._lru:
            self._lru.remove(key)

    def contains(self, key):
        return key in self._heap

    def lru_evict(self):
        if not self._lru:
            return None
        key = self._lru.pop(0)
        self._heap.pop(key, None)
        self._alloc_sizes.pop(key, None)
        return key

    def usage(self):
        return {
            "entries": len(self._heap),
            "keys": list(self._heap.keys()),
            "bytes_tracked": sum(self._alloc_sizes.values()),
            "lru_order": list(self._lru),
        }

    def _touch(self, key):
        if key in self._lru:
            self._lru.remove(key)
        self._lru.append(key)


# ── Device Bus ───────────────────────────────────────────────────────────────

class Device:
    """Generic device interface. Subclass to wrap any library."""

    def call(self, method, *args):
        raise DeviceFault(f"device does not support: {method}")

    def info(self):
        return {"type": "base", "methods": []}


class ClockDevice(Device):
    """Self-contained wall clock — no Python time library dependency.

    Counts PIT ticks from a fixed epoch (Jan 1, 1900 00:00:00 UTC).
    The PIT calls ``tick()`` on each IRQ0; CMOS reads ``seconds_now()``
    to derive year/month/day/hour/minute/second via pure arithmetic.

    Epoch: Jan 1, 1900 00:00:00 UTC = Unix timestamp -2208988800.
    """

    # Jan 1, 1900 as a Unix timestamp (negative — before Unix epoch)
    EPOCH_1900: int = -2208988800

    # Days in each month (index 1..12, 0 unused)
    _DAYS_IN_MONTH = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    def __init__(self, freq: int = 100, epoch_unix: int | None = None):
        """
        Args:
            freq: Ticks per second (matches PIT target_hz, default 100).
            epoch_unix: Unix timestamp for the starting wall-clock time.
                        Defaults to Jan 1, 1900.
        """
        self._freq = freq
        self._ticks = 0
        self._epoch = epoch_unix if epoch_unix is not None else self.EPOCH_1900

    # ── PIT interface ─────────────────────────────────────────────────

    def tick(self):
        """Called by PITDevice on each IRQ0.  Increments the tick counter."""
        self._ticks += 1

    # ── Time queries ──────────────────────────────────────────────────

    @property
    def ticks(self) -> int:
        return self._ticks

    @property
    def freq(self) -> int:
        return self._freq

    def seconds_now(self) -> float:
        """Total seconds since epoch (wall-clock time)."""
        return self._epoch + self._ticks / self._freq

    def set_time(self, year: int, month: int, day: int,
                 hour: int = 0, minute: int = 0, second: int = 0):
        """Set the wall clock to a specific date/time (useful for tests)."""
        self._epoch = self._date_to_unix(year, month, day, hour, minute, second)
        self._ticks = 0

    # ── Date decoding (pure arithmetic) ──────────────────────────────

    def decode(self, seconds: float | None = None) -> dict:
        """Decode total seconds since epoch into year/month/day/hour/minute/second/weekday.

        Args:
            seconds: Seconds since epoch.  Defaults to ``seconds_now()``.

        Returns:
            dict with keys: year, month, day, hour, minute, second, weekday.
            weekday: 0=Monday … 6=Sunday (ISO convention).
        """
        if seconds is None:
            seconds = self.seconds_now()
        return self._decode_unix(int(seconds))

    @classmethod
    def _decode_unix(cls, ts: int) -> dict:
        """Convert a Unix timestamp to calendar components.  Pure math."""
        ts = max(ts, 0)  # clamp to >= 1970 for arithmetic safety

        # ── Days since Unix epoch ──
        days = ts // 86400
        remaining = ts % 86400
        hour = remaining // 3600
        minute = (remaining % 3600) // 60
        second = remaining % 60

        # ── Day-of-week (Jan 1 1970 = Thursday = 3) ──
        weekday = (days + 3) % 7  # 0=Mon

        # ── Year ──
        year = 1970
        while True:
            days_in_year = 366 if cls._is_leap(year) else 365
            if days < days_in_year:
                break
            days -= days_in_year
            year += 1

        # ── Month ──
        month = 1
        for m in range(1, 13):
            dim = cls._days_in_month(year, m)
            if days < dim:
                month = m
                break
            days -= dim

        day = days + 1  # days is 0-indexed within the month

        return {
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "second": second,
            "weekday": weekday,
        }

    @classmethod
    def _date_to_unix(cls, year: int, month: int, day: int,
                      hour: int = 0, minute: int = 0, second: int = 0) -> int:
        """Encode a calendar date to a Unix timestamp.  Pure math."""
        # Days from 1970-01-01 to year-01-01
        days = 0
        for y in range(1970, year):
            days += 366 if cls._is_leap(y) else 365
        # Days from month start
        for m in range(1, month):
            days += cls._days_in_month(year, m)
        days += day - 1
        return days * 86400 + hour * 3600 + minute * 60 + second

    @staticmethod
    def _is_leap(year: int) -> bool:
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    @classmethod
    def _days_in_month(cls, year: int, month: int) -> int:
        if month == 2 and cls._is_leap(year):
            return 29
        return cls._DAYS_IN_MONTH[month]


class ConsoleDevice(Device):
    """Console I/O device — port 0 for stdin, port 1 for stdout.

    Registered as "0" (stdin) and "1" (stdout) on the device bus so that
    IN R0, 0 reads a line and OUT 1, R0 prints it.
    """

    def __init__(self, port: int, stdin_fn=None, stdout_fn=None):
        self._port = port
        self._stdin_fn = stdin_fn or (lambda: "")
        self._stdout_fn = stdout_fn or (lambda v: None)
        self._buffer: list = []

    def info(self):
        return {"type": "console", "port": self._port, "status": 1}

    def write(self, value):
        self._stdout_fn(value)

    def read(self):
        return self._stdin_fn()

    def call(self, method, *args):
        if method == "read":
            return self.read()
        if method == "write":
            return self.write(*args)
        return super().call(method, *args)


class FileDevice(Device):
    """File I/O device — provides read/write access to the host filesystem.

    Commands (via DEV_CALL):
      open(path, mode) -> fd
      read(fd, size) -> bytes
      write(fd, data) -> bytes_written
      close(fd)
      listdir(path) -> list[str]
      exists(path) -> bool
    """

    def __init__(self):
        self._files: dict[int, any] = {}
        self._next_fd: int = 1

    def info(self):
        return {"type": "file", "open_files": len(self._files)}

    def call(self, method, *args):
        if method == "open":
            path, mode = args[0], args[1] if len(args) > 1 else "r"
            fh = open(path, mode)
            fd = self._next_fd
            self._next_fd += 1
            self._files[fd] = fh
            return fd
        if method == "read":
            fd, size = args[0], args[1] if len(args) > 1 else 4096
            fh = self._files.get(fd)
            if fh is None:
                raise DeviceFault(f"bad fd: {fd}")
            return fh.read(size)
        if method == "write":
            fd, data = args[0], args[1]
            fh = self._files.get(fd)
            if fh is None:
                raise DeviceFault(f"bad fd: {fd}")
            return fh.write(data)
        if method == "close":
            fd = args[0]
            fh = self._files.pop(fd, None)
            if fh:
                fh.close()
            return True
        if method == "listdir":
            import os
            return os.listdir(args[0])
        if method == "exists":
            import os
            return os.path.exists(args[0])
        return super().call(method, *args)


class IRQDevice(Device):
    """Interrupt request device — fires timer and keyboard interrupts.

    Registers IRQ handlers on a CPU. When tick() is called, fires timer IRQ.
    Keyboard input is queued and fires keyboard IRQ.
    """

    TIMER_IRQ = 0
    KEYBOARD_IRQ = 1

    def __init__(self):
        self._tick_count = 0
        self._key_queue: list = []

    def info(self):
        return {"type": "irq", "ticks": self._tick_count, "keys_pending": len(self._key_queue)}

    def tick(self, cpu):
        """Fire timer interrupt every tick."""
        self._tick_count += 1
        if self._tick_count % 10 == 0:
            cpu.fire_irq(self.TIMER_IRQ)

    def push_key(self, key):
        """Queue a keypress and fire keyboard interrupt."""
        self._key_queue.append(key)

    def read_key(self):
        """Read next key from queue (returns 0 if empty)."""
        if self._key_queue:
            return self._key_queue.pop(0)
        return 0

    def call(self, method, *args):
        if method == "tick":
            return self._tick_count
        if method == "read_key":
            return self.read_key()
        return super().call(method, *args)


class VGADevice(Device):
    """VGA text mode device — memory-mapped display at 0xB8000.

    80x25 text mode with color attributes. Each character is 2 bytes:
      byte 0 = ASCII character, byte 1 = color attribute (fg|bg<<4).

    Colors:
      0=black 1=blue 2=green 3=cyan 4=red 5=magenta 6=brown 7=lightgray
      8=darkgray 9=lightblue 10=lightgreen 11=lightcyan
      12=lightred 13=lightmagenta 14=yellow 15=white

    Commands (via DEV_CALL):
      write(row, col, char, fg, bg)  — write character with color
      write_string(row, col, text, fg, bg)  — write string
      clear(fg, bg)  — clear screen
      scroll(n)  — scroll up n lines
      set_cursor(row, col)  — move cursor
      get_cursor() -> (row, col)
      get_screen() -> list of 25 lines of 80 chars
    """

    ROWS = 25
    COLS = 80
    VGA_BUFFER_ADDR = 0xB8000

    def __init__(self):
        self._screen = [[{'char': ' ', 'fg': 7, 'bg': 0} for _ in range(self.COLS)]
                        for _ in range(self.ROWS)]
        self._cursor_row = 0
        self._cursor_col = 0
        self._default_fg = 7  # light gray
        self._default_bg = 0  # black
        self._writes = 0

    def info(self):
        return {
            "type": "vga",
            "rows": self.ROWS,
            "cols": self.COLS,
            "cursor": (self._cursor_row, self._cursor_col),
            "writes": self._writes,
        }

    def call(self, method, *args):
        if method == "write":
            row, col, char = args[0], args[1], args[2]
            fg = args[3] if len(args) > 3 else self._default_fg
            bg = args[4] if len(args) > 4 else self._default_bg
            if 0 <= row < self.ROWS and 0 <= col < self.COLS:
                self._screen[row][col] = {'char': char, 'fg': fg, 'bg': bg}
                self._writes += 1
            return True
        if method == "write_string":
            row, col, text = args[0], args[1], args[2]
            fg = args[3] if len(args) > 3 else self._default_fg
            bg = args[4] if len(args) > 4 else self._default_bg
            for i, ch in enumerate(text):
                c = col + i
                if c < self.COLS and 0 <= row < self.ROWS:
                    self._screen[row][c] = {'char': ch, 'fg': fg, 'bg': bg}
                    self._writes += 1
            return True
        if method == "clear":
            fg = args[0] if args else self._default_fg
            bg = args[1] if len(args) > 1 else self._default_bg
            self._screen = [[{'char': ' ', 'fg': fg, 'bg': bg}
                             for _ in range(self.COLS)] for _ in range(self.ROWS)]
            self._cursor_row = 0
            self._cursor_col = 0
            self._writes += self.ROWS * self.COLS
            return True
        if method == "scroll":
            n = args[0] if args else 1
            for _ in range(n):
                self._screen.pop(0)
                self._screen.append([{'char': ' ', 'fg': 7, 'bg': 0}
                                     for _ in range(self.COLS)])
            self._writes += n * self.COLS
            return True
        if method == "set_cursor":
            self._cursor_row = max(0, min(args[0], self.ROWS - 1))
            self._cursor_col = max(0, min(args[1], self.COLS - 1))
            return True
        if method == "get_cursor":
            return (self._cursor_row, self._cursor_col)
        if method == "get_screen":
            lines = []
            for row in self._screen:
                lines.append(''.join(c['char'] for c in row))
            return lines
        return super().call(method, *args)


class PS2KeyboardDevice(Device):
    """PS/2 keyboard device — port 0x60 scancode input.

    Translates scancodes to ASCII. Supports key press and release.
    Special keys: arrow keys, enter, backspace, escape, tab.

    Commands (via DEV_CALL):
      read_key() -> ascii_code  — non-blocking read (0 if empty)
      read_key_blocking() -> ascii_code  — blocking read
      push_scancode(scancode)  — inject scancode (for testing)
      has_key() -> bool
      clear()  — flush input buffer
    """

    # PS/2 Set 1 scancodes → ASCII
    SCANCODE_TO_ASCII = {
        0x00: 0, 0x1E: ord('1'), 0x1F: ord('2'), 0x20: ord('3'),
        0x21: ord('4'), 0x22: ord('5'), 0x23: ord('6'), 0x24: ord('7'),
        0x25: ord('8'), 0x26: ord('9'), 0x27: ord('0'),
        0x10: ord('q'), 0x11: ord('w'), 0x12: ord('e'), 0x13: ord('r'),
        0x14: ord('t'), 0x15: ord('y'), 0x16: ord('u'), 0x17: ord('i'),
        0x18: ord('o'), 0x19: ord('p'),
        0x1E: ord('a'), 0x1F: ord('s'), 0x20: ord('d'), 0x21: ord('f'),
        0x22: ord('g'), 0x23: ord('h'), 0x24: ord('j'), 0x25: ord('k'),
        0x26: ord('l'),
        0x2C: ord('z'), 0x2D: ord('x'), 0x2E: ord('c'), 0x2F: ord('v'),
        0x30: ord('b'), 0x31: ord('n'), 0x32: ord('m'),
        0x39: ord(' '), 0x1C: 10, 0x0E: 8, 0x01: 27, 0x0F: 9,
        0x4B: 0x100, 0x4D: 0x101, 0x48: 0x102, 0x50: 0x103,  # arrows
    }

    def __init__(self):
        self._buffer: list[int] = []
        self._read_count = 0

    def info(self):
        return {
            "type": "ps2_keyboard",
            "buffered": len(self._buffer),
            "reads": self._read_count,
        }

    def call(self, method, *args):
        if method == "read_key":
            return self.read_key()
        if method == "push_scancode":
            scancode = args[0]
            if scancode < 0x80:  # key press only (not release)
                ascii_val = self.SCANCODE_TO_ASCII.get(scancode, 0)
                self._buffer.append(ascii_val)
            return True
        if method == "has_key":
            return len(self._buffer) > 0
        if method == "clear":
            self._buffer.clear()
            return True
        return super().call(method, *args)

    def read_key(self):
        if self._buffer:
            self._read_count += 1
            return self._buffer.pop(0)
        return 0


class BlockDevice(Device):
    """Sector-based block storage — 512-byte sectors.

    Provides read_sector/write_sector for raw I/O and read_block/write_block
    for higher-level access. Tracks I/O statistics.
    """

    SECTOR_SIZE = 512

    def __init__(self, num_sectors: int = 256):
        self._sectors = [bytearray(self.SECTOR_SIZE) for _ in range(num_sectors)]
        self._num_sectors = num_sectors
        self._reads = 0
        self._writes = 0

    def info(self):
        return {
            "type": "block",
            "sectors": self._num_sectors,
            "sector_size": self.SECTOR_SIZE,
            "reads": self._reads,
            "writes": self._writes,
        }

    def read_sector(self, sector_idx: int) -> bytearray:
        if not (0 <= sector_idx < self._num_sectors):
            raise DeviceFault(f"sector out of range: {sector_idx}")
        self._reads += 1
        return self._sectors[sector_idx]

    def write_sector(self, sector_idx: int, data: bytes) -> None:
        if not (0 <= sector_idx < self._num_sectors):
            raise DeviceFault(f"sector out of range: {sector_idx}")
        self._writes += 1
        if isinstance(data, str):
            data = data.encode('utf-8')
        self._sectors[sector_idx][:len(data)] = data[:self.SECTOR_SIZE]

    def read_block(self, sector_idx: int, size: int) -> bytes:
        data = self.read_sector(sector_idx)
        return bytes(data[:size])

    def write_block(self, sector_idx: int, data: bytes) -> None:
        self.write_sector(sector_idx, data)

    def call(self, method, *args):
        if method == "read_sector":
            return self.read_sector(*args)
        if method == "write_sector":
            return self.write_sector(*args)
        if method == "read_block":
            return self.read_block(*args)
        if method == "write_block":
            return self.write_block(*args)
        return super().call(method, *args)


class SerialDevice(Device):
    """UART COM1 serial communication device — ports 0x3F8–0x3FF.

    Provides byte-level serial I/O with TX/RX ring buffers and LSR
    (Line Status Register) for polling.

    I/O Ports:
        0x3F8 - RX/TX data register (read/write)
        0x3FD - LSR (Line Status Register)
            bit 0: RX data ready (byte available)
            bit 5: TX holding register empty (ready to send)

    Commands (via DEV_CALL):
      write_byte(byte)   — send a byte to the serial port
      read_byte() -> int — receive a byte (or -1 if empty)
      has_data() -> bool — check if RX buffer has data
      flush()            — clear both buffers
    """

    BASE_PORT = 0x3F8
    LSR_PORT = 0x3FD

    def __init__(self, cpu: X86CPU | None = None):
        self._tx_buffer: bytearray = bytearray()
        self._rx_buffer: bytearray = bytearray()
        self._tx_count = 0
        self._rx_count = 0
        self._cpu = cpu
        if cpu:
            cpu.register_io_in(self.BASE_PORT, self._read_data)
            cpu.register_io_out(self.BASE_PORT, self._write_data)
            cpu.register_io_in(self.LSR_PORT, self._read_lsr)

    def _read_data(self) -> int:
        if self._rx_buffer:
            self._rx_count += 1
            return self._rx_buffer.pop(0)
        return 0

    def _write_data(self, val: int):
        self._tx_buffer.append(val & 0xFF)
        self._tx_count += 1

    def _read_lsr(self) -> int:
        lsr = 0
        if self._rx_buffer:
            lsr |= 0x01  # bit 0: RX ready
        lsr |= 0x20      # bit 5: TX empty (always ready in simulation)
        return lsr

    def info(self):
        return {
            "type": "serial",
            "base_port": self.BASE_PORT,
            "tx_count": self._tx_count,
            "rx_count": self._rx_count,
            "rx_pending": len(self._rx_buffer),
        }

    def write_byte(self, b: int):
        self._tx_buffer.append(b & 0xFF)
        self._tx_count += 1

    def read_byte(self) -> int:
        if self._rx_buffer:
            self._rx_count += 1
            return self._rx_buffer.pop(0)
        return -1

    def push_byte(self, b: int):
        self._rx_buffer.append(b & 0xFF)

    def has_data(self) -> bool:
        return len(self._rx_buffer) > 0

    def flush(self):
        self._tx_buffer.clear()
        self._rx_buffer.clear()

    def call(self, method, *args):
        if method == "write_byte":
            self.write_byte(args[0])
            return True
        if method == "read_byte":
            return self.read_byte()
        if method == "push_byte":
            self.push_byte(args[0])
            return True
        if method == "has_data":
            return self.has_data()
        if method == "flush":
            self.flush()
            return True
        return super().call(method, *args)


class MouseDevice(Device):
    """PS/2 mouse device — relative X/Y movement and button state.

    Generates standard 3-byte PS/2 mouse packets:
        byte 0: [overflow_y, overflow_x, sign_y, sign_x, 1, middle, right, left]
        byte 1: X movement (signed)
        byte 2: Y movement (signed)

    Commands (via DEV_CALL):
      move(dx, dy)           — inject relative movement
      press(button)          — press button (1=left, 2=right, 4=middle)
      release(button)        — release button
      read_packet() -> bytes — read 3-byte packet (or empty bytes)
      get_state() -> dict    — current position and buttons
      reset()                — clear all state
    """

    def __init__(self):
        self._x: int = 0
        self._y: int = 0
        self._buttons: int = 0  # bit 0=left, bit 1=right, bit 2=middle
        self._packets: list[bytes] = []
        self._total_packets = 0

    def info(self):
        return {
            "type": "ps2_mouse",
            "x": self._x,
            "y": self._y,
            "buttons": self._buttons,
            "packets_pending": len(self._packets),
        }

    def move(self, dx: int, dy: int):
        self._x += dx
        self._y += dy
        packet = self._make_packet(dx, dy)
        self._packets.append(packet)
        self._total_packets += 1

    def press(self, button: int):
        self._buttons |= button

    def release(self, button: int):
        self._buttons &= ~button

    def read_packet(self) -> bytes:
        if self._packets:
            return self._packets.pop(0)
        return b''

    def get_state(self) -> dict:
        return {
            "x": self._x,
            "y": self._y,
            "buttons": self._buttons,
            "left": bool(self._buttons & 1),
            "right": bool(self._buttons & 2),
            "middle": bool(self._buttons & 4),
        }

    def reset(self):
        self._x = 0
        self._y = 0
        self._buttons = 0
        self._packets.clear()

    def _make_packet(self, dx: int, dy: int) -> bytes:
        byte0 = 0x08  # bit 3 always set (sync bit)
        byte0 |= self._buttons & 0x07
        if dx < 0:
            byte0 |= 0x10  # sign_x
            dx = dx & 0xFF
        if dy < 0:
            byte0 |= 0x20  # sign_y
            dy = dy & 0xFF
        return bytes([byte0, dx & 0xFF, dy & 0xFF])

    def call(self, method, *args):
        if method == "move":
            self.move(args[0], args[1])
            return True
        if method == "press":
            self.press(args[0])
            return True
        if method == "release":
            self.release(args[0])
            return True
        if method == "read_packet":
            return self.read_packet()
        if method == "get_state":
            return self.get_state()
        if method == "reset":
            self.reset()
            return True
        return super().call(method, *args)


class CMOSDevice(Device):
    """Intel MC146818 Real-Time Clock + CMOS RAM — ports 0x70/0x71.

    128 bytes of battery-backed CMOS RAM.  RTC registers occupy offsets
    0x00–0x0D.  The remaining bytes are general-purpose CMOS storage
    (BIOS setup, equipment list, boot order, etc.).

    I/O Ports:
        0x70 — Address register (write: select CMOS offset, bit 7 = NMI disable)
        0x71 — Data register    (read/write: access selected CMOS byte)

    RTC Register Layout (offsets in CMOS RAM):
        0x00 — Seconds        (00–59)
        0x01 — Alarm Seconds
        0x02 — Minutes        (00–59)
        0x03 — Alarm Minutes
        0x04 — Hours          (00–23 in 24h mode)
        0x05 — Alarm Hours
        0x06 — Day of Week    (1=Sunday … 7=Saturday)
        0x07 — Day of Month   (01–31)
        0x08 — Month          (01–12)
        0x09 — Year           (last 2 digits, 00–99)
        0x0A — Status Register A
            bit 7: UIP  (Update In Progress — set while RTC updates)
            bit 6-4: DV2-DV0 (divider select — default 010 = 1.024 kHz)
            bit 3-0: RS3-RS0 (rate select — default 0110 = 1024 Hz)
        0x0B — Status Register B
            bit 7: SET  (0 = update normally, 1 = inhibit updates)
            bit 6: PIE  (Periodic Interrupt Enable)
            bit 5: AIE  (Alarm Interrupt Enable)
            bit 4: UIE  (Update-Ended Interrupt Enable)
            bit 3: SQWE (Square Wave Enable)
            bit 2: DM   (Data Mode: 0 = BCD, 1 = Binary)
            bit 1: 24/12 (0 = 12h, 1 = 24h)
            bit 0: DSE  (Daylight Saving Enable — ignored)
        0x0C — Status Register C (read-only, cleared on read)
            bit 7: IRQF (Interrupt Request Flag)
            bit 6: PF   (Periodic Interrupt Flag)
            bit 5: AF   (Alarm Interrupt Flag)
            bit 4: UF   (Update-Ended Interrupt Flag)
        0x0D — Status Register D
            bit 7: VRT  (Valid RAM and Time — 1 = battery OK)

    CMOS Checksum (BIOS standard):
        Offset 0x2E-0x2F: 16-bit checksum of bytes 0x10-0x2D
        Offset 0x30-0x31: 16-bit extended checksum of bytes 0x30-0x3F

    Commands (via DEV_CALL):
      get_time() -> dict      — decoded RTC time
      get_unix_time() -> int  — Unix timestamp
      read_cmos(offset) -> int — read raw CMOS byte
      write_cmos(offset, val)  — write raw CMOS byte
      set_offset(seconds)     — adjust time offset (for testing)
      set_binary_mode(flag)   — True=binary, False=BCD (default)
    """

    PORT_ADDR = 0x70
    PORT_DATA = 0x71
    CMOS_SIZE = 128

    # RTC register offsets
    REG_SECONDS = 0x00
    REG_ALARM_SEC = 0x01
    REG_MINUTES = 0x02
    REG_ALARM_MIN = 0x03
    REG_HOURS = 0x04
    REG_ALARM_HRS = 0x05
    REG_WEEKDAY = 0x06
    REG_DAY = 0x07
    REG_MONTH = 0x08
    REG_YEAR = 0x09
    REG_STATUS_A = 0x0A
    REG_STATUS_B = 0x0B
    REG_STATUS_C = 0x0C
    REG_STATUS_D = 0x0D

    # BIOS standard: equipment list, base memory, extended memory etc.
    REG_EQUIPMENT = 0x14     # CMOS equipment word
    REG_BASE_MEM_LO = 0x15   # base memory (KB) low byte
    REG_BASE_MEM_HI = 0x16   # base memory (KB) high byte
    REG_EXT_MEM_LO = 0x17    # extended memory (KB) low byte
    REG_EXT_MEM_HI = 0x18    # extended memory (KB) high byte
    REG_BOOT_ORDER = 0x3D    # boot device order (4 bits each)

    def __init__(self, cpu: X86CPU | None = None,
                 clock: ClockDevice | None = None):
        self._cmos = bytearray(self.CMOS_SIZE)
        self._selected = 0       # current register offset (written to port 0x70)
        self._nmi_disabled = False
        self._clock = clock      # self-contained wall clock (no time.time)

        # Initialize Status Register D (VRT = battery OK)
        self._cmos[self.REG_STATUS_D] = 0x80

        # Initialize Status Register A (divider = 1.024 kHz, rate = 1024 Hz)
        # Default: DV=010 (bits 6-4), RS=0110 (bits 3-0)
        self._cmos[self.REG_STATUS_A] = 0x26

        # Initialize Status Register B: 24h mode, BCD
        # bit 1 = 1 (24h), bit 2 = 0 (BCD)
        self._cmos[self.REG_STATUS_B] = 0x02

        if cpu:
            cpu.register_io_in(self.PORT_DATA, self._read_data)
            cpu.register_io_out(self.PORT_ADDR, self._write_addr)
            cpu.register_io_out(self.PORT_DATA, self._write_data)

        # Register I/O ports for CMOS address port too
        # (reads from 0x70 return last written value per some implementations)
        if cpu:
            cpu.register_io_in(self.PORT_ADDR, lambda: self._selected | (0x80 if self._nmi_disabled else 0))

    def _write_addr(self, val: int):
        """Port 0x70 write: select CMOS register.  Bit 7 = NMI disable."""
        self._nmi_disabled = bool(val & 0x80)
        self._selected = val & 0x7F

    def _read_data(self) -> int:
        """Port 0x71 read: return selected CMOS byte.  Refreshes time registers."""
        reg = self._selected
        # Auto-refresh RTC registers from host clock on every read
        self._refresh_rtc()
        # Status register C is read-to-clear
        if reg == self.REG_STATUS_C:
            val = self._cmos[reg]
            self._cmos[reg] = 0x00
            return val
        return self._cmos[reg]

    def _write_data(self, val: int):
        """Port 0x71 write: store byte into selected CMOS register."""
        reg = self._selected
        if reg == self.REG_STATUS_A or reg == self.REG_STATUS_C:
            return  # Status A and C are read-only or write-ignored
        if reg == self.REG_STATUS_D:
            # Only bit 7 is writeable (VRT)
            self._cmos[reg] = (val & 0x80) | (self._cmos[reg] & 0x7F)
            return
        self._cmos[reg] = val & 0xFF

    def _refresh_rtc(self):
        """Update RTC register bytes from host system clock."""
        import time as _time
        t = _time.time() + self._offset
        parts = _time.gmtime(t)
        reg_b = self._cmos[self.REG_STATUS_B]
        binary = bool(reg_b & 0x04)    # DM bit: 0=BCD, 1=binary
        h24 = bool(reg_b & 0x02)       # 24/12 bit: 0=12h, 1=24h

        if binary:
            self._cmos[self.REG_SECONDS] = parts.tm_sec & 0xFF
            self._cmos[self.REG_MINUTES] = parts.tm_min & 0xFF
            self._cmos[self.REG_DAY] = parts.tm_mday & 0xFF
            self._cmos[self.REG_MONTH] = parts.tm_mon & 0xFF
            self._cmos[self.REG_YEAR] = parts.tm_year % 100
            self._cmos[self.REG_WEEKDAY] = (parts.tm_wday + 1) & 0xFF  # 1=Sun
            if h24:
                self._cmos[self.REG_HOURS] = parts.tm_hour & 0xFF
            else:
                h12 = parts.tm_hour % 12
                if h12 == 0:
                    h12 = 12
                self._cmos[self.REG_HOURS] = h12 | (0x80 if parts.tm_hour >= 12 else 0)
        else:
            self._cmos[self.REG_SECONDS] = self._to_bcd(parts.tm_sec)
            self._cmos[self.REG_MINUTES] = self._to_bcd(parts.tm_min)
            self._cmos[self.REG_DAY] = self._to_bcd(parts.tm_mday)
            self._cmos[self.REG_MONTH] = self._to_bcd(parts.tm_mon)
            self._cmos[self.REG_YEAR] = self._to_bcd(parts.tm_year % 100)
            self._cmos[self.REG_WEEKDAY] = (parts.tm_wday + 1) & 0xFF
            if h24:
                self._cmos[self.REG_HOURS] = self._to_bcd(parts.tm_hour)
            else:
                h12 = parts.tm_hour % 12
                if h12 == 0:
                    h12 = 12
                self._cmos[self.REG_HOURS] = self._to_bcd(h12) | (0x80 if parts.tm_hour >= 12 else 0)

    def _to_bcd(self, val: int) -> int:
        """Convert an integer (0–99) to BCD."""
        return ((val // 10) << 4) | (val % 10)

    def _from_bcd(self, val: int) -> int:
        """Convert a BCD byte to integer."""
        return ((val >> 4) * 10) + (val & 0x0F)

    # ── High-level API ───────────────────────────────────────────────────

    def get_time(self) -> dict:
        """Return decoded RTC time as a dict."""
        self._refresh_rtc()
        reg_b = self._cmos[self.REG_STATUS_B]
        binary = bool(reg_b & 0x04)
        h24 = bool(reg_b & 0x02)

        if binary:
            sec = self._cmos[self.REG_SECONDS]
            minute = self._cmos[self.REG_MINUTES]
            hour = self._cmos[self.REG_HOURS]
        else:
            sec = self._from_bcd(self._cmos[self.REG_SECONDS])
            minute = self._from_bcd(self._cmos[self.REG_MINUTES])
            hour = self._from_bcd(self._cmos[self.REG_HOURS] & 0x7F)

        if not h24:
            is_pm = bool(self._cmos[self.REG_HOURS] & 0x80)
            if is_pm and hour != 12:
                hour += 12
            elif not is_pm and hour == 12:
                hour = 0

        return {
            "year": 2000 + self._cmos[self.REG_YEAR] if self._cmos[self.REG_YEAR] < 100 else self._cmos[self.REG_YEAR],
            "month": self._cmos[self.REG_MONTH] if binary else self._from_bcd(self._cmos[self.REG_MONTH]),
            "day": self._cmos[self.REG_DAY] if binary else self._from_bcd(self._cmos[self.REG_DAY]),
            "hour": hour,
            "minute": minute,
            "second": sec,
            "weekday": self._cmos[self.REG_WEEKDAY],
        }

    def get_unix_time(self) -> int:
        """Return Unix timestamp from host clock."""
        import time as _time
        return int(_time.time() + self._offset)

    def read_cmos(self, offset: int) -> int:
        """Read a raw CMOS byte (bypasses I/O port interface)."""
        if 0 <= offset < self.CMOS_SIZE:
            self._refresh_rtc()
            return self._cmos[offset]
        return 0

    def write_cmos(self, offset: int, val: int):
        """Write a raw CMOS byte (bypasses I/O port interface)."""
        if 0 <= offset < self.CMOS_SIZE:
            if offset == self.REG_STATUS_A or offset == self.REG_STATUS_C:
                return  # read-only
            if offset == self.REG_STATUS_D:
                self._cmos[offset] = (val & 0x80) | (self._cmos[offset] & 0x7F)
                return
            self._cmos[offset] = val & 0xFF

    def set_offset(self, seconds: float):
        self._offset = seconds

    def set_binary_mode(self, binary: bool):
        """Switch between BCD (False, default) and binary (True) mode."""
        if binary:
            self._cmos[self.REG_STATUS_B] |= 0x04
        else:
            self._cmos[self.REG_STATUS_B] &= ~0x04

    def info(self):
        return {
            "type": "cmos",
            "nmi_disabled": self._nmi_disabled,
            "selected": self._selected,
            "unix_time": self.get_unix_time(),
        }

    def call(self, method, *args):
        if method == "get_time":
            return self.get_time()
        if method == "get_unix_time":
            return self.get_unix_time()
        if method == "read_cmos":
            return self.read_cmos(args[0])
        if method == "write_cmos":
            self.write_cmos(args[0], args[1])
            return True
        if method == "set_offset":
            self.set_offset(args[0])
            return True
        if method == "set_binary_mode":
            self.set_binary_mode(args[0])
            return True
        return super().call(method, *args)


class DiskDevice(Device):
    """ATA/IDE disk controller — provides structured disk I/O on a BlockDevice.

    Supports CHS (Cylinder/Head/Sector) addressing and basic ATA commands.

    Commands (via DEV_CALL):
      read_sectors(lba, count) -> bytes
      write_sectors(lba, data)
      get_geometry() -> dict   — {cylinders, heads, sectors_per_track, total_sectors}
      status() -> int          — status register (0x40=ready, 0x00=busy)
    """

    SECTOR_SIZE = 512

    def __init__(self, block_device: BlockDevice | None = None):
        self._block = block_device or BlockDevice(num_sectors=2048)
        self._total_sectors = self._block._num_sectors
        # CHS geometry (standard LBA-to-CHS mapping)
        self._heads = 16
        self._sectors_per_track = 63
        self._cylinders = self._total_sectors // (self._heads * self._sectors_per_track)
        self._reads = 0
        self._writes = 0
        self._status = 0x40  # ready

    def info(self):
        return {
            "type": "disk",
            "total_sectors": self._total_sectors,
            "reads": self._reads,
            "writes": self._writes,
            "status": self._status,
        }

    def read_sectors(self, lba: int, count: int) -> bytes:
        data = bytearray()
        for i in range(count):
            sector_data = self._block.read_sector(lba + i)
            data.extend(sector_data)
            self._reads += 1
        return bytes(data)

    def write_sectors(self, lba: int, data: bytes):
        for i in range(count := (len(data) + self.SECTOR_SIZE - 1) // self.SECTOR_SIZE):
            chunk = data[i * self.SECTOR_SIZE:(i + 1) * self.SECTOR_SIZE]
            self._block.write_sector(lba + i, chunk)
            self._writes += 1

    def get_geometry(self) -> dict:
        return {
            "cylinders": self._cylinders,
            "heads": self._heads,
            "sectors_per_track": self._sectors_per_track,
            "total_sectors": self._total_sectors,
            "sector_size": self.SECTOR_SIZE,
        }

    def status(self) -> int:
        return self._status

    def call(self, method, *args):
        if method == "read_sectors":
            return self.read_sectors(*args)
        if method == "write_sectors":
            return self.write_sectors(*args)
        if method == "get_geometry":
            return self.get_geometry()
        if method == "status":
            return self.status()
        return super().call(method, *args)


class NICDevice(Device):
    """Virtual network interface card — TX/RX packet buffers.

    Provides a simple send/receive packet interface. Packets are
    byte arrays (simulated Ethernet frames).

    Commands (via DEV_CALL):
      send_packet(data)      — transmit a packet
      recv_packet() -> bytes — receive a packet (or empty bytes)
      has_packet() -> bool   — check if RX buffer has packets
      get_stats() -> dict    — {tx_packets, rx_packets, tx_bytes, rx_bytes}
      flush()                — clear all buffers
    """

    MTU = 1500  # Maximum Transmission Unit

    def __init__(self):
        self._tx_buffer: list[bytes] = []
        self._rx_buffer: list[bytes] = []
        self._tx_packets = 0
        self._rx_packets = 0
        self._tx_bytes = 0
        self._rx_bytes = 0

    def info(self):
        return {
            "type": "nic",
            "mtu": self.MTU,
            "tx_packets": self._tx_packets,
            "rx_packets": self._rx_packets,
            "tx_pending": len(self._tx_buffer),
            "rx_pending": len(self._rx_buffer),
        }

    def send_packet(self, data: bytes) -> bool:
        if len(data) > self.MTU:
            return False
        self._tx_buffer.append(data)
        self._tx_packets += 1
        self._tx_bytes += len(data)
        return True

    def recv_packet(self) -> bytes:
        if self._rx_buffer:
            pkt = self._rx_buffer.pop(0)
            return pkt
        return b''

    def inject_packet(self, data: bytes):
        self._rx_buffer.append(data)
        self._rx_packets += 1
        self._rx_bytes += len(data)

    def has_packet(self) -> bool:
        return len(self._rx_buffer) > 0

    def get_stats(self) -> dict:
        return {
            "tx_packets": self._tx_packets,
            "rx_packets": self._rx_packets,
            "tx_bytes": self._tx_bytes,
            "rx_bytes": self._rx_bytes,
        }

    def flush(self):
        self._tx_buffer.clear()
        self._rx_buffer.clear()

    def call(self, method, *args):
        if method == "send_packet":
            return self.send_packet(args[0])
        if method == "recv_packet":
            return self.recv_packet()
        if method == "inject_packet":
            self.inject_packet(args[0])
            return True
        if method == "has_packet":
            return self.has_packet()
        if method == "get_stats":
            return self.get_stats()
        if method == "flush":
            self.flush()
            return True
        return super().call(method, *args)


class FlatFS:
    """Simple flat filesystem on top of a BlockDevice.

    File table stored in sector 0 (4 bytes: num_files, then per-file entry):
      [2 bytes: name_len] [name bytes] [2 bytes: start_sector] [2 bytes: num_sectors]

    File data starts at sector 1. Max 32 files, max 32 chars per name.
    """

    MAX_FILES = 32
    MAX_NAME = 32
    TABLE_SECTOR = 0
    DATA_START = 1

    def __init__(self, block_device: BlockDevice):
        self._block = block_device
        self._files: dict[str, tuple[int, int]] = {}  # name -> (start_sector, num_sectors)
        self._load_table()

    def _load_table(self):
        raw = bytes(self._block.read_sector(self.TABLE_SECTOR))
        if raw[:2] == b'\x00\x00':
            return
        n = int.from_bytes(raw[:2], 'big')
        pos = 2
        for _ in range(n):
            name_len = int.from_bytes(raw[pos:pos+2], 'big')
            pos += 2
            name = raw[pos:pos+name_len].decode('utf-8', errors='replace')
            pos += name_len
            start = int.from_bytes(raw[pos:pos+2], 'big')
            pos += 2
            count = int.from_bytes(raw[pos:pos+2], 'big')
            pos += 2
            self._files[name] = (start, count)

    def _save_table(self):
        data = len(self._files).to_bytes(2, 'big')
        for name, (start, count) in self._files.items():
            name_bytes = name.encode('utf-8')[:self.MAX_NAME]
            data += len(name_bytes).to_bytes(2, 'big')
            data += name_bytes
            data += start.to_bytes(2, 'big')
            data += count.to_bytes(2, 'big')
        # Pad to sector size
        data = data.ljust(self._block.SECTOR_SIZE, b'\x00')
        self._block.write_sector(self.TABLE_SECTOR, data)

    def list_files(self) -> list[str]:
        return list(self._files.keys())

    def exists(self, name: str) -> bool:
        return name in self._files

    def write(self, name: str, data: bytes) -> None:
        """Write data to a file, allocating sectors as needed."""
        sectors_needed = (len(data) + self._block.SECTOR_SIZE - 1) // self._block.SECTOR_SIZE

        # Find free sectors (simple: use sectors after all existing files)
        used = set()
        for _, (_, count) in self._files.items():
            for s in range(self.DATA_START, self.DATA_START + count):
                used.add(s)

        free_sectors = []
        s = self.DATA_START
        while len(free_sectors) < sectors_needed:
            if s not in used:
                free_sectors.append(s)
            s += 1
            if s >= self._block._num_sectors:
                raise DeviceFault("no space on disk")

        # Write data sectors
        for i, sector_idx in enumerate(free_sectors):
            chunk = data[i * self._block.SECTOR_SIZE:(i + 1) * self._block.SECTOR_SIZE]
            self._block.write_sector(sector_idx, chunk)

        self._files[name] = (free_sectors[0], sectors_needed)
        self._save_table()

    def read(self, name: str) -> bytes:
        """Read entire file contents."""
        if name not in self._files:
            raise DeviceFault(f"file not found: {name}")
        start, count = self._files[name]
        data = b''
        for i in range(count):
            data += bytes(self._block.read_sector(start + i))
        return data

    def delete(self, name: str) -> bool:
        """Delete a file and free its sectors."""
        if name not in self._files:
            return False
        del self._files[name]
        self._save_table()
        return True

    def size(self, name: str) -> int:
        if name not in self._files:
            return 0
        _, count = self._files[name]
        return count * self._block.SECTOR_SIZE


class DeviceBus:
    """Device registry and generic dispatch."""

    def __init__(self):
        self._devices = {}

    def register(self, name, device):
        self._devices[name] = device

    def register_console(self, stdin_fn=None, stdout_fn=None):
        """Register console I/O on ports 0 (stdin) and 1 (stdout)."""
        self._devices["0"] = ConsoleDevice(0, stdin_fn=stdin_fn, stdout_fn=stdout_fn)
        self._devices["1"] = ConsoleDevice(1, stdin_fn=stdin_fn, stdout_fn=stdout_fn)

    def open(self, name):
        if name not in self._devices:
            raise DeviceFault(f"no such device: {name}")
        return self._devices[name]

    def call(self, device, method, *args):
        return device.call(method, *args)

    def info(self, device):
        return device.info()

    def list_devices(self):
        return list(self._devices.keys())


# ── CPU ──────────────────────────────────────────────────────────────────────

class CPU:
    """Central processing unit with integer ALU, tensor ALU, and control flow."""

    def __init__(self, memory=None, devices=None):
        self.regs = [0] * NUM_REGS
        self.pc = 0
        self.sp = STACK_BASE
        self._cmp_flag = 0
        self._carry_flag = False
        self._call_stack = []
        self._stack = {}
        self._memory = memory or Memory()
        self._devices = devices or DeviceBus()
        self._instructions = []
        self._running = False
        self._step_count = 0
        self._max_instructions = MAX_INSTRUCTIONS
        self._output = []
        self._tracing = False
        self._trace = []
        self._irq_handlers: dict[int, callable] = {}
        self._irq_pending: list[int] = []

    def load_program(self, instructions):
        self._instructions = list(instructions)
        self.pc = 0
        self._cmp_flag = 0
        self._call_stack.clear()
        self._running = False
        self._step_count = 0
        self._trace.clear()

    def register_irq(self, irq_num: int, handler: callable) -> None:
        """Register an interrupt handler for IRQ number."""
        self._irq_handlers[irq_num] = handler

    def fire_irq(self, irq_num: int) -> None:
        """Queue an interrupt to be processed."""
        self._irq_pending.append(irq_num)

    def _process_irqs(self):
        """Process pending interrupts."""
        while self._irq_pending:
            irq = self._irq_pending.pop(0)
            handler = self._irq_handlers.get(irq)
            if handler:
                handler(self)

    def step(self):
        if self.pc >= len(self._instructions):
            if self._instructions:
                self._output.append(f"[VM] PC out of bounds: {self.pc}")
                self._running = False
            return False
        inst = self._instructions[self.pc]
        self._step_count += 1
        if self._tracing:
            self._record_trace(inst)
        old_pc = self.pc
        self._pc_changed = False
        try:
            self._dispatch(inst)
        except Halt:
            self._running = False
            return False
        if not self._pc_changed and self.pc == old_pc:
            self.pc += 1
        return True

    def run(self, max_steps=None):
        if max_steps is not None:
            self._max_instructions = max_steps
        self._running = True
        while self._running and self._step_count < self._max_instructions:
            self._process_irqs()
            if not self.step():
                break
        if self._step_count >= self._max_instructions:
            self._output.append(f"[VM] instruction limit ({self._max_instructions})")
        return self._output

    def _dispatch(self, inst):
        handler = _OPCODE_TABLE.get(inst.opcode)
        if handler is None:
            self._output.append(f"[VM] unknown opcode: {inst.opcode}")
            return
        handler(self, inst.operands)

    def _reg(self, operand):
        if isinstance(operand, str) and operand.startswith("R") and operand[1:].isdigit():
            idx = int(operand[1:])
            if 0 <= idx < NUM_REGS:
                return idx
        raise InsFault(f"invalid register: {operand}")

    def _val(self, operand):
        if isinstance(operand, str) and operand.startswith("R") and operand[1:].isdigit():
            return self.regs[int(operand[1:])]
        return operand

    def _check_arity(self, ops, expected):
        if len(ops) < expected:
            raise InsFault(f"expected {expected} operands, got {len(ops)}")

    def _truthy(self, val):
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return val != 0
        if isinstance(val, np.ndarray):
            return val.size > 0 and bool(np.any(val))
        return bool(val)

    def _parse_tensor(self, val):
        if isinstance(val, np.ndarray):
            return val
        if isinstance(val, list):
            return np.array(val, dtype=np.float64)
        if isinstance(val, (int, float)):
            return np.float64(val)
        raise InsFault(f"cannot parse as tensor: {type(val).__name__}")

    def _record_trace(self, inst):
        regs_snapshot = {}
        for i, v in enumerate(self.regs):
            if v != 0:
                if isinstance(v, np.ndarray):
                    regs_snapshot[f"R{i}"] = np.array2string(v, precision=3, suppress_small=True)
                else:
                    regs_snapshot[f"R{i}"] = v
        self._trace.append(TraceEntry(
            cycle=self._step_count,
            pc=self.pc,
            instruction=f"{inst.opcode} {', '.join(str(o) for o in inst.operands)}",
            registers=regs_snapshot,
            heap_keys=list(self._memory._heap.keys()),
        ))

    def get_trace(self):
        return list(self._trace)

    def format_trace(self):
        lines = []
        for e in self._trace:
            regs = " ".join(f"{k}={v}" for k, v in e.registers.items())
            lines.append(f"[{e.cycle:05d}] PC={e.pc:04d}  {e.instruction:<36s}  {regs}")
        return lines


# ── Assembler ────────────────────────────────────────────────────────────────

class Assembler:
    """Assembly text to Instruction list."""

    _RE_REGISTER = re.compile(r"^R(\d+)$")
    _RE_INT = re.compile(r"^-?\d+$")
    _RE_FLOAT = re.compile(r"^-?\d+\.\d*$")
    _RE_STRING = re.compile(r'^"(.*)"$')
    _RE_TENSOR = re.compile(r"^\[(.+)\]$")

    @staticmethod
    def _is_int(s):
        try:
            int(s, 0)
            return True
        except ValueError:
            return False

    def assemble(self, source):
        lines = source.strip().split("\n")

        # Pass 1: collect labels
        labels = {}
        raw_lines = []
        for line in lines:
            line = line.split(";")[0].split("#")[0].strip()
            if not line:
                continue
            if ":" in line and not line.startswith("["):
                prefix, _, rest = line.partition(":")
                prefix = prefix.strip()
                if prefix and " " not in prefix and "," not in prefix:
                    labels[prefix] = len(raw_lines)
                    line = rest.strip()
                    if not line:
                        continue
            elif line.endswith(":") and " " not in line and "," not in line:
                labels[line[:-1]] = len(raw_lines)
                continue
            raw_lines.append(line)

        # Pass 2: parse instructions with resolved labels
        instructions = []
        for line_num, line in enumerate(raw_lines):
            parts = line.split(None, 1)
            opcode = parts[0].upper()
            operand_str = parts[1].strip() if len(parts) > 1 else ""

            operands = self._parse_operands(operand_str, labels)
            instructions.append(Instruction(
                opcode=opcode, operands=operands,
                line_num=line_num, raw=line,
            ))

        return instructions

    def _parse_operands(self, text, labels):
        if not text:
            return []
        operands = []
        for part in self._split_operands(text):
            part = part.strip()
            if not part:
                continue
            if part in labels:
                operands.append(labels[part])
            elif self._RE_REGISTER.match(part):
                operands.append(part)
            elif self._is_int(part):
                operands.append(int(part, 0))
            elif self._RE_FLOAT.match(part):
                operands.append(float(part))
            elif self._RE_STRING.match(part):
                operands.append(self._RE_STRING.match(part).group(1))
            elif self._RE_TENSOR.match(part):
                operands.append(part)
            else:
                operands.append(part)
        return operands

    def _split_operands(self, text):
        result = []
        current = []
        in_string = False
        in_tensor = 0
        for ch in text:
            if ch == '"':
                in_string = not in_string
                current.append(ch)
            elif ch == '[' and not in_string:
                in_tensor += 1
                current.append(ch)
            elif ch == ']' and not in_string:
                in_tensor -= 1
                current.append(ch)
            elif ch == ',' and not in_string and in_tensor == 0:
                result.append("".join(current))
                current = []
            else:
                current.append(ch)
        if current:
            result.append("".join(current))
        return result


ProgramLoader = Assembler  # backward-compatible alias


# ── VM Runner ────────────────────────────────────────────────────────────────

class VMRunner:
    """Convenience: assemble + run + trace."""

    def __init__(self, devices=None):
        self._assembler = Assembler()
        self._devices = devices or DeviceBus()
        self.cpu = None

    def assemble_and_run(self, source, trace=False, max_steps=None):
        instructions = self._assembler.assemble(source)
        self.cpu = CPU(devices=self._devices)
        self.cpu._tracing = trace
        self.cpu.load_program(instructions)
        return self.cpu.run(max_steps=max_steps)

    def disassemble(self, source):
        instructions = self._assembler.assemble(source)
        lines = []
        for i, inst in enumerate(instructions):
            ops = ", ".join(str(o) for o in inst.operands) if inst.operands else ""
            lines.append(f"  {i:04d}: {inst.opcode:<12s} {ops}")
        return lines


# ── Backward Compat Aliases ──────────────────────────────────────────────────

ProgramLoader = Assembler
VirtualCPU = CPU
Assembler.load = Assembler.assemble


# ═══════════════════════════════════════════════════════════════════════════════
# Opcode Handlers (formerly vm_alu.py)
# ═══════════════════════════════════════════════════════════════════════════════


# ── Integer ALU ──────────────────────────────────────────────────────────────

def _op_iadd(cpu, ops):
    cpu._check_arity(ops, 3)
    a, b = int(cpu._val(ops[1])), int(cpu._val(ops[2]))
    result = a + b
    cpu._carry_flag = result > 0xFFFFFFFF
    cpu.regs[cpu._reg(ops[0])] = result & 0xFFFFFFFF

def _op_isub(cpu, ops):
    cpu._check_arity(ops, 3)
    a, b = int(cpu._val(ops[1])), int(cpu._val(ops[2]))
    result = a - b
    cpu._carry_flag = result < 0
    cpu.regs[cpu._reg(ops[0])] = result

def _op_imul(cpu, ops):
    cpu._check_arity(ops, 3)
    cpu.regs[cpu._reg(ops[0])] = int(cpu._val(ops[1])) * int(cpu._val(ops[2]))

def _op_idiv(cpu, ops):
    cpu._check_arity(ops, 3)
    a, b = int(cpu._val(ops[1])), int(cpu._val(ops[2]))
    cpu.regs[cpu._reg(ops[0])] = a // b if b != 0 else 0

def _op_iand(cpu, ops):
    cpu._check_arity(ops, 3)
    cpu.regs[cpu._reg(ops[0])] = int(cpu._val(ops[1])) & int(cpu._val(ops[2]))

def _op_ior(cpu, ops):
    cpu._check_arity(ops, 3)
    cpu.regs[cpu._reg(ops[0])] = int(cpu._val(ops[1])) | int(cpu._val(ops[2]))

def _op_ixor(cpu, ops):
    cpu._check_arity(ops, 3)
    cpu.regs[cpu._reg(ops[0])] = int(cpu._val(ops[1])) ^ int(cpu._val(ops[2]))

def _op_ishl(cpu, ops):
    cpu._check_arity(ops, 3)
    cpu.regs[cpu._reg(ops[0])] = int(cpu._val(ops[1])) << int(cpu._val(ops[2]))

def _op_ishr(cpu, ops):
    cpu._check_arity(ops, 3)
    cpu.regs[cpu._reg(ops[0])] = int(cpu._val(ops[1])) >> int(cpu._val(ops[2]))

def _op_ineg(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = -int(cpu._val(ops[1]))

def _op_inc(cpu, ops):
    cpu._check_arity(ops, 1)
    idx = cpu._reg(ops[0])
    cpu.regs[idx] = int(cpu.regs[idx]) + 1

def _op_dec(cpu, ops):
    cpu._check_arity(ops, 1)
    idx = cpu._reg(ops[0])
    cpu.regs[idx] = int(cpu.regs[idx]) - 1

def _op_icmp(cpu, ops):
    cpu._check_arity(ops, 2)
    a, b = int(cpu._val(ops[0])), int(cpu._val(ops[1]))
    cpu._cmp_flag = -1 if a < b else (1 if a > b else 0)


# ── Float ALU ──────────────────────────────────────────────────────────────

def _op_fadd(cpu, ops):
    cpu._check_arity(ops, 3)
    cpu.regs[cpu._reg(ops[0])] = float(cpu._val(ops[1])) + float(cpu._val(ops[2]))

def _op_fsub(cpu, ops):
    cpu._check_arity(ops, 3)
    cpu.regs[cpu._reg(ops[0])] = float(cpu._val(ops[1])) - float(cpu._val(ops[2]))

def _op_fmul(cpu, ops):
    cpu._check_arity(ops, 3)
    cpu.regs[cpu._reg(ops[0])] = float(cpu._val(ops[1])) * float(cpu._val(ops[2]))

def _op_fdiv(cpu, ops):
    cpu._check_arity(ops, 3)
    b = float(cpu._val(ops[2]))
    if b == 0:
        raise InsFault("division by zero")
    cpu.regs[cpu._reg(ops[0])] = float(cpu._val(ops[1])) / b

def _op_fcmp(cpu, ops):
    cpu._check_arity(ops, 2)
    a, b = float(cpu._val(ops[0])), float(cpu._val(ops[1]))
    cpu._cmp_flag = -1 if a < b else (1 if a > b else 0)


# ── Stack Operations ───────────────────────────────────────────────────────

def _op_push(cpu, ops):
    cpu._check_arity(ops, 1)
    if cpu.sp <= 0:
        raise InsFault("stack overflow")
    cpu.sp -= 1
    cpu._stack[cpu.sp] = cpu._val(ops[0])

def _op_pop(cpu, ops):
    cpu._check_arity(ops, 1)
    if cpu.sp >= STACK_BASE:
        raise InsFault("stack underflow")
    cpu.regs[cpu._reg(ops[0])] = cpu._stack[cpu.sp]
    cpu.sp += 1


# ── Memory Operations ──────────────────────────────────────────────────────

def _op_alloc(cpu, ops):
    cpu._check_arity(ops, 2)
    size = int(cpu._val(ops[1]))
    name = f"_alloc_{cpu._step_count}"
    cpu._memory.store(name, np.zeros(size, dtype=np.float64))
    cpu.regs[cpu._reg(ops[0])] = size

def _op_meminfo(cpu, ops):
    cpu._check_arity(ops, 1)
    usage = cpu._memory.usage()
    cpu.regs[cpu._reg(ops[0])] = usage.get("entries", 0)


# ── I/O Operations ─────────────────────────────────────────────────────────

def _op_in(cpu, ops):
    cpu._check_arity(ops, 2)
    port = int(cpu._val(ops[1]))
    try:
        device = cpu._devices._devices.get(str(port))
        if device:
            if hasattr(device, 'read'):
                val = device.read()
                try:
                    val = int(val)
                except (ValueError, TypeError):
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        pass
                cpu.regs[cpu._reg(ops[0])] = val
            else:
                cpu.regs[cpu._reg(ops[0])] = device.info().get("status", 0)
        else:
            cpu.regs[cpu._reg(ops[0])] = 0
    except Exception:
        cpu.regs[cpu._reg(ops[0])] = 0

def _op_out(cpu, ops):
    cpu._check_arity(ops, 2)
    port = int(cpu._val(ops[0]))
    val = cpu._val(ops[1])
    try:
        device = cpu._devices._devices.get(str(port))
        if device and hasattr(device, 'write'):
            device.write(val)
    except Exception:
        pass


# ── Tensor ALU ───────────────────────────────────────────────────────────────

def _op_add(cpu, ops):
    cpu._check_arity(ops, 3)
    rd = cpu._reg(ops[0])
    a, b = cpu._val(ops[1]), cpu._val(ops[2])
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        cpu.regs[rd] = cpu._parse_tensor(a) + cpu._parse_tensor(b)
    else:
        cpu.regs[rd] = (a or 0) + (b or 0)

def _op_sub(cpu, ops):
    cpu._check_arity(ops, 3)
    rd = cpu._reg(ops[0])
    a, b = cpu._val(ops[1]), cpu._val(ops[2])
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        cpu.regs[rd] = cpu._parse_tensor(a) - cpu._parse_tensor(b)
    else:
        cpu.regs[rd] = (a or 0) - (b or 0)

def _op_mul(cpu, ops):
    cpu._check_arity(ops, 3)
    rd = cpu._reg(ops[0])
    a, b = cpu._val(ops[1]), cpu._val(ops[2])
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        cpu.regs[rd] = cpu._parse_tensor(a) * cpu._parse_tensor(b)
    else:
        cpu.regs[rd] = (a or 0) * (b or 0)

def _op_div(cpu, ops):
    cpu._check_arity(ops, 3)
    rd = cpu._reg(ops[0])
    a = cpu._parse_tensor(cpu._val(ops[1]))
    b = cpu._parse_tensor(cpu._val(ops[2]))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = a / b
        result = np.where(np.isinf(result), 0.0, result)
        result = np.where(np.isnan(result), 0.0, result)
    cpu.regs[rd] = result

def _op_neg(cpu, ops):
    cpu._check_arity(ops, 2)
    rd = cpu._reg(ops[0])
    a = cpu._val(ops[1])
    cpu.regs[rd] = -cpu._parse_tensor(a) if isinstance(a, np.ndarray) else -(a or 0)

def _op_abs(cpu, ops):
    cpu._check_arity(ops, 2)
    rd = cpu._reg(ops[0])
    a = cpu._val(ops[1])
    cpu.regs[rd] = np.abs(a) if isinstance(a, np.ndarray) else abs(a or 0)

def _op_matmul(cpu, ops):
    cpu._check_arity(ops, 3)
    rd = cpu._reg(ops[0])
    a = cpu._parse_tensor(cpu._val(ops[1]))
    b = cpu._parse_tensor(cpu._val(ops[2]))
    if a.ndim == 0:
        a = a.reshape(1, 1)
    if b.ndim == 0:
        b = b.reshape(1, 1)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    cpu.regs[rd] = a @ b

def _op_transpose(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = cpu._parse_tensor(cpu._val(ops[1])).T

def _op_dot(cpu, ops):
    cpu._check_arity(ops, 3)
    a = cpu._parse_tensor(cpu._val(ops[1])).ravel()
    b = cpu._parse_tensor(cpu._val(ops[2])).ravel()
    cpu.regs[cpu._reg(ops[0])] = float(np.dot(a, b))

def _op_norm(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = float(np.linalg.norm(cpu._parse_tensor(cpu._val(ops[1]))))

def _op_sum(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = float(np.sum(cpu._parse_tensor(cpu._val(ops[1]))))

def _op_mean(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = float(np.mean(cpu._parse_tensor(cpu._val(ops[1]))))

def _op_max(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = float(np.max(cpu._parse_tensor(cpu._val(ops[1]))))

def _op_argmax(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = int(np.argmax(cpu._parse_tensor(cpu._val(ops[1]))))

def _op_reshape(cpu, ops):
    cpu._check_arity(ops, 4)
    a = cpu._parse_tensor(cpu._val(ops[1]))
    rows = int(ops[2]) if isinstance(ops[2], (int, float)) else -1
    cols = int(ops[3]) if isinstance(ops[3], (int, float)) else -1
    cpu.regs[cpu._reg(ops[0])] = a.reshape(rows, cols)

def _op_shape(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = list(cpu._parse_tensor(cpu._val(ops[1])).shape)

def _op_size(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = int(cpu._parse_tensor(cpu._val(ops[1])).size)

def _op_relu(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = np.maximum(0, cpu._parse_tensor(cpu._val(ops[1])))

def _op_gelu(cpu, ops):
    cpu._check_arity(ops, 2)
    a = cpu._parse_tensor(cpu._val(ops[1]))
    cpu.regs[cpu._reg(ops[0])] = 0.5 * a * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (a + 0.044715 * a ** 3)))

def _op_sigmoid(cpu, ops):
    cpu._check_arity(ops, 2)
    a = cpu._parse_tensor(cpu._val(ops[1]))
    cpu.regs[cpu._reg(ops[0])] = 1.0 / (1.0 + np.exp(-np.clip(a, -500, 500)))

def _op_tanh(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = np.tanh(cpu._parse_tensor(cpu._val(ops[1])))

def _op_softmax(cpu, ops):
    cpu._check_arity(ops, 2)
    a = cpu._parse_tensor(cpu._val(ops[1]))
    shifted = a - np.max(a)
    exp_a = np.exp(shifted)
    cpu.regs[cpu._reg(ops[0])] = exp_a / np.sum(exp_a)

def _op_layernorm(cpu, ops):
    cpu._check_arity(ops, 2)
    a = cpu._parse_tensor(cpu._val(ops[1]))
    cpu.regs[cpu._reg(ops[0])] = (a - np.mean(a)) / np.sqrt(np.var(a) + 1e-5)

def _op_rmsnorm(cpu, ops):
    cpu._check_arity(ops, 2)
    a = cpu._parse_tensor(cpu._val(ops[1]))
    cpu.regs[cpu._reg(ops[0])] = a / np.sqrt(np.mean(a ** 2) + 1e-5)

def _op_randn(cpu, ops):
    cpu._check_arity(ops, 3)
    r = int(ops[1]) if isinstance(ops[1], (int, float)) else 1
    c = int(ops[2]) if isinstance(ops[2], (int, float)) else 1
    cpu.regs[cpu._reg(ops[0])] = np.random.randn(r, c)

def _op_randunif(cpu, ops):
    cpu._check_arity(ops, 5)
    r = int(ops[1]) if isinstance(ops[1], (int, float)) else 1
    c = int(ops[2]) if isinstance(ops[2], (int, float)) else 1
    lo = float(ops[3]) if isinstance(ops[3], (int, float)) else 0.0
    hi = float(ops[4]) if isinstance(ops[4], (int, float)) else 1.0
    cpu.regs[cpu._reg(ops[0])] = np.random.uniform(lo, hi, (r, c))


# ── Comparison ───────────────────────────────────────────────────────────────

def _op_cmp(cpu, ops):
    cpu._check_arity(ops, 2)
    a, b = cpu._val(ops[0]), cpu._val(ops[1])
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        diff = a - b
        cpu._cmp_flag = -1 if diff < 0 else (1 if diff > 0 else 0)
        if hasattr(cpu, '_update_flags_sub'):
            cpu._update_flags_sub(a, b, diff, 32)
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if np.array_equal(a, b):
            cpu._cmp_flag = 0
        elif np.all(a < b):
            cpu._cmp_flag = -1
        elif np.all(a > b):
            cpu._cmp_flag = 1
        else:
            cpu._cmp_flag = 0
    else:
        cpu._cmp_flag = -1 if str(a) < str(b) else (1 if str(a) > str(b) else 0)

def _op_test(cpu, ops):
    cpu._check_arity(ops, 1)
    cpu._cmp_flag = 1 if cpu._truthy(cpu._val(ops[0])) else 0


# ── Control Flow ─────────────────────────────────────────────────────────────

def _resolve_label(cpu, operand):
    if isinstance(operand, int):
        return operand
    if isinstance(operand, str) and operand.isdigit():
        return int(operand)
    raise InsFault(f"invalid jump target: {operand}")

def _op_jmp(cpu, ops):
    cpu._check_arity(ops, 1)
    cpu.pc = _resolve_label(cpu, ops[0])
    cpu._pc_changed = True

def _op_jz(cpu, ops):
    cpu._check_arity(ops, 1)
    if cpu._cmp_flag == 0:
        cpu.pc = _resolve_label(cpu, ops[0])
        cpu._pc_changed = True

def _op_jnz(cpu, ops):
    cpu._check_arity(ops, 1)
    if cpu._cmp_flag != 0:
        cpu.pc = _resolve_label(cpu, ops[0])
        cpu._pc_changed = True

def _op_jgt(cpu, ops):
    cpu._check_arity(ops, 1)
    if cpu._cmp_flag > 0:
        cpu.pc = _resolve_label(cpu, ops[0])
        cpu._pc_changed = True

def _op_jge(cpu, ops):
    cpu._check_arity(ops, 1)
    if cpu._cmp_flag >= 0:
        cpu.pc = _resolve_label(cpu, ops[0])
        cpu._pc_changed = True

def _op_jlt(cpu, ops):
    cpu._check_arity(ops, 1)
    if cpu._cmp_flag < 0:
        cpu.pc = _resolve_label(cpu, ops[0])
        cpu._pc_changed = True

def _op_jle(cpu, ops):
    cpu._check_arity(ops, 1)
    if cpu._cmp_flag <= 0:
        cpu.pc = _resolve_label(cpu, ops[0])
        cpu._pc_changed = True

def _op_call(cpu, ops):
    cpu._check_arity(ops, 1)
    if len(cpu._call_stack) >= MAX_CALL_DEPTH:
        cpu._output.append("[VM] call stack overflow")
        cpu._running = False
        raise Halt("call stack overflow")
    cpu._call_stack.append(cpu.pc + 1)
    cpu.pc = _resolve_label(cpu, ops[0])
    cpu._pc_changed = True

def _op_ret(cpu, ops):
    if not cpu._call_stack:
        raise InsFault("ret with empty call stack")
    cpu.pc = cpu._call_stack.pop()
    cpu._pc_changed = True

def _op_loop(cpu, ops):
    cpu._check_arity(ops, 2)
    idx = cpu._reg(ops[0])
    cpu.regs[idx] = int(cpu.regs[idx]) - 1
    if cpu.regs[idx] != 0:
        cpu.pc = _resolve_label(cpu, ops[1])
        cpu._pc_changed = True

def _op_halt(cpu, ops):
    raise Halt()


# ── System Calls ─────────────────────────────────────────────────────────────

# Syscall numbers (match kernel_syscall.SyscallNumber)
SYS_PRINT = 111
SYS_EXIT = 2
SYS_ALLOC = 20
SYS_FREE = 21
SYS_OPEN = 120
SYS_READ = 121
SYS_WRITE = 122
SYS_CLOSE = 123
SYS_UPTIME = 200
SYS_STATS = 201

# Kernel-provided syscall handler (set by Kernel or VirtualSystem)
_syscall_handler = None


def set_syscall_handler(handler):
    """Set the global syscall handler function.

    The handler receives (syscall_num, args) and returns a value.
    """
    global _syscall_handler
    _syscall_handler = handler


def _op_syscall(cpu, ops):
    """SYSCALL — software interrupt for kernel services.

    Convention:
      R7 = syscall number
      R0-R5 = arguments
      R0 = return value
    """
    num = int(cpu.regs[7])
    args = [cpu.regs[i] for i in range(6)]

    if _syscall_handler is not None:
        result = _syscall_handler(num, args)
        cpu.regs[0] = result if result is not None else 0
    else:
        cpu.regs[0] = 0


# ── Data Movement ────────────────────────────────────────────────────────────

def _op_load_const(cpu, ops):
    cpu._check_arity(ops, 2)
    rd = cpu._reg(ops[0])
    val = ops[1]
    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
        import json as _json
        try:
            parsed = _json.loads(val)
            if isinstance(parsed, list):
                cpu.regs[rd] = np.array(parsed, dtype=np.float64)
                return
        except (ValueError, TypeError):
            pass
        inner = val[1:-1].strip()
        if inner:
            parts = []
            for item in inner.split(","):
                item = item.strip()
                try:
                    parts.append(int(item))
                except ValueError:
                    try:
                        parts.append(float(item))
                    except ValueError:
                        parts.append(item)
            cpu.regs[rd] = np.array(parts, dtype=np.float64)
        else:
            cpu.regs[rd] = np.array([], dtype=np.float64)
    else:
        cpu.regs[rd] = val

def _op_load_shape(cpu, ops):
    cpu._check_arity(ops, 3)
    r = int(ops[1]) if isinstance(ops[1], (int, float)) else 1
    c = int(ops[2]) if isinstance(ops[2], (int, float)) else 1
    cpu.regs[cpu._reg(ops[0])] = np.zeros((r, c))

def _op_mov(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = cpu._val(ops[1])

def _op_store(cpu, ops):
    cpu._check_arity(ops, 2)
    key = str(ops[1])
    cpu._memory.store(key, cpu._val(ops[0]))

def _op_load(cpu, ops):
    cpu._check_arity(ops, 2)
    key = str(ops[1])
    cpu.regs[cpu._reg(ops[0])] = cpu._memory.load(key)

def _op_free(cpu, ops):
    cpu._check_arity(ops, 1)
    cpu._memory.free(str(ops[0]))

def _op_print(cpu, ops):
    cpu._check_arity(ops, 1)
    val = cpu._val(ops[0])
    if isinstance(val, np.ndarray):
        cpu._output.append(np.array2string(val, precision=4, suppress_small=True))
    else:
        cpu._output.append(str(val))

def _op_nop(cpu, ops):
    pass


# ── Device Bus Ops ───────────────────────────────────────────────────────────

def _op_dev_open(cpu, ops):
    cpu._check_arity(ops, 2)
    rd = cpu._reg(ops[0])
    name = str(cpu._val(ops[1]))
    dev = cpu._devices.open(name)
    cpu.regs[rd] = name

def _op_dev_call(cpu, ops):
    cpu._check_arity(ops, 3)
    rd = cpu._reg(ops[0])
    handle = str(cpu._val(ops[1]))
    method = str(cpu._val(ops[2]))
    extra_args = [cpu._val(o) for o in ops[3:]]
    dev = cpu._devices.open(handle)
    cpu.regs[rd] = cpu._devices.call(dev, method, *extra_args)

def _op_dev_close(cpu, ops):
    cpu._check_arity(ops, 1)
    pass

def _op_dev_info(cpu, ops):
    cpu._check_arity(ops, 2)
    rd = cpu._reg(ops[0])
    handle = str(cpu._val(ops[1]))
    dev = cpu._devices.open(handle)
    cpu.regs[rd] = cpu._devices.info(dev)


# ── Opcode Table ─────────────────────────────────────────────────────────────

_OPCODE_TABLE = {
    "LOAD_CONST": _op_load_const, "LOAD_SHAPE": _op_load_shape,
    "MOV": _op_mov, "STORE": _op_store, "LOAD": _op_load,
    "FREE": _op_free, "PRINT": _op_print, "NOP": _op_nop,
    "IADD": _op_iadd, "ISUB": _op_isub, "IMUL": _op_imul, "IDIV": _op_idiv,
    "IAND": _op_iand, "IOR": _op_ior, "IXOR": _op_ixor,
    "ISHL": _op_ishl, "ISHR": _op_ishr, "INEG": _op_ineg,
    "INC": _op_inc, "DEC": _op_dec, "ICMP": _op_icmp,
    "ADD": _op_add, "SUB": _op_sub, "MUL": _op_mul, "DIV": _op_div,
    "NEG": _op_neg, "ABS": _op_abs,
    "MATMUL": _op_matmul, "TRANSPOSE": _op_transpose,
    "DOT": _op_dot, "NORM": _op_norm,
    "SUM": _op_sum, "MEAN": _op_mean, "MAX": _op_max, "ARGMAX": _op_argmax,
    "RESHAPE": _op_reshape, "SHAPE": _op_shape, "SIZE": _op_size,
    "RELU": _op_relu, "GELU": _op_gelu, "SIGMOID": _op_sigmoid,
    "TANH": _op_tanh, "SOFTMAX": _op_softmax,
    "LAYERNORM": _op_layernorm, "RMSNORM": _op_rmsnorm,
    "RANDN": _op_randn, "RANDUNIF": _op_randunif,
    "CMP": _op_cmp, "TEST": _op_test,
    "JMP": _op_jmp, "JZ": _op_jz, "JNZ": _op_jnz,
    "JGT": _op_jgt, "JGE": _op_jge, "JLT": _op_jlt, "JLE": _op_jle,
    "CALL": _op_call, "RET": _op_ret, "LOOP": _op_loop, "HALT": _op_halt,
    "DEV_OPEN": _op_dev_open, "DEV_CALL": _op_dev_call,
    "DEV_CLOSE": _op_dev_close, "DEV_INFO": _op_dev_info,
    "PUSH": _op_push, "POP": _op_pop,
    "FADD": _op_fadd, "FSUB": _op_fsub, "FMUL": _op_fmul, "FDIV": _op_fdiv,
    "FCMP": _op_fcmp,
    "ALLOC": _op_alloc, "MEMINFO": _op_meminfo,
    "IN": _op_in, "OUT": _op_out,
    "SYSCALL": _op_syscall,
}


# ── x86 Assembler ────────────────────────────────────────────────────────────

class X86Assembler:
    """x86-32 real mode assembler — compiles assembly to machine code bytes.

    Two-pass assembler:
      Pass 1: Collect labels and calculate addresses
      Pass 2: Generate machine code bytes

    Supports:
      - Instructions: mov, add, sub, and, or, xor, cmp, test,
        jmp, jz, jnz, jg, jl, jge, jle, ja, jb, jae, jbe,
        push, pop, call, ret, int, in, out, cli, sti, hlt, nop,
        inc, dec, neg, not, shl, shr, mul, div, imul, idiv,
        lea, xchg, pusha, popa, iret, lgdt, lidt, ltr, mov cr/dr,
        iret, ljmp, lcall
      - Data: db, dw, dd, dq, times
      - Directives: [BITS 16/32], [ORG addr], equ, section
      - Register encoding: eax/ecx/edx/ebx/esp/ebp/esi/edi
                         ax/cx/dx/bx/sp/bp/si/di
                         al/cl/dl/bl/ah/ch/dh/bh
      - ModR/M addressing: [eax], [eax+disp8], [eax+disp32],
        [base+index*scale+disp], etc.
      - String literals in db
    """

    # ── Register encoding ────────────────────────────────────────────────

    _REG8 = {"al": 0, "cl": 1, "dl": 2, "bl": 3, "ah": 4, "ch": 5, "dh": 6, "bh": 7}
    _REG16 = {"ax": 0, "cx": 1, "dx": 2, "bx": 3, "sp": 4, "bp": 5, "si": 6, "di": 7}
    _REG32 = {"eax": 0, "ecx": 1, "edx": 2, "ebx": 3, "esp": 4, "ebp": 5, "esi": 6, "edi": 7}
    _SEG_REGS = {"es": 0, "cs": 1, "ss": 2, "ds": 3, "fs": 4, "gs": 5}
    _CTRL_REGS = {"cr0": 0, "cr1": 1, "cr2": 2, "cr3": 3, "cr4": 4, "cr5": 5, "cr6": 6, "cr7": 7}
    _DBG_REGS = {"dr0": 0, "dr1": 1, "dr2": 2, "dr3": 3, "dr4": 4, "dr5": 5, "dr6": 6, "dr7": 7}

    # ── Condition codes ──────────────────────────────────────────────────

    _CC = {
        "jz": 0x4, "je": 0x4, "jnz": 0x5, "jne": 0x5,
        "jg": 0xf, "jnle": 0xf, "jge": 0xd, "jnl": 0xd,
        "jl": 0xc, "jnge": 0xc, "jle": 0xe, "jng": 0xe,
        "ja": 0x7, "jnbe": 0x7, "jae": 0x3, "jnb": 0x3,
        "jb": 0x2, "jnae": 0x2, "jbe": 0x6, "jna": 0x6,
        "js": 0x8, "jns": 0x9, "jo": 0x0, "jno": 0x1,
        "jp": 0xa, "jpe": 0xa, "jnp": 0xb, "jpo": 0xb,
        "loop": 0xe0, "loope": 0xe1, "loopz": 0xe1,
        "loopne": 0xe2, "loopnz": 0xe2, "jcxz": 0xe3,
    }

    def __init__(self):
        self._bits = 16
        self._org = 0
        self._labels = {}
        self._output = bytearray()

    def _pfx(self, reg):
        """Return True if register needs 0x66 operand-size prefix in current bits mode."""
        if reg in self._REG16:
            return self._bits == 32
        if reg in self._REG32:
            return self._bits == 16
        return False
        self._reloc = []  # (offset, label, type)

    def assemble(self, source: str) -> bytearray:
        """Assemble x86 source to machine code bytes.

        Multi-pass converging assembler:
          Each pass generates code using current labels, then re-records labels.
          Repeats until label addresses stabilize (or max 10 passes).
        """
        self._bits = 16
        self._org = 0
        self._labels = {}
        self._output = bytearray()
        self._reloc = []

        lines = source.split("\n")

        prev_labels = {}
        for iteration in range(10):
            self._output = bytearray()
            self._bits = 16
            self._org = 0
            self._pass = max(iteration, 1)

            for line in lines:
                clean = line.split(";")[0].strip()
                if not clean:
                    continue
                if clean.startswith("["):
                    self._handle_directive(clean)
                    continue
                if ":" in clean and not clean.startswith("db ") and not clean.startswith("dw ") and not clean.startswith("dd "):
                    colon_idx = clean.index(":")
                    after_colon = clean[colon_idx + 1:colon_idx + 2]
                    if after_colon and after_colon not in (" ", "\t", ""):
                        pass
                    else:
                        prefix, _, rest = clean.partition(":")
                        if prefix and " " not in prefix:
                            self._labels[prefix] = self._org + len(self._output)
                        clean = rest.strip()
                        if not clean:
                            continue
                if " equ " in clean.lower():
                    parts_eq = clean.split(None, 2)
                    if len(parts_eq) >= 3:
                        const_name = parts_eq[0].strip()
                        const_val = self._parse_imm(parts_eq[2].strip())
                        self._labels[const_name] = const_val
                    continue
                elif clean.startswith("times"):
                    self._emit_times(clean)
                elif clean.startswith("db ") or clean.startswith("dw ") or clean.startswith("dd "):
                    self._emit_data(clean)
                else:
                    self._emit_instruction(clean)

            if self._labels == prev_labels:
                break
            prev_labels = dict(self._labels)

        return bytes(self._output)

    def _handle_directive(self, line):
        line = line.strip("[]").strip()
        if line.upper().startswith("BITS"):
            self._bits = int(line.split()[1])
        elif line.upper().startswith("ORG"):
            self._org = self._parse_imm(line.split(None, 1)[1])

    def _estimate_times_size(self, line):
        parts = line.split(None, 2)
        count = self._parse_imm(parts[1])
        return count * self._estimate_data_size(parts[2])

    def _estimate_data_size(self, line):
        if line.startswith("times"):
            return self._estimate_times_size(line)
        if line.startswith("db"):
            inner = line[2:].strip()
            total = 0
            for item in inner.split(","):
                item = item.strip()
                if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):
                    total += len(item) - 2  # strip quotes
                elif item.startswith('"') or item.startswith("'"):
                    total += len(item) - 1  # strip leading quote
                else:
                    total += 1
            return total
        if line.startswith("dw"):
            return 2 * len(line[2:].strip().split(","))
        if line.startswith("dd"):
            return 4 * len(line[2:].strip().split(","))
        return 1

    def _estimate_insn_size(self, line):
        parts = line.split(None, 1)
        op = parts[0].lower()
        if op in ("nop", "hlt", "cli", "sti", "ret", "iret", "pusha", "popa", "cld", "std",
                   "lodsb", "lodsw", "stosb", "stosw", "movsb", "movsw", "cmpsb", "scasb"):
            return 1
        if op == "rep":
            return 2  # rep prefix + string instruction
        if op in ("retf",):
            return 1
        if op in self._CC:
            return 2  # short jump
        if op == "int":
            return 2
        if op == "push":
            # push reg = 1, push imm = 3
            operand = parts[1].strip() if len(parts) > 1 else ""
            if operand in self._REG16 or operand in self._REG32:
                return 1
            return 3
        if op == "pop":
            return 1
        if op == "jmp":
            # Far jump: jmp seg:off → EA off16 seg16 = 5 bytes
            if len(parts) > 1 and ":" in parts[1] and not parts[1].strip().startswith("["):
                return 5
            if self._bits == 16:
                return 3  # jmp rel16 (worst case in 16-bit mode)
            return 2  # jmp rel8 (short jump, always 2 bytes)
        if op == "call":
            if self._bits == 16:
                return 3  # call rel16 in 16-bit mode
            return 5
        if op in ("in", "out"):
            return 2
        if op == "mov":
            return self._estimate_mov_size(parts[1] if len(parts) > 1 else "")
        if op in ("add", "sub", "and", "or", "xor", "cmp", "test"):
            return self._estimate_alu_size(parts[1] if len(parts) > 1 else "")
        if op in ("inc", "dec", "neg", "not", "shl", "shr", "sal", "sar", "rol", "ror", "rcl", "rcr"):
            return 2
        return 3  # default

    def _estimate_mov_size(self, operands):
        parts = self._split_ops(operands)
        if len(parts) < 2:
            return 2
        dst, src = parts[0].strip(), parts[1].strip()
        # MOV Sreg, r/m16 or MOV r/m16, Sreg — 2 bytes (8E/8C + ModRM)
        if dst in self._SEG_REGS or src in self._SEG_REGS:
            return 2
        if dst in self._REG32 or src in self._REG32:
            return 5
        if dst in self._REG16 or src in self._REG16:
            return 3
        return 2

    def _estimate_alu_size(self, operands):
        parts = self._split_ops(operands)
        if len(parts) < 2:
            return 2
        dst, src = parts[0].strip(), parts[1].strip()
        # reg, reg → 2 bytes (31/r for xor, 08/r for or, etc.)
        if (dst in self._REG32 and src in self._REG32) or \
           (dst in self._REG16 and src in self._REG16) or \
           (dst in self._REG8 and src in self._REG8):
            return 2
        # reg, imm (small) → 3 bytes (83 /x ib)
        # reg, imm (large) → 6 bytes (81 /x id)
        return 3

    def _emit_times(self, line):
        parts = line.split(None, 2)
        expr = parts[1].strip()
        inner = parts[2].strip()
        # Resolve $ (current position) and $$ (section start = 0)
        expr = expr.replace("$$", "0").replace("$", str(len(self._output)))
        count = self._eval_expr(expr)
        # Handle instructions
        single_byte_insns = {"nop": 0x90, "hlt": 0xF4, "cli": 0xFA, "sti": 0xFB, "ret": 0xC3}
        if inner in single_byte_insns:
            for _ in range(count):
                self._output.append(single_byte_insns[inner])
        elif inner.startswith("db ") or inner.startswith("dw ") or inner.startswith("dd "):
            for _ in range(count):
                self._emit_data(inner)
        else:
            val = self._parse_imm(inner)
            for _ in range(count):
                self._output.append(val & 0xFF)

    def _emit_data(self, line):
        if line.startswith("db"):
            inner = line[2:].strip()
            items = self._parse_db_items(inner)
            for item in items:
                if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):
                    s = item[1:-1]
                    for ch in s:
                        self._output.append(ord(ch))
                elif item.startswith('"') or item.startswith("'"):
                    s = item[1:]
                    for ch in s:
                        self._output.append(ord(ch))
                else:
                    val = self._parse_imm(item)
                    self._output.append(val & 0xFF)
        elif line.startswith("dw"):
            inner = line[2:].strip()
            for item in inner.split(","):
                val = self._parse_imm(item.strip())
                self._output.extend(val.to_bytes(2, "little"))
        elif line.startswith("dd"):
            inner = line[2:].strip()
            for item in inner.split(","):
                val = self._parse_imm(item.strip())
                self._output.extend((val & 0xFFFFFFFF).to_bytes(4, "little"))

    def _parse_db_items(self, inner):
        items = []
        i = 0
        while i < len(inner):
            if inner[i] in ('"', "'"):
                quote = inner[i]
                j = inner.index(quote, i + 1) if quote in inner[i+1:] else len(inner)
                items.append(inner[i:j+1])
                i = j + 1
                if i < len(inner) and inner[i] == ',':
                    i += 1
            elif inner[i] == ',':
                i += 1
            else:
                j = inner.index(',', i) if ',' in inner[i:] else len(inner)
                items.append(inner[i:j].strip())
                i = j + 1
        return items

    def _emit_instruction(self, line):
        parts = line.split(None, 1)
        op = parts[0].lower()
        operands = self._split_ops(parts[1].strip()) if len(parts) > 1 else []

        if op == "nop":
            self._output.append(0x90)
        elif op == "hlt":
            self._output.append(0xF4)
        elif op == "cli":
            self._output.append(0xFA)
        elif op == "sti":
            self._output.append(0xFB)
        elif op == "cld":
            self._output.append(0xFC)
        elif op == "std":
            self._output.append(0xFD)
        elif op == "ret":
            self._output.append(0xC3)
        elif op == "retf":
            self._output.append(0xCB)
        elif op == "iret":
            self._output.append(0xCF)
        elif op in ("pusha", "pushad"):
            self._output.append(0x60)
        elif op in ("popa", "popad"):
            self._output.append(0x61)
        elif op == "push":
            self._emit_push(operands)
        elif op == "pop":
            self._emit_pop(operands)
        elif op == "int":
            self._emit_int(operands)
        elif op == "in":
            self._emit_in(operands)
        elif op == "out":
            self._emit_out(operands)
        elif op in self._CC:
            self._emit_cc_jump(op, operands)
        elif op == "jmp":
            self._emit_jmp(operands)
        elif op == "call":
            self._emit_call(operands)
        elif op == "mov":
            self._emit_mov(operands)
        elif op in ("add", "sub", "and", "or", "xor", "cmp", "test"):
            self._emit_alu(op, operands)
        elif op in ("inc", "dec"):
            self._emit_inc_dec(op, operands)
        elif op in ("neg", "not", "mul", "imul", "div", "idiv"):
            self._emit_unary(op, operands)
        elif op in ("shl", "shr", "sal", "sar", "rol", "ror", "rcl", "rcr"):
            self._emit_shift(op, operands)
        elif op == "lea":
            self._emit_lea(operands)
        elif op == "xchg":
            self._emit_xchg(operands)
        elif op == "lodsb":
            self._output.append(0xAC)
        elif op == "lodsw":
            if self._pfx("ax"):
                self._output.append(0x66)
            self._output.append(0xAD)
        elif op == "stosb":
            self._output.append(0xAA)
        elif op == "stosw":
            if self._pfx("ax"):
                self._output.append(0x66)
            self._output.append(0xAB)
        elif op == "movsb":
            self._output.append(0xA4)
        elif op == "movsw":
            if self._pfx("ax"):
                self._output.append(0x66)
            self._output.append(0xA5)
        elif op == "cmpsb":
            self._output.append(0xA6)
        elif op == "scasb":
            self._output.append(0xAE)
        elif op == "rep":
            # rep prefix + string instruction
            self._output.append(0xF3)
            inner = operands[0].lower() if operands else ""
            if inner == "movsb":
                self._output.append(0xA4)
            elif inner == "movsw":
                if self._pfx("ax"):
                    self._output.append(0x66)
                self._output.append(0xA5)
            elif inner == "stosb":
                self._output.append(0xAA)
            elif inner == "stosw":
                if self._pfx("ax"):
                    self._output.append(0x66)
                self._output.append(0xAB)
            elif inner == "lodsb":
                self._output.append(0xAC)
            elif inner == "lodsw":
                if self._pfx("ax"):
                    self._output.append(0x66)
                self._output.append(0xAD)
            elif inner == "cmpsb":
                self._output.append(0xA6)
            elif inner == "scasb":
                self._output.append(0xAE)
            else:
                # Unknown rep target — just emit prefix
                pass
        elif op == "lgdt":
            self._emit_lgdt(operands)
        elif op == "lidt":
            self._emit_lidt(operands)
        elif op == "ltr":
            self._emit_ltr(operands)
        else:
            # Unknown instruction — emit NOP placeholder
            self._output.append(0x90)

    def _emit_push(self, ops):
        if not ops:
            return
        reg = ops[0].lower()
        if reg in self._REG32:
            if self._pfx(reg):
                self._output.append(0x66)
            self._output.append(0x50 + self._REG32[reg])
        elif reg in self._REG16:
            if self._pfx(reg):
                self._output.append(0x66)
            self._output.append(0x50 + self._REG16[reg])
        elif reg in self._SEG_REGS:
            self._output.append(0x06 + self._SEG_REGS[reg] * 8)
        else:
            # Push immediate
            val = self._parse_imm(ops[0])
            if -128 <= val <= 127:
                self._output.append(0x6A)
                self._output.append(val & 0xFF)
            else:
                self._output.append(0x68)
                if self._bits == 16:
                    self._output.extend(struct.pack("<H", val & 0xFFFF))
                else:
                    self._output.extend((val & 0xFFFFFFFF).to_bytes(4, "little"))

    def _emit_pop(self, ops):
        if not ops:
            return
        reg = ops[0].lower()
        if reg in self._REG32:
            if self._pfx(reg):
                self._output.append(0x66)
            self._output.append(0x58 + self._REG32[reg])
        elif reg in self._REG16:
            if self._pfx(reg):
                self._output.append(0x66)
            self._output.append(0x58 + self._REG16[reg])
        elif reg in self._SEG_REGS:
            self._output.append(0x07 + self._SEG_REGS[reg] * 8)

    def _emit_int(self, ops):
        if not ops:
            return
        val = self._parse_imm(ops[0])
        self._output.append(0xCD)
        self._output.append(val & 0xFF)

    def _emit_in(self, ops):
        if len(ops) < 2:
            return
        dst, src = ops[0].lower(), ops[1].strip().lower()
        if src in ("dx", "edx"):
            if dst == "al":
                self._output.append(0xEC)
            elif dst in ("ax", "eax"):
                self._output.append(0xED)
            else:
                self._output.append(0xEC)
        elif dst in ("al", "ax", "eax"):
            val = self._parse_imm(src)
            if dst == "al":
                self._output.append(0xE4)
                self._output.append(val & 0xFF)
            elif dst in ("ax", "eax"):
                self._output.append(0xE5)
                self._output.append(val & 0xFF)
            else:
                self._output.append(0xE4)
                self._output.append(val & 0xFF)

    def _emit_out(self, ops):
        if len(ops) < 2:
            return
        dst, src = ops[0].strip().lower(), ops[1].lower()
        if dst in ("dx", "edx"):
            if src == "al":
                self._output.append(0xEE)
            elif src in ("ax", "eax"):
                self._output.append(0xEF)
            else:
                self._output.append(0xEE)
        else:
            val = self._parse_imm(dst)
            if src == "al":
                self._output.append(0xE6)
                self._output.append(val & 0xFF)
            elif src in ("ax", "eax"):
                self._output.append(0xE7)
                self._output.append(val & 0xFF)
            else:
                self._output.append(0xE6)
                self._output.append(val & 0xFF)

    def _emit_cc_jump(self, op, ops):
        if not ops:
            return
        cc = self._CC[op]
        target = self._parse_label(ops[0])

        # Special opcodes: loop/jcxz use their own encoding (0xE0-0xE3)
        if cc >= 0xE0:
            offset = target - (self._org + len(self._output) + 2)
            if -128 <= offset <= 127:
                self._output.append(cc)
                self._output.append(offset & 0xFF)
            else:
                # Near jump for loop/jcxz (not standard, but use JMP)
                self._output.append(0xEB)
                near_offset = target - (self._org + len(self._output) + 2)
                self._output.append(near_offset & 0xFF)
            return

        offset = target - (self._org + len(self._output) + 2)
        if -128 <= offset <= 127:
            self._output.append(0x70 + cc)
            self._output.append(offset & 0xFF)
        elif self._bits == 16:
            # In 16-bit mode, use JMP near (EB) as fallback to keep 2-byte estimate
            self._output.append(0xEB)
            near_offset = target - (self._org + len(self._output) + 2)
            self._output.append(near_offset & 0xFF)
        else:
            # Near jump in 32-bit mode
            offset = target - (self._org + len(self._output) + 6)
            self._output.append(0x0F)
            self._output.append(0x80 + cc)
            self._output.extend((offset & 0xFFFFFFFF).to_bytes(4, "little"))

    def _emit_jmp(self, ops):
        if not ops:
            return
        target_str = ops[0].strip()
        # Far jump: jmp seg:off
        if ":" in target_str and not target_str.startswith("["):
            seg_str, _, off_str = target_str.partition(":")
            seg_val = self._parse_label(seg_str)
            off_val = self._parse_label(off_str)
            if self._bits == 16:
                # 16-bit mode: need 0x66 prefix for 32-bit offset
                # (required when jumping to a 32-bit code segment for PM switch)
                self._output.append(0x66)
                self._output.append(0xEA)
                self._output.extend((off_val & 0xFFFFFFFF).to_bytes(4, "little"))
                self._output.extend(seg_val.to_bytes(2, "little"))
            else:
                # 32-bit mode: 16-bit offset is default, need 0x66 for 16-bit offset
                self._output.append(0xEA)
                self._output.extend((off_val & 0xFFFFFFFF).to_bytes(4, "little"))
                self._output.extend(seg_val.to_bytes(2, "little"))
            return
        target = self._parse_label(target_str)
        if self._bits == 16:
            # Always use near jump (3 bytes) in 16-bit mode to keep
            # code sizes stable across assembly passes.
            offset = target - (self._org + len(self._output) + 3)
            self._output.append(0xE9)
            self._output.extend(offset.to_bytes(2, "little", signed=True))
        else:
            offset = target - (self._org + len(self._output) + 2)
            if -128 <= offset <= 127:
                self._output.append(0xEB)
                self._output.append(offset & 0xFF)
            else:
                offset = target - (self._org + len(self._output) + 5)
                self._output.append(0xE9)
                self._output.extend((offset & 0xFFFFFFFF).to_bytes(4, "little"))

    def _emit_call(self, ops):
        if not ops:
            return
        target = self._parse_label(ops[0])
        if self._bits == 16:
            # call rel16: E8 + 2-byte offset = 3 bytes
            offset = target - (self._org + len(self._output) + 3)
            self._output.append(0xE8)
            self._output.extend(offset.to_bytes(2, "little", signed=True))
        else:
            # call rel32: E8 + 4-byte offset = 5 bytes
            offset = target - (self._org + len(self._output) + 5)
            self._output.append(0xE8)
            self._output.extend((offset & 0xFFFFFFFF).to_bytes(4, "little"))

    def _emit_mov(self, ops):
        if len(ops) < 2:
            return
        dst, src = ops[0].lower().strip(), ops[1].strip()

        # MOV reg, reg (must check before reg, imm)
        if dst in self._SEG_REGS and src in self._REG16:
            # MOV Sreg, r/m16 — opcode 8E, ModRM with /r = segment reg
            reg_field = self._SEG_REGS[dst]
            rm_field = self._REG16[src]
            modrm = (0xC0 | (reg_field << 3) | rm_field)
            self._output.append(0x8E)
            self._output.append(modrm)
        elif dst in self._REG16 and src in self._SEG_REGS:
            # MOV r/m16, Sreg — opcode 8C, ModRM with /r = segment reg
            reg_field = self._SEG_REGS[src]
            rm_field = self._REG16[dst]
            modrm = (0xC0 | (reg_field << 3) | rm_field)
            self._output.append(0x8C)
            self._output.append(modrm)
        elif dst in self._REG32 and src in self._REG32:
            if self._pfx(dst):
                self._output.append(0x66)
            self._output.append(0x89)
            modrm = (0xC0 | (self._REG32[src] << 3) | self._REG32[dst])
            self._output.append(modrm)
        elif dst in self._REG16 and src in self._REG16:
            if self._pfx(dst):
                self._output.append(0x66)
            self._output.append(0x89)
            modrm = (0xC0 | (self._REG16[src] << 3) | self._REG16[dst])
            self._output.append(modrm)
        # MOV reg, CRn / MOV CRn, reg (control register moves)
        elif dst in self._REG32 and src in self._CTRL_REGS:
            # MOV r32, CRn — opcode 0F 20, ModRM with reg=CRn, rm=r32
            self._output.append(0x0F)
            self._output.append(0x20)
            modrm = (0xC0 | (self._CTRL_REGS[src] << 3) | self._REG32[dst])
            self._output.append(modrm)
        elif dst in self._CTRL_REGS and src in self._REG32:
            # MOV CRn, r32 — opcode 0F 22, ModRM with reg=CRn, rm=r32
            self._output.append(0x0F)
            self._output.append(0x22)
            modrm = (0xC0 | (self._CTRL_REGS[dst] << 3) | self._REG32[src])
            self._output.append(modrm)
        # MOV reg, DRn / MOV DRn, reg (debug register moves)
        elif dst in self._REG32 and src in self._DBG_REGS:
            # MOV r32, DRn — opcode 0F 21, ModRM with reg=DRn, rm=r32
            self._output.append(0x0F)
            self._output.append(0x21)
            modrm = (0xC0 | (self._DBG_REGS[src] << 3) | self._REG32[dst])
            self._output.append(modrm)
        elif dst in self._DBG_REGS and src in self._REG32:
            # MOV DRn, r32 — opcode 0F 23, ModRM with reg=DRn, rm=r32
            self._output.append(0x0F)
            self._output.append(0x23)
            modrm = (0xC0 | (self._DBG_REGS[dst] << 3) | self._REG32[src])
            self._output.append(modrm)
        # MOV [mem], reg — size-prefixed memory stores with register source
        elif dst.startswith("byte") and "[" in dst and src in self._REG8:
            mem = dst[4:].strip()
            self._output.append(0x88)
            self._emit_modrm_mem(src, mem)
        elif dst.startswith("word") and "[" in dst and src in self._REG16:
            mem = dst[4:].strip()
            if self._pfx(src):
                self._output.append(0x66)
            self._output.append(0x89)
            self._emit_modrm_mem(src, mem)
        elif dst.startswith("dword") and "[" in dst and src in self._REG32:
            mem = dst[5:].strip()
            if self._pfx(src):
                self._output.append(0x66)
            self._output.append(0x89)
            self._emit_modrm_mem(src, mem)
        # MOV [mem], imm — size-prefixed memory stores (C6/C7 /0)
        elif dst.startswith("byte") and "[" in dst:
            val = self._parse_imm(src)
            mem = dst[4:].strip()
            self._output.append(0xC6)  # MOV r/m8, imm8
            self._emit_modrm_mem("eax", mem)
            self._output.append(val & 0xFF)
        elif dst.startswith("word") and "[" in dst:
            val = self._parse_imm(src)
            mem = dst[4:].strip()
            self._output.append(0x66)  # operand-size prefix
            self._output.append(0xC7)  # MOV r/m16, imm16
            self._emit_modrm_mem("eax", mem)
            self._output.extend(struct.pack("<H", val & 0xFFFF))
        elif dst.startswith("dword") and "[" in dst:
            val = self._parse_imm(src)
            mem = dst[5:].strip()
            self._output.append(0xC7)  # MOV r/m32, imm32
            self._emit_modrm_mem("eax", mem)
            self._output.extend((val & 0xFFFFFFFF).to_bytes(4, "little"))
        # MOV reg, imm (resolve labels) — must come after [mem] checks
        elif dst in self._REG32 and not src.startswith("["):
            val = self._parse_label(src)
            if self._pfx(dst):
                self._output.append(0x66)
            self._output.append(0xB8 + self._REG32[dst])
            self._output.extend((val & 0xFFFFFFFF).to_bytes(4, "little"))
        elif dst in self._REG16 and not src.startswith("["):
            val = self._parse_label(src)
            if self._pfx(dst):
                self._output.append(0x66)
            self._output.append(0xB8 + self._REG16[dst])
            self._output.extend(struct.pack("<H", val & 0xFFFF))
        elif dst in self._REG8 and src in self._REG8:
            # MOV r8, r8 — opcode 88, ModRM
            self._output.append(0x88)
            modrm = (0xC0 | (self._REG8[src] << 3) | self._REG8[dst])
            self._output.append(modrm)
        elif dst in self._REG8 and src.startswith("["):
            self._output.append(0x8A)
            self._emit_modrm_mem(dst, src)
        elif dst in self._REG8:
            val = self._parse_imm(src)
            self._output.append(0xB0 + self._REG8[dst])
            self._output.append(val & 0xFF)
        # MOV reg, [mem] / MOV [mem], reg
        elif dst in self._REG32 and src.startswith("["):
            if self._pfx(dst):
                self._output.append(0x66)
            self._output.append(0x8B)
            self._emit_modrm_mem(dst, src)
        elif dst in self._REG16 and src.startswith("["):
            inner = src.strip("[]").strip()
            if inner.startswith("0x") or (inner.isdigit() and int(inner) > 15):
                addr = int(inner, 16) if inner.startswith("0x") else int(inner)
                self._output.append(0xA1)
                self._output.extend(struct.pack("<H", addr & 0xFFFF))
            else:
                self._output.append(0x8B)
                self._emit_modrm_mem(dst, src)
        elif dst.startswith("[") and src in self._REG32:
            if self._pfx(src):
                self._output.append(0x66)
            self._output.append(0x89)
            self._emit_modrm_mem(src, dst)
        elif dst.startswith("[") and src in self._REG16:
            inner = dst.strip("[]").strip()
            if inner.startswith("0x") or (inner.isdigit() and int(inner) > 15):
                addr = int(inner, 16) if inner.startswith("0x") else int(inner)
                self._output.append(0xA3)
                self._output.extend(struct.pack("<H", addr & 0xFFFF))
            else:
                self._output.append(0x89)
                self._emit_modrm_mem(src, dst)
        elif dst.startswith("[") and src in self._REG8:
            self._output.append(0x88)
            self._emit_modrm_mem(src, dst)
        # MOV reg, [imm] — direct address load
        elif dst in self._REG32 and (src.startswith("0x") or (src.startswith("[") and src[1:-1].strip().startswith("0x"))):
            inner = src.strip("[]") if src.startswith("[") else src
            addr = int(inner, 16) if inner.startswith("0x") else int(inner)
            self._output.append(0xA1)
            self._output.extend((addr & 0xFFFFFFFF).to_bytes(4, "little"))
        elif dst in self._REG16 and (src.startswith("0x") or (src.startswith("[") and src[1:-1].strip().startswith("0x"))):
            inner = src.strip("[]") if src.startswith("[") else src
            addr = int(inner, 16) if inner.startswith("0x") else int(inner)
            self._output.append(0xA1)
            self._output.extend(struct.pack("<H", addr & 0xFFFF))
        else:
            # Fallback: MOV reg, imm
            val = self._parse_imm(src)
            if dst in self._REG32:
                if self._pfx(dst):
                    self._output.append(0x66)
                self._output.append(0xB8 + self._REG32[dst])
                self._output.extend((val & 0xFFFFFFFF).to_bytes(4, "little"))

    def _emit_modrm_mem(self, reg, mem):
        """Emit ModR/M byte for [memory] operand."""
        inner = mem.strip("[]").strip()
        # Resolve register encoding (works for both 8-bit and 32-bit registers)
        reg_enc = self._REG32.get(reg, self._REG8.get(reg, 0))
        if inner in self._REG32:
            # [reg]
            modrm = (0x00 | (reg_enc << 3) | self._REG32[inner])
            self._output.append(modrm)
        elif "+" in inner:
            parts = inner.split("+")
            base = parts[0].strip()
            other = parts[1].strip() if len(parts) > 1 else ""
            # Check which part is a register
            if base in self._REG32:
                # [reg + imm/label]
                disp = self._parse_imm(other) if other else 0
                if disp == 0:
                    modrm = (0x00 | (reg_enc << 3) | self._REG32[base])
                    self._output.append(modrm)
                elif -128 <= disp <= 127:
                    modrm = (0x40 | (reg_enc << 3) | self._REG32[base])
                    self._output.append(modrm)
                    self._output.append(disp & 0xFF)
                else:
                    modrm = (0x80 | (reg_enc << 3) | self._REG32[base])
                    self._output.append(modrm)
                    self._output.extend((disp & 0xFFFFFFFF).to_bytes(4, "little"))
            elif other in self._REG32:
                # [reg + label] — swapped order like "input_buf + ebx"
                disp = self._parse_imm(base) if base else 0
                if disp == 0:
                    modrm = (0x00 | (reg_enc << 3) | self._REG32[other])
                    self._output.append(modrm)
                elif -128 <= disp <= 127:
                    modrm = (0x40 | (reg_enc << 3) | self._REG32[other])
                    self._output.append(modrm)
                    self._output.append(disp & 0xFF)
                else:
                    modrm = (0x80 | (reg_enc << 3) | self._REG32[other])
                    self._output.append(modrm)
                    self._output.extend((disp & 0xFFFFFFFF).to_bytes(4, "little"))
            else:
                # [label + label] — treat as direct address sum
                addr = self._parse_imm(inner)
                modrm = (0x00 | (reg_enc << 3) | 0x05)
                self._output.append(modrm)
                self._output.extend((addr & 0xFFFFFFFF).to_bytes(4, "little"))
        else:
            # Direct address [imm32]
            modrm = (0x00 | (reg_enc << 3) | 0x05)
            self._output.append(modrm)
            addr = self._parse_imm(inner)
            self._output.extend((addr & 0xFFFFFFFF).to_bytes(4, "little"))

    def _emit_alu(self, op, ops):
        if len(ops) < 2:
            return
        dst, src = ops[0].lower().strip(), ops[1].strip()
        alu_op = {"add": 0, "or": 1, "adc": 2, "sbb": 3,
                  "and": 4, "sub": 5, "xor": 6, "cmp": 7, "test": 0}

        # reg, reg
        if dst in self._REG32 and src in self._REG32:
            if op == "test":
                if self._pfx(dst):
                    self._output.append(0x66)
                self._output.append(0x85)
                modrm = (0xC0 | (self._REG32[src] << 3) | self._REG32[dst])
                self._output.append(modrm)
            else:
                if self._pfx(dst):
                    self._output.append(0x66)
                self._output.append(0x01 + alu_op[op] * 8)
                modrm = (0xC0 | (self._REG32[src] << 3) | self._REG32[dst])
                self._output.append(modrm)
        elif dst in self._REG16 and src in self._REG16:
            if op == "test":
                if self._pfx(dst):
                    self._output.append(0x66)
                self._output.append(0x85)
                modrm = (0xC0 | (self._REG16[src] << 3) | self._REG16[dst])
                self._output.append(modrm)
            else:
                if self._pfx(dst):
                    self._output.append(0x66)
                self._output.append(0x01 + alu_op[op] * 8)
                modrm = (0xC0 | (self._REG16[src] << 3) | self._REG16[dst])
                self._output.append(modrm)
        elif dst in self._REG8 and src in self._REG8:
            if op == "test":
                self._output.append(0x84)
                modrm = (0xC0 | (self._REG8[src] << 3) | self._REG8[dst])
                self._output.append(modrm)
            else:
                self._output.append(0x00 + alu_op[op] * 8)
                modrm = (0xC0 | (self._REG8[src] << 3) | self._REG8[dst])
                self._output.append(modrm)
        # reg, imm
        elif dst in self._REG32:
            val = self._parse_imm(src)
            if op == "test":
                # TEST r/m32, imm32 — F7 /0 id
                if self._pfx(dst):
                    self._output.append(0x66)
                self._output.append(0xF7)
                modrm = (0xC0 | (0 << 3) | self._REG32[dst])
                self._output.append(modrm)
                self._output.extend((val & 0xFFFFFFFF).to_bytes(4, "little"))
            elif op == "sub" and -128 <= val <= 127:
                if self._pfx(dst):
                    self._output.append(0x66)
                self._output.append(0x83)
                modrm = (0xE8 | self._REG32[dst])
                self._output.append(modrm)
                self._output.append(val & 0xFF)
            elif -128 <= val <= 127:
                if self._pfx(dst):
                    self._output.append(0x66)
                self._output.append(0x83)
                modrm = (0xC0 | (alu_op[op] << 3) | self._REG32[dst])
                self._output.append(modrm)
                self._output.append(val & 0xFF)
            else:
                if self._pfx(dst):
                    self._output.append(0x66)
                self._output.append(0x81)
                modrm = (0xC0 | (alu_op[op] << 3) | self._REG32[dst])
                self._output.append(modrm)
                self._output.extend((val & 0xFFFFFFFF).to_bytes(4, "little"))
        elif dst in self._REG16:
            val = self._parse_imm(src)
            if op == "test":
                # TEST r/m16, imm16 — F7 /0 iw
                if self._pfx(dst):
                    self._output.append(0x66)
                self._output.append(0xF7)
                modrm = (0xC0 | (0 << 3) | self._REG16[dst])
                self._output.append(modrm)
                self._output.extend(struct.pack("<H", val & 0xFFFF))
            elif -128 <= val <= 127:
                if self._pfx(dst):
                    self._output.append(0x66)
                self._output.append(0x83)
                modrm = (0xC0 | (alu_op[op] << 3) | self._REG16[dst])
                self._output.append(modrm)
                self._output.append(val & 0xFF)
            else:
                if self._pfx(dst):
                    self._output.append(0x66)
                self._output.append(0x81)
                modrm = (0xC0 | (alu_op[op] << 3) | self._REG16[dst])
                self._output.append(modrm)
                self._output.extend(val.to_bytes(2, "little", signed=True))
        elif dst in self._REG8:
            val = self._parse_imm(src)
            if op == "test":
                # TEST r/m8, imm8 — F6 /0 ib
                self._output.append(0xF6)
                modrm = (0xC0 | (0 << 3) | self._REG8[dst])
                self._output.append(modrm)
                self._output.append(val & 0xFF)
            elif -128 <= val <= 127:
                self._output.append(0x80)
                modrm = (0xC0 | (alu_op[op] << 3) | self._REG8[dst])
                self._output.append(modrm)
                self._output.append(val & 0xFF)
            else:
                self._output.append(0x81)
                modrm = (0xC0 | (alu_op[op] << 3) | self._REG8[dst])
                self._output.append(modrm)
                self._output.append(val & 0xFF)
        # [mem], reg or [mem], imm — ALU r/m32, r/imm
        elif "[" in dst:
            mem = dst
            for pfx in ("dword", "word", "byte"):
                if mem.startswith(pfx):
                    mem = mem[len(pfx):].strip()
            # Resolve memory operand: returns (modrm_rm, disp_bytes)
            inner = mem.strip("[]").strip()
            if inner in self._REG32:
                modrm_rm = self._REG32[inner]
                disp_bytes = b""
            elif "+" in inner:
                parts = inner.split("+")
                base = parts[0].strip()
                other = parts[1].strip() if len(parts) > 1 else ""
                if base in self._REG32:
                    modrm_rm = self._REG32[base]
                    disp = self._parse_imm(other) if other else 0
                    if disp != 0:
                        if -128 <= disp <= 127:
                            disp_bytes = bytes([disp & 0xFF])
                        else:
                            disp_bytes = (disp & 0xFFFFFFFF).to_bytes(4, "little")
                    else:
                        disp_bytes = b""
                elif other in self._REG32:
                    modrm_rm = self._REG32[other]
                    disp = self._parse_imm(base) if base else 0
                    if disp != 0:
                        if -128 <= disp <= 127:
                            disp_bytes = bytes([disp & 0xFF])
                        else:
                            disp_bytes = (disp & 0xFFFFFFFF).to_bytes(4, "little")
                    else:
                        disp_bytes = b""
                else:
                    modrm_rm = 5
                    addr = self._parse_imm(inner)
                    disp_bytes = (addr & 0xFFFFFFFF).to_bytes(4, "little")
            else:
                modrm_rm = 5
                addr = self._parse_imm(inner)
                disp_bytes = (addr & 0xFFFFFFFF).to_bytes(4, "little")
            # [mem], reg — ALU r/m32, r32
            if src in self._REG32:
                self._output.append(0x01 + alu_op[op] * 8)
                self._output.append(0x00 | (self._REG32[src] << 3) | modrm_rm)
                self._output.extend(disp_bytes)
            elif src in self._REG16:
                self._output.append(0x66)
                self._output.append(0x01 + alu_op[op] * 8)
                self._output.append(0x00 | (self._REG16[src] << 3) | modrm_rm)
                self._output.extend(disp_bytes)
            elif src in self._REG8:
                self._output.append(0x00 + alu_op[op] * 8)
                self._output.append(0x00 | (self._REG8[src] << 3) | modrm_rm)
                self._output.extend(disp_bytes)
            else:
                # [mem], imm — ALU r/m32, imm32
                val = self._parse_imm(src)
                if op == "test":
                    self._output.append(0xF7)
                    self._output.append(0x00 | (0 << 3) | modrm_rm)
                    self._output.extend(disp_bytes)
                    self._output.extend((val & 0xFFFFFFFF).to_bytes(4, "little"))
                elif -128 <= val <= 127:
                    self._output.append(0x83)
                    self._output.append(0x00 | (alu_op[op] << 3) | modrm_rm)
                    self._output.extend(disp_bytes)
                    self._output.append(val & 0xFF)
                else:
                    self._output.append(0x81)
                    self._output.append(0x00 | (alu_op[op] << 3) | modrm_rm)
                    self._output.extend(disp_bytes)
                    self._output.extend((val & 0xFFFFFFFF).to_bytes(4, "little"))

    def _emit_inc_dec(self, op, ops):
        if not ops:
            return
        reg = ops[0].lower()
        # inc/dec register — 40+rd / 48+rd
        if reg in self._REG32:
            if self._pfx(reg):
                self._output.append(0x66)
            base = 0x40 if op == "inc" else 0x48
            self._output.append(base + self._REG32[reg])
        elif reg in self._REG16:
            if self._pfx(reg):
                self._output.append(0x66)
            base = 0x40 if op == "inc" else 0x48
            self._output.append(base + self._REG16[reg])
        # inc/dec dword [mem] / word [mem] — FF /0 or FF /1
        elif "[" in reg:
            func = 0 if op == "inc" else 1
            mem = reg
            # strip size prefix
            for pfx in ("dword", "word", "byte"):
                if mem.startswith(pfx):
                    mem = mem[len(pfx):].strip()
            self._output.append(0xFF)
            modrm = (0x00 | (func << 3) | 0x05)  # /0 or /1, [disp32]
            self._output.append(modrm)
            addr = self._parse_imm(mem.strip("[]"))
            self._output.extend((addr & 0xFFFFFFFF).to_bytes(4, "little"))

    def _emit_unary(self, op, ops):
        if not ops:
            return
        reg = ops[0].lower()
        func = {"not": 2, "neg": 3, "mul": 4, "imul": 5, "div": 6, "idiv": 7}.get(op, 2)
        if reg in self._REG32:
            if self._pfx(reg):
                self._output.append(0x66)
            self._output.append(0xF7)
            modrm = (0xC0 | (func << 3) | self._REG32[reg])
            self._output.append(modrm)
        elif reg in self._REG16:
            if self._pfx(reg):
                self._output.append(0x66)
            self._output.append(0xF7)
            modrm = (0xC0 | (func << 3) | self._REG16[reg])
            self._output.append(modrm)
        elif reg in self._REG8:
            self._output.append(0xF6)
            modrm = (0xC0 | (func << 3) | self._REG8[reg])
            self._output.append(modrm)

    def _emit_shift(self, op, ops):
        if len(ops) < 2:
            return
        reg = ops[0].lower()
        count = ops[1].strip()
        # /0=ROL, /1=ROR, /2=RCL, /3=RCR, /4=SHL, /5=SHR, /7=SAR
        ext_map = {"rol": 0, "ror": 1, "rcl": 2, "rcr": 3,
                   "shl": 4, "sal": 4, "shr": 5, "sar": 7}
        ext = ext_map.get(op, 4)
        if reg in self._REG32:
            if self._pfx(reg):
                self._output.append(0x66)
            if count == "cl":
                self._output.append(0xD3)
                modrm = 0xC0 | (ext << 3) | self._REG32[reg]
                self._output.append(modrm)
            else:
                val = self._parse_imm(count)
                self._output.append(0xC1)
                modrm = 0xC0 | (ext << 3) | self._REG32[reg]
                self._output.append(modrm)
                self._output.append(val & 0xFF)
        elif reg in self._REG16:
            if self._pfx(reg):
                self._output.append(0x66)
            if count == "cl":
                self._output.append(0xD3)
                modrm = 0xC0 | (ext << 3) | self._REG16[reg]
                self._output.append(modrm)
            else:
                val = self._parse_imm(count)
                self._output.append(0xC1)
                modrm = 0xC0 | (ext << 3) | self._REG16[reg]
                self._output.append(modrm)
                self._output.append(val & 0xFF)
        elif reg in self._REG8:
            if count == "cl":
                self._output.append(0xD2)
                modrm = 0xC0 | (ext << 3) | self._REG8[reg]
                self._output.append(modrm)
            else:
                val = self._parse_imm(count)
                self._output.append(0xC0)
                modrm = 0xC0 | (ext << 3) | self._REG8[reg]
                self._output.append(modrm)
                self._output.append(val & 0xFF)
        elif "[" in reg:
            # [mem], count
            mem = reg
            for pfx in ("dword", "word", "byte"):
                if mem.startswith(pfx):
                    mem = mem[len(pfx):].strip()
            inner = mem.strip("[]").strip()
            if inner in self._REG32:
                modrm_rm = self._REG32[inner]
                disp_bytes = b""
            elif "+" in inner:
                parts = inner.split("+")
                base = parts[0].strip()
                other = parts[1].strip() if len(parts) > 1 else ""
                if base in self._REG32:
                    modrm_rm = self._REG32[base]
                    disp = self._parse_imm(other) if other else 0
                    if disp != 0:
                        if -128 <= disp <= 127:
                            disp_bytes = bytes([disp & 0xFF])
                        else:
                            disp_bytes = (disp & 0xFFFFFFFF).to_bytes(4, "little")
                    else:
                        disp_bytes = b""
                else:
                    modrm_rm = 5
                    addr = self._parse_imm(inner)
                    disp_bytes = (addr & 0xFFFFFFFF).to_bytes(4, "little")
            else:
                modrm_rm = 5
                addr = self._parse_imm(inner)
                disp_bytes = (addr & 0xFFFFFFFF).to_bytes(4, "little")
            # Determine if byte or dword based on prefix
            is_byte = "byte" in reg.lower()
            if is_byte:
                if count == "cl":
                    self._output.append(0xD2)
                else:
                    self._output.append(0xC0)
                    val = self._parse_imm(count)
                self._output.append(0x00 | (ext << 3) | modrm_rm)
                self._output.extend(disp_bytes)
                if count != "cl":
                    self._output.append(val & 0xFF)
            else:
                if count == "cl":
                    self._output.append(0xD3)
                else:
                    self._output.append(0xC1)
                    val = self._parse_imm(count)
                self._output.append(0x00 | (ext << 3) | modrm_rm)
                self._output.extend(disp_bytes)
                if count != "cl":
                    self._output.append(val & 0xFF)

    def _emit_lea(self, ops):
        if len(ops) < 2:
            return
        dst = ops[0].lower()
        src = ops[1].strip()
        if dst in self._REG32 and src.startswith("["):
            self._output.append(0x8D)
            self._emit_modrm_mem(dst, src)

    def _emit_xchg(self, ops):
        if len(ops) < 2:
            return
        r1, r2 = ops[0].lower(), ops[1].lower()
        if r1 in self._REG32 and r2 in self._REG32:
            self._output.append(0x87)
            modrm = (0xC0 | (self._REG32[r1] << 3) | self._REG32[r2])
            self._output.append(modrm)

    def _emit_lgdt(self, ops):
        if not ops:
            return
        # LGDT [addr] — 0F 01 /2 with ModR/M for [disp32]
        # In 16-bit mode, r/m=101 with mod=00 means [DI], not [disp32].
        # We must emit 0x67 address-size prefix to use 32-bit addressing
        # so that r/m=101 with mod=00 means [disp32].
        if self._bits == 16:
            self._output.append(0x67)
        self._output.append(0x0F)
        self._output.append(0x01)
        self._output.append(0x15)
        # Parse the memory operand [label] → emit 32-bit displacement
        addr_str = ops[0].strip()
        if addr_str.startswith("[") and addr_str.endswith("]"):
            addr_str = addr_str[1:-1]
        addr = self._parse_label(addr_str)
        self._output.extend((addr & 0xFFFFFFFF).to_bytes(4, "little"))

    def _emit_lidt(self, ops):
        if not ops:
            return
        # LIDT [addr] — 0F 01 /3 with ModR/M for [disp32]
        # Same as LGDT: need 0x67 prefix in 16-bit mode for [disp32] addressing.
        if self._bits == 16:
            self._output.append(0x67)
        self._output.append(0x0F)
        self._output.append(0x01)
        self._output.append(0x1D)
        addr_str = ops[0].strip()
        if addr_str.startswith("[") and addr_str.endswith("]"):
            addr_str = addr_str[1:-1]
        addr = self._parse_label(addr_str)
        self._output.extend((addr & 0xFFFFFFFF).to_bytes(4, "little"))

    def _emit_ltr(self, ops):
        if not ops:
            return
        reg = ops[0].lower()
        if reg in self._REG16:
            self._output.append(0x0F)
            self._output.append(0x00)
            self._output.append(0xD8 | self._REG16[reg])

    def _parse_label(self, text):
        text = text.strip()
        if text == "$":
            return self._org + len(self._output)
        if text in self._labels:
            return self._labels[text]
        # Always resolve numeric literals (hex, binary, decimal) even in pass 1
        if text.startswith("0x") or text.startswith("0X"):
            return int(text, 16)
        if text.startswith("0b") or text.startswith("0B"):
            return int(text, 2)
        # Only treat as hex literal if the prefix is purely hex digits (e.g. "10h", "FFh")
        import re
        if re.match(r'^[0-9a-fA-F]+[hH]$', text):
            return int(text[:-1], 16)
        if re.match(r'^[01]+[bB]$', text):
            return int(text[:-1], 2)
        try:
            return int(text, 0)
        except ValueError:
            pass
        # In pass 1, forward references to labels use placeholder (0)
        if getattr(self, '_pass', 2) == 1:
            return 0
        return self._parse_imm(text)

    def _eval_expr(self, text):
        """Evaluate a simple arithmetic expression with +, -, *, /, parentheses."""
        text = text.strip()
        # Replace hex literals
        import re
        text = re.sub(r'0[xX]([0-9a-fA-F]+)', lambda m: str(int(m.group(1), 16)), text)
        text = re.sub(r'0[bB]([01]+)', lambda m: str(int(m.group(1), 2)), text)
        text = re.sub(r'([0-9]+)[hH]', lambda m: str(int(m.group(1), 16)), text)
        # Only allow digits, operators, parentheses, spaces
        if re.match(r'^[\d\s\+\-\*\/\(\)]+$', text):
            try:
                return int(eval(text))
            except:
                return 0
        return 0

    def _parse_imm(self, text):
        text = text.strip()
        # Character literals: 'A', '0', '\n', etc.
        if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
            inner = text[1:-1]
            if len(inner) == 1:
                return ord(inner)
            elif inner == '\\n':
                return 10
            elif inner == '\\r':
                return 13
            elif inner == '\\t':
                return 9
            elif inner == '\\0':
                return 0
            elif inner == '\\\\':
                return 92
            elif inner.startswith('\\x'):
                return int(inner[2:], 16)
            return ord(inner[0]) if inner else 0
        if text.startswith("0x") or text.startswith("0X"):
            return int(text, 16)
        if text.startswith("0b") or text.startswith("0B"):
            return int(text, 2)
        if text.endswith("h") or text.endswith("H"):
            return int(text[:-1], 16)
        if text.endswith("b") or text.endswith("B"):
            return int(text[:-1], 2)
        try:
            return int(text, 0)
        except ValueError:
            pass
        # Try direct label lookup (no recursion through _parse_label)
        if text == "$":
            return self._org + len(self._output)
        if text in self._labels:
            return self._labels[text]
        # Try arithmetic expression with label references
        import re
        expr = text
        for name, addr in self._labels.items():
            expr = re.sub(r'\b' + re.escape(name) + r'\b', str(addr), expr)
        if re.match(r'^[\d\s\+\-\*\/\(\)]+$', expr):
            try:
                return int(eval(expr))
            except Exception:
                pass
        return 0

    def _split_ops(self, text):
        if not text:
            return []
        result = []
        depth = 0
        current = ""
        for ch in text:
            if ch == "[":
                depth += 1
                current += ch
            elif ch == "]":
                depth -= 1
                current += ch
            elif ch == "," and depth == 0:
                result.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            result.append(current.strip())
        return result

    def run(self, source: str, org: int = 0, max_steps: int = 100000,
            memory_size: int = 1024 * 1024, **kwargs) -> "X86CPU":
        """Assemble, load, and run source on an X86CPU. Returns the CPU."""
        code = self.assemble(source)
        cpu = X86CPU(memory_size=memory_size)
        cpu.load(code, org)
        cpu.run(max_steps=max_steps, **kwargs)
        return cpu


# ── x86 CPU Simulator ───────────────────────────────────────────────────────

# EFLAGS bit positions
FLAG_CF = 0x0001
FLAG_PF = 0x0004
FLAG_ZF = 0x0040
FLAG_SF = 0x0080
FLAG_OF = 0x0800
FLAG_DF = 0x0400
FLAG_IF = 0x0200

# Register indices
_REG_I32 = {"eax": 0, "ecx": 1, "edx": 2, "ebx": 3,
            "esp": 4, "ebp": 5, "esi": 6, "edi": 7}
_REG_I16 = {"ax": 0, "cx": 1, "dx": 2, "bx": 3,
            "sp": 4, "bp": 5, "si": 6, "di": 7}
_REG_I8L = {"al": 0, "cl": 1, "dl": 2, "bl": 3}
_REG_I8H = {"ah": 4, "ch": 5, "dh": 6, "bh": 7}


def _parity(v: int) -> bool:
    """Return True if low 8 bits of v have even parity."""
    return bin(v & 0xFF).count("1") % 2 == 0


# PS/2 scancode set 1 lookup (make codes only, no break codes)
_CHAR_TO_SCANCODE = {
    'a': 0x1E, 'b': 0x30, 'c': 0x2E, 'd': 0x20, 'e': 0x12, 'f': 0x21,
    'g': 0x22, 'h': 0x23, 'i': 0x17, 'j': 0x24, 'k': 0x25, 'l': 0x26,
    'm': 0x32, 'n': 0x31, 'o': 0x18, 'p': 0x19, 'q': 0x10, 'r': 0x13,
    's': 0x1F, 't': 0x14, 'u': 0x16, 'v': 0x2F, 'w': 0x11, 'x': 0x2D,
    'y': 0x15, 'z': 0x2C,
    '0': 0x0B, '1': 0x02, '2': 0x03, '3': 0x04, '4': 0x05,
    '5': 0x06, '6': 0x07, '7': 0x08, '8': 0x09, '9': 0x0A,
    ' ': 0x39, '\n': 0x1C, '\r': 0x1C, '\t': 0x0F,
    '-': 0x0C, '=': 0x0D, '[': 0x1A, ']': 0x1B, '\\': 0x2B,
    ';': 0x27, "'": 0x28, ',': 0x33, '.': 0x34, '/': 0x35,
    '`': 0x29,
}

_SCANDATA = [
    0x00, 0x1B, '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 0x08, 0x09,
    'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', 0x0D, 0x00,
    'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', "'", '`', 0x00, '\\',
    'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/', 0x00, '*', 0x00, ' ',
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    '7', '8', '9', '-', '4', '5', '6', '+', '1', '2', '3', '0', '.',
]


def _char_to_scancode(char: str) -> int:
    """Convert a character to PS/2 set-1 make scancode. Returns -1 if unknown."""
    return _CHAR_TO_SCANCODE.get(char.lower(), -1)


def _scancode_to_char(sc: int) -> str:
    """Convert a PS/2 set-1 make scancode to character. Returns 0 if unknown."""
    if 0 <= sc < len(_SCANDATA):
        ch = _SCANDATA[sc]
        if isinstance(ch, str):
            return ch
        if isinstance(ch, int) and ch != 0:
            return chr(ch)
    return '\0'


def _default_kbd_handler(cpu):
    """Default IRQ1 handler: reads scancode from buffer, stores ASCII in [0x400]."""
    if cpu._kbd_buffer:
        sc = cpu._kbd_buffer.pop(0)
        ch = _scancode_to_char(sc)
        if ch and ch != '\0':
            cpu._mem[0x400] = ord(ch)  # keyboard buffer at 0x400
            cpu._mem[0x401] = sc       # scancode at 0x401


class X86CPU:
    """32-bit x86 CPU simulator — flat memory model, ring 0 only.

    Designed to run assembled X86Assembler output. The CPU reads bytes
    from a flat address space, decodes x86 instructions, and executes
    them, maintaining full register and flag state.

    Usage::

        cpu = X86CPU()
        code = assembler.assemble(source)
        cpu.load(code, org=0x1000)
        cpu.run(max_steps=10000)
        print(cpu.reg_dump())        # register snapshot
        print(hex(cpu.eax))          # individual register
        print(cpu.eflags_str())      # flags display
    """

    def __init__(self, memory_size: int = 1024 * 1024):
        self._mem = bytearray(memory_size)
        self._mem_size = memory_size
        self.reset()

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self):
        """Zero all registers, flags, and memory."""
        self._regs = [0] * 8          # EAX..EDI
        self._regs[4] = self._mem_size - 4  # ESP starts at top of memory
        self._eip = 0
        self._eflags = 0x0002         # Bit 1 always set per x86 spec
        self._segregs = [0] * 6       # ES, CS, SS, DS, FS, GS
        self._cr = [0] * 8            # CR0..CR7 (control registers)
        self._dr = [0] * 8            # DR0..DR7 (debug registers)
        self._running = False
        self._step_count = 0
        self._max_steps = 1_000_000
        self._io_in: dict[int, callable] = {}
        self._io_out: dict[int, callable] = {}
        self._mem[:] = bytearray(len(self._mem))
        # IDT / Interrupt support
        self._idt_base = 0
        self._idt_limit = 0
        self._idt_handlers: dict[int, callable] = {}  # int_num → handler(cpu)
        self._irq_pending: list[int] = []
        self._gdt_base = 0
        self._gdt_limit = 0
        # Keyboard buffer (PS/2 scancode queue fed from Python side)
        self._kbd_buffer: list[int] = []
        # Auto-register default keyboard handler (IRQ1)
        self._idt_handlers[1] = _default_kbd_handler

    # ── Interrupt support ──────────────────────────────────────────────────

    def register_handler(self, int_num: int, handler):
        """Register a Python callback for interrupt int_num. handler(cpu)."""
        self._idt_handlers[int_num] = handler

    def fire_irq(self, irq: int):
        """Queue a hardware IRQ. Will be dispatched when IF=1 between instructions."""
        self._irq_pending.append(irq)

    def push_key(self, char: str):
        """Push a character into the keyboard buffer. Does NOT fire IRQ."""
        scancode = _char_to_scancode(char)
        if scancode >= 0:
            self._kbd_buffer.append(scancode)

    def push_scancode(self, scancode: int):
        """Push a raw PS/2 scancode into the keyboard buffer. Does NOT fire IRQ."""
        self._kbd_buffer.append(scancode)

    def transfer_key(self):
        """Transfer one key from _kbd_buffer to [0x400] if buffer is empty.

        Call this between CPU steps to feed keyboard input to the polling loop.
        """
        if self._mem[0x400] == 0 and self._kbd_buffer:
            sc = self._kbd_buffer.pop(0)
            ch = _scancode_to_char(sc)
            if ch and ch != '\0':
                self._mem[0x400] = ord(ch)
                self._mem[0x401] = sc

    def _raise_interrupt(self, int_num: int):
        """Dispatch interrupt int_num — save state, jump to handler."""
        # Push EFLAGS, CS, EIP (like real x86 INT)
        self._push32(self._eflags)
        self._push32(0)  # CS (flat model, always 0)
        self._push32(self._eip)
        # Save IF state and clear IF on hardware interrupts (IRQs 0-15)
        saved_if = self._flag(FLAG_IF)
        if int_num < 16:
            self._set_flag(FLAG_IF, False)
        # Dispatch
        handler = self._idt_handlers.get(int_num)
        if handler:
            handler(self)
        # Simulate IRET: restore IF to pre-interrupt state
        if int_num < 16:
            self._set_flag(FLAG_IF, saved_if)

    def _check_pending_irqs(self):
        """Dispatch pending hardware IRQs if IF is set."""
        if not self._flag(FLAG_IF):
            return
        while self._irq_pending:
            irq = self._irq_pending.pop(0)
            self._raise_interrupt(irq)

    # ── Register access ──────────────────────────────────────────────────

    def _get32(self, idx: int) -> int:
        return self._regs[idx] & 0xFFFFFFFF

    def _set32(self, idx: int, val: int):
        self._regs[idx] = val & 0xFFFFFFFF

    def _get16(self, idx: int) -> int:
        return self._regs[idx] & 0xFFFF

    def _set16(self, idx: int, val: int):
        self._regs[idx] = (self._regs[idx] & 0xFFFF0000) | (val & 0xFFFF)

    def _get8l(self, idx: int) -> int:
        return self._regs[idx] & 0xFF

    def _set8l(self, idx: int, val: int):
        self._regs[idx] = (self._regs[idx] & 0xFFFFFF00) | (val & 0xFF)

    def _get8h(self, idx: int) -> int:
        return (self._regs[idx] >> 8) & 0xFF

    def _set8h(self, idx: int, val: int):
        self._regs[idx] = (self._regs[idx] & 0xFFFF00FF) | ((val & 0xFF) << 8)

    _SEG_REGS = {"es": 0, "cs": 1, "ss": 2, "ds": 3, "fs": 4, "gs": 5}

    def _get_seg(self, idx: int) -> int:
        return self._segregs[idx] & 0xFFFF

    def _set_seg(self, idx: int, val: int):
        self._segregs[idx] = val & 0xFFFF

    def _reg_index(self, name: str) -> int:
        """Resolve register name to index. Supports 32/16/8-bit names."""
        n = name.lower()
        if n in _REG_I32:
            return _REG_I32[n]
        if n in _REG_I16:
            return _REG_I16[n]
        if n in _REG_I8L:
            return _REG_I8L[n]
        if n in _REG_I8H:
            return _REG_I8H[n]
        raise ValueError(f"unknown register: {name}")

    def _read_reg(self, name: str) -> int:
        n = name.lower()
        if n in _REG_I32:
            return self._get32(_REG_I32[n])
        if n in _REG_I16:
            return self._get16(_REG_I16[n])
        if n in _REG_I8L:
            return self._get8l(_REG_I8L[n])
        if n in _REG_I8H:
            return self._get8h(_REG_I8H[n])
        raise ValueError(f"unknown register: {name}")

    def _write_reg(self, name: str, val: int):
        n = name.lower()
        if n in _REG_I32:
            self._set32(_REG_I32[n], val)
        elif n in _REG_I16:
            self._set16(_REG_I16[n], val)
        elif n in _REG_I8L:
            self._set8l(_REG_I8L[n], val)
        elif n in _REG_I8H:
            self._set8h(_REG_I8H[n], val)
        else:
            raise ValueError(f"unknown register: {name}")

    @property
    def eax(self) -> int: return self._get32(0)
    @eax.setter
    def eax(self, v: int): self._set32(0, v)
    @property
    def ecx(self) -> int: return self._get32(1)
    @ecx.setter
    def ecx(self, v: int): self._set32(1, v)
    @property
    def edx(self) -> int: return self._get32(2)
    @edx.setter
    def edx(self, v: int): self._set32(2, v)
    @property
    def ebx(self) -> int: return self._get32(3)
    @ebx.setter
    def ebx(self, v: int): self._set32(3, v)
    @property
    def esp(self) -> int: return self._get32(4)
    @esp.setter
    def esp(self, v: int): self._set32(4, v)
    @property
    def ebp(self) -> int: return self._get32(5)
    @ebp.setter
    def ebp(self, v: int): self._set32(5, v)
    @property
    def esi(self) -> int: return self._get32(6)
    @esi.setter
    def esi(self, v: int): self._set32(6, v)
    @property
    def edi(self) -> int: return self._get32(7)
    @edi.setter
    def edi(self, v: int): self._set32(7, v)
    @property
    def eip(self) -> int: return self._eip
    @eip.setter
    def eip(self, v: int): self._eip = v & 0xFFFFFFFF
    @property
    def esp_val(self) -> int: return self._get32(4)

    # ── EFLAGS ───────────────────────────────────────────────────────────

    def _flag(self, mask: int) -> bool:
        return bool(self._eflags & mask)

    def _set_flag(self, mask: int, val: bool):
        if val:
            self._eflags |= mask
        else:
            self._eflags &= ~mask

    @property
    def cf(self) -> bool: return self._flag(FLAG_CF)
    @property
    def zf(self) -> bool: return self._flag(FLAG_ZF)
    @property
    def sf(self) -> bool: return self._flag(FLAG_SF)
    @property
    def of(self) -> bool: return self._flag(FLAG_OF)
    @property
    def df(self) -> bool: return self._flag(FLAG_DF)
    @property
    def if_(self) -> bool: return self._flag(FLAG_IF)

    def _update_flags_add(self, a: int, b: int, result: int, bits: int = 32):
        mask = (1 << bits) - 1
        sign = bits - 1
        r = result & mask
        self._set_flag(FLAG_CF, result > mask)
        self._set_flag(FLAG_ZF, r == 0)
        self._set_flag(FLAG_SF, bool(r & (1 << sign)))
        self._set_flag(FLAG_OF, bool(((a ^ r) & (b ^ r)) & (1 << sign)))
        self._set_flag(FLAG_PF, _parity(r))

    def _update_flags_sub(self, a: int, b: int, result: int, bits: int = 32):
        mask = (1 << bits) - 1
        sign = bits - 1
        r = result & mask
        self._set_flag(FLAG_CF, a < b)
        self._set_flag(FLAG_ZF, r == 0)
        self._set_flag(FLAG_SF, bool(r & (1 << sign)))
        self._set_flag(FLAG_OF, bool(((a ^ b) & (a ^ r)) & (1 << sign)))
        self._set_flag(FLAG_PF, _parity(r))

    def _update_flags_logic(self, result: int, bits: int = 32):
        mask = (1 << bits) - 1
        sign = bits - 1
        r = result & mask
        self._set_flag(FLAG_CF, False)
        self._set_flag(FLAG_OF, False)
        self._set_flag(FLAG_ZF, r == 0)
        self._set_flag(FLAG_SF, bool(r & (1 << sign)))
        self._set_flag(FLAG_PF, _parity(r))

    # ── Memory access ────────────────────────────────────────────────────

    def _read8(self, addr: int) -> int:
        return self._mem[addr & 0xFFFFFFFF]

    def _write8(self, addr: int, val: int):
        self._mem[addr & 0xFFFFFFFF] = val & 0xFF

    def _read32(self, addr: int) -> int:
        a = addr & 0xFFFFFFFF
        return struct.unpack_from("<I", self._mem, a)[0]

    def _write32(self, addr: int, val: int):
        a = addr & 0xFFFFFFFF
        struct.pack_into("<I", self._mem, a, val & 0xFFFFFFFF)

    def _push32(self, val: int):
        self._set32(4, self._get32(4) - 4)
        self._write32(self._get32(4), val)

    def _pop32(self) -> int:
        val = self._read32(self._get32(4))
        self._set32(4, self._get32(4) + 4)
        return val

    # ── ModRM decoder ────────────────────────────────────────────────────

    def _fetch_byte(self) -> int:
        v = self._read8(self._eip)
        self._eip = (self._eip + 1) & 0xFFFFFFFF
        return v

    def _fetch_word(self) -> int:
        v = struct.unpack_from("<H", self._mem, self._eip)[0]
        self._eip = (self._eip + 2) & 0xFFFFFFFF
        return v

    def _fetch_dword(self) -> int:
        v = struct.unpack_from("<I", self._mem, self._eip)[0]
        self._eip = (self._eip + 4) & 0xFFFFFFFF
        return v

    def _decode_modrm(self):
        """Decode ModR/M byte + optional SIB + displacement.

        Returns (reg_field, rm_is_reg, rm_index_or_addr).
        """
        modrm = self._fetch_byte()
        mod = (modrm >> 6) & 3
        reg = (modrm >> 3) & 7
        rm = modrm & 7

        rm_is_reg = (mod == 3)
        if rm_is_reg:
            return reg, True, rm

        # Memory operand — resolve effective address
        if rm == 4:
            # SIB byte follows
            sib = self._fetch_byte()
            sib_scale = (sib >> 6) & 3
            sib_index = (sib >> 3) & 7
            sib_base = sib & 7
            base_val = self._regs[sib_base] if sib_base != 5 or mod != 0 else 0
            if mod == 0 and sib_base == 5:
                base_val = self._fetch_dword()
            index_val = 0
            if sib_index != 4:
                index_val = self._regs[sib_index] << sib_scale
            addr = (base_val + index_val) & 0xFFFFFFFF
        elif rm == 5 and mod == 0:
            addr = self._fetch_dword()
        else:
            addr = self._regs[rm]

        if mod == 1:
            disp = self._fetch_byte()
            if disp & 0x80:
                disp |= 0xFFFFFF00
            addr = (addr + disp) & 0xFFFFFFFF
        elif mod == 2:
            disp = self._fetch_dword()
            addr = (addr + disp) & 0xFFFFFFFF

        return reg, False, addr

    def _resolve_rm(self, modrm_reg, rm_is_reg, rm_val, width):
        """Read or write the r/m operand."""
        if rm_is_reg:
            return self._read_rm_reg(rm_val, width)
        return self._read_rm_mem(rm_val, width)

    def _read_rm_reg(self, idx, width):
        if width == 32: return self._get32(idx)
        if width == 16: return self._get16(idx)
        if width == 8 and idx < 4: return self._get8l(idx)
        if width == 8: return self._get8h(idx - 4)
        return 0

    def _write_rm_reg(self, idx, width, val):
        if width == 32: self._set32(idx, val)
        elif width == 16: self._set16(idx, val)
        elif width == 8 and idx < 4: self._set8l(idx, val)
        elif width == 8: self._set8h(idx - 4, val)

    def _read_rm_mem(self, addr, width):
        if width == 32: return self._read32(addr)
        if width == 16: return self._read16(addr)
        return self._read8(addr)

    def _write_rm_mem(self, addr, width, val):
        if width == 32: self._write32(addr, val)
        elif width == 16:
            struct.pack_into("<H", self._mem, addr & 0xFFFFFFFF, val & 0xFFFF)
        else: self._write8(addr, val)

    def _read16(self, addr: int) -> int:
        return struct.unpack_from("<H", self._mem, addr & 0xFFFFFFFF)[0]

    # ── IO ports ─────────────────────────────────────────────────────────

    def _port_in(self, port: int) -> int:
        handler = self._io_in.get(port)
        if handler:
            return handler()
        return 0xFF

    def _port_out(self, port: int, val: int):
        handler = self._io_out.get(port)
        if handler:
            handler(val)

    def register_io_in(self, port: int, fn: callable):
        self._io_in[port] = fn

    def register_io_out(self, port: int, fn: callable):
        self._io_out[port] = fn

    # ── Load ─────────────────────────────────────────────────────────────

    def load(self, code: bytes, org: int = 0):
        """Load binary code into memory at address org."""
        end = org + len(code)
        if end > self._mem_size:
            raise ValueError(f"code overflows memory: {end} > {self._mem_size}")
        self._mem[org:end] = code
        self._eip = org

    # ── Execute ──────────────────────────────────────────────────────────

    def step(self) -> bool:
        """Execute one instruction. Returns False on HLT or error."""
        self._check_pending_irqs()
        start_eip = self._eip
        try:
            self._exec_one()
        except Halt:
            return False
        except Exception as e:
            logger.error(f"CPU fault at EIP=0x{start_eip:X}: {e}")
            return False
        self._step_count += 1
        return True

    def run(self, max_steps: int = 0) -> int:
        """Run until HLT or max_steps. Returns steps executed."""
        limit = max_steps or self._max_steps
        self._running = True
        while self._running and self._step_count < limit:
            if not self.step():
                break
        return self._step_count

    def _exec_one(self):
        """Fetch and execute one instruction."""
        opcode = self._fetch_byte()

        # ── NOP ──
        if opcode == 0x90:
            return

        # ── HLT ──
        if opcode == 0xF4:
            raise Halt("HLT")

        # ── CLI / STI / CLD / STD ──
        if opcode == 0xFA:
            self._set_flag(FLAG_IF, False); return
        if opcode == 0xFB:
            self._set_flag(FLAG_IF, True); return
        if opcode == 0xFC:
            self._set_flag(FLAG_DF, False); return
        if opcode == 0xFD:
            self._set_flag(FLAG_DF, True); return

        # ── PUSHAD / POPAD ──
        if opcode == 0x60:
            # PUSHAD: push EAX, ECX, EDX, EBX, ESP(orig), EBP, ESI, EDI
            esp_save = self._regs[4]
            for i in [0, 1, 2, 3, 4, 5, 6, 7]:
                self._push32(self._regs[i] if i != 4 else esp_save)
            return
        if opcode == 0x61:
            # POPAD: pop EDI, ESI, EBP, (skip ESP), EBX, EDX, ECX, EAX
            vals = [self._pop32() for _ in range(8)]
            # vals[0]=EDI, [1]=ESI, [2]=EBP, [3]=skip(ESP), [4]=EBX, [5]=EDX, [6]=ECX, [7]=EAX
            self._regs[7] = vals[0]  # EDI
            self._regs[6] = vals[1]  # ESI
            self._regs[5] = vals[2]  # EBP
            # vals[3] is old ESP — discard
            self._regs[3] = vals[4]  # EBX
            self._regs[2] = vals[5]  # EDX
            self._regs[1] = vals[6]  # ECX
            self._regs[0] = vals[7]  # EAX
            return

        # ── RET / RETF ──
        if opcode == 0xC3:
            self._eip = self._pop32(); return
        if opcode == 0xCB:
            self._eip = self._pop32(); self._pop32()  # pop CS (ignored)
            return

        # ── INT imm8 ──
        if opcode == 0xCD:
            int_num = self._fetch_byte()
            self._raise_interrupt(int_num)
            return

        # ── IRET ──
        if opcode == 0xCF:
            self._eip = self._pop32()
            self._pop32()  # pop CS
            self._eflags = self._pop32() & 0xFFFFFFFF
            return

        # ── Group 4: INC/DEC r/m8 (FE /0 /1) ──
        if opcode == 0xFE:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            if reg_f == 0:  # INC r/m8
                if rm_is_reg:
                    v = self._get8l(rm_val)
                    r = (v + 1) & 0xFF
                    self._set8l(rm_val, r)
                    self._update_flags_add(v, 1, r)
                else:
                    v = self._read8(rm_val)
                    r = (v + 1) & 0xFF
                    self._write8(rm_val, r)
                    self._update_flags_add(v, 1, r)
                return
            if reg_f == 1:  # DEC r/m8
                if rm_is_reg:
                    v = self._get8l(rm_val)
                    r = (v - 1) & 0xFF
                    self._set8l(rm_val, r)
                    self._update_flags_sub(v, 1, v - 1)
                else:
                    v = self._read8(rm_val)
                    r = (v - 1) & 0xFF
                    self._write8(rm_val, r)
                    self._update_flags_sub(v, 1, v - 1)
                return

        # ── PUSH r/m32 (FF /6) ──
        if opcode == 0xFF:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            if reg_f == 6:  # PUSH r/m
                if rm_is_reg:
                    v = self._get32(rm_val)
                else:
                    v = self._read32(rm_val)
                self._push32(v)
                return
            if reg_f == 4:  # JMP r/m
                if rm_is_reg:
                    self._eip = self._get32(rm_val)
                else:
                    self._eip = self._read32(rm_val)
                return
            if reg_f == 2:  # CALL r/m
                if rm_is_reg:
                    target = self._get32(rm_val)
                else:
                    target = self._read32(rm_val)
                self._push32(self._eip)
                self._eip = target
                return
            if reg_f == 0:  # INC r/m32
                if rm_is_reg:
                    v = self._get32(rm_val)
                    r = (v + 1) & 0xFFFFFFFF
                    self._set32(rm_val, r)
                    self._update_flags_add(v, 1, r)
                else:
                    v = self._read32(rm_val)
                    r = (v + 1) & 0xFFFFFFFF
                    self._write32(rm_val, r)
                    self._update_flags_add(v, 1, r)
                return
            if reg_f == 1:  # DEC r/m32
                if rm_is_reg:
                    v = self._get32(rm_val)
                    r = (v - 1) & 0xFFFFFFFF
                    self._set32(rm_val, r)
                    self._update_flags_sub(v, 1, v - 1)
                else:
                    v = self._read32(rm_val)
                    r = (v - 1) & 0xFFFFFFFF
                    self._write32(rm_val, r)
                    self._update_flags_sub(v, 1, v - 1)
                return
            # Group 5 fallback
            return

        # ── CALL rel32 ──
        if opcode == 0xE8:
            offset = self._fetch_dword()
            if offset & 0x80000000:
                offset |= 0xFFFFFFFF00000000  # sign extend
                offset = offset - 0x100000000
            target = (self._eip + offset) & 0xFFFFFFFF
            self._push32(self._eip)
            self._eip = target
            return

        # ── JMP rel32 ──
        if opcode == 0xE9:
            offset = self._fetch_dword()
            if offset & 0x80000000:
                offset = offset - 0x100000000
            self._eip = (self._eip + offset) & 0xFFFFFFFF
            return

        # ── JMP rel8 ──
        if opcode == 0xEB:
            offset = self._fetch_byte()
            if offset & 0x80:
                offset = offset - 0x100
            self._eip = (self._eip + offset) & 0xFFFFFFFF
            return

        # ── Jcc rel8 (conditional jumps) ──
        if 0x70 <= opcode <= 0x7F:
            cond = self._cc_condition(opcode - 0x70)
            offset = self._fetch_byte()
            if offset & 0x80:
                offset = offset - 0x100
            if cond:
                self._eip = (self._eip + offset) & 0xFFFFFFFF
            return

        # ── JO/JNO/JB/JAE/JE/JNE/JBE/JA/JS/JNS/JP/JNP/JL/JGE/JLE/JG rel8 ──
        # (0x0F 0x8x rel32 — two-byte near jumps)
        if opcode == 0x0F:
            opcode2 = self._fetch_byte()
            if 0x80 <= opcode2 <= 0x8F:
                cond = self._cc_condition(opcode2 - 0x80)
                offset = self._fetch_dword()
                if offset & 0x80000000:
                    offset = offset - 0x100000000
                if cond:
                    self._eip = (self._eip + offset) & 0xFFFFFFFF
                return
            # 0x0F 0x31 = RDTSC (stub)
            if opcode2 == 0x31:
                self._set32(0, 0)  # eax = 0
                self._set32(2, 0)  # edx = 0
                return
            # ── LGDT/LIDT/LTR (0F 01 /2 /3 /3) ──
            if opcode2 == 0x01:
                reg_f, rm_is_reg, rm_val = self._decode_modrm()
                if not rm_is_reg:
                    addr = rm_val & 0xFFFFFFFF
                    limit = struct.unpack_from("<H", self._mem, addr)[0]
                    base = struct.unpack_from("<I", self._mem, addr + 2)[0]
                    if reg_f == 2:  # LGDT
                        self._gdt_base = base
                        self._gdt_limit = limit
                    elif reg_f == 3:  # LIDT
                        self._idt_base = base
                        self._idt_limit = limit
                return
            # ── MOV r32, CRn (0F 20) / MOV CRn, r32 (0F 22) ──
            if opcode2 in (0x20, 0x22):
                reg_f, rm_is_reg, rm_val = self._decode_modrm()
                if opcode2 == 0x20:  # MOV r32, CRn
                    self._set32(rm_val, self._cr[reg_f])
                else:  # MOV CRn, r32
                    self._cr[reg_f] = self._get32(rm_val) & 0xFFFFFFFF
                return
            # ── MOV r32, DRn (0F 21) / MOV DRn, r32 (0F 23) ──
            if opcode2 in (0x21, 0x23):
                reg_f, rm_is_reg, rm_val = self._decode_modrm()
                if opcode2 == 0x21:  # MOV r32, DRn
                    self._set32(rm_val, self._dr[reg_f])
                else:  # MOV DRn, r32
                    self._dr[reg_f] = self._get32(rm_val) & 0xFFFFFFFF
                return
            # ── MOVZX r32, r/m8 (0F B6) ──
            if opcode2 == 0xB6:
                reg_f, rm_is_reg, rm_val = self._decode_modrm()
                if rm_is_reg:
                    v = self._read_rm_reg(rm_val, 8)
                else:
                    v = self._read_rm_mem(rm_val, 8)
                self._set32(reg_f, v)
                return
            # ── MOVZX r32, r/m16 (0F B7) ──
            if opcode2 == 0xB7:
                reg_f, rm_is_reg, rm_val = self._decode_modrm()
                if rm_is_reg:
                    v = self._get16(rm_val)
                else:
                    v = self._read16(rm_val)
                self._set32(reg_f, v)
                return
            # ── MOVSX r32, r/m8 (0F BE) ──
            if opcode2 == 0xBE:
                reg_f, rm_is_reg, rm_val = self._decode_modrm()
                if rm_is_reg:
                    v = self._read_rm_reg(rm_val, 8)
                else:
                    v = self._read_rm_mem(rm_val, 8)
                if v & 0x80:
                    v |= 0xFFFFFF00
                self._set32(reg_f, v)
                return
            # ── MOVSX r32, r/m16 (0F BF) ──
            if opcode2 == 0xBF:
                reg_f, rm_is_reg, rm_val = self._decode_modrm()
                if rm_is_reg:
                    v = self._get16(rm_val)
                else:
                    v = self._read16(rm_val)
                if v & 0x8000:
                    v |= 0xFFFF0000
                self._set32(reg_f, v)
                return
            # ── IMUL r32, r/m32 (0F AF) ──
            if opcode2 == 0xAF:
                reg_f, rm_is_reg, rm_val = self._decode_modrm()
                if rm_is_reg:
                    b = self._get32(rm_val)
                else:
                    b = self._read32(rm_val)
                a = self._get32(reg_f)
                result = (a * b) & 0xFFFFFFFF
                self._set32(reg_f, result)
                return
            # ── BSF (0F BC) / BSR (0F BD) — stub ──
            if opcode2 in (0xBC, 0xBD):
                reg_f, rm_is_reg, rm_val = self._decode_modrm()
                if rm_is_reg:
                    v = self._get32(rm_val)
                else:
                    v = self._read32(rm_val)
                if v == 0:
                    self._set_flag(FLAG_ZF, True)
                else:
                    self._set_flag(FLAG_ZF, False)
                    if opcode2 == 0xBC:
                        for bit in range(32):
                            if v & (1 << bit):
                                self._set32(reg_f, bit)
                                break
                    else:
                        for bit in range(31, -1, -1):
                            if v & (1 << bit):
                                self._set32(reg_f, bit)
                                break
                return
            return

        # ── MOV r/m8, r8 / MOV r/m32, r32 (opcode 88/89) ──
        if opcode == 0x88:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            src = self._read_rm_reg(reg_f, 8)
            if rm_is_reg:
                self._write_rm_reg(rm_val, 8, src)
            else:
                self._write_rm_mem(rm_val, 8, src)
            return
        if opcode == 0x89:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            src = self._read_rm_reg(reg_f, 32)
            if rm_is_reg:
                self._write_rm_reg(rm_val, 32, src)
            else:
                self._write_rm_mem(rm_val, 32, src)
            return

        # ── MOV r8, r/m8 / MOV r32, r/m32 (opcode 8A/8B) ──
        if opcode == 0x8A:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            if rm_is_reg:
                src = self._read_rm_reg(rm_val, 8)
            else:
                src = self._read_rm_mem(rm_val, 8)
            self._write_rm_reg(reg_f, 8, src)
            return
        if opcode == 0x8B:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            if rm_is_reg:
                src = self._read_rm_reg(rm_val, 32)
            else:
                src = self._read_rm_mem(rm_val, 32)
            self._write_rm_reg(reg_f, 32, src)
            return

        # ── MOV r/m16, Sreg (8C) ──
        if opcode == 0x8C:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            seg_val = self._get_seg(reg_f)
            if rm_is_reg:
                self._set16(rm_val, seg_val & 0xFFFF)
            else:
                self._write16(rm_val, seg_val & 0xFFFF)
            return

        # ── MOV Sreg, r/m16 (8E) ──
        if opcode == 0x8E:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            if rm_is_reg:
                val = self._get16(rm_val)
            else:
                val = self._read16(rm_val)
            self._set_seg(reg_f, val)
            return

        # ── MOV r/m8, imm8 (C6 /0) ──
        if opcode == 0xC6:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            imm = self._fetch_byte()
            if rm_is_reg:
                self._write_rm_reg(rm_val, 8, imm)
            else:
                self._write_rm_mem(rm_val, 8, imm)
            return

        # ── MOV r/m32, imm32 (C7 /0) ──
        if opcode == 0xC7:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            imm = self._fetch_dword()
            if rm_is_reg:
                self._write_rm_reg(rm_val, 32, imm)
            else:
                self._write_rm_mem(rm_val, 32, imm)
            return

        # ── MOV r8, imm8 (B0-B7) ──
        if 0xB0 <= opcode <= 0xB7:
            val = self._fetch_byte()
            self._write_rm_reg(opcode - 0xB0, 8, val)
            return

        # ── MOV r32, imm32 (B8-BF) ──
        if 0xB8 <= opcode <= 0xBF:
            val = self._fetch_dword()
            self._set32(opcode - 0xB8, val)
            return

        # ── MOV r/m16, imm16 (66 C7 /0) ──
        if opcode == 0x66:
            opcode2 = self._fetch_byte()
            if opcode2 == 0xC7:
                reg_f, rm_is_reg, rm_val = self._decode_modrm()
                imm = self._fetch_word()
                if rm_is_reg:
                    self._set16(rm_val, imm)
                else:
                    struct.pack_into("<H", self._mem, rm_val & 0xFFFFFFFF, imm)
                return
            if opcode2 == 0x89:
                reg_f, rm_is_reg, rm_val = self._decode_modrm()
                src = self._read_rm_reg(reg_f, 16)
                if rm_is_reg:
                    self._set16(rm_val, src)
                else:
                    struct.pack_into("<H", self._mem, rm_val & 0xFFFFFFFF, src)
                return
            if opcode2 == 0x8B:
                reg_f, rm_is_reg, rm_val = self._decode_modrm()
                if rm_is_reg:
                    src = self._get16(rm_val)
                else:
                    src = self._read16(rm_val)
                self._set16(reg_f, src)
                return
            # ── MOV r16, imm16 (66 B8-BF) ──
            if 0xB8 <= opcode2 <= 0xBF:
                val = self._fetch_word()
                self._set16(opcode2 - 0xB8, val)
                return
            return

        # ── MOV AL, [addr] (A1) / MOV EAX, [addr] (A1 with 66 prefix handled above) ──
        if opcode == 0xA1:
            addr = self._fetch_dword()
            self._set32(0, self._read32(addr))
            return

        # ── MOV [addr], AL (A3) / MOV [addr], EAX ──
        if opcode == 0xA3:
            addr = self._fetch_dword()
            self._write32(addr, self._get32(0))
            return

        # ── LEA r32, m ──
        if opcode == 0x8D:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            if not rm_is_reg:
                self._set32(reg_f, rm_val)
            return

        # ── XCHG (90+r) ──
        if 0x91 <= opcode <= 0x97:
            other = opcode - 0x90
            v0 = self._get32(0)
            self._set32(0, self._get32(other))
            self._set32(other, v0)
            return

        # ── Group 1: ALU r/m32, imm32 (81) / ALU r/m32, imm8 (83) ──
        if opcode == 0x81:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            imm = self._fetch_dword()
            if imm & 0x80000000:
                imm = imm - 0x100000000
            if rm_is_reg:
                a = self._get32(rm_val)
            else:
                a = self._read32(rm_val)
            r = self._alu(reg_f, a, imm, 32)
            if rm_is_reg:
                self._set32(rm_val, r & 0xFFFFFFFF)
            else:
                self._write32(rm_val, r & 0xFFFFFFFF)
            return
        if opcode == 0x83:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            imm = self._fetch_byte()
            if imm & 0x80:
                imm |= 0xFFFFFF00
            if rm_is_reg:
                a = self._get32(rm_val)
            else:
                a = self._read32(rm_val)
            r = self._alu(reg_f, a, imm, 32)
            if rm_is_reg:
                self._set32(rm_val, r & 0xFFFFFFFF)
            else:
                self._write32(rm_val, r & 0xFFFFFFFF)
            return

        # ── Group 1: ALU r/m8, imm8 (80) ──
        if opcode == 0x80:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            imm = self._fetch_byte()
            if rm_is_reg:
                a = self._get8l(rm_val)
            else:
                a = self._read8(rm_val)
            r = self._alu(reg_f, a, imm, 8)
            if rm_is_reg:
                self._set8l(rm_val, r & 0xFF)
            else:
                self._write_rm_mem(rm_val, 8, r & 0xFF)
            return

        # ── Group 1: ALU r32, r/m32, imm32 (69 /4=IMUL) ──
        if opcode == 0x69:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            imm = self._fetch_dword()
            if imm & 0x80000000:
                imm = imm - 0x100000000
            if rm_is_reg:
                a = self._get32(rm_val)
            else:
                a = self._read32(rm_val)
            r = (a * imm) & 0xFFFFFFFF
            self._set32(reg_f, r)
            return
        if opcode == 0x6B:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            imm = self._fetch_byte()
            if imm & 0x80:
                imm |= 0xFFFFFF00
            if rm_is_reg:
                a = self._get32(rm_val)
            else:
                a = self._read32(rm_val)
            r = (a * imm) & 0xFFFFFFFF
            self._set32(reg_f, r)
            return

        # ── ADD/SUB/AND/OR/XOR/CMP/TEST r/m32, r32 (01-03, 09-0B, 21-23, 29-2B, 31-33, 39-3B, 84-85) ──
        alu_ops = {
            0x01: (0, 32), 0x03: (0, 32),  # ADD
            0x09: (1, 32), 0x0B: (1, 32),  # OR
            0x21: (4, 32), 0x23: (4, 32),  # AND
            0x29: (5, 32), 0x2B: (5, 32),  # SUB
            0x31: (6, 32), 0x33: (6, 32),  # XOR
            0x39: (7, 32), 0x3B: (7, 32),  # CMP
            0x00: (0, 8),  0x02: (0, 8),   # ADD8
            0x08: (1, 8),  0x0A: (1, 8),   # OR8
            0x20: (4, 8),  0x22: (4, 8),   # AND8
            0x28: (5, 8),  0x2A: (5, 8),   # SUB8
            0x30: (6, 8),  0x32: (6, 8),   # XOR8
            0x38: (7, 8),  0x3A: (7, 8),   # CMP8
        }
        if opcode in alu_ops:
            alu_op, width = alu_ops[opcode]
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            if rm_is_reg:
                rm_val_reg = self._read_rm_reg(rm_val, width)
            else:
                rm_val_reg = self._read_rm_mem(rm_val, width)
            reg_val = self._read_rm_reg(reg_f, width)
            # Direction bit is bit 1 of opcode (not bit 0, which is w/word-size).
            # d=0: r/m is dest (r/m ← r/m OP reg)   e.g. 01/09/21/29/31/39
            # d=1: reg is dest (reg ← reg OP r/m)   e.g. 03/0B/23/2B/33/3B
            d = (opcode >> 1) & 1
            if d == 1:
                r = self._alu(alu_op, reg_val, rm_val_reg, width)
                if alu_op != 7:
                    self._write_rm_reg(reg_f, width, r & ((1 << width) - 1))
            else:
                r = self._alu(alu_op, rm_val_reg, reg_val, width)
                if alu_op != 7:
                    if rm_is_reg:
                        self._write_rm_reg(rm_val, width, r & ((1 << width) - 1))
                    else:
                        self._write_rm_mem(rm_val, width, r & ((1 << width) - 1))
            return

        # ── TEST r/m32, r32 (85) / TEST r/m8, r8 (84) ──
        if opcode == 0x85:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            if rm_is_reg:
                b = self._get32(rm_val)
            else:
                b = self._read32(rm_val)
            a = self._get32(reg_f)
            self._update_flags_logic(a & b, 32)
            return
        if opcode == 0x84:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            if rm_is_reg:
                b = self._get8l(rm_val)
            else:
                b = self._read8(rm_val)
            a = self._get8l(reg_f)
            self._update_flags_logic(a & b, 8)
            return

        # ── INC/DEC r32 (40-4F) ──
        if 0x40 <= opcode <= 0x47:
            idx = opcode - 0x40
            v = self._get32(idx)
            r = (v + 1) & 0xFFFFFFFF
            self._set32(idx, r)
            self._update_flags_add(v, 1, r)
            return
        if 0x48 <= opcode <= 0x4F:
            idx = opcode - 0x48
            v = self._get32(idx)
            r = (v - 1) & 0xFFFFFFFF
            self._set32(idx, r)
            self._update_flags_sub(v, 1, v - 1)
            return

        # ── PUSH/POP r32 (50-57 / 58-5F) ──
        if 0x50 <= opcode <= 0x57:
            self._push32(self._get32(opcode - 0x50))
            return
        if 0x58 <= opcode <= 0x5F:
            self._set32(opcode - 0x58, self._pop32())
            return

        # ── MOV r/m, r / MOV r, r/m (88-8B already handled above) ──
        # ── MOVSXD r32, r/m32 (63) ──
        if opcode == 0x63:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            if rm_is_reg:
                v = self._get32(rm_val)
            else:
                v = self._read32(rm_val)
            self._set32(reg_f, v)
            return

        # ── MOVZX r32, r/m8 (0F B6) / MOVZX r32, r/m16 (0F B7) ──
        # Already caught by 0x0F prefix above — handled there

        # ── IMUL r32, r/m32 (0F AF) ──
        # Already caught by 0x0F prefix — needs handling

        # ── PUSH imm8 (6A) ──
        if opcode == 0x6A:
            imm = self._fetch_byte()
            if imm & 0x80:
                imm |= 0xFFFFFF00
            self._push32(imm)
            return

        # ── PUSH imm32 (68) ──
        if opcode == 0x68:
            imm = self._fetch_dword()
            self._push32(imm)
            return

        # ── POP r/m (8F /0) ──
        if opcode == 0x8F:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            val = self._pop32()
            if rm_is_reg:
                self._set32(rm_val, val)
            else:
                self._write32(rm_val, val)
            return

        # ── SAHF (9E) ──
        if opcode == 0x9E:
            ah = self._get8h(0)
            self._eflags = (self._eflags & 0xFFFFFF00) | (ah & 0xD5) | 0x02
            return

        # ── LAHF (9F) ──
        if opcode == 0x9F:
            self._set8h(0, self._eflags & 0xFF)
            return

        # ── CDQ (99) — sign extend EAX into EDX:EAX ──
        if opcode == 0x99:
            if self._get32(0) & 0x80000000:
                self._set32(2, 0xFFFFFFFF)
            else:
                self._set32(2, 0)
            return

        # ── DAA/DAS/AAA/AAS (27/2F/37/3F) — BCD ops, stub ──
        if opcode in (0x27, 0x2F, 0x37, 0x3F):
            return

        # ── LODSB (AC) ──
        if opcode == 0xAC:
            al = self._read8(self._get32(6))
            self._set8l(0, al)
            if self._flag(FLAG_DF):
                self._set32(6, (self._get32(6) - 1) & 0xFFFFFFFF)
            else:
                self._set32(6, (self._get32(6) + 1) & 0xFFFFFFFF)
            return

        # ── STOSB (AA) ──
        if opcode == 0xAA:
            self._write8(self._get32(7), self._get8l(0))
            if self._flag(FLAG_DF):
                self._set32(7, (self._get32(7) - 1) & 0xFFFFFFFF)
            else:
                self._set32(7, (self._get32(7) + 1) & 0xFFFFFFFF)
            return

        # ── STOSW (AB) ──
        if opcode == 0xAB:
            struct.pack_into("<H", self._mem, self._get32(7) & 0xFFFFFFFF,
                             self._get16(0))
            if self._flag(FLAG_DF):
                self._set32(7, (self._get32(7) - 2) & 0xFFFFFFFF)
            else:
                self._set32(7, (self._get32(7) + 2) & 0xFFFFFFFF)
            return

        # ── CMPSB (A6) — compare [ESI] with [EDI], advance both ──
        if opcode == 0xA6:
            esi = self._regs[6] & 0xFFFFFFFF
            edi = self._regs[7] & 0xFFFFFFFF
            a = self._read8(esi)
            b = self._read8(edi)
            r = (a - b) & 0xFF
            self._update_flags_sub(a, b, r)
            df = -1 if self._flag(FLAG_DF) else 1
            self._regs[6] = (self._regs[6] + df) & 0xFFFFFFFF
            self._regs[7] = (self._regs[7] + df) & 0xFFFFFFFF
            return
        # ── CMPSW (A7) — compare [ESI] with [EDI] as word ──
        if opcode == 0xA7:
            esi = self._regs[6] & 0xFFFFFFFF
            edi = self._regs[7] & 0xFFFFFFFF
            a = self._read16(esi)
            b = self._read16(edi)
            r = (a - b) & 0xFFFF
            self._update_flags_sub(a, b, r)
            df = -2 if self._flag(FLAG_DF) else 2
            self._regs[6] = (self._regs[6] + df) & 0xFFFFFFFF
            self._regs[7] = (self._regs[7] + df) & 0xFFFFFFFF
            return
        # ── SCASB (AE) — compare AL with [EDI], advance EDI ──
        if opcode == 0xAE:
            al = self._get8l(0)
            edi = self._regs[7] & 0xFFFFFFFF
            b = self._read8(edi)
            r = (al - b) & 0xFF
            self._update_flags_sub(al, b, r)
            df = -1 if self._flag(FLAG_DF) else 1
            self._regs[7] = (self._regs[7] + df) & 0xFFFFFFFF
            return
        # ── SCASW (AF) — compare AX with [EDI] as word ──
        if opcode == 0xAF:
            ax = self._get16(0)
            edi = self._regs[7] & 0xFFFFFFFF
            b = self._read16(edi)
            r = (ax - b) & 0xFFFF
            self._update_flags_sub(ax, b, r)
            df = -2 if self._flag(FLAG_DF) else 2
            self._regs[7] = (self._regs[7] + df) & 0xFFFFFFFF
            return

        # ── REP/REPE/REPNE prefix (F2/F3) ──
        if opcode in (0xF2, 0xF3):
            opcode2 = self._fetch_byte()
            count = self._get32(1)  # ECX
            if opcode2 == 0xA4:  # REP MOVSB
                while count > 0:
                    self._write8(self._get32(7), self._read8(self._get32(6)))
                    self._set32(6, (self._get32(6) + (1 if not self._flag(FLAG_DF) else -1)) & 0xFFFFFFFF)
                    self._set32(7, (self._get32(7) + (1 if not self._flag(FLAG_DF) else -1)) & 0xFFFFFFFF)
                    count -= 1
                self._set32(1, 0)
                return
            if opcode2 == 0xAC:  # REP LODSB
                while count > 0:
                    self._set8l(0, self._read8(self._get32(6)))
                    self._set32(6, (self._get32(6) + (1 if not self._flag(FLAG_DF) else -1)) & 0xFFFFFFFF)
                    count -= 1
                self._set32(1, 0)
                return
            if opcode2 == 0xAA:  # REP STOSB
                while count > 0:
                    self._write8(self._get32(7), self._get8l(0))
                    self._set32(7, (self._get32(7) + (1 if not self._flag(FLAG_DF) else -1)) & 0xFFFFFFFF)
                    count -= 1
                self._set32(1, 0)
                return
            if opcode2 == 0xC3:  # REP RET (unusual but valid)
                self._eip = self._pop32()
                return
            return

        # ── IN AL, imm8 (E4) / IN EAX, imm8 (E5) ──
        if opcode == 0xE4:
            port = self._fetch_byte()
            self._set8l(0, self._port_in(port) & 0xFF)
            return
        if opcode == 0xE5:
            port = self._fetch_byte()
            self._set32(0, self._port_in(port) & 0xFFFFFFFF)
            return

        # ── IN AL, DX (EC) / IN EAX, DX (ED) ──
        if opcode == 0xEC:
            port = self._get16(2)
            self._set8l(0, self._port_in(port) & 0xFF)
            return
        if opcode == 0xED:
            port = self._get16(2)
            self._set32(0, self._port_in(port) & 0xFFFFFFFF)
            return

        # ── OUT imm8, AL (E6) / OUT imm8, EAX (E7) ──
        if opcode == 0xE6:
            port = self._fetch_byte()
            self._port_out(port, self._get8l(0))
            return
        if opcode == 0xE7:
            port = self._fetch_byte()
            self._port_out(port, self._get32(0))
            return

        # ── OUT DX, AL (EE) / OUT DX, EAX (EF) ──
        if opcode == 0xEE:
            port = self._get16(2)
            self._port_out(port, self._get8l(0))
            return
        if opcode == 0xEF:
            port = self._get16(2)
            self._port_out(port, self._get32(0))
            return

        # ── Group 1: shift/rotate r/m32, imm8 (C1) / by 1 (D1) ──
        if opcode == 0xC1:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            imm = self._fetch_byte() & 0x1F
            if rm_is_reg:
                a = self._get32(rm_val)
            else:
                a = self._read32(rm_val)
            r = self._shift(reg_f, a, imm, 32)
            if rm_is_reg:
                self._set32(rm_val, r & 0xFFFFFFFF)
            else:
                self._write32(rm_val, r & 0xFFFFFFFF)
            return
        if opcode == 0xD1:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            if rm_is_reg:
                a = self._get32(rm_val)
            else:
                a = self._read32(rm_val)
            r = self._shift(reg_f, a, 1, 32)
            if rm_is_reg:
                self._set32(rm_val, r & 0xFFFFFFFF)
            else:
                self._write32(rm_val, r & 0xFFFFFFFF)
            return

        # ── Group 1: shift r/m8, imm8 (C0) / by 1 (D0) / by CL (D2) ──
        if opcode == 0xC0:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            imm = self._fetch_byte() & 0x1F
            if rm_is_reg:
                a = self._get8l(rm_val)
            else:
                a = self._read8(rm_val)
            r = self._shift(reg_f, a, imm, 8)
            if rm_is_reg:
                self._set8l(rm_val, r & 0xFF)
            else:
                self._write8(rm_val, r & 0xFF)
            return
        if opcode == 0xD0:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            if rm_is_reg:
                a = self._get8l(rm_val)
            else:
                a = self._read8(rm_val)
            r = self._shift(reg_f, a, 1, 8)
            if rm_is_reg:
                self._set8l(rm_val, r & 0xFF)
            else:
                self._write8(rm_val, r & 0xFF)
            return
        if opcode == 0xD2:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            count = self._get8l(1) & 0x1F  # CL
            if rm_is_reg:
                a = self._get8l(rm_val)
            else:
                a = self._read8(rm_val)
            r = self._shift(reg_f, a, count, 8)
            if rm_is_reg:
                self._set8l(rm_val, r & 0xFF)
            else:
                self._write8(rm_val, r & 0xFF)
            return

        # ── Group 1: shift r/m32, CL (D3) ──
        if opcode == 0xD3:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            count = self._get8l(1) & 0x1F  # CL
            if rm_is_reg:
                a = self._get32(rm_val)
            else:
                a = self._read32(rm_val)
            r = self._shift(reg_f, a, count, 32)
            if rm_is_reg:
                self._set32(rm_val, r & 0xFFFFFFFF)
            else:
                self._write32(rm_val, r & 0xFFFFFFFF)
            return

        # ── MUL r/m8 (F6 /4) / MUL r/m32 (F7 /4) ──
        if opcode == 0xF6:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            if reg_f == 4:  # MUL r/m8
                if rm_is_reg:
                    a = self._get8l(rm_val)
                else:
                    a = self._read8(rm_val)
                b = self._get8l(0)
                result = a * b
                self._set16(0, result & 0xFFFF)
                self._set_flag(FLAG_CF, bool(result & 0xFF00))
                self._set_flag(FLAG_OF, bool(result & 0xFF00))
                return
            if reg_f == 5:  # IMUL r/m8
                if rm_is_reg:
                    a = self._get8l(rm_val)
                else:
                    a = self._read8(rm_val)
                b = self._get8l(0)
                if a & 0x80: a |= 0xFFFFFF00
                if b & 0x80: b |= 0xFFFFFF00
                result = (a * b) & 0xFFFFFFFF
                self._set16(0, result & 0xFFFF)
                sign = ((result >> 15) & 1) == ((result >> 7) & 1)
                self._set_flag(FLAG_CF, not sign)
                self._set_flag(FLAG_OF, not sign)
                return
            if reg_f == 6:  # DIV r/m8
                if rm_is_reg:
                    a = self._get8l(rm_val)
                else:
                    a = self._read8(rm_val)
                if a == 0:
                    raise InsFault("DIV by zero")
                ax = self._get16(0)
                q = ax // a
                if q > 0xFF:
                    raise InsFault("DIV overflow")
                self._set8l(0, q & 0xFF)
                self._set8h(0, ax % a)
                return
            if reg_f == 7:  # IDIV r/m8
                if rm_is_reg:
                    a = self._get8l(rm_val)
                else:
                    a = self._read8(rm_val)
                if a == 0:
                    raise InsFault("IDIV by zero")
                ax = self._get16(0)
                if ax & 0x8000:
                    ax |= 0xFFFF0000
                a_s = a if a < 0x80 else a - 0x100
                q = int(ax / a_s)
                r = ax - q * a_s
                self._set8l(0, q & 0xFF)
                self._set8h(0, r & 0xFF)
                return
            return

        if opcode == 0xF7:
            reg_f, rm_is_reg, rm_val = self._decode_modrm()
            if reg_f == 0:  # TEST r/m32, imm32
                imm = self._fetch_dword()
                if rm_is_reg:
                    a = self._get32(rm_val)
                else:
                    a = self._read32(rm_val)
                self._update_flags_logic(a & imm, 32)
                return
            if reg_f == 2:  # NOT r/m32
                if rm_is_reg:
                    v = self._get32(rm_val)
                    self._set32(rm_val, ~v)
                else:
                    v = self._read32(rm_val)
                    self._write32(rm_val, ~v)
                return
            if reg_f == 3:  # NEG r/m32
                if rm_is_reg:
                    v = self._get32(rm_val)
                    r = (-v) & 0xFFFFFFFF
                    self._set32(rm_val, r)
                else:
                    v = self._read32(rm_val)
                    r = (-v) & 0xFFFFFFFF
                    self._write32(rm_val, r)
                self._set_flag(FLAG_CF, v != 0)
                self._update_flags_sub(0, v, r)
                return
            if reg_f == 4:  # MUL r/m32
                if rm_is_reg:
                    a = self._get32(rm_val)
                else:
                    a = self._read32(rm_val)
                b = self._get32(0)
                result = a * b
                self._set32(0, result & 0xFFFFFFFF)
                self._set32(2, (result >> 32) & 0xFFFFFFFF)
                self._set_flag(FLAG_CF, bool(self._get32(2)))
                self._set_flag(FLAG_OF, bool(self._get32(2)))
                return
            if reg_f == 5:  # IMUL r/m32
                if rm_is_reg:
                    a = self._get32(rm_val)
                else:
                    a = self._read32(rm_val)
                b = self._get32(0)
                a_s = a if a < 0x80000000 else a - 0x100000000
                b_s = b if b < 0x80000000 else b - 0x100000000
                result = a_s * b_s
                self._set32(0, result & 0xFFFFFFFF)
                self._set32(2, (result >> 32) & 0xFFFFFFFF)
                hi = self._get32(2)
                self._set_flag(FLAG_CF, hi != 0 and hi != 0xFFFFFFFF)
                self._set_flag(FLAG_OF, hi != 0 and hi != 0xFFFFFFFF)
                return
            if reg_f == 6:  # DIV r/m32
                if rm_is_reg:
                    divisor = self._get32(rm_val)
                else:
                    divisor = self._read32(rm_val)
                if divisor == 0:
                    raise InsFault("DIV by zero")
                dividend = (self._get32(2) << 32) | self._get32(0)
                quotient = dividend // divisor
                if quotient > 0xFFFFFFFF:
                    raise InsFault("DIV overflow")
                self._set32(0, quotient & 0xFFFFFFFF)
                self._set32(2, dividend % divisor)
                return
            if reg_f == 7:  # IDIV r/m32
                if rm_is_reg:
                    divisor = self._get32(rm_val)
                else:
                    divisor = self._read32(rm_val)
                if divisor == 0:
                    raise InsFault("IDIV by zero")
                dividend = (self._get32(2) << 32) | self._get32(0)
                d_s = divisor if divisor < 0x80000000 else divisor - 0x100000000
                dd_s = dividend if dividend < 0x10000000000000000 else dividend
                quotient = int(dd_s / d_s)
                remainder = dd_s - quotient * d_s
                self._set32(0, quotient & 0xFFFFFFFF)
                self._set32(2, remainder & 0xFFFFFFFF)
                return
            return

        # ── Unknown opcode — skip 1 byte and try again ──
        logger.warning(f"CPU: unknown opcode 0x{opcode:02X} at EIP=0x{(self._eip - 1) & 0xFFFFFFFF:X}")
        self._eip = (self._eip + 1) & 0xFFFFFFFF

    # ── ALU operations ───────────────────────────────────────────────────

    def _alu(self, op: int, a: int, b: int, bits: int = 32) -> int:
        mask = (1 << bits) - 1
        a &= mask
        b &= mask
        sign = bits - 1

        if op == 0:  # ADD
            result = a + b
            self._update_flags_add(a, b, result, bits)
            return result & mask
        elif op == 1:  # OR
            r = a | b
            self._update_flags_logic(r, bits)
            return r
        elif op == 4:  # AND
            r = a & b
            self._update_flags_logic(r, bits)
            return r
        elif op == 5:  # SUB
            result = a - b
            self._update_flags_sub(a, b, result, bits)
            return result & mask
        elif op == 6:  # XOR
            r = a ^ b
            self._update_flags_logic(r, bits)
            return r
        elif op == 7:  # CMP
            result = a - b
            self._update_flags_sub(a, b, result, bits)
            return a  # CMP doesn't write result
        return a

    # ── Shift operations ─────────────────────────────────────────────────

    def _shift(self, op: int, a: int, count: int, bits: int = 32) -> int:
        mask = (1 << bits) - 1
        a &= mask
        if count == 0:
            return a

        if op == 0:  # ROL
            r = ((a << count) | (a >> (bits - count))) & mask
            self._set_flag(FLAG_CF, bool(r & 1))
            return r
        elif op == 1:  # ROR
            r = ((a >> count) | (a << (bits - count))) & mask
            self._set_flag(FLAG_CF, bool(r & (1 << (bits - 1))))
            return r
        elif op == 4:  # SHL / SAL
            self._set_flag(FLAG_CF, bool(a & (1 << (bits - count))))
            r = (a << count) & mask
            self._update_flags_logic(r, bits)
            self._set_flag(FLAG_CF, bool(a & (1 << (bits - count))))
            return r
        elif op == 5:  # SHR
            self._set_flag(FLAG_CF, bool(a & (1 << (count - 1))))
            r = (a >> count) & mask
            self._update_flags_logic(r, bits)
            return r
        elif op == 7:  # SAR
            sign = a & (1 << (bits - 1))
            r = (a >> count) & mask
            if sign:
                r |= mask << (bits - count) & mask
            self._set_flag(FLAG_CF, bool(a & (1 << (count - 1))))
            self._update_flags_logic(r, bits)
            return r
        return a

    # ── Condition code evaluation ────────────────────────────────────────

    def _cc_condition(self, cc: int) -> bool:
        """Evaluate x86 condition code (0-15) against current flags."""
        if cc == 0x0: return self._flag(FLAG_OF)             # JO
        if cc == 0x1: return not self._flag(FLAG_OF)         # JNO
        if cc == 0x2: return self._flag(FLAG_CF)             # JB/JC
        if cc == 0x3: return not self._flag(FLAG_CF)         # JAE/JNC
        if cc == 0x4: return self._flag(FLAG_ZF)             # JE/JZ
        if cc == 0x5: return not self._flag(FLAG_ZF)         # JNE/JNZ
        if cc == 0x6: return (self._flag(FLAG_CF) or         # JBE
                              self._flag(FLAG_ZF))
        if cc == 0x7: return (not self._flag(FLAG_CF) and    # JA
                              not self._flag(FLAG_ZF))
        if cc == 0x8: return self._flag(FLAG_SF)             # JS
        if cc == 0x9: return not self._flag(FLAG_SF)         # JNS
        if cc == 0xA: return self._flag(FLAG_PF)             # JP
        if cc == 0xB: return not self._flag(FLAG_PF)         # JNP
        if cc == 0xC: return self._flag(FLAG_SF) != self._flag(FLAG_OF)  # JL
        if cc == 0xD: return self._flag(FLAG_SF) == self._flag(FLAG_OF)  # JGE
        if cc == 0xE: return self._flag(FLAG_ZF) or self._flag(FLAG_SF) != self._flag(FLAG_OF)  # JLE
        if cc == 0xF: return not self._flag(FLAG_ZF) and self._flag(FLAG_SF) == self._flag(FLAG_OF)  # JG
        return False

    # ── Debug / inspection ───────────────────────────────────────────────

    def reg_dump(self) -> str:
        """Return formatted register dump."""
        lines = []
        lines.append(f"EAX={self.eax:08X}  ECX={self.ecx:08X}  EDX={self.edx:08X}  EBX={self.ebx:08X}")
        lines.append(f"ESP={self.esp:08X}  EBP={self.ebp:08X}  ESI={self.esi:08X}  EDI={self.edi:08X}")
        lines.append(f"EIP={self.eip:08X}  EFLAGS={self._eflags:08X} [{self.eflags_str()}]")
        return "\n".join(lines)

    def eflags_str(self) -> str:
        """Return flags as character string (e.g. 'POZA')."""
        flags = ""
        flags += "C" if self._flag(FLAG_CF) else "c"
        flags += "P" if self._flag(FLAG_PF) else "p"
        flags += "Z" if self._flag(FLAG_ZF) else "z"
        flags += "S" if self._flag(FLAG_SF) else "s"
        flags += "T" if self._flag(FLAG_DF) else "t"
        flags += "I" if self._flag(FLAG_IF) else "i"
        flags += "O" if self._flag(FLAG_OF) else "o"
        return flags

    def mem_dump(self, addr: int, length: int = 64) -> str:
        """Return hex dump of memory at addr."""
        lines = []
        for offset in range(0, length, 16):
            a = (addr + offset) & 0xFFFFFFFF
            hex_bytes = " ".join(f"{self._read8(a + i):02X}" for i in range(16))
            ascii_bytes = "".join(
                chr(self._read8(a + i)) if 32 <= self._read8(a + i) < 127 else "."
                for i in range(16)
            )
            lines.append(f"{a:08X}  {hex_bytes}  |{ascii_bytes}|")
        return "\n".join(lines)


# ── Interactive Shell ─────────────────────────────────────────────────────────

import time
import threading

# Minimal x86 kernel shell assembly — reads keyboard, echoes to screen
_SHELL_ASM = """\
[BITS 32]
[ORG 0x1000]

start:
    sti
    ; Print prompt
    mov esi, prompt
    call print
.loop:
    ; Wait for key at 0x400
    mov al, [0x400]
    cmp al, 0
    je .loop
    ; Get character
    mov bl, al
    mov byte [0x400], 0
    ; Echo to screen
    cmp bl, 0x0D   ; Enter?
    je .enter
    cmp bl, 0x08   ; Backspace?
    je .bs
    ; Store in line buffer
    mov edi, [line_pos]
    cmp edi, 126
    jae .loop
    mov [line_buf + edi], bl
    inc dword [line_pos]
    ; Print char
    mov al, bl
    call putchar
    jmp .loop
.enter:
    ; Null-terminate
    mov edi, [line_pos]
    mov byte [line_buf + edi], 0
    ; Newline
    mov al, 0x0D
    call putchar
    mov al, 0x0A
    call putchar
    ; Process command
    call process_cmd
    ; Reset buffer
    mov dword [line_pos], 0
    ; Print prompt
    mov esi, prompt
    call print
    jmp .loop
.bs:
    mov edi, [line_pos]
    cmp edi, 0
    je .loop
    dec dword [line_pos]
    ; Erase on screen: BS SP BS
    mov al, 0x08
    call putchar
    mov al, ' '
    call putchar
    mov al, 0x08
    call putchar
    jmp .loop

print:
    lodsb
    cmp al, 0
    je .done
    call putchar
    jmp print
.done:
    ret

putchar:
    ; Write char at cursor position in VGA memory (0xB8000)
    push edi
    mov edi, [cursor]
    shl edi, 1
    mov [edi + 0xB8000], al
    mov byte [edi + 0xB8001], 0x07
    inc dword [cursor]
    pop edi
    ret

process_cmd:
    ; Compare first byte of line_buf
    mov al, [line_buf]
    cmp al, 'h'
    je .cmd_help
    cmp al, 'q'
    je .cmd_quit
    cmp al, 'r'
    je .cmd_reboot
    ; Unknown command
    mov esi, err_msg
    call print
    ret
.cmd_help:
    mov esi, help_msg
    call print
    ret
.cmd_quit:
    hlt
    ret
.cmd_reboot:
    jmp 0xFFFF:0

prompt:   db '> ', 0
help_msg: db 'Commands: h=help q=quit r=reboot', 0x0D, 0x0A, 0
err_msg:  db 'Unknown command', 0x0D, 0x0A, 0
line_pos: dd 0
cursor:   dd 0
line_buf: times 128 db 0
"""


class X86Shell:
    """Interactive x86 shell — wraps X86CPU with keyboard/screen I/O.

    The shell assembles a minimal kernel, loads it into the CPU,
    and provides Python methods for keyboard input and screen output.
    The CPU runs in a background thread.

    Usage::

        shell = X86Shell()
        shell.start()           # start CPU in background
        shell.type_keys("hello\\n")  # simulate keyboard input
        print(shell.read_screen())  # read VGA text output
        shell.stop()
    """

    def __init__(self, source: str = None, memory_size: int = 1024 * 1024):
        self._asm = X86Assembler()
        self._source = source or _SHELL_ASM
        self._cpu = X86CPU(memory_size=memory_size)
        self._thread = None
        self._running = False

    def start(self, max_steps: int = 1_000_000):
        """Assemble, load, and start the CPU in a background thread."""
        code = self._asm.assemble(self._source)
        self._cpu.load(code, 0x1000)
        self._running = True
        self._thread = threading.Thread(target=self._run_loop,
                                         args=(max_steps,), daemon=True)
        self._thread.start()

    def _run_loop(self, max_steps: int):
        """CPU run loop — executes until stop() or max_steps."""
        cpu = self._cpu
        step = 0
        while self._running and step < max_steps:
            cpu.transfer_key()
            if not cpu.step():
                break
            step += 1
        self._running = False

    def stop(self):
        """Stop the CPU."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def type_keys(self, text: str):
        """Send characters to the keyboard buffer."""
        for ch in text:
            self._cpu.push_key(ch)
            time.sleep(0.001)  # small delay to let CPU process

    def read_screen(self, width: int = 80, height: int = 25) -> str:
        """Read VGA text mode screen (0xB8000) as a string."""
        lines = []
        for row in range(height):
            line = ""
            for col in range(width):
                offset = (row * width + col) * 2
                ch = self._cpu._mem[0xB8000 + offset]
                line += chr(ch) if 32 <= ch < 127 else ' '
            lines.append(line.rstrip())
        return "\n".join(lines)

    @property
    def running(self) -> bool:
        return self._running


# ── Program Loader ───────────────────────────────────────────────────────────

class DiskProgramLoader:
    """Load and execute programs from a FlatFS filesystem.

    Programs are stored as assembly source (.asm) files. The loader reads
    the source, assembles it, and can execute it directly or return the
    compiled instructions.
    """

    def __init__(self, filesystem: FlatFS):
        self._fs = filesystem
        self._assembler = Assembler()

    def list_programs(self) -> list[str]:
        """List all .asm files on the filesystem."""
        return [f for f in self._fs.list_files() if f.endswith('.asm')]

    def load_source(self, name: str) -> str:
        """Read assembly source from filesystem."""
        if not name.endswith('.asm'):
            name = name + '.asm'
        data = self._fs.read(name)
        return data.decode('utf-8', errors='replace').rstrip('\x00')

    def assemble(self, source: str) -> list:
        """Assemble source into instructions."""
        return self._assembler.assemble(source)

    def run(self, name: str, max_steps: int = 10000,
            stdin_fn=None, stdout_fn=None) -> dict:
        """Load, assemble, and run a program. Returns output and stats."""
        source = self.load_source(name)
        instructions = self.assemble(source)

        bus = DeviceBus()
        if stdout_fn or stdin_fn:
            bus.register_console(stdin_fn=stdin_fn, stdout_fn=stdout_fn)
        cpu = CPU(devices=bus)
        cpu.load_program(instructions)
        output = cpu.run(max_steps=max_steps)

        return {
            "name": name,
            "output": output,
            "steps": cpu._step_count,
            "source": source,
        }

    def save_program(self, name: str, source: str) -> None:
        """Save assembly source to filesystem."""
        if not name.endswith('.asm'):
            name = name + '.asm'
        self._fs.write(name, source.encode('utf-8'))


# ── Integrated Virtual System ────────────────────────────────────────────────

class VirtualSystem:
    """Integrated virtual computer — CPU + Memory + DeviceBus + optional devices.

    Wires together the components into a single runnable system.
    """

    def __init__(self, enable_block: bool = False, enable_console: bool = True,
                 stdin_fn=None, stdout_fn=None, syscall_handler=None):
        self.memory = Memory()
        self.bus = DeviceBus()

        if enable_console:
            self.bus.register_console(stdin_fn=stdin_fn, stdout_fn=stdout_fn)
        if enable_block:
            self.block = BlockDevice()
            self.bus.register("block", self.block)

        self.cpu = CPU(memory=self.memory, devices=self.bus)

        if syscall_handler is not None:
            set_syscall_handler(syscall_handler)

    def load_program(self, source: str) -> int:
        """Assemble and load program. Returns instruction count."""
        assembler = Assembler()
        instructions = assembler.assemble(source)
        self.cpu.load_program(instructions)
        return len(instructions)

    def run(self, max_steps: int = 10000) -> list[str]:
        """Run the loaded program. Returns printed output."""
        return self.cpu.run(max_steps=max_steps)

    def status(self) -> dict:
        return {
            "pc": self.cpu.pc,
            "sp": self.cpu.sp,
            "regs": list(self.cpu.regs),
            "cmp_flag": self.cpu._cmp_flag,
            "carry_flag": self.cpu._carry_flag,
            "steps": self.cpu._step_count,
            "devices": self.bus.list_devices(),
            "heap_entries": len(self.memory._heap),
        }

    def reset(self) -> None:
        self.cpu = CPU(memory=self.memory, devices=self.bus)


# ══════════════════════════════════════════════════════════════════════════════
# OS Layer — processes, scheduling, syscalls, memory management
# ══════════════════════════════════════════════════════════════════════════════


# ── Page Frame Allocator ─────────────────────────────────────────────────────

class PageFrameAllocator:
    """Bitmap-based physical page frame allocator.

    Manages a flat 32-bit address space divided into 4 KB pages.  The bitmap
    tracks which frames are allocated.  Supports first-fit allocation and
    coalescing on free.

    Memory layout (default 4 MB):
        0x000000 - 0x000FFF  Page 0   (reserved / IVT / BDA)
        0x001000 - 0x001FFF  Page 1   (reserved / GDT)
        ...
        0x004000 - 0x004FFF  Page 4   (keyboard buffer area)
        ...
        0x0F0000 - 0x0FFFFF  Page 240  (VGA region — not allocatable)
        0x100000 - ...       Page 256+ (high memory)
    """

    PAGE_SIZE = 0x1000  # 4 KB

    def __init__(self, total_memory: int = 4 * 1024 * 1024,
                 reserved_ranges: list[tuple[int, int]] | None = None):
        """Initialize the page frame allocator.

        Args:
            total_memory: Total physical memory in bytes.
            reserved_ranges: List of (start, end) byte ranges to mark reserved.
        """
        self._total_memory = total_memory
        self._total_frames = total_memory // self.PAGE_SIZE
        # Bitmap: 1 bit per frame. 1 = allocated, 0 = free.
        self._bitmap = bytearray((self._total_frames + 7) // 8)
        self._allocated_count = 0

        # Reserve page 0 (IVT/BDA area) always
        self._mark_reserved(0)

        # Reserve caller-specified ranges
        if reserved_ranges:
            for start, end in reserved_ranges:
                start_frame = start // self.PAGE_SIZE
                end_frame = (end + self.PAGE_SIZE - 1) // self.PAGE_SIZE
                for f in range(start_frame, min(end_frame, self._total_frames)):
                    self._mark_reserved(f)

    def _mark_reserved(self, frame: int):
        """Mark a frame as allocated (reserved)."""
        byte_idx = frame >> 3
        bit_idx = frame & 7
        if byte_idx < len(self._bitmap):
            if not (self._bitmap[byte_idx] & (1 << bit_idx)):
                self._bitmap[byte_idx] |= (1 << bit_idx)
                self._allocated_count += 1

    def _is_allocated(self, frame: int) -> bool:
        byte_idx = frame >> 3
        bit_idx = frame & 7
        return bool(self._bitmap[byte_idx] & (1 << bit_idx))

    def alloc(self, count: int = 1) -> int | None:
        """Allocate contiguous page frames. Returns base address or None."""
        if count <= 0:
            return None
        run = 0
        start = 0
        for f in range(self._total_frames):
            if not self._is_allocated(f):
                if run == 0:
                    start = f
                run += 1
                if run == count:
                    for i in range(start, start + count):
                        self._mark_reserved(i)
                    return start * self.PAGE_SIZE
            else:
                run = 0
        return None

    def free(self, address: int, count: int = 1):
        """Free page frames starting at address."""
        frame = address // self.PAGE_SIZE
        for i in range(frame, frame + count):
            byte_idx = i >> 3
            bit_idx = i & 7
            if byte_idx < len(self._bitmap) and (self._bitmap[byte_idx] & (1 << bit_idx)):
                self._bitmap[byte_idx] &= ~(1 << bit_idx)
                self._allocated_count -= 1

    def free_single(self, address: int):
        """Free a single page frame."""
        self.free(address, 1)

    @property
    def free_frames(self) -> int:
        return self._total_frames - self._allocated_count

    @property
    def free_bytes(self) -> int:
        return self.free_frames * self.PAGE_SIZE

    @property
    def used_frames(self) -> int:
        return self._allocated_count

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def stats(self) -> dict:
        return {
            "total_frames": self._total_frames,
            "allocated": self._allocated_count,
            "free": self.free_frames,
            "total_bytes": self._total_memory,
            "free_bytes": self.free_bytes,
            "page_size": self.PAGE_SIZE,
        }


# ── Process Control Block ───────────────────────────────────────────────────

class ProcessState:
    """Process lifecycle states."""
    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    WAITING = "waiting"    # blocked on I/O
    TERMINATED = "terminated"


class ProcessControlBlock:
    """Per-process state for context switching.

    Stores the full CPU register state so a process can be suspended
    and resumed transparently.
    """

    _next_pid = 1

    def __init__(self, name: str = "unnamed", priority: int = 0):
        self.pid = ProcessControlBlock._next_pid
        ProcessControlBlock._next_pid += 1
        self.name = name
        self.priority = priority
        self.state = ProcessState.CREATED

        # Register state (saved on context switch)
        self.eax = 0
        self.ecx = 0
        self.edx = 0
        self.ebx = 0
        self.esp = 0
        self.ebp = 0
        self.esi = 0
        self.edi = 0
        self.eip = 0
        self.eflags = 0x00000002  # bit 1 always set

        # Memory
        self.stack_base = 0       # start of stack region
        self.stack_size = 0       # size in bytes
        self.heap_base = 0        # start of heap region
        self.heap_size = 0

        # I/O
        self.stdin_fd = -1
        self.stdout_fd = -1

        # Scheduling
        self.time_slice = 0       # remaining ticks in current quantum
        self.total_ticks = 0      # total CPU ticks consumed
        self.wait_ticks = 0       # ticks spent waiting

        # Accounting
        self.exit_code = 0
        self.children: list[int] = []  # child PIDs
        self.parent_pid = 0

    def save_from_cpu(self, cpu: X86CPU):
        """Save CPU registers into this PCB."""
        self.eax = cpu._regs[0]
        self.ecx = cpu._regs[1]
        self.edx = cpu._regs[2]
        self.ebx = cpu._regs[3]
        self.esp = cpu._regs[4]
        self.ebp = cpu._regs[5]
        self.esi = cpu._regs[6]
        self.edi = cpu._regs[7]
        self.eip = cpu._eip
        self.eflags = cpu._eflags

    def restore_to_cpu(self, cpu: X86CPU):
        """Restore PCB registers into CPU."""
        cpu._regs[0] = self.eax & 0xFFFFFFFF
        cpu._regs[1] = self.ecx & 0xFFFFFFFF
        cpu._regs[2] = self.edx & 0xFFFFFFFF
        cpu._regs[3] = self.ebx & 0xFFFFFFFF
        cpu._regs[4] = self.esp & 0xFFFFFFFF
        cpu._regs[5] = self.ebp & 0xFFFFFFFF
        cpu._regs[6] = self.esi & 0xFFFFFFFF
        cpu._regs[7] = self.edi & 0xFFFFFFFF
        cpu._eip = self.eip & 0xFFFFFFFF
        cpu._eflags = self.eflags & 0xFFFFFFFF

    def __repr__(self):
        return (f"PCB(pid={self.pid}, name={self.name!r}, "
                f"state={self.state}, eip=0x{self.eip:08X})")


# ── Process Table ────────────────────────────────────────────────────────────

class ProcessTable:
    """Central process table — the kernel's view of all processes."""

    def __init__(self):
        self._processes: dict[int, ProcessControlBlock] = {}
        self._by_name: dict[str, list[int]] = {}

    def create(self, name: str = "unnamed", priority: int = 0) -> ProcessControlBlock:
        """Create a new process. Returns the PCB."""
        pcb = ProcessControlBlock(name=name, priority=priority)
        self._processes[pcb.pid] = pcb
        self._by_name.setdefault(name, []).append(pcb.pid)
        return pcb

    def get(self, pid: int) -> ProcessControlBlock | None:
        return self._processes.get(pid)

    def get_by_name(self, name: str) -> list[ProcessControlBlock]:
        pids = self._by_name.get(name, [])
        return [self._processes[p] for p in pids if p in self._processes]

    def remove(self, pid: int) -> ProcessControlBlock | None:
        pcb = self._processes.pop(pid, None)
        if pcb:
            name_list = self._by_name.get(pcb.name, [])
            if pid in name_list:
                name_list.remove(pid)
            if not name_list:
                self._by_name.pop(pcb.name, None)
        return pcb

    def all(self) -> list[ProcessControlBlock]:
        return list(self._processes.values())

    def by_state(self, state: str) -> list[ProcessControlBlock]:
        return [p for p in self._processes.values() if p.state == state]

    def count(self) -> int:
        return len(self._processes)

    def alive_count(self) -> int:
        return sum(1 for p in self._processes.values()
                   if p.state != ProcessState.TERMINATED)


# ── Scheduler ────────────────────────────────────────────────────────────────

class Scheduler:
    """Round-robin preemptive scheduler.

    Uses a time quantum (tick count).  When a process exhausts its quantum,
    it is preempted and moved to the back of the ready queue.

    Integration: call `tick(cpu)` from the PIT interrupt handler.  It saves
    the current process, advances the quantum, and context-switches if needed.
    """

    def __init__(self, process_table: ProcessTable, quantum: int = 10):
        self._ptable = process_table
        self._quantum = quantum
        self._current_pid: int | None = None
        self._ready_queue: list[int] = []  # PIDs in ready order
        self._tick_count = 0

    @property
    def current(self) -> ProcessControlBlock | None:
        if self._current_pid is None:
            return None
        return self._ptable.get(self._current_pid)

    def enqueue(self, pid: int):
        """Add a process to the ready queue."""
        if pid not in self._ready_queue:
            pcb = self._ptable.get(pid)
            if pcb and pcb.state != ProcessState.TERMINATED:
                self._ready_queue.append(pid)
                pcb.state = ProcessState.READY
                pcb.time_slice = self._quantum

    def dequeue(self) -> int | None:
        """Remove and return the next process from the ready queue."""
        if self._ready_queue:
            return self._ready_queue.pop(0)
        return None

    def start(self, cpu: X86CPU):
        """Start scheduling — pick the first process and load it."""
        pid = self.dequeue()
        if pid is None:
            return
        self._current_pid = pid
        pcb = self._ptable.get(pid)
        if pcb:
            pcb.state = ProcessState.RUNNING
            pcb.time_slice = self._quantum
            pcb.restore_to_cpu(cpu)

    def tick(self, cpu: X86CPU):
        """Called on every timer interrupt. Handles preemption."""
        self._tick_count += 1
        current = self.current
        if current is None:
            # Try to pick a new process
            self.start(cpu)
            return

        current.total_ticks += 1
        current.time_slice -= 1

        if current.time_slice <= 0:
            # Quantum expired — preempt
            self._preempt(cpu)

    def _preempt(self, cpu: X86CPU):
        """Save current process, put it at back of ready queue, switch."""
        current = self.current
        if current is None:
            return

        current.save_from_cpu(cpu)
        current.state = ProcessState.READY
        current.time_slice = self._quantum

        # Put at back of ready queue
        self._ready_queue.append(current.pid)

        # Pick next process
        self._current_pid = None
        self.start(cpu)

    def switch_to(self, cpu: X86CPU, pid: int) -> bool:
        """Explicitly switch to a specific process. Returns True on success."""
        pcb = self._ptable.get(pid)
        if pcb is None or pcb.state == ProcessState.TERMINATED:
            return False

        # Save current
        current = self.current
        if current and current.state == ProcessState.RUNNING:
            current.save_from_cpu(cpu)
            current.state = ProcessState.READY
            current.time_slice = self._quantum
            self._ready_queue.append(current.pid)

        # Remove target from ready queue if present
        if pid in self._ready_queue:
            self._ready_queue.remove(pid)

        # Switch
        self._current_pid = pid
        pcb.state = ProcessState.RUNNING
        pcb.time_slice = self._quantum
        pcb.restore_to_cpu(cpu)
        return True

    def exit_current(self, cpu: X86CPU, exit_code: int = 0):
        """Terminate the current process and switch to next."""
        current = self.current
        if current is None:
            return

        current.save_from_cpu(cpu)
        current.state = ProcessState.TERMINATED
        current.exit_code = exit_code
        self._current_pid = None

        # Pick next process
        self.start(cpu)

    def block_current(self, cpu: X86CPU):
        """Block current process (e.g. waiting for I/O)."""
        current = self.current
        if current is None:
            return

        current.save_from_cpu(cpu)
        current.state = ProcessState.WAITING
        current.time_slice = self._quantum
        self._current_pid = None

        # Pick next process
        self.start(cpu)

    def unblock(self, pid: int):
        """Move a blocked process back to ready queue."""
        pcb = self._ptable.get(pid)
        if pcb and pcb.state == ProcessState.WAITING:
            self.enqueue(pid)

    def stats(self) -> dict:
        return {
            "tick_count": self._tick_count,
            "quantum": self._quantum,
            "current_pid": self._current_pid,
            "ready_queue": list(self._ready_queue),
            "ready_count": len(self._ready_queue),
            "processes": self._ptable.alive_count(),
        }


# ── Syscall Handler (INT 0x80) ──────────────────────────────────────────────

class X86SyscallHandler:
    """INT 0x80 syscall dispatcher for x86.

    Convention:
        EAX = syscall number
        EBX = arg1
        ECX = arg2
        EDX = arg3
        ESI = arg4
        EDI = arg5
        Return: EAX

    Syscall numbers:
        1  - exit          (EBX = exit code)
        2  - read          (EBX = fd, ECX = buf_addr, EDX = count) → EAX = bytes read
        3  - write         (EBX = fd, ECX = buf_addr, EDX = count) → EAX = bytes written
        4  - open          (EBX = name_addr, ECX = mode) → EAX = fd
        5  - close         (EBX = fd)
        6  - fork          () → EAX = child pid
        7  - exec          (EBX = name_addr) → EAX = 0 ok, -1 error
        8  - wait          () → EAX = child pid
        9  - brk           (EBX = new_heap_end) → EAX = 0 ok
        10 - getpid        () → EAX = pid
        11 - sbrk          (EBX = increment) → EAX = old_break
        12 - yield         ()
        13 - kill          (EBX = pid, ECX = signal)
        14 - gettimeofday  (EBX = buf_addr) → EAX = ticks
        15 - malloc        (EBX = size) → EAX = addr
        16 - free          (EBX = addr)
        17 - readdir       (EBX = buf_addr, ECX = max_entries) → EAX = count
        18 - uname         (EBX = buf_addr)
    """

    SYS_EXIT = 1
    SYS_READ = 2
    SYS_WRITE = 3
    SYS_OPEN = 4
    SYS_CLOSE = 5
    SYS_FORK = 6
    SYS_EXEC = 7
    SYS_WAIT = 8
    SYS_BRK = 9
    SYS_GETPID = 10
    SYS_SBRK = 11
    SYS_YIELD = 12
    SYS_KILL = 13
    SYS_GETTIMEOFDAY = 14
    SYS_MALLOC = 15
    SYS_FREE = 16
    SYS_READDIR = 17
    SYS_UNAME = 18
    SYS_SERIAL_WRITE = 19
    SYS_SERIAL_READ = 20
    SYS_MOUSE_READ = 21
    SYS_RTC_GETTIME = 22
    SYS_DISK_READ = 23
    SYS_DISK_WRITE = 24
    SYS_NET_SEND = 25
    SYS_NET_RECV = 26

    def __init__(self, cpu: X86CPU, process_table: ProcessTable,
                 scheduler: Scheduler, memory: 'PageFrameAllocator',
                 filesystem: FlatFS | None = None):
        self._cpu = cpu
        self._ptable = process_table
        self._scheduler = scheduler
        self._memory = memory
        self._fs = filesystem

        # Simple heap: address → size
        self._heap: dict[int, int] = {}
        self._heap_break = 0x400000  # 4 MB default heap start

        # File descriptor table per process (fd → filename)
        self._fd_table: dict[int, str] = {}
        self._next_fd = 3  # 0=stdin, 1=stdout, 2=stderr reserved

        # Clock ticks
        self._ticks = 0

        # I/O devices (set by X86VirtualSystem after construction)
        self._serial: SerialDevice | None = None
        self._mouse: MouseDevice | None = None
        self._rtc: CMOSDevice | None = None
        self._disk: DiskDevice | None = None
        self._nic: NICDevice | None = None

    def handle(self):
        """Dispatch syscall based on EAX. Called from INT 0x80 handler."""
        num = self._cpu._regs[0]  # EAX
        arg1 = self._cpu._regs[3]  # EBX
        arg2 = self._cpu._regs[1]  # ECX
        arg3 = self._cpu._regs[2]  # EDX
        arg4 = self._cpu._regs[6]  # ESI
        arg5 = self._cpu._regs[7]  # EDI

        handlers = {
            self.SYS_EXIT: lambda: self._sys_exit(arg1),
            self.SYS_READ: lambda: self._sys_read(arg1, arg2, arg3),
            self.SYS_WRITE: lambda: self._sys_write(arg1, arg2, arg3),
            self.SYS_OPEN: lambda: self._sys_open(arg1, arg2),
            self.SYS_CLOSE: lambda: self._sys_close(arg1),
            self.SYS_FORK: lambda: self._sys_fork(),
            self.SYS_EXEC: lambda: self._sys_exec(arg1),
            self.SYS_WAIT: lambda: self._sys_wait(),
            self.SYS_BRK: lambda: self._sys_brk(arg1),
            self.SYS_GETPID: lambda: self._sys_getpid(),
            self.SYS_SBRK: lambda: self._sys_sbrk(arg1),
            self.SYS_YIELD: lambda: self._sys_yield(),
            self.SYS_KILL: lambda: self._sys_kill(arg1, arg2),
            self.SYS_GETTIMEOFDAY: lambda: self._sys_gettimeofday(arg1),
            self.SYS_MALLOC: lambda: self._sys_malloc(arg1),
            self.SYS_FREE: lambda: self._sys_free(arg1),
            self.SYS_READDIR: lambda: self._sys_readdir(arg1, arg2),
            self.SYS_UNAME: lambda: self._sys_uname(arg1),
            self.SYS_SERIAL_WRITE: lambda: self._sys_serial_write(arg1),
            self.SYS_SERIAL_READ: lambda: self._sys_serial_read(),
            self.SYS_MOUSE_READ: lambda: self._sys_mouse_read(arg1),
            self.SYS_RTC_GETTIME: lambda: self._sys_rtc_gettime(arg1),
            self.SYS_DISK_READ: lambda: self._sys_disk_read(arg1, arg2, arg3),
            self.SYS_DISK_WRITE: lambda: self._sys_disk_write(arg1, arg2, arg3),
            self.SYS_NET_SEND: lambda: self._sys_net_send(arg1, arg2),
            self.SYS_NET_RECV: lambda: self._sys_net_recv(arg1, arg2),
        }

        handler = handlers.get(num)
        if handler:
            result = handler()
            self._cpu._regs[0] = result & 0xFFFFFFFF  # EAX = return
        else:
            self._cpu._regs[0] = 0xFFFFFFFF  # -1 = unknown syscall

    def tick(self):
        """Increment clock. Called by PIT."""
        self._ticks += 1

    def _read_string(self, addr: int) -> str:
        """Read a null-terminated string from CPU memory."""
        chars = []
        for _ in range(1024):
            b = self._cpu._read8(addr)
            if b == 0:
                break
            chars.append(chr(b))
            addr += 1
        return ''.join(chars)

    def _write_string(self, addr: int, s: str):
        """Write a string to CPU memory (null-terminated)."""
        for ch in s:
            self._cpu._write8(addr, ord(ch) & 0xFF)
            addr += 1
        self._cpu._write8(addr, 0)

    # ── Syscall implementations ───────────────────────────────────────────

    def _sys_exit(self, code: int) -> int:
        self._scheduler.exit_current(self._cpu, exit_code=code)
        return 0

    def _sys_read(self, fd: int, buf_addr: int, count: int) -> int:
        if fd < 0 or count <= 0:
            return -1
        if fd == 0:
            # stdin — read from keyboard buffer
            read = 0
            for _ in range(count):
                ch = self._cpu._read8(0x400)
                if ch == 0:
                    break
                self._cpu._write8(buf_addr, ch)
                self._cpu._write8(0x400, 0)  # consume
                buf_addr += 1
                read += 1
            return read
        elif self._fs and fd in self._fd_table:
            filename = self._fd_table[fd]
            data = self._fs.read(filename)
            to_read = min(count, len(data))
            for i in range(to_read):
                self._cpu._write8(buf_addr + i, data[i])
            return to_read
        return -1

    def _sys_write(self, fd: int, buf_addr: int, count: int) -> int:
        if fd < 0 or count <= 0:
            return -1
        data = bytes(self._cpu._read8(buf_addr + i) for i in range(count))
        if fd == 1 or fd == 2:
            # stdout/stderr — write to console
            text = data.decode('ascii', errors='replace')
            print(text, end='', flush=True)
            return count
        elif self._fs and fd in self._fd_table:
            filename = self._fd_table[fd]
            existing = self._fs.read(filename)
            self._fs.write(filename, existing + data)
            return count
        return -1

    def _sys_open(self, name_addr: int, mode: int) -> int:
        if not self._fs:
            return -1
        filename = self._read_string(name_addr)
        if not filename:
            return -1
        # mode: 0=read, 1=write, 2=create+write
        if mode == 0 and not self._fs.exists(filename):
            return -1
        fd = self._next_fd
        self._next_fd += 1
        self._fd_table[fd] = filename
        if mode == 2:
            if not self._fs.exists(filename):
                self._fs.write(filename, b'')
        return fd

    def _sys_close(self, fd: int) -> int:
        if fd in self._fd_table:
            del self._fd_table[fd]
            return 0
        return -1

    def _sys_fork(self) -> int:
        current = self._scheduler.current
        if current is None:
            return -1
        # Allocate stack for child
        child_stack = self._memory.alloc(4)  # 4 pages = 16 KB
        if child_stack is None:
            return -1

        child = self._ptable.create(name=current.name + "_child",
                                     priority=current.priority)
        child.parent_pid = current.pid
        current.children.append(child.pid)

        # Copy registers (fork returns 0 in child, child_pid in parent)
        child.eax = 0  # child return value
        child.ecx = current.ecx
        child.edx = current.edx
        child.ebx = current.ebx
        child.esp = child_stack + 0x4000  # top of child stack
        child.ebp = current.ebp
        child.esi = current.esi
        child.edi = current.edi
        child.eip = current.eip  # same instruction pointer
        child.eflags = current.eflags

        child.stack_base = child_stack
        child.stack_size = 0x4000
        child.heap_base = current.heap_base
        child.heap_size = current.heap_size

        self._scheduler.enqueue(child.pid)
        return child.pid  # parent gets child PID

    def _sys_exec(self, name_addr: int) -> int:
        """Replace current process with a new program from the filesystem.

        Assembles the source file, allocates fresh memory, loads the
        binary, resets the instruction pointer and stack, and returns 0.
        Returns -1 on any error (file not found, assembly failure, OOM).
        """
        if not self._fs:
            return -1
        filename = self._read_string(name_addr)
        if not self._fs.exists(filename):
            return -1

        current = self._scheduler.current
        if current is None:
            return -1

        # Free old code/stack memory if it was allocated
        if current.stack_base and self._memory:
            old_pages = (current.stack_size + 0xFFF) // 0x1000
            if old_pages > 0:
                self._memory.free(current.stack_base, old_pages)

        # Read and assemble the source
        data = self._fs.read(filename)
        source = data.decode('utf-8', errors='replace')

        # Default to 32-bit mode if no BITS directive present
        if '[BITS' not in source.upper():
            source = '[BITS 32]\n' + source

        asm = X86Assembler()
        try:
            code = asm.assemble(source)
        except Exception:
            return -1

        if not code:
            return -1

        code_size = len(code)
        pages_needed = (code_size + 0x1000) // 0x1000 + 1  # code + stack
        base = self._memory.alloc(pages_needed)
        if base is None:
            return -1

        # Load code at base address
        self._cpu._mem[base:base + code_size] = code

        # Reset process state for new program
        current.eip = base
        current.stack_base = base + code_size
        current.stack_size = pages_needed * 0x1000 - code_size
        current.esp = current.stack_base + current.stack_size - 4
        current.ebp = current.esp
        # Zero general-purpose registers for fresh execution
        current.eax = 0
        current.ecx = 0
        current.edx = 0
        current.ebx = 0
        current.esi = 0
        current.edi = 0

        return 0

    def _sys_wait(self) -> int:
        current = self._scheduler.current
        if current is None:
            return -1
        # Check if any child is terminated
        for child_pid in list(current.children):
            child = self._ptable.get(child_pid)
            if child and child.state == ProcessState.TERMINATED:
                current.children.remove(child_pid)
                self._ptable.remove(child_pid)
                return child_pid
        # No terminated child — block until one exits
        self._scheduler.block_current(self._cpu)
        return 0

    def _sys_brk(self, new_end: int) -> int:
        current = self._scheduler.current
        if current is None:
            return -1
        current.heap_break = new_end
        return 0

    def _sys_getpid(self) -> int:
        current = self._scheduler.current
        return current.pid if current else 0

    def _sys_sbrk(self, increment: int) -> int:
        current = self._scheduler.current
        if current is None:
            return -1
        old_break = self._heap_break
        self._heap_break += increment
        return old_break

    def _sys_yield(self) -> int:
        self._scheduler.tick(self._cpu)
        return 0

    def _sys_kill(self, pid: int, signal: int) -> int:
        pcb = self._ptable.get(pid)
        if pcb is None:
            return -1
        if signal == 9:  # SIGKILL
            self._scheduler.exit_current(self._cpu, exit_code=-1)
            return 0
        return 0

    def _sys_gettimeofday(self, buf_addr: int) -> int:
        if buf_addr:
            self._cpu._write32(buf_addr, self._ticks)
            self._cpu._write32(buf_addr + 4, 0)  # seconds (high)
        return self._ticks

    def _sys_malloc(self, size: int) -> int:
        if size <= 0:
            return 0
        # Align to 16 bytes
        aligned = (size + 15) & ~15
        # Find free region in heap
        heap_start = 0x400000
        addr = heap_start
        while addr < heap_start + 0x100000:  # 1 MB heap limit
            if addr not in self._heap:
                self._heap[addr] = aligned
                return addr
            addr += self._heap[addr]
            addr = (addr + 15) & ~15
        return 0  # out of memory

    def _sys_free(self, addr: int) -> int:
        if addr in self._heap:
            del self._heap[addr]
            return 0
        return -1

    def _sys_readdir(self, buf_addr: int, max_entries: int) -> int:
        if not self._fs:
            return 0
        files = self._fs.list_files()
        count = min(len(files), max_entries)
        for i in range(count):
            self._write_string(buf_addr + i * 32, files[i][:31])
        return count

    def _sys_uname(self, buf_addr: int) -> int:
        """Write system info to buffer. Simple Linux-like uname."""
        info = {
            "sysname": "SloughOS",
            "nodename": "sloughvm",
            "release": "0.1.0",
            "version": "#1 SMP",
            "machine": "i686",
        }
        offset = 0
        for field in ["sysname", "nodename", "release", "version", "machine"]:
            self._write_string(buf_addr + offset, info[field])
            offset += 65  # Linux struct utsname field size
        return 0

    # ── Serial I/O syscalls ──────────────────────────────────────────────

    def _sys_serial_write(self, byte_val: int) -> int:
        """Write a byte to the serial port. Returns 0 on success."""
        if self._serial is None:
            return -1
        self._serial.write_byte(byte_val)
        return 0

    def _sys_serial_read(self) -> int:
        """Read a byte from the serial port. Returns byte or -1 if empty."""
        if self._serial is None:
            return -1
        return self._serial.read_byte()

    # ── Mouse syscalls ───────────────────────────────────────────────────

    def _sys_mouse_read(self, buf_addr: int) -> int:
        """Read a 3-byte mouse packet into buf_addr. Returns 0 on success, -1 if no packet."""
        if self._mouse is None:
            return -1
        pkt = self._mouse.read_packet()
        if not pkt:
            return -1
        for i, b in enumerate(pkt):
            self._cpu._write8(buf_addr + i, b)
        return 0

    # ── RTC syscalls ─────────────────────────────────────────────────────

    def _sys_rtc_gettime(self, buf_addr: int) -> int:
        """Write Unix timestamp to buf_addr. Returns the timestamp."""
        if self._rtc is None:
            return -1
        ts = self._rtc.get_unix_time()
        if buf_addr:
            self._cpu._write32(buf_addr, ts)
        return ts

    # ── Disk I/O syscalls ────────────────────────────────────────────────

    def _sys_disk_read(self, lba: int, buf_addr: int, count: int) -> int:
        """Read `count` sectors from LBA into memory at buf_addr. Returns bytes read."""
        if self._disk is None or count <= 0:
            return -1
        data = self._disk.read_sectors(lba, count)
        for i, b in enumerate(data):
            self._cpu._write8(buf_addr + i, b)
        return len(data)

    def _sys_disk_write(self, lba: int, buf_addr: int, count: int) -> int:
        """Write `count` sectors from memory at buf_addr to LBA. Returns bytes written."""
        if self._disk is None or count <= 0:
            return -1
        data = bytes(self._cpu._read8(buf_addr + i) for i in range(count * 512))
        self._disk.write_sectors(lba, data)
        return len(data)

    # ── Network syscalls ─────────────────────────────────────────────────

    def _sys_net_send(self, buf_addr: int, length: int) -> int:
        """Send a network packet from memory. Returns 0 on success."""
        if self._nic is None or length <= 0:
            return -1
        data = bytes(self._cpu._read8(buf_addr + i) for i in range(length))
        ok = self._nic.send_packet(data)
        return 0 if ok else -1

    def _sys_net_recv(self, buf_addr: int, max_len: int) -> int:
        """Receive a network packet into memory. Returns packet length or -1 if empty."""
        if self._nic is None:
            return -1
        pkt = self._nic.recv_packet()
        if not pkt:
            return -1
        to_read = min(len(pkt), max_len)
        for i in range(to_read):
            self._cpu._write8(buf_addr + i, pkt[i])
        return to_read


# ── PIT (Programmable Interval Timer) ───────────────────────────────────────

class PITDevice:
    """Intel 8254 Programmable Interval Timer.

    I/O ports:
        0x40 - Channel 0 data (counter)
        0x41 - Channel 1 data
        0x42 - Channel 2 data
        0x43 - Command register

    Default frequency: 100 Hz (10 ms per tick).  Fires IRQ 0 to the CPU
    on each tick.  The scheduler and syscall handler both advance on this.
    """

    BASE_PORT = 0x40
    CMD_PORT = 0x43
    FREQ = 1193182  # Base PIT frequency in Hz

    def __init__(self, cpu: X86CPU, scheduler: Scheduler,
                 syscall_handler: X86SyscallHandler | None = None,
                 target_hz: int = 100,
                 clock: ClockDevice | None = None):
        self._cpu = cpu
        self._scheduler = scheduler
        self._syscall = syscall_handler
        self._clock = clock
        self._target_hz = target_hz
        self._tick_count = 0
        self._divider = self.FREQ // target_hz

        # Counter channels (3 channels, each 16-bit)
        self._counters = [self._divider, self._divider, self._divider]
        self._latch = [0, 0, 0]  # latched values

        # Register I/O ports
        for i in range(3):
            cpu.register_io_in(self.BASE_PORT + i, lambda ch=i: self._read_counter(ch))
            cpu.register_io_out(self.BASE_PORT + i, lambda val, ch=i: self._write_counter(ch, val))
        cpu.register_io_out(self.CMD_PORT, self._write_command)

    def _read_counter(self, channel: int) -> int:
        if self._latch[channel]:
            val = self._latch[channel] & 0xFF
            self._latch[channel] >>= 8
            return val
        return self._counters[channel] & 0xFF

    def _write_counter(self, channel: int, val: int):
        self._counters[channel] = (self._counters[channel] & 0xFF00) | (val & 0xFF)

    def _write_command(self, val: int):
        channel = (val >> 6) & 3
        if channel < 3:
            self._latch[channel] = self._counters[channel]

    def tick(self):
        """Called every instruction cycle. Fires IRQ when counter reaches 0."""
        for ch in range(3):
            self._counters[ch] -= 1
            if self._counters[ch] <= 0:
                self._counters[ch] = self._divider
                if ch == 0:
                    # Channel 0 → IRQ 0 (timer)
                    self._tick_count += 1
                    if self._clock:
                        self._clock.tick()
                    if self._syscall:
                        self._syscall.tick()
                    self._scheduler.tick(self._cpu)


# ── X86 Virtual System (OS-integrated) ──────────────────────────────────────

class X86VirtualSystem:
    """Integrated x86 virtual computer with OS layer.

    Wires together: X86CPU + memory allocator + process table +
    scheduler + syscalls + PIT + VGA + keyboard + filesystem.

    Usage::

        vs = X86VirtualSystem()
        vs.load_kernel(kernel_asm)
        vs.run(max_cycles=100000)

    Or for a shell::

        vs = X86VirtualSystem()
        vs.start_shell(shell_asm)
    """

    def __init__(self, memory_size: int = 4 * 1024 * 1024,
                 filesystem: FlatFS | None = None,
                 timer_hz: int = 100,
                 quantum: int = 10):
        # Memory allocator
        # Reserve: 0-0x40000 (low 256 KB), 0xB8000-0xC0000 (VGA)
        self._allocator = PageFrameAllocator(
            total_memory=memory_size,
            reserved_ranges=[
                (0, 0x40000),
                (0xB8000, 0xC0000),
            ]
        )

        # Filesystem
        if filesystem is not None:
            self._fs = filesystem
            self._block = filesystem._block if hasattr(filesystem, '_block') else BlockDevice()
        else:
            self._block = BlockDevice()
            self._fs = FlatFS(self._block)

        # Process table + scheduler
        self._ptable = ProcessTable()
        self._scheduler = Scheduler(self._ptable, quantum=quantum)

        # CPU
        self._cpu = X86CPU(memory_size=memory_size)

        # Syscall handler
        self._syscall = X86SyscallHandler(
            cpu=self._cpu,
            process_table=self._ptable,
            scheduler=self._scheduler,
            memory=self._allocator,
            filesystem=self._fs,
        )

        # I/O devices
        self._serial = SerialDevice(cpu=self._cpu)
        self._mouse = MouseDevice()
        self._rtc = CMOSDevice(cpu=self._cpu)
        self._disk = DiskDevice(block_device=self._block)
        self._nic = NICDevice()

        # Wire devices into syscall handler
        self._syscall._serial = self._serial
        self._syscall._mouse = self._mouse
        self._syscall._rtc = self._rtc
        self._syscall._disk = self._disk
        self._syscall._nic = self._nic

        # PIT timer
        self._pit = PITDevice(
            cpu=self._cpu,
            scheduler=self._scheduler,
            syscall_handler=self._syscall,
            target_hz=timer_hz,
        )

        # Register INT 0x80 handler
        self._cpu.register_handler(0x80, self._syscall.handle)

        # Keyboard interrupt handler
        def _keyboard_handler():
            pass  # Default handler already built into X86CPU

        self._cpu.register_handler(1, _keyboard_handler)

        # Kernel process (PID 1)
        self._kernel = self._ptable.create(name="kernel", priority=10)
        self._kernel.state = ProcessState.RUNNING
        self._kernel.stack_base = memory_size - 0x10000
        self._kernel.stack_size = 0x10000
        self._kernel.eip = 0x1000  # kernel loads at 0x1000
        self._kernel.esp = memory_size - 4
        self._scheduler._current_pid = self._kernel.pid

    @property
    def cpu(self) -> X86CPU:
        return self._cpu

    @property
    def scheduler(self) -> Scheduler:
        return self._scheduler

    @property
    def process_table(self) -> ProcessTable:
        return self._ptable

    @property
    def filesystem(self) -> FlatFS:
        return self._fs

    @property
    def serial(self) -> SerialDevice:
        return self._serial

    @property
    def mouse(self) -> MouseDevice:
        return self._mouse

    @property
    def rtc(self) -> CMOSDevice:
        return self._rtc

    @property
    def disk(self) -> DiskDevice:
        return self._disk

    @property
    def nic(self) -> NICDevice:
        return self._nic

    def load_kernel(self, source: str, org: int = 0x1000):
        """Assemble and load kernel code."""
        asm = X86Assembler()
        code = asm.assemble(source)
        self._cpu.load(code, org)
        self._kernel.eip = org
        self._kernel.esp = self._cpu._mem_size - 4

    def spawn(self, name: str, source: str, org: int = 0x100000) -> int | None:
        """Assemble a user program and spawn it as a new process.

        Returns the new PID, or None on failure.
        """
        # Allocate memory for the process
        code = X86Assembler().assemble(source)
        code_size = len(code)
        # Round up to page boundary
        pages_needed = (code_size + 0x1000) // 0x1000 + 1  # code + stack
        base = self._allocator.alloc(pages_needed)
        if base is None:
            return None

        # Load code at the base address
        self._cpu._mem[base:base + code_size] = code

        # Create process
        pcb = self._ptable.create(name=name)
        pcb.eip = base
        pcb.stack_base = base + code_size
        pcb.stack_size = pages_needed * 0x1000 - code_size
        pcb.esp = pcb.stack_base + pcb.stack_size - 4

        self._scheduler.enqueue(pcb.pid)
        return pcb.pid

    def run(self, max_cycles: int = 100000):
        """Run the system for N CPU cycles (instructions)."""
        self._kernel.eip = 0x1000
        self._kernel.esp = self._cpu._mem_size - 4
        self._kernel.restore_to_cpu(self._cpu)

        cycles = 0
        while cycles < max_cycles:
            # Run a batch of instructions
            batch = min(1000, max_cycles - cycles)
            for _ in range(batch):
                if not self._cpu.step():
                    return cycles
                # PIT tick every 100 instructions (simulates timer)
                if cycles % 100 == 0:
                    self._pit.tick()
                cycles += 1

            # Check if kernel is the only process
            alive = self._ptable.alive_count()
            if alive <= 1 and self._scheduler.current is None:
                break

        return cycles

    def status(self) -> dict:
        """Return comprehensive system status."""
        return {
            "cpu": {
                "eip": self._cpu._eip,
                "esp": self._cpu._regs[4],
                "eax": self._cpu._regs[0],
                "eflags": f"0x{self._cpu._eflags:08X}",
            },
            "memory": self._allocator.stats(),
            "scheduler": self._scheduler.stats(),
            "processes": {
                "total": self._ptable.count(),
                "alive": self._ptable.alive_count(),
                "list": [
                    {"pid": p.pid, "name": p.name, "state": p.state,
                     "eip": f"0x{p.eip:08X}"}
                    for p in self._ptable.all()
                ],
            },
            "pit_ticks": self._pit._tick_count,
            "syscall_ticks": self._syscall._ticks,
        }

    def reset(self):
        """Reset the entire system."""
        self._allocator = PageFrameAllocator(
            total_memory=self._cpu._mem_size,
            reserved_ranges=[(0, 0x40000), (0xB8000, 0xC0000)],
        )
        self._ptable = ProcessTable()
        self._scheduler = Scheduler(self._ptable,
                                     quantum=self._pit._divider if hasattr(self, '_pit') else 10)
        self._cpu = X86CPU(memory_size=self._cpu._mem_size)
        # Re-register everything...
        self._syscall = X86SyscallHandler(
            cpu=self._cpu, process_table=self._ptable,
            scheduler=self._scheduler, memory=self._allocator,
            filesystem=self._fs,
        )
        self._kernel = self._ptable.create(name="kernel", priority=10)
        self._kernel.state = ProcessState.RUNNING


# ── Re-exports from submodules ──────────────────────────────────────────────

from .vm_programs import (  # noqa: E402, F401
    HELLO_ASM, CLASSICAL_ASM, TENSOR_MATH_ASM, MATRIX_MUL_ASM,
    NEURAL_NET_ASM, LOOP_ASM, FUNCTION_ASM, MIXED_ASM,
    NPU_PROGRAM_ASM, COUNTER_ASM, FIB_ASM, COLLATZ_ASM,
    self_test,
)

from .vm_devices import (  # noqa: E402, F401
    TensorDevice, PythonExecDevice, SlonetDevice,
    MultimodalDevice, EngineDevice, SlonetTrainingDevice, NPUVMDevice,
)
