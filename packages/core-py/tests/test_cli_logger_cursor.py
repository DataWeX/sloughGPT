"""Tests for CLILogger cursor control methods."""
import threading
from unittest.mock import MagicMock

import pytest


class TestCursorControl:
    """Tests for CLILogger cursor control methods."""

    def test_cursor_up_writes_ansi(self):
        from domains.logging.cli_logger import CLILogger

        stream = MagicMock()
        stream.isatty.return_value = True
        stream.write = MagicMock()
        stream.flush = MagicMock()

        logger = CLILogger.__new__(CLILogger)
        logger._stream = stream
        logger._colors = False
        logger._lock = threading.Lock()

        logger.cursor_up(3)
        stream.write.assert_called_with("\033[3A")

    def test_cursor_down_writes_ansi(self):
        from domains.logging.cli_logger import CLILogger

        stream = MagicMock()
        stream.isatty.return_value = True
        stream.write = MagicMock()
        stream.flush = MagicMock()

        logger = CLILogger.__new__(CLILogger)
        logger._stream = stream
        logger._colors = False
        logger._lock = threading.Lock()

        logger.cursor_down(2)
        stream.write.assert_called_with("\033[2B")

    def test_clear_line_writes_ansi(self):
        from domains.logging.cli_logger import CLILogger

        stream = MagicMock()
        stream.isatty.return_value = True
        stream.write = MagicMock()
        stream.flush = MagicMock()

        logger = CLILogger.__new__(CLILogger)
        logger._stream = stream
        logger._colors = False
        logger._lock = threading.Lock()

        logger.clear_line()
        stream.write.assert_called_with("\033[2K")

    def test_clear_lines_writes_multiple(self):
        from domains.logging.cli_logger import CLILogger

        stream = MagicMock()
        stream.isatty.return_value = True
        stream.write = MagicMock()
        stream.flush = MagicMock()

        logger = CLILogger.__new__(CLILogger)
        logger._stream = stream
        logger._colors = False
        logger._lock = threading.Lock()

        logger.clear_lines(3)
        # Each iteration: cursor_up(1) + clear_line() = 2 writes per line
        # 3 lines * 2 writes = 6 calls
        assert stream.write.call_count == 6

    def test_save_restore_position_write_ansi(self):
        from domains.logging.cli_logger import CLILogger

        stream = MagicMock()
        stream.isatty.return_value = True
        stream.write = MagicMock()
        stream.flush = MagicMock()

        logger = CLILogger.__new__(CLILogger)
        logger._stream = stream
        logger._colors = False
        logger._lock = threading.Lock()

        logger.save_position()
        stream.write.assert_called_with("\033[s")

        logger.restore_position()
        stream.write.assert_called_with("\033[u")

    def test_cursor_methods_skip_on_non_tty(self):
        from domains.logging.cli_logger import CLILogger

        stream = MagicMock()
        stream.isatty.return_value = False
        stream.write = MagicMock()
        stream.flush = MagicMock()

        logger = CLILogger.__new__(CLILogger)
        logger._stream = stream
        logger._colors = False
        logger._lock = threading.Lock()

        logger.cursor_up(1)
        logger.cursor_down(1)
        logger.clear_line()
        logger.clear_lines(1)
        logger.save_position()
        logger.restore_position()

        stream.write.assert_not_called()
