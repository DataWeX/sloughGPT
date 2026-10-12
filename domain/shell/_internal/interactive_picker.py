"""Date, time and scalar pickers. — InteractivePrompt mixin.

date_picker and its calendar, the time/year/month/week/quarter/duration pickers, colour pickers, the range variants, and the locale pickers (timezone, currency, language) — the last of which reuse _select_with_search_raw.
"""

from __future__ import annotations

import sys
import termios

from domain.shell._internal.interactive_base import InteractiveBase
from domain.shell._internal.interactive_select import _SelectPrompts
from domain.shell._internal.interactive_term import (
    _BOLD,
    _CURSOR_UP,
    _CYAN,
    _DIM,
    _ERASE_LINE,
    _GREEN,
    _HIDE_CURSOR,
    _KEY_CTRL_C,
    _KEY_DOWN,
    _KEY_END,
    _KEY_ENTER,
    _KEY_ESC,
    _KEY_HOME,
    _KEY_LEFT,
    _KEY_RIGHT,
    _KEY_UP,
    _RED,
    _RESET,
    _SHOW_CURSOR,
    _rewind,
)


class _PickerPrompts(_SelectPrompts, InteractiveBase):
    """Date, time and scalar pickers..

    Split out of InteractivePrompt by .splitwork/split_interactive.py.
    """

    def date_picker(self, message: str, default: str = "") -> str:
        """Interactive date picker with year/month/day navigation.

        Args:
            message: prompt text
            default: default date as "YYYY-MM-DD" (empty = today)

        Returns:
            selected date as "YYYY-MM-DD"
        """
        import datetime as _dt

        if not self._is_tty:
            return self._ask_fallback(message, default or _dt.date.today().isoformat())
        return self._date_picker_raw(message, default)

    def _date_picker_raw(self, message: str, default: str) -> str:
        import datetime as _dt

        if default:
            try:
                parts = default.split("-")
                cur = _dt.date(int(parts[0]), int(parts[1]), int(parts[2]))
            except (ValueError, IndexError):
                cur = _dt.date.today()
        else:
            cur = _dt.date.today()
        focus = 0  # 0=year, 1=month, 2=day

        labels = ["Year", "Month", "Day"]

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            y_color = _CYAN if focus == 0 else ""
            m_color = _CYAN if focus == 1 else ""
            d_color = _CYAN if focus == 2 else ""
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  "
                f"{y_color}{_BOLD}{cur.year:>4}{_RESET}-"
                f"{m_color}{_BOLD}{cur.month:>02}{_RESET}-"
                f"{d_color}{_BOLD}{cur.day:>02}{_RESET}  "
                f"{_DIM}{labels[focus]}{_RESET}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return default or _dt.date.today().isoformat()
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return cur.isoformat()
                    if key.matches(_KEY_LEFT):
                        focus = max(0, focus - 1)
                    elif key.matches(_KEY_RIGHT):
                        focus = min(2, focus + 1)
                    elif key.matches(_KEY_UP):
                        if focus == 0:
                            cur = cur.replace(year=cur.year + 1)
                        elif focus == 1:
                            m = cur.month % 12 + 1
                            cur = cur.replace(month=m)
                        else:
                            try:
                                cur = cur.replace(day=cur.day + 1)
                            except ValueError:
                                cur = cur.replace(day=1)
                    elif key.matches(_KEY_DOWN):
                        if focus == 0:
                            cur = cur.replace(year=cur.year - 1)
                        elif focus == 1:
                            m = (cur.month - 2) % 12 + 1
                            cur = cur.replace(month=m)
                        else:
                            try:
                                cur = cur.replace(day=cur.day - 1)
                            except ValueError:
                                import calendar

                                cur = cur.replace(day=calendar.monthrange(cur.year, cur.month)[1])
        except (termios.error, OSError):
            return default or _dt.date.today().isoformat()

    def color_picker_rgb(self, message: str, default: str = "#ffffff") -> str:
        """Interactive hex color picker with RGB sliders.

        Args:
            message: prompt text
            default: default hex color (e.g. "#ff0000")

        Returns:
            selected hex color string
        """
        if not self._is_tty:
            return self._ask_fallback(message, default)
        return self._color_picker_rgb_raw(message, default)

    def _color_picker_rgb_raw(self, message: str, default: str) -> str:
        r, g, b = self._hex_to_rgb(default)
        focus = 0  # 0=r, 1=g, 2=b
        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            hex_str = f"#{r:02x}{g:02x}{b:02x}"
            bar_w = 20

            def _bar(val: int, color: str) -> str:
                filled = int(val / 255 * bar_w)
                block = "\u2588" * filled
                gap = " " * (bar_w - filled)
                return f"\u2588{color}{block}{_RESET}\u2591{gap}"

            r_bar = _bar(r, _RED if focus == 0 else "")
            g_bar = _bar(g, _GREEN if focus == 1 else "")
            b_bar = _bar(b, _CYAN if focus == 2 else "")
            lines = [
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_BOLD}{hex_str}{_RESET}",
                f"{_ERASE_LINE}\r  {_RED if focus == 0 else ''}R {r:>3}{_RESET} {r_bar}",
                f"{_ERASE_LINE}\r  {_GREEN if focus == 1 else ''}G {g:>3}{_RESET} {g_bar}",
                f"{_ERASE_LINE}\r  {_CYAN if focus == 2 else ''}B {b:>3}{_RESET} {b_bar}",
                f"{_ERASE_LINE}\r  {_DIM}Left/Right: channel  Up/Down: adjust  Enter: confirm{_RESET}{_HIDE_CURSOR}",
            ]
            return "\n".join(lines)

        def _clear() -> None:
            # `up` then `erase` clears the four rows *above* the footer and
            # leaves the one the cursor is sitting on. `_rewind` starts at the
            # cursor's own row and walks up from there.
            _rewind(prev_rows)
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    # Undo the frame just written before writing the next —
                    # `_ERASE_LINE` alone clears only the footer row, so each
                    # key used to drop the copy below the previous one.
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return default
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return f"#{r:02x}{g:02x}{b:02x}"
                    if key.matches(_KEY_LEFT):
                        focus = (focus + 1) % 3
                    elif key.matches(_KEY_RIGHT):
                        focus = (focus + 2) % 3
                    elif key.matches(_KEY_UP):
                        vals = [r, g, b]
                        vals[focus] = min(255, vals[focus] + 5)
                        r, g, b = vals
                    elif key.matches(_KEY_DOWN):
                        vals = [r, g, b]
                        vals[focus] = max(0, vals[focus] - 5)
                        r, g, b = vals
        except (termios.error, OSError):
            return default

    def time_picker(self, message: str, default: str = "") -> str:
        """Interactive time picker with hour/minute/AM-PM navigation."""
        import datetime as _dt

        if not self._is_tty:
            now = _dt.datetime.now()
            return self._ask_fallback(message, default or now.strftime("%I:%M %p"))
        return self._time_picker_raw(message, default)

    def _time_picker_raw(self, message: str, default: str) -> str:
        import datetime as _dt

        now = _dt.datetime.now()
        h, m = now.hour, now.minute
        ampm = "AM" if h < 12 else "PM"
        if default:
            try:
                parts = default.replace("  ", " ").split()
                hm = parts[0].split(":")
                h, m = int(hm[0]), int(hm[1])
                ampm = parts[1].upper() if len(parts) > 1 else ampm
            except (ValueError, IndexError):
                pass
        if h > 12:
            h -= 12
        if h == 0:
            h = 12
        focus = 0

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            hc = _CYAN if focus == 0 else ""
            mc = _CYAN if focus == 1 else ""
            ac = _CYAN if focus == 2 else ""
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  "
                f"{hc}{_BOLD}{h:>2}{_RESET}:"
                f"{mc}{_BOLD}{m:>02}{_RESET} "
                f"{ac}{_BOLD}{ampm}{_RESET}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return default or now.strftime("%I:%M %p")
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return f"{h:>2}:{m:02d} {ampm}"
                    if key.matches(_KEY_LEFT):
                        focus = (focus - 1) % 3
                    elif key.matches(_KEY_RIGHT):
                        focus = (focus + 1) % 3
                    elif key.matches(_KEY_UP):
                        if focus == 0:
                            h = h % 12 + 1
                        elif focus == 1:
                            m = (m + 5) % 60
                        else:
                            ampm = "PM" if ampm == "AM" else "AM"
                    elif key.matches(_KEY_DOWN):
                        if focus == 0:
                            h = (h - 2) % 12 + 1
                        elif focus == 1:
                            m = (m - 5) % 60
                        else:
                            ampm = "PM" if ampm == "AM" else "AM"
        except (termios.error, OSError):
            return default or now.strftime("%I:%M %p")

    def year_picker(
        self, message: str, default: int = 0, min_year: int = 1900, max_year: int = 2100
    ) -> int:
        """Interactive year picker with arrow keys.

        Args:
            message: prompt text
            default: default year (0 = current year)
            min_year: minimum selectable year
            max_year: maximum selectable year

        Returns:
            selected year as integer
        """
        import datetime as _dt

        if not self._is_tty:
            return int(self._ask_fallback(message, str(default or _dt.date.today().year)))
        return self._year_picker_raw(message, default or _dt.date.today().year, min_year, max_year)

    def _year_picker_raw(self, message: str, year: int, min_y: int, max_y: int) -> int:

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            return f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_BOLD}{_CYAN}{year:>4}{_RESET}{_HIDE_CURSOR}"

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return year
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return year
                    if key.matches(_KEY_UP):
                        year = min(max_y, year + 1)
                    elif key.matches(_KEY_DOWN):
                        year = max(min_y, year - 1)
                    elif key.kind == _KEY_HOME:
                        year = min_y
                    elif key.kind == _KEY_END:
                        year = max_y
        except (termios.error, OSError):
            return year

    def month_picker(self, message: str, default: int = 0) -> int:
        """Interactive month picker (1-12) with names.

        Args:
            message: prompt text
            default: default month (0 = current month)

        Returns:
            selected month as integer (1-12)
        """
        import datetime as _dt

        if not self._is_tty:
            return int(self._ask_fallback(message, str(default or _dt.date.today().month)))
        return self._month_picker_raw(message, default or _dt.date.today().month)

    def _month_picker_raw(self, message: str, month: int) -> int:
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  "
                f"{_BOLD}{_CYAN}{months[month - 1]:>9}{_RESET} ({month:>2}/12){_HIDE_CURSOR}"
            )

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return month
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return month
                    if key.matches(_KEY_UP):
                        month = month % 12 + 1
                    elif key.matches(_KEY_DOWN):
                        month = (month - 2) % 12 + 1
                    elif key.matches(_KEY_LEFT):
                        month = max(1, month - 1)
                    elif key.matches(_KEY_RIGHT):
                        month = min(12, month + 1)
        except (termios.error, OSError):
            return month

    def duration_picker(self, message: str, default: int = 0) -> int:
        """Interactive duration picker in seconds with h/m/s display.

        Args:
            message: prompt text
            default: default duration in seconds

        Returns:
            selected duration in seconds
        """
        if not self._is_tty:
            return int(self._ask_fallback(message, str(default)))
        return self._duration_picker_raw(message, default)

    def _duration_picker_raw(self, message: str, total: int) -> int:
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        focus = 0  # 0=h, 1=m, 2=s

        def _fmt() -> str:
            return f"{h:>2}h {m:>02}m {s:>02}s"

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            parts = [f"{h:>2}h", f"{m:>02}m", f"{s:>02}s"]
            colored = []
            for i, p in enumerate(parts):
                if i == focus:
                    colored.append(f"{_BOLD}{_CYAN}{p}{_RESET}")
                else:
                    colored.append(p)
            return f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {' '.join(colored)}{_HIDE_CURSOR}"

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return total
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return h * 3600 + m * 60 + s
                    if key.matches(_KEY_LEFT):
                        focus = (focus - 1) % 3
                    elif key.matches(_KEY_RIGHT):
                        focus = (focus + 1) % 3
                    elif key.matches(_KEY_UP):
                        if focus == 0:
                            h = min(99, h + 1)
                        elif focus == 1:
                            m = (m + 5) % 60
                        else:
                            s = (s + 5) % 60
                    elif key.matches(_KEY_DOWN):
                        if focus == 0:
                            h = max(0, h - 1)
                        elif focus == 1:
                            m = (m - 5) % 60
                        else:
                            s = (s - 5) % 60
        except (termios.error, OSError):
            return total

    def week_picker(self, message: str, default: int = 0) -> int:
        """Interactive week-of-year picker (1-52).

        Args:
            message: prompt text
            default: default week (0 = current week)

        Returns:
            selected week number (1-52)
        """
        import datetime as _dt

        if not self._is_tty:
            return int(
                self._ask_fallback(message, str(default or _dt.date.today().isocalendar()[1]))
            )
        return self._week_picker_raw(message, default or _dt.date.today().isocalendar()[1])

    def _week_picker_raw(self, message: str, week: int) -> int:

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            import datetime as _dt

            jan1 = _dt.date(_dt.date.today().year, 1, 1)
            start = jan1 + _dt.timedelta(weeks=week - 1)
            end = start + _dt.timedelta(days=6)
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  "
                f"{_BOLD}{_CYAN}Week {week:>2}{_RESET}  "
                f"{_DIM}{start.strftime('%b %d')} - {end.strftime('%b %d')}{_RESET}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return week
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return week
                    if key.matches(_KEY_UP):
                        week = min(52, week + 1)
                    elif key.matches(_KEY_DOWN):
                        week = max(1, week - 1)
        except (termios.error, OSError):
            return week

    def quarter_picker(self, message: str, default: int = 0) -> int:
        """Interactive quarter picker (1-4).

        Args:
            message: prompt text
            default: default quarter (0 = current quarter)

        Returns:
            selected quarter (1-4)
        """
        import datetime as _dt

        if not self._is_tty:
            return int(
                self._ask_fallback(message, str(default or (_dt.date.today().month - 1) // 3 + 1))
            )
        return self._quarter_picker_raw(message, default or (_dt.date.today().month - 1) // 3 + 1)

    def _quarter_picker_raw(self, message: str, q: int) -> int:
        quarters = {
            1: "Q1 (Jan-Mar)",
            2: "Q2 (Apr-Jun)",
            3: "Q3 (Jul-Sep)",
            4: "Q4 (Oct-Dec)",
        }

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            return f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_BOLD}{_CYAN}{quarters[q]}{_RESET}{_HIDE_CURSOR}"

        def _clear() -> None:
            sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}{_SHOW_CURSOR}")
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return q
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return q
                    if key.matches(_KEY_UP, _KEY_RIGHT):
                        q = q % 4 + 1
                    elif key.matches(_KEY_DOWN, _KEY_LEFT):
                        q = (q - 2) % 4 + 1
        except (termios.error, OSError):
            return q

    def timezone_picker(self, message: str, default: str = "UTC") -> str:
        """Interactive timezone picker from common timezones.

        Args:
            message: prompt text
            default: default timezone

        Returns:
            selected timezone string
        """
        timezones = [
            "UTC",
            "US/Eastern",
            "US/Central",
            "US/Mountain",
            "US/Pacific",
            "Europe/London",
            "Europe/Paris",
            "Europe/Berlin",
            "Europe/Moscow",
            "Asia/Tokyo",
            "Asia/Shanghai",
            "Asia/Kolkata",
            "Asia/Dubai",
            "Australia/Sydney",
            "Pacific/Auckland",
            "America/Sao_Paulo",
            "Africa/Cairo",
            "Africa/Lagos",
            "Asia/Singapore",
            "Asia/Seoul",
        ]
        if not self._is_tty:
            return self._ask_fallback(message, default)
        return self._select_with_search_raw(message, timezones)

    def currency_picker(self, message: str, default: str = "USD") -> str:
        """Interactive currency picker from common currencies.

        Args:
            message: prompt text
            default: default currency code

        Returns:
            selected 3-letter currency code
        """
        currencies = [
            "USD - US Dollar",
            "EUR - Euro",
            "GBP - British Pound",
            "JPY - Japanese Yen",
            "CNY - Chinese Yuan",
            "KRW - Korean Won",
            "INR - Indian Rupee",
            "BRL - Brazilian Real",
            "CAD - Canadian Dollar",
            "AUD - Australian Dollar",
            "CHF - Swiss Franc",
            "MXN - Mexican Peso",
            "SGD - Singapore Dollar",
            "HKD - Hong Kong Dollar",
            "SEK - Swedish Krona",
            "NOK - Norwegian Krone",
            "DKK - Danish Krone",
            "PLN - Polish Zloty",
            "THB - Thai Baht",
            "ZAR - South African Rand",
        ]
        if not self._is_tty:
            return self._ask_fallback(message, default)
        result = self._select_with_search_raw(message, currencies)
        return result[:3] if result else default

    def language_picker(self, message: str, default: str = "en") -> str:
        """Interactive language picker from common languages.

        Args:
            message: prompt text
            default: default language code

        Returns:
            selected 2-letter language code
        """
        languages = [
            "en - English",
            "es - Spanish",
            "fr - French",
            "de - German",
            "it - Italian",
            "pt - Portuguese",
            "ru - Russian",
            "zh - Chinese",
            "ja - Japanese",
            "ko - Korean",
            "ar - Arabic",
            "hi - Hindi",
            "nl - Dutch",
            "sv - Swedish",
            "pl - Polish",
            "tr - Turkish",
            "vi - Vietnamese",
            "th - Thai",
            "uk - Ukrainian",
            "cs - Czech",
        ]
        if not self._is_tty:
            return self._ask_fallback(message, default)
        result = self._select_with_search_raw(message, languages)
        return result[:2] if result else default

    def date_range_picker(
        self, message: str, default_start: str = "", default_end: str = ""
    ) -> tuple[str, str]:
        """Pick a date range (start and end dates).

        Args:
            message: prompt text
            default_start: default start date (YYYY-MM-DD)
            default_end: default end date (YYYY-MM-DD)

        Returns:
            tuple of (start_date, end_date) strings
        """
        if not self._is_tty:
            s = self._ask_fallback(f"{message} start", default_start or "2025-01-01")
            e = self._ask_fallback(f"{message} end", default_end or "2025-12-31")
            return (s.strip(), e.strip())
        return self._date_range_picker_raw(message, default_start, default_end)

    def _date_range_picker_raw(self, message: str, ds: str, de: str) -> tuple[str, str]:
        import datetime as _dt

        today = _dt.date.today()
        if not ds:
            ds = today.strftime("%Y-%m-%d")
        if not de:
            de = today.strftime("%Y-%m-%d")
        start = self.date_picker(f"{message} start date", default=ds)
        end = self.date_picker(f"{message} end date", default=de)
        return (start, end)

    def color_picker(self, message: str, default: str = "#ffffff") -> str:
        """Interactive color picker with palette and hex input.

        Args:
            message: prompt text
            default: default hex color

        Returns:
            selected hex color string
        """
        if not self._is_tty:
            return self._ask_fallback(message, default)
        return self._color_picker_raw(message, default)

    def _color_picker_raw(self, message: str, default: str) -> str:
        palette = [
            ("#ff0000", "Red"),
            ("#ff8000", "Orange"),
            ("#ffff00", "Yellow"),
            ("#00ff00", "Green"),
            ("#00ffff", "Cyan"),
            ("#0080ff", "Blue"),
            ("#8000ff", "Purple"),
            ("#ff00ff", "Magenta"),
            ("#ffffff", "White"),
            ("#808080", "Gray"),
            ("#000000", "Black"),
            ("#804000", "Brown"),
        ]
        colors = [c[0] for c in palette]
        names = [c[1] for c in palette]
        idx = 0
        if default in colors:
            idx = colors.index(default)

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            c = colors[idx]
            n = names[idx]
            _r, _g, _b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  {_DIM}(arrows, enter){_RESET}\n"
                f"  {c} {n}  "
                f"{_BOLD}████{_RESET}  "
                f"{_BOLD}{_CYAN}>>{_RESET} {c}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            _rewind(prev_rows)
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return colors[idx]
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return colors[idx]
                    if key.matches(_KEY_UP, _KEY_RIGHT):
                        idx = (idx + 1) % len(colors)
                    elif key.matches(_KEY_DOWN, _KEY_LEFT):
                        idx = (idx - 1) % len(colors)
        except (termios.error, OSError):
            return default

    def time_range_picker(
        self, message: str, default_start: str = "", default_end: str = ""
    ) -> tuple[str, str]:
        """Pick a time range (start and end times).

        Args:
            message: prompt text
            default_start: default start time (HH:MM)
            default_end: default end time (HH:MM)

        Returns:
            tuple of (start_time, end_time) strings
        """
        if not self._is_tty:
            s = self._ask_fallback(f"{message} start", default_start or "09:00")
            e = self._ask_fallback(f"{message} end", default_end or "17:00")
            return (s.strip(), e.strip())
        return self._time_range_picker_raw(message, default_start, default_end)

    def _time_range_picker_raw(self, message: str, ds: str, de: str) -> tuple[str, str]:
        import datetime as _dt

        now = _dt.datetime.now()
        if not ds:
            ds = now.strftime("%H:%M")
        if not de:
            de = now.strftime("%H:%M")
        start = self.time_picker(f"{message} start time", default=ds)
        end = self.time_picker(f"{message} end time", default=de)
        return (start, end)

    def number_range_picker(
        self,
        message: str,
        min_val: int = 0,
        max_val: int = 100,
        default: int = 0,
        step: int = 1,
    ) -> int:
        """Pick a number from a range using arrow keys.

        Args:
            message: prompt text
            min_val: minimum value
            max_val: maximum value
            default: default value
            step: increment step

        Returns:
            selected number
        """
        if not self._is_tty:
            return int(self._ask_fallback(message, str(default)))
        return self._number_range_picker_raw(message, min_val, max_val, default, step)

    def _number_range_picker_raw(
        self, message: str, min_val: int, max_val: int, default: int, step: int
    ) -> int:
        val = default

        prev_rows = 0  # rows the frame on screen occupies

        def _render() -> str:
            pct = (val - min_val) / max(max_val - min_val, 1)
            bar_w = 20
            filled = int(pct * bar_w)
            bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
            return (
                f"{_ERASE_LINE}\r  {_CYAN}{message}{_RESET}  "
                f"{_BOLD}{_CYAN}{val}{_RESET}\n"
                f"  {bar}  {_DIM}{min_val} - {max_val}{_RESET}{_HIDE_CURSOR}"
            )

        def _clear() -> None:
            _rewind(prev_rows)
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()

        try:
            with self._session.raw():
                sys.stdout.write(_HIDE_CURSOR)
                sys.stdout.flush()
                while True:
                    _rewind(prev_rows)
                    frame = _render()
                    sys.stdout.write(f"\r{_ERASE_LINE}\r{frame}")
                    sys.stdout.flush()
                    prev_rows = frame.count("\n") + 1
                    key = self._session.read_key()
                    if key.kind in (_KEY_CTRL_C, _KEY_ESC):
                        _clear()
                        return val
                    if key.kind == _KEY_ENTER:
                        _clear()
                        return val
                    if key.matches(_KEY_UP, _KEY_RIGHT):
                        val = min(max_val, val + step)
                    elif key.matches(_KEY_DOWN, _KEY_LEFT):
                        val = max(min_val, val - step)
                    elif key.kind == _KEY_HOME:
                        val = min_val
                    elif key.kind == _KEY_END:
                        val = max_val
        except (termios.error, OSError):
            return default
