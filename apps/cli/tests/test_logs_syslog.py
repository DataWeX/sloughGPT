"""Tests for syslog line parsing in the logs command."""

import json
import sys
from pathlib import Path

# Add CLI src to path (mirrors other CLI tests)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from commands.logs import (
    _format_line,
    _matches_filters,
    _parse_line,
    _read_logs,
)


def test_parse_full_syslog_line():
    line = (
        "Sep 18 09:31:02 slo.startup[385542]: INFO [START] model loaded rid=abc id=Qwen layers=24"
    )
    rec = _parse_line(line)
    assert rec is not None
    assert rec["level"] == "INFO"
    assert rec["logger"] == "slo.startup"
    assert rec["msg"] == "model loaded"
    assert rec["tag"] == "START"
    assert rec["request_id"] == "abc"
    assert rec["ctx"] == {"id": "Qwen", "layers": "24"}
    assert rec["ts"].startswith(str(__import__("datetime").datetime.now().year))


def test_parse_minimal_syslog_line():
    rec = _parse_line("Sep  8 04:33:03 systemd[1]: INFO boot done")
    assert rec is not None
    assert rec["level"] == "INFO"
    assert rec["logger"] == "systemd"
    assert rec["msg"] == "boot done"
    assert rec["tag"] == ""
    assert rec["request_id"] == ""
    assert rec["ctx"] == {}


def test_parse_quoted_ctx_values():
    rec = _parse_line('Sep 18 09:31:02 slo.x[1]: WARNING [INFRA] slow path="/a b/c" n=3')
    assert rec is not None
    assert rec["msg"] == "slow"
    assert rec["ctx"] == {"path": "/a b/c", "n": "3"}


def test_parse_rejects_bad_level():
    assert _parse_line("Sep 18 09:31:02 slo.x[1]: BOGUS something") is None
    assert _parse_line("not a log line") is None
    assert _parse_line("") is None


def test_parse_json_back_compat():
    line = json.dumps(
        {
            "ts": "2026-09-18T09:31:02",
            "level": "ERROR",
            "logger": "slo.api",
            "msg": "boom",
            "tag": "REQ",
            "request_id": "r1",
            "ctx": {"path": "/x"},
        }
    )
    rec = _parse_line(line)
    assert rec is not None
    assert rec["level"] == "ERROR"
    assert rec["msg"] == "boom"
    assert rec["request_id"] == "r1"


def test_parse_json_v1_envelope():
    line = json.dumps(
        {
            "v": 1,
            "ts": "2026-09-18T09:31:02",
            "lvl": "INFO",
            "op": "model.load",
            "corr": "c9",
            "logger": "slo.model",
            "msg": "loaded",
        }
    )
    rec = _parse_line(line)
    assert rec is not None
    assert rec["level"] == "INFO"
    assert rec["request_id"] == "c9"


def test_round_trip_formatter_to_parser():
    import logging

    from domain.logging._internal.config import SyslogFormatter

    fmt = SyslogFormatter()
    record = logging.LogRecord(
        name="slo.test",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg="disk slow",
        args=(),
        exc_info=None,
    )
    record.process = 1234
    record.tag = "INFRA"
    record.request_id = "r7"
    record.path = "/mnt/x y"
    out = fmt.format(record)
    rec = _parse_line(out)
    assert rec is not None
    assert rec["level"] == "WARNING"
    assert rec["logger"] == "slo.test"
    assert rec["tag"] == "INFRA"
    assert rec["request_id"] == "r7"
    assert rec["msg"] == "disk slow"
    assert rec["ctx"]["path"] == "/mnt/x y"


def test_filters_and_format_on_parsed_record():
    rec = _parse_line("Sep 18 09:31:02 slo.api[9]: ERROR [REQ] timeout rid=z9")
    assert _matches_filters(rec, {"level": "error"})
    assert _matches_filters(rec, {"tag": "REQ"})
    assert _matches_filters(rec, {"search": "time"})
    assert not _matches_filters(rec, {"level": "info"})
    shown = _format_line(rec, use_color=False)
    assert "ERROR" in shown and "timeout" in shown


def test_read_logs_attaches_traceback_continuation(tmp_path, capsys):
    logf = tmp_path / "test.log"
    logf.write_text(
        "Sep 18 09:31:02 slo.api[9]: ERROR [REQ] boom rid=z9\n"
        "Traceback (most recent call last):\n"
        '  File "x.py", line 1, in <module>\n'
        "Sep 18 09:31:03 slo.api[9]: INFO [REQ] fine\n",
        encoding="utf-8",
    )
    _read_logs(logf, tail=50, filters={}, output_json=False, use_color=False)
    out = capsys.readouterr().out
    assert "boom" in out
    assert "Traceback (most recent call last)" in out
    assert "fine" in out


def test_read_logs_json_still_works(tmp_path, capsys):
    logf = tmp_path / "old.log"
    logf.write_text(
        json.dumps(
            {
                "ts": "2026-09-18T09:31:02",
                "level": "INFO",
                "logger": "slo.x",
                "msg": "legacy",
                "tag": "",
                "request_id": "",
                "ctx": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _read_logs(logf, tail=50, filters={"level": "INFO"}, output_json=False, use_color=False)
    assert "legacy" in capsys.readouterr().out
