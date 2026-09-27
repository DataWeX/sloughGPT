from test_support import get_test_client

client = get_test_client()


def _data(resp):
    """Unwrap the success_response() envelope."""
    body = resp.json()
    return body.get("data", body)


class TestMetrics:
    def test_get_metrics_structure(self):
        resp = client.get("/system/metrics")
        assert resp.status_code == 200
        data = _data(resp)
        expected = {"cpu_percent", "memory_percent", "memory_used_gb", "memory_total_gb"}
        assert expected.issubset(data.keys())
        assert isinstance(data["cpu_percent"], (int, float))
        assert isinstance(data["memory_used_gb"], (int, float))

    def test_metrics_caching(self):
        resp1 = client.get("/system/metrics")
        resp2 = client.get("/system/metrics")
        assert _data(resp1) == _data(resp2)


class TestInfo:
    def test_get_info_structure(self):
        resp = client.get("/system/info")
        assert resp.status_code == 200
        data = _data(resp)
        expected = {
            "platform",
            "platform_release",
            "platform_version",
            "architecture",
            "processor",
            "cpu_count",
        }
        assert expected.issubset(data.keys())
        assert isinstance(data["platform"], str)
        assert isinstance(data["cpu_count"], int)


class TestDisk:
    def test_get_disk_structure(self):
        resp = client.get("/system/disk")
        assert resp.status_code == 200
        data = _data(resp)
        expected = {"total_gb", "used_gb", "free_gb", "percent"}
        assert expected.issubset(data.keys())
        assert 0 <= data["percent"] <= 100


class TestLifecycle:
    def test_get_lifecycle_structure(self):
        resp = client.get("/system/lifecycle")
        assert resp.status_code == 200
        data = _data(resp)
        assert "phase" in data
        assert "profile" in data
        assert "uptime" in data
        assert "in_flight" in data
        assert "hooks" in data
        assert "gates" in data
        assert data["phase"] in ("init", "running", "starting")
        assert data["profile"] == "full"
        assert isinstance(data["in_flight"], int)
        assert "startup" in data["hooks"]
        assert "shutdown" in data["hooks"]
        assert "preview" in data["hooks"]
        assert "total" in data["gates"]


class TestTailOutput:
    def test_tail_output_returns_lines(self):
        resp = client.get("/system/output")
        assert resp.status_code == 200
        data = _data(resp)
        assert "lines" in data
        assert "size" in data
        assert "seq" in data
        assert isinstance(data["lines"], list)

    def test_tail_output_with_limit(self):
        resp = client.get("/system/output?n=10")
        assert resp.status_code == 200
        assert isinstance(_data(resp)["lines"], list)


class TestExecutorStatus:
    def test_executor_status_uninitialized(self):
        resp = client.get("/system/executor")
        assert resp.status_code == 200
        data = _data(resp)
        assert "initialized" in data
        assert "active_jobs" in data
        assert "jobs" in data


class TestExecutorJobNotFound:
    def test_get_nonexistent_job(self):
        resp = client.get("/system/executor/nonexistent_job_id")
        assert resp.status_code in (404, 500, 503)

    def test_get_nonexistent_job_result(self):
        resp = client.get("/system/executor/nonexistent_job_id/result")
        assert resp.status_code in (404, 500, 503)


class TestExecutorPurge:
    def test_purge_returns_count(self):
        resp = client.post("/system/executor/purge")
        assert resp.status_code == 200
        data = _data(resp)
        assert "purged" in data


class TestExecutorCancel:
    def test_cancel_nonexistent_job(self):
        resp = client.post("/system/executor/nonexistent_job_id/cancel")
        assert resp.status_code == 200
        data = _data(resp)
        assert "cancelled" in data


class TestInferencePool:
    def test_pool_status(self):
        resp = client.get("/system/inference-pool")
        assert resp.status_code == 200
        data = _data(resp)
        assert "initialized" in data
        assert "max_workers" in data
        assert "queue_timeout" in data


class TestBattery:
    def test_get_battery_structure(self):
        resp = client.get("/system/battery")
        assert resp.status_code == 200
        data = _data(resp)
        assert {"status", "control", "advice"}.issubset(data.keys())

        status = data["status"]
        assert 0 <= status["level"] <= 100
        assert status["source"] in ("sysfs", "simulated")
        assert status["level_band"] in ("low", "ok", "high", "full")
        assert isinstance(status["is_charging"], bool)

        control = data["control"]
        assert {"supported", "writable", "reason"}.issubset(control.keys())
        assert isinstance(control["supported"], bool)

        advice = data["advice"]
        assert advice["action"] in ("unplug", "cap_at_80", "plug_in", "maintain")
        assert advice["reason"]
        assert 1 <= advice["limit"] <= 100

    def test_get_battery_reports_capability_not_error(self):
        """Unsupported kernels must answer 200 with supported=false, not 5xx."""
        resp = client.get("/system/battery")
        assert resp.status_code == 200
        control = _data(resp)["control"]
        if not control["supported"]:
            assert control["reason"]

    def test_set_limit_returns_control_result(self, monkeypatch):
        import chargectl
        from chargectl import ControlResult

        seen = {}

        def fake_set_limit(percent, sys_base=None):
            seen["percent"] = percent
            return ControlResult(True, True, percent, "stubbed")

        monkeypatch.setattr(chargectl, "set_limit", fake_set_limit)
        resp = client.post("/system/battery/limit", params={"percent": 80})
        assert resp.status_code == 200
        data = _data(resp)
        assert data == {
            "applied": True,
            "supported": True,
            "limit": 80,
            "reason": "stubbed",
            "path": None,
        }
        assert seen["percent"] == 80

    def test_set_limit_rejects_out_of_range(self):
        assert client.post("/system/battery/limit", params={"percent": 0}).status_code == 422
        assert client.post("/system/battery/limit", params={"percent": 101}).status_code == 422

    def test_set_limit_defaults_to_80(self, monkeypatch):
        import chargectl
        from chargectl import ControlResult

        seen = {}

        def fake_set_limit(percent, sys_base=None):
            seen["percent"] = percent
            return ControlResult(True, True, percent, "stubbed")

        monkeypatch.setattr(chargectl, "set_limit", fake_set_limit)
        resp = client.post("/system/battery/limit")
        assert resp.status_code == 200
        assert seen["percent"] == 80
