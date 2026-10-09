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
            "floor_limit": None,
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


class TestBatteryPolicy:
    def test_get_battery_includes_policy_and_daemon(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CHARGECTL_POLICY", str(tmp_path / "policy.json"))
        monkeypatch.setenv("CHARGECTL_STATE", str(tmp_path / "state.json"))

        data = _data(client.get("/system/battery"))
        assert {"policy", "daemon"}.issubset(data.keys())

        policy = data["policy"]
        assert policy["enabled"] is False
        assert policy["band"] == "40-80"
        assert policy["error"] is None
        assert policy["explain"]
        assert str(policy["file"]).endswith("policy.json")

        assert data["daemon"]["present"] is False
        assert data["daemon"]["active"] is False
        assert str(data["daemon"]["state_file"]).endswith("state.json")

    def test_get_battery_reports_an_unreadable_policy(self, monkeypatch, tmp_path):
        path = tmp_path / "policy.json"
        path.write_text("{ not json")
        monkeypatch.setenv("CHARGECTL_POLICY", str(path))

        policy = _data(client.get("/system/battery"))["policy"]
        assert policy["error"]
        assert policy["enabled"] is False  # falls back to defaults, still 200

    def _write_state(self, tmp_path, monkeypatch, age_seconds: float) -> dict:
        import time

        from chargectl import write_state

        state_path = tmp_path / "state.json"
        write_state(
            {
                "pid": 4242,
                "updated_at": time.time() - age_seconds,
                "owned": True,
                "dry_run": False,
                "explain": "policy on — holding 40-80%",
                "action": {
                    "action": "set_ceiling",
                    "value": 80,
                    "reason": "charge cap is 100% — re-asserting 80% (band mode)",
                },
                "result": {"applied": True, "reason": "charge threshold set to 80%"},
            },
            state_path,
        )
        monkeypatch.setenv("CHARGECTL_STATE", str(state_path))
        monkeypatch.setenv("CHARGECTL_POLICY", str(tmp_path / "policy.json"))
        return _data(client.get("/system/battery"))["daemon"]

    def test_fresh_daemon_state_reads_as_active(self, monkeypatch, tmp_path):
        daemon = self._write_state(tmp_path, monkeypatch, age_seconds=5)
        assert daemon["present"] is True
        assert daemon["active"] is True
        assert daemon["pid"] == 4242
        assert daemon["owned"] is True
        assert daemon["dry_run"] is False
        assert daemon["last_action"] == "set_ceiling"
        assert daemon["last_value"] == 80
        assert "re-asserting" in daemon["last_reason"]
        assert "set to 80%" in daemon["last_outcome"]
        assert "40-80" in daemon["explain"]

    def test_stale_daemon_state_reads_as_inactive(self, monkeypatch, tmp_path):
        daemon = self._write_state(tmp_path, monkeypatch, age_seconds=60 * 60)
        assert daemon["present"] is True
        assert daemon["active"] is False  # policy set, nobody enforcing

    def test_put_policy_persists_the_band(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CHARGECTL_POLICY", str(tmp_path / "policy.json"))

        resp = client.put(
            "/system/battery/policy",
            params={"enabled": "true", "floor": 45, "ceiling": 70},
        )
        assert resp.status_code == 200
        data = _data(resp)
        assert data["ok"] is True
        assert data["error"] is None
        assert data["policy"]["enabled"] is True
        assert data["policy"]["band"] == "45-70"
        assert data["explain"]

        # the next GET reflects it — policy is shared through the file, not memory
        assert _data(client.get("/system/battery"))["policy"]["band"] == "45-70"

    def test_put_policy_is_partial(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CHARGECTL_POLICY", str(tmp_path / "policy.json"))
        client.put("/system/battery/policy", params={"floor": 50, "ceiling": 75})
        data = _data(client.put("/system/battery/policy", params={"ceiling": 90}))
        assert data["ok"] is True
        assert data["policy"]["floor"] == 50
        assert data["policy"]["ceiling"] == 90

    def test_put_policy_rejects_an_inverted_band(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CHARGECTL_POLICY", str(tmp_path / "policy.json"))
        client.put("/system/battery/policy", params={"floor": 45, "ceiling": 75})

        data = _data(client.put("/system/battery/policy", params={"floor": 90, "ceiling": 60}))
        assert data["ok"] is False
        assert "floor" in data["error"]
        # the previously saved policy is untouched
        assert data["policy"]["band"] == "45-75"

    def test_put_policy_validates_ranges(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CHARGECTL_POLICY", str(tmp_path / "policy.json"))
        assert client.put("/system/battery/policy", params={"floor": 0}).status_code == 422
        assert client.put("/system/battery/policy", params={"ceiling": 101}).status_code == 422

    def test_put_policy_can_be_disabled(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CHARGECTL_POLICY", str(tmp_path / "policy.json"))
        client.put("/system/battery/policy", params={"enabled": "true"})
        data = _data(client.put("/system/battery/policy", params={"enabled": "false"}))
        assert data["ok"] is True and data["policy"]["enabled"] is False
