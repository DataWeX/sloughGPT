"""
Unit tests for the Comprehensive Training Orchestrator and Computer-Use Agent.

Tests the core logic without requiring a running server.
"""
import json
import tempfile
import time

FAST_CONFIG = {
    "method": "sft", "data_quality_threshold": 0.0, "epochs": 1,
    "batch_size": 8, "block_size": 32, "max_steps": 2,
    "n_embed": 32, "n_layer": 2, "n_head": 2,
}

# ── Comprehensive Trainer Unit Tests ────────────────────────────────

class TestComprehensiveTrainer:
    """Unit tests for ComprehensiveTrainer."""

    def test_config_to_dict(self):
        from domains.training.comprehensive_trainer import TrainingConfig

        cfg = TrainingConfig(method="lora", epochs=5, use_lora=True)
        d = cfg.to_dict()
        assert d["method"] == "lora"
        assert d["epochs"] == 5
        assert d["use_lora"] is True
        assert "learning_rate" in d

    def test_config_defaults(self):
        from domains.training.comprehensive_trainer import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.method == "sft"
        assert cfg.epochs == 3
        assert cfg.learning_rate == 2e-4
        assert cfg.use_lora is False
        assert cfg.use_ewc is False

    def test_training_method_enum(self):
        from domains.training.comprehensive_trainer import TrainingMethod

        assert TrainingMethod.SFT.value == "sft"
        assert TrainingMethod.RLHF.value == "rlhf"
        assert TrainingMethod.DPO.value == "dpo"
        assert TrainingMethod.LORA.value == "lora"
        assert TrainingMethod.AUTO.value == "auto"

    def test_training_phase_enum(self):
        from domains.training.comprehensive_trainer import TrainingPhase

        phases = list(TrainingPhase)
        assert TrainingPhase.IDLE in phases
        assert TrainingPhase.TRAINING in phases
        assert TrainingPhase.COMPLETE in phases
        assert TrainingPhase.FAILED in phases

    def test_phase_result_dataclass(self):
        from domains.training.comprehensive_trainer import PhaseResult, TrainingPhase

        pr = PhaseResult(
            phase=TrainingPhase.VALIDATING,
            success=True,
            duration_s=1.5,
            data={"text_length": 1000},
        )
        assert pr.success is True
        assert pr.duration_s == 1.5
        assert pr.error is None

    def test_comprehensive_result_summary(self):
        from domains.training.comprehensive_trainer import (
            ComprehensiveResult,
            PhaseResult,
            TrainingPhase,
        )

        result = ComprehensiveResult(
            run_id="abc123",
            success=True,
            phases=[
                PhaseResult(phase=TrainingPhase.VALIDATING, success=True, duration_s=0.5),
                PhaseResult(phase=TrainingPhase.TRAINING, success=True, duration_s=10.0),
            ],
            method="sft",
            config={},
            final_loss=1.23,
            quality_score=0.85,
            total_duration_s=12.0,
        )
        summary = result.summary()
        assert "abc123" in summary
        assert "OK" in summary
        assert "1.23" in summary
        assert "sft" in summary

    def test_comprehensive_result_failure_summary(self):
        from domains.training.comprehensive_trainer import (
            ComprehensiveResult,
        )

        result = ComprehensiveResult(
            run_id="fail1",
            success=False,
            phases=[],
            method="sft",
            config={},
            error="Data file not found",
        )
        summary = result.summary()
        assert "FAILED" in summary
        assert "Data file not found" in summary

    def test_trainer_validates_missing_file(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        trainer = ComprehensiveTrainer()
        result = trainer.run_full_cycle(
            data_path="/nonexistent/file.jsonl",
            config={"method": "sft"},
        )
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_trainer_validates_short_data(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("short")
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(
                data_path=f.name,
                config={"method": "sft"},
            )
            assert result.success is False
            assert "too short" in result.error.lower()

    def test_trainer_validates_empty_data(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(
                data_path=f.name,
                config={"method": "sft"},
            )
            assert result.success is False

    def test_trainer_runs_validation_phase(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer, TrainingPhase

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("The quick brown fox jumps over the lazy dog. " * 200)
            f.flush()
            trainer = ComprehensiveTrainer()
            phases_recorded = []
            trainer.on_phase(lambda p, r: phases_recorded.append(p))
            trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert TrainingPhase.VALIDATING in phases_recorded

    def test_trainer_records_outcome(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            text = "The quick brown fox jumps over the lazy dog. " * 100
            text += "Pack my box with five dozen liquor jugs. " * 100
            text += "How vexingly quick daft zebras jump! " * 100
            f.write(text)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert result.success is True
            assert result.final_loss is not None

    def test_trainer_adaptive_config(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("The quick brown fox jumps over the lazy dog. " * 200)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(
                data_path=f.name,
                config={**FAST_CONFIG, "adaptive": True, "auto_config": False},
            )
            assert result.success is True

    def test_trainer_cancel(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        trainer = ComprehensiveTrainer()
        trainer.cancel()
        assert trainer._cancel_event.is_set()

    def test_trainer_phase_callback(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("The quick brown fox jumps over the lazy dog. " * 200)
            f.flush()
            trainer = ComprehensiveTrainer()
            called_phases = []
            trainer.on_phase(lambda p, r: called_phases.append(p.value))
            trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert "validating" in called_phases
            assert "configuring" in called_phases

    def test_trainer_jsonl_data(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(20):
                f.write(json.dumps({"input": f"question {i} about topic", "output": f"answer {i} with detail"}) + "\n")
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert result.success is True

    def test_trainer_multiple_methods(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        for method in ["sft", "rlhf", "dpo", "lora"]:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write("The quick brown fox jumps over the lazy dog. " * 200)
                f.flush()
                trainer = ComprehensiveTrainer()
                result = trainer.run_full_cycle(
                    data_path=f.name,
                    config={**FAST_CONFIG, "method": method},
                )
                assert result.method == method


# ── Computer-Use Agent Unit Tests ──────────────────────────────────

    def test_trainer_performance_metrics(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("The quick brown fox jumps over the lazy dog. " * 200)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert result.success is True
            assert "performance" in result.__dict__
            assert "total_duration_s" in result.performance
            assert "phase_durations" in result.performance
            assert result.performance["phases_completed"] >= 4

    def test_trainer_progress_callbacks(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("The quick brown fox jumps over the lazy dog. " * 200)
            f.flush()
            trainer = ComprehensiveTrainer()
            progress_events = []
            trainer.on_progress(lambda d: progress_events.append(d))
            trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert len(progress_events) >= 4
            assert progress_events[0]["phase"] == "validating"
            assert progress_events[-1]["phase"] == "complete"
            assert progress_events[-1]["progress"] == 1.0

    def test_trainer_result_summary_with_performance(self):
        from domains.training.comprehensive_trainer import (
            ComprehensiveResult,
            PhaseResult,
            TrainingPhase,
        )

        result = ComprehensiveResult(
            run_id="perf123",
            success=True,
            phases=[PhaseResult(phase=TrainingPhase.TRAINING, success=True, duration_s=5.0)],
            method="sft",
            config={},
            performance={"total_duration_s": 5.0, "slowest_phase": "training"},
        )
        summary = result.summary()
        assert "Performance:" in summary
        assert "total_duration_s" in summary


class TestComputerUseAgent:
    """Unit tests for ComputerUseAgent (without Playwright)."""

    def test_agent_init_defaults(self):
        from domains.agents.computer_use import ComputerUseAgent

        agent = ComputerUseAgent()
        assert agent.base_url == "http://localhost:3000"
        assert agent.api_url == "http://localhost:8000"
        assert agent.headless is True
        assert agent._page is None

    def test_agent_init_custom(self):
        from domains.agents.computer_use import ComputerUseAgent

        agent = ComputerUseAgent(
            base_url="http://localhost:4000",
            api_url="http://localhost:9000",
            headless=False,
            viewport_width=1920,
            viewport_height=1080,
        )
        assert agent.base_url == "http://localhost:4000"
        assert agent.api_url == "http://localhost:9000"
        assert agent.headless is False
        assert agent.viewport_width == 1920

    def test_devtools_report_empty(self):
        from domains.agents.computer_use import ComputerUseAgent

        agent = ComputerUseAgent()
        report = agent.devtools_report()
        assert report.total_console == 0
        assert report.total_requests == 0
        assert report.total_errors == 0
        assert "Console: 0" in report.summary

    def test_devtools_log_tracking(self):
        from domains.agents.computer_use import ComputerUseAgent

        agent = ComputerUseAgent()
        agent._console_messages.append({"type": "log", "text": "hello", "timestamp": time.time()})
        agent._network_requests.append({"url": "http://test", "method": "GET", "timestamp": time.time()})
        agent._errors.append({"message": "err", "timestamp": time.time()})

        report = agent.devtools_report()
        assert report.total_console == 1
        assert report.total_requests == 1
        assert report.total_errors == 1

    def test_clear_logs(self):
        from domains.agents.computer_use import ComputerUseAgent

        agent = ComputerUseAgent()
        agent._console_messages.append({"type": "log", "text": "x", "timestamp": time.time()})
        agent._errors.append({"message": "y", "timestamp": time.time()})
        agent.clear_logs()
        report = agent.devtools_report()
        assert report.total_console == 0
        assert report.total_errors == 0

    def test_step_counter(self):
        from domains.agents.computer_use import ComputerUseAgent

        agent = ComputerUseAgent()
        s1 = agent.step("navigate to training")
        s2 = agent.step("click train button")
        assert s1["step"] == 1
        assert s2["step"] == 2
        assert s1["description"] == "navigate to training"

    def test_full_log(self):
        from domains.agents.computer_use import ComputerUseAgent, DevToolsEntry

        agent = ComputerUseAgent()
        agent._devtools_log.append(
            DevToolsEntry(kind="console", timestamp=time.time(), data={"text": "hi"})
        )
        log = agent.get_full_log()
        assert len(log) == 1
        assert log[0]["kind"] == "console"

    def test_navigation_result(self):
        from domains.agents.computer_use import NavigationResult

        nr = NavigationResult(
            url="http://test.com",
            status=200,
            body_length=1000,
            duration_s=0.5,
        )
        assert nr.status == 200
        assert nr.errors == []

    def test_click_result(self):
        from domains.agents.computer_use import ClickResult

        cr = ClickResult(element="Train", found=True, duration_s=0.1)
        assert cr.found is True
        assert cr.error is None

    def test_fill_result(self):
        from domains.agents.computer_use import FillResult

        fr = FillResult(element="epochs", value="5", success=True)
        assert fr.success is True

    def test_devtools_report_summary(self):
        from domains.agents.computer_use import ComputerUseAgent

        agent = ComputerUseAgent()
        agent._console_messages = [{"type": "log"}] * 5
        agent._network_requests = [{"url": "x"}] * 3
        agent._errors = [{"message": "e"}] * 2
        report = agent.devtools_report()
        assert "Console: 5" in report.summary
        assert "Network: 3" in report.summary
        assert "Errors: 2" in report.summary

    def test_devtools_report_console_errors(self):
        from domains.agents.computer_use import ComputerUseAgent

        agent = ComputerUseAgent()
        agent._console_messages = [
            {"type": "log", "text": "ok"},
            {"type": "error", "text": "bad"},
            {"type": "error", "text": "worse"},
        ]
        report = agent.devtools_report()
        assert len(report.console_messages) == 3
        err_msgs = [m for m in report.console_messages if m["type"] == "error"]
        assert len(err_msgs) == 2


# ── DevToolsReport Unit Tests ───────────────────────────────────────

class TestDevToolsReport:
    """Unit tests for the DevToolsReport dataclass."""

    def test_report_creation(self):
        from domains.agents.computer_use import DevToolsReport

        report = DevToolsReport(
            console_messages=[],
            network_requests=[],
            errors=[],
            performance={"memoryUsed": 1024 * 1024},
            total_console=0,
            total_requests=0,
            total_errors=0,
            summary="test summary",
        )
        assert report.total_console == 0
        assert report.performance["memoryUsed"] == 1024 * 1024
        assert report.summary == "test summary"
