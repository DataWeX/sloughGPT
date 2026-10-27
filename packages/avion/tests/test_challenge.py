"""Challenge→human policy: automation yields where a human must take over.

Four seams, four contracts:
1. ``detect_challenge`` — pure, evidence-quoted, false-positive-guarded
   (vendor names alone match nothing).
2. ``page_challenge`` — best-effort page read: a broken checker skips
   the check, it never fails or fakes the run it protects.
3. Replay/flow — a challenge page halts the run mid-flow with the
   evidence; ``yield_on_challenge=False`` is the explicit drive-through.
4. Agent loop — the model never gets to ACT on a challenge page
   (``predict_action`` is not called); ``avion run``/``avion agent``
   exit 3 (a human must take over), distinct from 1 (failed).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from avion.ai.agent import Agent, AgentConfig
from avion.challenge import Challenge, detect_challenge, page_challenge
from avion.events.models import Event, EventType
from avion.events.replay import EventReplay, ReplayConfig, ReplayResult
from avion.flow.complete import FlowOutcome

# ── fakes ───────────────────────────────────────────────────────────────


class FakeBackend:
    """Answers the three reads page_challenge makes, or refuses to."""

    def __init__(self, *, body="", title="", url="https://example.com/app", raises=False):
        self._body, self._title, self._url = body, title, url
        self._raises = raises

    async def get_url(self):
        if self._raises:
            raise RuntimeError("connection lost")
        return self._url

    async def start(self):
        pass

    async def stop(self):
        pass

    async def evaluate(self, expr):
        if self._raises:
            raise RuntimeError("connection lost")
        return self._title if "title" in expr else self._body

    async def get_accessibility_tree(self):
        if self._raises:
            raise RuntimeError("connection lost")
        return {"role": "document", "text": self._body}

    async def screenshot(self):
        return b""


class FakeSession:
    """Minimal session for EventReplay: backend + a goto that lands."""

    def __init__(self, backend):
        self._backend = backend
        self.goto_calls = 0

    @property
    def backend(self):
        return self._backend

    async def goto(self, path):
        self.goto_calls += 1
        return True


class NeverCalledModel:
    """Stands in for the brain: calling it on a challenge page is a bug."""

    model_name = "never"

    def __init__(self):
        self.calls = 0

    async def predict_action(self, **kwargs):
        self.calls += 1
        raise AssertionError("model must not act on a challenge page")


class DriveThroughModel:
    """Answers FAIL so the loop records one step and the test can prove
    the model was reached when the yield is off."""

    model_name = "drive"

    def __init__(self):
        self.calls = 0

    async def predict_action(self, **kwargs):
        self.calls += 1
        from avion.ai.models import Action, ActionType

        return Action(action_type=ActionType.FAIL, reasoning="stopped by test")


# ── 1. detector: pure, evidence-quoted, false-positive-guarded ─────────


class TestDetectChallenge:
    def test_cloudflare_interstitial(self):
        ch = detect_challenge(title="Example", text="Just a moment... checking your browser")
        assert ch is not None
        assert ch.kind == "cloudflare"
        assert "just a moment" in ch.evidence

    def test_captcha_wall(self):
        ch = detect_challenge(text="please verify you are human to continue")
        assert ch is not None and ch.kind == "captcha"

    def test_case_insensitive(self):
        assert detect_challenge(text="VERIFY YOU ARE HUMAN") is not None

    def test_healthy_page_is_none(self):
        assert detect_challenge(title="Dashboard", text="3 new messages") is None

    def test_vendor_name_alone_never_matches(self):
        # "recaptcha" appears in healthy pages' script tags — the
        # false-positive guard this module exists by.
        assert detect_challenge(text='<script src="/assets/recaptcha.js">') is None
        assert detect_challenge(text="hcaptcha enabled on this site") is None

    def test_login_wall_is_structural(self):
        ch = detect_challenge(url="https://x.com/i/flow/login")
        assert ch is not None and ch.kind == "login_wall"

    def test_priority_first_marker_wins(self):
        ch = detect_challenge(text="just a moment — verify you are human")
        assert ch is not None and ch.kind == "cloudflare"

    def test_to_dict_shape(self):
        ch = detect_challenge(url="https://e.com/", text="slide to verify")
        assert ch is not None
        assert set(ch.to_dict()) == {"kind", "evidence", "url"}
        assert ch.kind == "slider"

    def test_frozen_and_exported(self):
        assert isinstance(
            detect_challenge(text="just a moment"), Challenge
        )  # ships on the package surface
        import avion

        assert avion.Challenge is Challenge and avion.detect_challenge is detect_challenge


# ── 2. page_challenge: best-effort, never fatal ────────────────────────


class TestPageChallenge:
    def test_reads_live_page(self):
        ch = asyncio.run(
            page_challenge(FakeBackend(body="Just a moment...", title="Wait"))
        )
        assert ch is not None and ch.kind == "cloudflare"

    def test_a11y_text_substitutes_for_body_read(self):
        ch = asyncio.run(
            page_challenge(
                FakeBackend(body="healthy body"),
                a11y_text=json.dumps({"text": "please verify you are human"}),
            )
        )
        assert ch is not None and ch.kind == "captcha"

    def test_broken_checker_skips_the_check(self):
        assert asyncio.run(page_challenge(FakeBackend(raises=True))) is None

    def test_no_backend_is_none(self):
        assert asyncio.run(page_challenge(None)) is None

    def test_healthy_live_page(self):
        assert asyncio.run(page_challenge(FakeBackend(body="all good"))) is None


# ── 3. replay: a challenge halts the flow mid-run ──────────────────────


class TestReplayYield:
    EVENTS = [
        Event(type=EventType.NAVIGATE, name="http://example.com/app", seq=1),
        Event(type=EventType.NAVIGATE, name="http://example.com/next", seq=2),
    ]

    def test_challenge_stops_before_any_step(self):
        session = FakeSession(FakeBackend(body="Just a moment..."))
        result = asyncio.run(EventReplay(session).replay(self.EVENTS))
        assert result.needs_human
        assert result.success is False
        assert result.challenge is not None and result.challenge["kind"] == "cloudflare"
        assert result.steps == []  # nothing was driven
        assert session.goto_calls == 0

    def test_healthy_page_runs_to_the_end(self):
        session = FakeSession(FakeBackend(body="welcome"))
        result = asyncio.run(EventReplay(session).replay(self.EVENTS))
        assert not result.needs_human
        assert result.success and result.passed == 2

    def test_yield_off_is_an_explicit_drive_through(self):
        session = FakeSession(FakeBackend(body="Just a moment..."))
        cfg = ReplayConfig(yield_on_challenge=False)
        result = asyncio.run(EventReplay(session, cfg).replay(self.EVENTS))
        assert result.challenge is None
        assert session.goto_calls == 2

    def test_dry_run_never_yields(self):
        # A dress rehearsal resolves locators on whatever page is open —
        # it acts on nothing, so a challenge cannot trap it.
        session = FakeSession(FakeBackend(body="Just a moment..."))
        cfg = ReplayConfig(dry_run=True)
        result = asyncio.run(EventReplay(session, cfg).replay(self.EVENTS))
        assert result.challenge is None


# ── 3b. FlowOutcome surfaces the handoff ───────────────────────────────


class TestFlowOutcome:
    def test_needs_human_and_to_dict_carry_the_evidence(self):
        challenge = {"kind": "cloudflare", "evidence": "'just a moment' shown on page", "url": "u"}
        outcome = FlowOutcome(
            result=ReplayResult(challenge=challenge, success=False),
            final_url="https://e.com/",
            steps_total=2,
        )
        assert outcome.needs_human and not outcome.completed
        assert outcome.to_dict()["challenge"] == challenge

    def test_clean_run_has_no_challenge(self):
        outcome = FlowOutcome(result=ReplayResult(success=True), steps_total=2)
        assert not outcome.needs_human
        assert outcome.to_dict()["challenge"] is None


# ── 4. agent loop: the model never acts on a challenge page ────────────


def _agent(model, tmp_path, **cfg):
    return Agent(
        model=model,
        config=AgentConfig(
            save_trajectories=False,
            output_dir=str(tmp_path / "traj"),
            **cfg,
        ),
    )


class TestAgentYield:
    def test_model_never_called_and_stops_needs_human(self, tmp_path):
        model = NeverCalledModel()
        agent = _agent(model, tmp_path)

        async def go():
            await agent.start(backend=FakeBackend(body="Just a moment..."))
            try:
                return await agent.run("sign in")
            finally:
                await agent.stop()

        result = asyncio.run(go())
        assert model.calls == 0, "the loop yielded before the model could act"
        assert result.stop_reason == "needs_human"
        assert result.success is False
        assert result.challenge is not None and result.challenge["kind"] == "cloudflare"
        assert "needs a human" in result.error
        assert result.to_dict()["challenge"]["kind"] == "cloudflare"

    def test_yield_off_reaches_the_model(self, tmp_path):
        model = DriveThroughModel()
        agent = _agent(model, tmp_path, yield_on_challenge=False)

        async def go():
            await agent.start(backend=FakeBackend(body="Just a moment..."))
            try:
                return await agent.run("anything")
            finally:
                await agent.stop()

        result = asyncio.run(go())
        assert model.calls >= 1, "yield_on_challenge=False must reach the model"
        assert result.stop_reason != "needs_human"
        assert result.challenge is None

    def test_healthy_page_runs_normally(self, tmp_path):
        model = DriveThroughModel()
        agent = _agent(model, tmp_path)

        async def go():
            await agent.start(backend=FakeBackend(body="welcome"))
            try:
                return await agent.run("anything")
            finally:
                await agent.stop()

        result = asyncio.run(go())
        assert result.stop_reason != "needs_human"
        assert result.challenge is None


# ── 5. CLI: exit 3 = a human must take over (distinct from 1) ──────────


class TestCliExitCodes:
    FLOW = {
        "events": [{"type": "navigate", "name": "http://example.com", "seq": 1}]
    }

    def _args(self, tmp_path: Path):
        import argparse

        flow = tmp_path / "flow.json"
        flow.write_text(json.dumps(self.FLOW))
        return argparse.Namespace(
            flow=str(flow),
            profile=None,
            cdp="ws://127.0.0.1:9",  # never reached: complete_flow is faked
            base_url="",
            dry_run=False,
            json=False,
            headed=False,
            arg=[],
        )

    def test_run_exits_3_on_challenge(self, tmp_path, monkeypatch, capsys):
        from avion.cli import _cmd_run

        async def fake_flow(events, target, *, base_url="", config=None):
            return FlowOutcome(
                result=ReplayResult(
                    challenge={"kind": "cloudflare", "evidence": "'just a moment'", "url": "u"},
                    success=False,
                ),
                final_url="https://e.com/challenge",
                steps_total=1,
            )

        monkeypatch.setattr("avion.flow.complete_flow", fake_flow)
        code = asyncio.run(_cmd_run(self._args(tmp_path)))
        assert code == 3
        out = capsys.readouterr().out
        assert "needs a human" in out and "just a moment" in out

    def test_run_still_exits_1_on_plain_failure(self, tmp_path, monkeypatch):
        from avion.cli import _cmd_run

        async def fake_flow(events, target, *, base_url="", config=None):
            return FlowOutcome(result=ReplayResult(success=False), steps_total=1)

        monkeypatch.setattr("avion.flow.complete_flow", fake_flow)
        assert asyncio.run(_cmd_run(self._args(tmp_path))) == 1

    def test_run_still_exits_0_on_success(self, tmp_path, monkeypatch):
        from avion.cli import _cmd_run

        async def fake_flow(events, target, *, base_url="", config=None):
            return FlowOutcome(result=ReplayResult(success=True), steps_total=1)

        monkeypatch.setattr("avion.flow.complete_flow", fake_flow)
        assert asyncio.run(_cmd_run(self._args(tmp_path))) == 0

    def test_json_mode_carries_the_challenge(self, tmp_path, monkeypatch, capsys):
        from avion.cli import _cmd_run

        async def fake_flow(events, target, *, base_url="", config=None):
            return FlowOutcome(
                result=ReplayResult(
                    challenge={"kind": "login_wall", "evidence": "cookies dead", "url": "u"},
                    success=False,
                ),
                steps_total=1,
            )

        monkeypatch.setattr("avion.flow.complete_flow", fake_flow)
        args = self._args(tmp_path)
        args.json = True
        assert asyncio.run(_cmd_run(args)) == 3
        payload = json.loads(capsys.readouterr().out)
        assert payload["challenge"]["kind"] == "login_wall"
