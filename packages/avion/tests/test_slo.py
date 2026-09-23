"""SloModel tests — prompt, parsing, adapters, agent integration."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from avion.ai.models import ActionType
from avion.ai.slo import SloModel, build_prompt, parse_action
from test_autoclicker import FakeBackend


def run(coro):
    return asyncio.run(coro)


class TestPrompt:
    def test_contains_task_cards_history(self):
        from avion.ai.models import Action, Step

        history = [
            Step(step_number=0, action=Action(ActionType.MOUSE_CLICK, {"x": 1, "y": 2}),
                 observation="clicked", reward=0.1)
        ]
        prompt = build_prompt("do it", {"role": "button"}, history)
        assert "do it" in prompt
        assert "mouse_click" in prompt
        assert "clicked" in prompt
        assert "button" in prompt

    def test_empty_state(self):
        prompt = build_prompt("do it")
        assert "(first step)" in prompt
        assert "(no accessibility tree)" in prompt

    def test_action_subset(self):
        prompt = build_prompt("t", available_actions=[ActionType.NAVIGATE])
        assert "navigate" in prompt and "mouse_click" not in prompt
        assert "done" in prompt  # terminators always listed

    def test_tree_trimmed(self):
        prompt = build_prompt("t", {"blob": "x" * 5000})
        assert "[trimmed]" in prompt

    def test_max_chars_budgets(self):
        from avion.ai.models import Action, Step

        history = [
            Step(
                step_number=i,
                action=Action(ActionType.WAIT, {"ms": 1}),
                observation="w" * 200,
            )
            for i in range(5)
        ]
        full = build_prompt("t", {"blob": "y" * 5000}, history)
        small = build_prompt("t", {"blob": "y" * 5000}, history, max_chars=200)
        assert len(small) < len(full)
        assert "Task: t" in small  # fixed instructions survive
        assert "[trimmed]" in small

    def test_model_level_budget_and_subset(self):
        seen = []

        async def fake(prompt):
            seen.append(prompt)
            return '{"action_type": "done"}'

        m = SloModel(
            generate=fake,
            available_actions=[ActionType.DONE],
            max_prompt_chars=200,
        )
        a = run(m.predict_action("t", "", {"blob": "z" * 5000}))
        assert a.action_type == ActionType.DONE
        assert len(seen[0]) < 1200
        assert "mouse_click" not in seen[0]


class TestParse:
    def test_plain_json(self):
        a = parse_action('{"action_type": "done", "reasoning": "ok"}')
        assert a.action_type == ActionType.DONE and a.reasoning == "ok"

    def test_wrapped_in_prose(self):
        a = parse_action('Sure! {"action_type": "wait", "params": {"ms": 5}} done.')
        assert a.action_type == ActionType.WAIT and a.params == {"ms": 5}

    def test_echo_plus_answer(self):
        text = (
            'Reply format: {"action_type": "<name>", "params": {}} '
            '<|im_start|>assistant {"action_type": "done"}<|im_end|>'
        )
        a = parse_action(text)
        assert a.action_type == ActionType.DONE

    def test_garbage_becomes_wait(self):
        a = parse_action("hello there no json")
        assert a.action_type == ActionType.WAIT
        assert a.confidence < 0.5

    def test_unknown_type_becomes_fail(self):
        a = parse_action('{"action_type": "teleport"}')
        assert a.action_type == ActionType.FAIL

    def test_invalid_params_become_wait(self):
        a = parse_action('{"action_type": "mouse_click", "params": {}}')
        assert a.action_type == ActionType.WAIT
        assert "missing params" in a.reasoning


class TestStripEcho:
    def test_strips_echoed_prompt(self):
        from avion.ai.slo import strip_echo

        prompt = "Pick one:\n1. click\n2. done\nNumber:"
        text = prompt + "2"
        assert strip_echo(text, prompt) == "2"

    def test_no_echo_unchanged(self):
        from avion.ai.slo import strip_echo

        assert strip_echo('{"action_type": "done"}', "Pick one:") == '{"action_type": "done"}'

    def test_echo_does_not_create_phantom_action(self):
        async def echo_sampler(prompt):
            return prompt + " done"

        m = SloModel(generate=echo_sampler, available_actions=[ActionType.DONE])
        a = run(m.predict_action("t", ""))
        # "done" prose without JSON is not an action
        assert a.action_type == ActionType.WAIT


class TestAdapters:
    def test_generate_fn(self):
        async def fake(prompt):
            assert "my task" in prompt
            return '{"action_type": "done", "reasoning": "ok"}'

        m = SloModel(generate=fake, model_name="test")
        assert m.model_name == "test" and not m.supports_vision
        a = run(m.predict_action("my task", ""))
        assert a.action_type == ActionType.DONE

    def test_generate_error_becomes_fail(self):
        async def broken(prompt):
            raise RuntimeError("down")

        a = run(SloModel(generate=broken).predict_action("t", ""))
        assert a.action_type == ActionType.FAIL

    def test_from_inference_client(self):
        class StubClient:
            model_id = "stub-1"

            async def chat(self, messages, max_tokens=512, temperature=0.3):
                assert messages[0]["role"] == "user"
                return '{"action_type": "navigate", "params": {"url": "http://x"}}'

        m = SloModel.from_inference_client(StubClient())
        assert m.model_name == "stub-1"
        a = run(m.predict_action("t", ""))
        assert a.action_type == ActionType.NAVIGATE

    def test_from_http(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                body = json.dumps({"text": '{"action_type": "done"}'}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        httpd = HTTPServer(("127.0.0.1", 0), Handler)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            m = SloModel.from_http(f"http://127.0.0.1:{port}")
            a = run(m.predict_action("t", ""))
            assert a.action_type == ActionType.DONE
        finally:
            httpd.shutdown()


class TestAgentIntegration:
    def test_slo_drives_agent_to_done(self):
        from avion.ai import Agent, AgentConfig

        script = [
            '{"action_type": "keyboard_type", "params": {"text": "hi"}}',
            '{"action_type": "done", "reasoning": "typed"}',
        ]

        async def fake(prompt):
            return script.pop(0)

        import tempfile

        cfg = AgentConfig(
            action_delay_ms=0,
            screenshot_on_each_step=False,
            output_dir=tempfile.mkdtemp(prefix="avion-slo-"),
        )
        agent = Agent(model=SloModel(generate=fake), config=cfg)

        class Backend(FakeBackend):
            async def screenshot(self, path=None):
                return b"png"

        run(agent.start(backend=Backend()))
        try:
            result = run(agent.run("type hi"))
            assert result.success is True
            assert result.stop_reason == "done"
            assert agent._backend.typed == ["hi"]
        finally:
            run(agent.stop())

    def test_slo_garbage_does_not_hang(self):
        from avion.ai import Agent, AgentConfig

        async def garbage(prompt):
            return "nonsense without json"

        import tempfile

        cfg = AgentConfig(
            action_delay_ms=0,
            screenshot_on_each_step=False,
            output_dir=tempfile.mkdtemp(prefix="avion-slo-"),
        )
        agent = Agent(model=SloModel(generate=garbage), config=cfg)

        class Backend(FakeBackend):
            async def screenshot(self, path=None):
                return b"png"

        run(agent.start(backend=Backend()))
        try:
            result = run(agent.run("t"))
            assert result.stop_reason == "no_progress"
            assert result.success is False
        finally:
            run(agent.stop())
