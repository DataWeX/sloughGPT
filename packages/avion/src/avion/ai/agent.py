"""Computer-use agent — observe, decide, act, learn.

Perception-action loop: screenshot + accessibility tree in, model picks
an action, interaction primitives execute it, experience is recorded.
"""

from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass
from typing import Any, Callable

from avion.ai.learning import Experience, ExperienceBuffer, FeedbackLoop
from avion.ai.models import (
    Action,
    ActionType,
    EchoModel,
    Step,
    Trajectory,
    VisionModel,
    validate_action,
)
from avion.ai.verifier import Verifier
from avion.interact.primitives import (
    Coordinate,
    Keyboard,
    Mouse,
)
from avion.vision.detector import ImageAnalyzer, MatchMethod, VisualMatch


@dataclass
class AgentConfig:
    """Configuration for the computer-use agent."""

    max_steps: int = 50
    step_timeout: float = 10.0
    action_delay_ms: float = 200
    screenshot_on_each_step: bool = True
    save_trajectories: bool = True
    retry_on_failure: bool = True
    max_retries: int = 2
    viewport_width: int = 1280
    viewport_height: int = 720
    deadline_ms: float = 120_000
    no_progress_limit: int = 3  # identical actions in a row = stuck (0 disables)
    output_dir: str = "avion_output/trajectories"


@dataclass
class AgentResult:
    """Result of an agent run."""

    task: str
    trajectory: Trajectory
    success: bool = False
    error: str = ""
    total_duration_ms: float = 0.0
    steps_taken: int = 0
    screenshots_captured: int = 0
    stop_reason: str = ""  # done|fail|max_steps|deadline|no_progress|unverified|aborted

    def summary(self) -> str:
        return (
            f"AgentResult('{self.task}'): "
            f"{'SUCCESS' if self.success else 'FAILED'} "
            f"({self.steps_taken} steps, "
            f"{self.total_duration_ms:.0f}ms, "
            f"{self.screenshots_captured} screenshots)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "success": self.success,
            "error": self.error,
            "steps_taken": self.steps_taken,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "screenshots_captured": self.screenshots_captured,
            "stop_reason": self.stop_reason,
            "trajectory": self.trajectory.to_dict(),
        }


def _action_signature(action: Action) -> str:
    """Stable hash of tool + args for no-progress detection."""
    import json

    return action.action_type.value + ":" + json.dumps(action.params, sort_keys=True, default=str)


class Agent:
    """Computer-use agent — observe, decide, act, learn."""

    def __init__(
        self,
        model: VisionModel | None = None,
        config: AgentConfig | None = None,
        base_url: str = "http://localhost:3000",
        api_url: str = "http://localhost:8000",
        headless: bool = True,
    ):
        self._model = model or EchoModel()
        self._config = config or AgentConfig()
        self._base_url = base_url
        self._api_url = api_url
        self._headless = headless

        self._backend = None
        self._mouse: Mouse | None = None
        self._keyboard: Keyboard | None = None
        self._experience_buffer = ExperienceBuffer()
        self._feedback_loop = FeedbackLoop()
        self._analyzer = ImageAnalyzer()

        self._running = False
        self._current_url = ""
        self._last_screenshot_b64 = ""
        self._step_count = 0
        self._callbacks: list[Callable] = []

    @property
    def model(self) -> VisionModel:
        return self._model

    @model.setter
    def model(self, new_model: VisionModel) -> None:
        """Switch the AI model at runtime."""
        self._model = new_model

    @property
    def mouse(self) -> Mouse:
        return self._mouse

    @property
    def keyboard(self) -> Keyboard:
        return self._keyboard

    @property
    def experience_buffer(self) -> ExperienceBuffer:
        return self._experience_buffer

    @property
    def feedback_loop(self) -> FeedbackLoop:
        return self._feedback_loop

    def on_step(self, callback: Callable[[Step], None]) -> None:
        """Register a callback for each step."""
        self._callbacks.append(callback)

    async def start(self, backend=None) -> None:
        """Initialize the backend and interaction controllers.

        Args:
            backend: optional pre-built backend (fakes in tests).
                     Defaults to a headless PlaywrightBackend.
        """
        if backend is None:
            from avion.backends.playwright import PlaywrightBackend

            backend = PlaywrightBackend(headless=self._headless)
        self._backend = backend
        await self._backend.start()
        self._mouse = Mouse(self._backend)
        self._keyboard = Keyboard(self._backend)
        self._running = True

    async def stop(self) -> None:
        """Shut down the backend."""
        self._running = False
        if self._backend:
            await self._backend.stop()

    async def __aenter__(self) -> Agent:
        await self.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self.stop()

    # ── Main loop ───────────────────────────────────────────────────────

    async def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Execute a task using the perception-action loop.

        Stops on the first of: DONE/FAIL from the model, max_steps,
        deadline_ms elapsed, or no_progress_limit identical actions
        in a row. Every step is appended to a JSONL transcript.
        """
        import json
        import os
        import re

        trajectory = Trajectory(task=task, metadata=context or {})
        stop_reason = "aborted"
        run_start = time.monotonic()
        start_time = time.perf_counter()
        recent_signatures: list[str] = []

        transcript_path = ""
        transcript = None
        if self._config.save_trajectories:
            slug = re.sub(r"[^a-z0-9]+", "_", task.lower()).strip("_")[:40] or "task"
            os.makedirs(self._config.output_dir, exist_ok=True)
            transcript_path = os.path.join(self._config.output_dir, f"{slug}.jsonl")
            transcript = open(transcript_path, "w")

        try:
            for step_num in range(self._config.max_steps):
                if not self._running:
                    break

                # Deadline budget.
                if (time.monotonic() - run_start) * 1000 >= self._config.deadline_ms:
                    stop_reason = "deadline"
                    trajectory.metadata["error"] = "deadline exceeded"
                    break

                step_start = time.perf_counter()

                # 1. OBSERVE
                screenshot_b64 = ""
                a11y_tree = None
                if self._config.screenshot_on_each_step or step_num == 0:
                    screenshot_b64 = await self._capture_screenshot()
                    a11y_tree = await self._capture_accessibility_tree()

                # 2. DECIDE
                try:
                    action = await self._model.predict_action(
                        task=task,
                        screenshot_b64=screenshot_b64,
                        accessibility_tree=a11y_tree,
                        history=trajectory.steps,
                    )
                except Exception as e:
                    action = Action(
                        action_type=ActionType.FAIL,
                        reasoning=f"Model error: {e}",
                        confidence=0.0,
                    )

                # No-progress detection: identical actions in a row.
                recent_signatures.append(_action_signature(action))
                window = self._config.no_progress_limit
                if (
                    window > 0
                    and len(recent_signatures) >= window
                    and len(set(recent_signatures[-window:])) == 1
                ):
                    stop_reason = "no_progress"
                    trajectory.metadata["error"] = f"no progress: same action {window}x in a row"
                    break

                # 3. ACT (validated against the action's tool card)
                validation_error = validate_action(action)
                goal_error = ""
                if validation_error is None and action.action_type == ActionType.DONE:
                    goal_error = await self._check_goal(context)
                if validation_error:
                    step_result = {
                        "observation": f"Invalid action: {validation_error}",
                        "reward": -0.5,
                    }
                elif goal_error:
                    step_result = {
                        "observation": f"Unverified: {goal_error}",
                        "reward": 0.0,
                    }
                else:
                    step_result = await self._execute_action(action, step_num)

                # 4. LEARN
                step = Step(
                    step_number=step_num,
                    screenshot_b64=screenshot_b64,
                    accessibility_tree=a11y_tree,
                    action=action,
                    observation=step_result.get("observation", ""),
                    reward=step_result.get("reward", 0.0),
                    done=action.action_type in (ActionType.DONE, ActionType.FAIL),
                    duration_ms=(time.perf_counter() - step_start) * 1000,
                )
                trajectory.add_step(step)
                self._step_count += 1

                self._experience_buffer.add(
                    Experience(
                        task=task,
                        screenshot_b64=screenshot_b64,
                        accessibility_tree=a11y_tree,
                        action=action,
                        reward=step.reward,
                        done=step.done,
                    )
                )

                if transcript is not None:
                    transcript.write(
                        json.dumps(
                            {
                                "task": task,
                                "step_number": step.step_number,
                                "action": action.to_dict(),
                                "observation": step.observation,
                                "reward": step.reward,
                                "done": step.done,
                            },
                            default=str,
                        )
                        + "\n"
                    )
                    transcript.flush()

                for cb in self._callbacks:
                    try:
                        cb(step)
                    except Exception:
                        pass

                if action.action_type == ActionType.DONE:
                    if goal_error:
                        trajectory.success = False
                        trajectory.metadata["error"] = goal_error
                        stop_reason = "unverified"
                    else:
                        trajectory.success = True
                        stop_reason = "done"
                    break
                if action.action_type == ActionType.FAIL:
                    trajectory.success = False
                    trajectory.metadata["error"] = action.reasoning
                    stop_reason = "fail"
                    break

                if self._config.action_delay_ms > 0:
                    await asyncio.sleep(self._config.action_delay_ms / 1000)
            else:
                stop_reason = "max_steps"
        finally:
            if transcript is not None:
                transcript.close()

        total_ms = (time.perf_counter() - start_time) * 1000
        return AgentResult(
            task=task,
            trajectory=trajectory,
            success=trajectory.success,
            error=trajectory.metadata.get("error", ""),
            total_duration_ms=total_ms,
            steps_taken=len(trajectory.steps),
            screenshots_captured=sum(1 for s in trajectory.steps if s.screenshot_b64),
            stop_reason=stop_reason,
        )

    # ── Action execution ────────────────────────────────────────────────

    async def _execute_action(self, action: Action, step_num: int) -> dict[str, Any]:
        """Execute a single action and return the result."""
        at = action.action_type
        params = action.params
        result = {"observation": "", "reward": 0.0}

        try:
            if at == ActionType.MOUSE_MOVE:
                await self._mouse.move_to(Coordinate(params["x"], params["y"]))
                result["observation"] = f"Moved to ({params['x']}, {params['y']})"

            elif at == ActionType.MOUSE_CLICK:
                x, y = params.get("x", 0), params.get("y", 0)
                await self._mouse.click(Coordinate(x, y))
                result["observation"] = f"Clicked at ({x}, {y})"
                result["reward"] = 0.1

            elif at == ActionType.MOUSE_DOUBLE_CLICK:
                x, y = params.get("x", 0), params.get("y", 0)
                await self._mouse.double_click(Coordinate(x, y))
                result["observation"] = f"Double-clicked at ({x}, {y})"

            elif at == ActionType.MOUSE_RIGHT_CLICK:
                x, y = params.get("x", 0), params.get("y", 0)
                await self._mouse.right_click(Coordinate(x, y))
                result["observation"] = f"Right-clicked at ({x}, {y})"

            elif at == ActionType.MOUSE_DRAG:
                start = Coordinate(params["start_x"], params["start_y"])
                end = Coordinate(params["end_x"], params["end_y"])
                await self._mouse.drag(start, end)
                result["observation"] = f"Dragged from {start} to {end}"

            elif at == ActionType.MOUSE_SCROLL:
                await self._mouse.scroll(
                    params.get("x", 640),
                    params.get("y", 360),
                    delta_y=params.get("delta_y", -3),
                )
                result["observation"] = f"Scrolled delta_y={params.get('delta_y', -3)}"

            elif at == ActionType.KEYBOARD_TYPE:
                text = params.get("text", "")
                await self._keyboard.type_text(text)
                result["observation"] = f"Typed '{text[:50]}'"

            elif at == ActionType.KEYBOARD_PRESS:
                key = params.get("key", "Enter")
                await self._keyboard.press(key)
                result["observation"] = f"Pressed {key}"

            elif at == ActionType.KEYBOARD_HOTKEY:
                keys = params.get("keys", [])
                if keys:
                    await self._keyboard.hotkey(*keys)
                    result["observation"] = f"Hotkey {'+'.join(keys)}"

            elif at == ActionType.SCREENSHOT:
                screenshot = await self._backend.screenshot()
                self._last_screenshot_b64 = base64.b64encode(screenshot).decode()
                result["observation"] = "Captured screenshot"

            elif at == ActionType.WAIT:
                ms = params.get("ms", 1000)
                await asyncio.sleep(ms / 1000)
                result["observation"] = f"Waited {ms}ms"

            elif at == ActionType.NAVIGATE:
                url = params.get("url", "")
                await self._backend.navigate(url)
                self._current_url = url
                result["observation"] = f"Navigated to {url}"

            elif at == ActionType.DONE:
                result["observation"] = "Task completed"
                result["reward"] = 1.0

            elif at == ActionType.FAIL:
                result["observation"] = f"Task failed: {action.reasoning}"
                result["reward"] = -1.0

        except Exception as e:
            result["observation"] = f"Error executing {at.value}: {e}"
            result["reward"] = -0.5

        return result

    # ── Goal verification ─────────────────────────────────────────────

    async def _check_goal(self, context: dict[str, Any] | None) -> str:
        """Verify the run goal against the live page. '' = pass/no goal."""
        goal = (context or {}).get("goal")
        if not goal:
            return ""
        try:
            url = await self._backend.get_url()
        except Exception:
            url = ""
        try:
            text = str(await self._backend.evaluate("document.body.innerText") or "")
        except Exception:
            text = ""

        async def find(locator):
            try:
                return await self._backend.find_element(locator)
            except Exception:
                return None

        result = await Verifier().verify(goal, url=url, text=text, find=find)
        return "" if result.passed else "; ".join(result.failed)

    # ── Perception ──────────────────────────────────────────────────────

    async def _capture_screenshot(self) -> str:
        """Capture screenshot and return as base64."""
        screenshot_bytes = await self._backend.screenshot()
        self._last_screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
        return self._last_screenshot_b64

    async def _capture_accessibility_tree(self) -> dict[str, Any] | None:
        """Capture accessibility tree."""
        try:
            return await self._backend.get_accessibility_tree()
        except Exception:
            return None

    # ── Vision helpers ──────────────────────────────────────────────────

    def analyze_screenshot(self, screenshot_b64: str) -> dict[str, Any]:
        """Analyze a screenshot for visual elements."""
        img_bytes = base64.b64decode(screenshot_b64)
        return {
            "hash": self._analyzer.compute_hash(img_bytes),
            "size_bytes": len(img_bytes),
        }

    def find_text_in_screenshot(self, screenshot_b64: str, text: str) -> list[VisualMatch]:
        """Find text regions in a screenshot using OCR."""
        img_bytes = base64.b64decode(screenshot_b64)
        results = self._analyzer.ocr_text(img_bytes)
        return [
            VisualMatch(bbox=bbox, confidence=0.9, method=MatchMethod.TEXT_REGION, label=ocr_text)
            for ocr_text, bbox in results
            if text.lower() in ocr_text.lower()
        ]

    def pixel_diff(self, screenshot1_b64: str, screenshot2_b64: str) -> float:
        """Compare two screenshots for differences."""
        img1 = base64.b64decode(screenshot1_b64)
        img2 = base64.b64decode(screenshot2_b64)
        return self._analyzer.pixel_diff_bytes(img1, img2)

    # ── Learning ────────────────────────────────────────────────────────

    def feedback(self, step: Step, rating: float, comment: str = "") -> None:
        """Provide feedback on a step for learning."""
        self._feedback_loop.record_feedback(step, rating, comment)

    def correct(self, step: Step, corrected_action: Action) -> None:
        """Provide a corrected action for learning."""
        self._feedback_loop.record_correction(step, corrected_action)

    def save_learning(self, path: str) -> None:
        """Save experience buffer and feedback to disk."""
        import os

        os.makedirs(path, exist_ok=True)
        self._experience_buffer.save(os.path.join(path, "experiences.json"))
        self._feedback_loop.save(os.path.join(path, "feedback.json"))

    def load_learning(self, path: str) -> None:
        """Load experience buffer and feedback from disk."""
        import os

        exp_path = os.path.join(path, "experiences.json")
        fb_path = os.path.join(path, "feedback.json")
        if os.path.exists(exp_path):
            self._experience_buffer.load(exp_path)
        if os.path.exists(fb_path):
            self._feedback_loop.load(fb_path)
