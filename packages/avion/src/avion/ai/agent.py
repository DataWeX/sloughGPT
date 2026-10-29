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
from avion.ai.tools import ToolExecutor, ToolRegistry, normalize_result
from avion.ai.verifier import Verifier
from avion.interact.primitives import (
    Coordinate,
    Keyboard,
    Mouse,
)
from avion.vision.detector import ImageAnalyzer, MatchMethod, VisualMatch


class _DeadlineExceeded(Exception):
    """No budget remains for this run — stop before acting."""


class _StepTimeout(Exception):
    """A single await exceeded the allowed budget (step_timeout or the
    remaining deadline). The step degrades or is recorded as failed; the
    run continues."""


@dataclass
class AgentConfig:
    """Configuration for the computer-use agent."""

    max_steps: int = 50
    step_timeout: float = 10.0
    action_delay_ms: float = 200
    screenshot_on_each_step: bool = True
    save_trajectories: bool = True
    # Transcript flush policy: 1 = flush every step (crash-durable per step,
    # the benchmark's "durability cost"); N>1 = group-commit — flush every N
    # written steps so a fast loop pays one write(2) per N steps instead of
    # one per step. A crash loses at most the pending batch; the run's
    # close() in finally always flushes the remainder, so every completed
    # run's transcript is complete and parseable.
    transcript_flush_steps: int = 1
    retry_on_failure: bool = True
    max_retries: int = 2
    viewport_width: int = 1280
    viewport_height: int = 720
    deadline_ms: float = 120_000
    no_progress_limit: int = 3  # identical actions in a row = stuck (0 disables)
    output_dir: str = "arken_output/trajectories"
    # on_step callbacks failing this many times in a row are skipped
    # (degraded) for the rest of the run; totals stay visible.
    callback_degrade_after: int = 5


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
    transcript_path: str = ""  # where this run's transcript landed ('' = not saved)
    transcript_failures: int = 0  # write/close errors counted, never fatal

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
            "transcript_path": self.transcript_path,
            "transcript_failures": self.transcript_failures,
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
        tools: ToolRegistry | None = None,
        executor: ToolExecutor | None = None,
    ):
        self._model = model or EchoModel()
        self._config = config or AgentConfig()
        self._base_url = base_url
        self._api_url = api_url
        self._headless = headless
        self._tools = tools
        self._executor = executor

        self._backend = None
        self._mouse: Mouse | None = None
        self._keyboard: Keyboard | None = None
        self._experience_buffer = ExperienceBuffer()
        self._feedback_loop = FeedbackLoop()
        self._analyzer = ImageAnalyzer()

        self._running = False
        self._run_active = False  # re-entrancy guard: one run() at a time
        self._current_url = ""
        self._last_screenshot_b64 = ""
        self._step_count = 0
        self._transcript_failures = 0
        self._callbacks: list[Callable] = []
        # Parallel to _callbacks: totals, consecutive failures, degraded flag.
        self._cb_failures: list[int] = []
        self._cb_consecutive: list[int] = []
        self._cb_disabled: list[bool] = []

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
        """Register a callback for each step.

        A callback that keeps failing is never allowed to break the loop:
        failures are counted, and after ``callback_degrade_after``
        consecutive failures it is skipped for the rest of the run
        (revived at the next run start). See ``callback_status()``.
        """
        self._callbacks.append(callback)
        self._cb_failures.append(0)
        self._cb_consecutive.append(0)
        self._cb_disabled.append(False)

    def callback_status(self) -> list[dict[str, Any]]:
        """Per-callback health: cumulative failures + degraded flag."""
        return [
            {
                "index": index,
                "failures": self._cb_failures[index],
                "disabled": self._cb_disabled[index],
            }
            for index in range(len(self._callbacks))
        ]

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
        in a row. Every await is bounded by ``min(step_timeout,
        remaining deadline)`` — a hung backend or model can never wedge
        the run. Model predictions are retried (``retry_on_failure``);
        side-effecting actions are never auto-retried. Each run claims
        its own transcript file (earlier runs are never truncated) and
        the path comes back on ``AgentResult.transcript_path``.
        """
        import json

        if self._run_active:
            raise RuntimeError("Agent.run() is already in progress — one run at a time")
        self._run_active = True
        self._step_count = 0
        self._transcript_failures = 0
        # Every run gives degraded callbacks a fresh chance; failure
        # totals in _cb_failures keep accumulating.
        self._cb_consecutive = [0] * len(self._callbacks)
        self._cb_disabled = [False] * len(self._callbacks)

        trajectory = Trajectory(task=task, metadata=context or {})
        stop_reason = "aborted"
        run_start = time.monotonic()
        start_time = time.perf_counter()
        recent_signatures: list[str] = []

        transcript_path = ""
        transcript = None
        pending_flush = 0  # transcript writes since last flush (group-commit)
        try:
            if self._config.save_trajectories:
                transcript_path, transcript = self._open_transcript(task)

            for step_num in range(self._config.max_steps):
                if not self._running:
                    break

                # Deadline budget: raises once the run is out of time, so
                # even a slow await cannot push past it unnoticed.
                if (time.monotonic() - run_start) * 1000 >= self._config.deadline_ms:
                    raise _DeadlineExceeded("deadline exceeded")

                step_start = time.perf_counter()

                # 1. OBSERVE — bounded; a hung capture degrades to an
                #    empty frame instead of stalling the run.
                screenshot_b64 = ""
                a11y_tree = None
                if self._config.screenshot_on_each_step or step_num == 0:
                    screenshot_b64, a11y_tree = await self._observe(run_start)

                # 2. DECIDE — bounded + retried (predict is a pure call,
                #    so retrying it has no side effects)
                action = await self._predict(
                    task, screenshot_b64, a11y_tree, trajectory.steps, run_start
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
                    try:
                        goal_error = await self._bounded(
                            lambda: self._check_goal(context), run_start
                        )
                    except _StepTimeout as exc:
                        # Honest failure: an unverified goal must never
                        # read as a pass.
                        goal_error = f"goal verification timed out: {exc}"
                if validation_error:
                    step_result: dict[str, Any] = {
                        "observation": f"Invalid action: {validation_error}",
                        "reward": -0.5,
                    }
                elif goal_error:
                    step_result = {
                        "observation": f"Unverified: {goal_error}",
                        "reward": 0.0,
                    }
                else:
                    try:
                        step_result = await self._bounded(
                            lambda: self._execute_action(action, step_num), run_start
                        )
                    except _StepTimeout as exc:
                        # Cancelled mid-flight: side effects are
                        # uncertain, so the step is recorded as a failure
                        # and NOT re-issued (a retried click might land
                        # twice).
                        step_result = {"observation": f"Timeout: {exc}", "reward": -0.5}

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
                    try:
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
                        pending_flush += 1
                        if pending_flush >= max(1, self._config.transcript_flush_steps):
                            transcript.flush()
                            pending_flush = 0
                    except Exception:
                        # A failing log path must not take the run down
                        # (Tier-1 containment): count it, keep going.
                        self._transcript_failures += 1

                for index, cb in enumerate(self._callbacks):
                    if self._cb_disabled[index]:
                        continue
                    try:
                        cb(step)
                    except Exception:
                        # An observer may never break the observed run:
                        # count, then degrade after the threshold.
                        self._cb_failures[index] += 1
                        self._cb_consecutive[index] += 1
                        if self._cb_consecutive[index] >= max(
                            1, self._config.callback_degrade_after
                        ):
                            self._cb_disabled[index] = True
                    else:
                        self._cb_consecutive[index] = 0

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
                    # Bounded: even the inter-step delay cannot overshoot
                    # the run deadline.
                    await asyncio.sleep(
                        min(
                            self._config.action_delay_ms / 1000,
                            self._budget(run_start),
                        )
                    )
            else:
                stop_reason = "max_steps"
        except _DeadlineExceeded as exc:
            stop_reason = "deadline"
            trajectory.metadata["error"] = str(exc) or "deadline exceeded"
        finally:
            self._run_active = False
            if transcript is not None:
                try:
                    transcript.close()
                except Exception:
                    self._transcript_failures += 1

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
            transcript_path=transcript_path,
            transcript_failures=self._transcript_failures,
        )

    # ── Bounded execution ──────────────────────────────────────────────

    def _open_transcript(self, task: str) -> tuple[str, Any]:
        """Claim a fresh transcript path for this run.

        Created with ``O_CREAT | O_EXCL`` — atomic, so a re-run of the
        same task lands in ``slug-1.jsonl`` instead of truncating the
        previous trajectory, and concurrent claimers cannot collide.
        """
        import os
        import re

        slug = re.sub(r"[^a-z0-9]+", "_", task.lower()).strip("_")[:40] or "task"
        os.makedirs(self._config.output_dir, exist_ok=True)
        base = os.path.join(self._config.output_dir, slug)
        n = 0
        while True:
            path = f"{base}.jsonl" if n == 0 else f"{base}-{n}.jsonl"
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                n += 1
                continue
            return path, os.fdopen(fd, "w")

    def _budget(self, run_start: float) -> float:
        """Seconds allowed for the next await: ``step_timeout``, but never
        more than the run's remaining deadline — this is what makes
        ``deadline_ms`` enforceable when an await hangs."""
        remaining_ms = self._config.deadline_ms - (time.monotonic() - run_start) * 1000
        if remaining_ms <= 0:
            return 0.0
        remaining_s = remaining_ms / 1000.0
        if self._config.step_timeout <= 0:  # disabled: bound by deadline only
            return remaining_s
        return min(self._config.step_timeout, remaining_s)

    async def _bounded(self, factory: Callable[[], Any], run_start: float) -> Any:
        """Await one factory-produced call under the current budget.

        The factory form matters: when the budget is already gone we can
        raise without ever creating a coroutine that would go un-awaited.
        """
        budget = self._budget(run_start)
        if budget <= 0:
            raise _DeadlineExceeded("deadline exceeded")
        try:
            return await asyncio.wait_for(factory(), timeout=budget)
        except TimeoutError as exc:
            raise _StepTimeout(f"no result within {budget:.2f}s") from exc

    async def _observe(self, run_start: float) -> tuple[str, dict[str, Any] | None]:
        """Bounded OBSERVE. A hung capture degrades (empty frame / no
        tree); an expired deadline still propagates."""
        screenshot_b64 = ""
        a11y_tree: dict[str, Any] | None = None
        try:
            screenshot_b64 = await self._bounded(self._capture_screenshot, run_start)
        except _StepTimeout:
            pass  # model sees no screenshot this step; loop continues
        try:
            a11y_tree = await self._bounded(self._capture_accessibility_tree, run_start)
        except _StepTimeout:
            a11y_tree = None
        return screenshot_b64, a11y_tree

    async def _predict(
        self,
        task: str,
        screenshot_b64: str,
        a11y_tree: dict[str, Any] | None,
        history: list[Step],
        run_start: float,
    ) -> Action:
        """Deadline-bounded prediction with retries.

        Transient model errors and timeouts are retried up to
        ``max_retries`` when ``retry_on_failure`` is on; only after every
        attempt fails does the loop see a FAIL action. An exhausted
        deadline always wins — retries never outlive the run's budget.
        """
        attempts = 1 + (
            max(0, self._config.max_retries) if self._config.retry_on_failure else 0
        )
        last_error = "no attempt made"
        for _ in range(attempts):
            try:
                action: Action = await self._bounded(
                    lambda: self._model.predict_action(
                        task=task,
                        screenshot_b64=screenshot_b64,
                        accessibility_tree=a11y_tree,
                        history=history,
                    ),
                    run_start,
                )
                return action
            except _DeadlineExceeded:
                raise
            except _StepTimeout as exc:
                last_error = str(exc)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        return Action(
            action_type=ActionType.FAIL,
            reasoning=f"Model error: {last_error}",
            confidence=0.0,
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

            elif at == ActionType.TOOL_CALL:
                name = str(params.get("tool", ""))
                spec = self._tools.get(name) if self._tools is not None else None
                if spec is None:
                    result["observation"] = f"Unknown tool: {name}"
                    result["reward"] = -0.1
                elif spec.requires_approval:
                    result["observation"] = f"Tool '{name}' needs approval"
                    result["reward"] = 0.0
                elif self._executor is None:
                    result["observation"] = f"No executor for tool '{name}'"
                    result["reward"] = 0.0
                else:
                    try:
                        raw = await self._executor.run(
                            name, params.get("args") or {}
                        )
                    except Exception as e:
                        result["observation"] = f"error: {e}"
                        result["reward"] = -0.5
                    else:
                        text, ok = normalize_result(raw)
                        result["observation"] = text
                        result["reward"] = 0.5 if ok else -0.5

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
