"""Tests for FeedbackController._wire_model — wiring the active model into the workflow."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from controllers.feedback import FeedbackController


def _fake_module(**attrs) -> ModuleType:
    mod = ModuleType("fake")
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


class TestWireModel:
    def test_wires_server_model_when_no_student(self):
        controller = FeedbackController(repo_root=Path("/tmp/fb-wiring-test"))
        controller._workflow = MagicMock()
        server_state = _fake_module(model="the-model", tokenizer="the-tokenizer")
        at_state = _fake_module(state=_fake_module(student_net=None))
        with patch.dict(
            sys.modules,
            {
                "state": server_state,
                "routers.auto_train": at_state,
                "routers": _fake_module(),
            },
        ):
            controller._wire_model()
        controller._workflow.set_model.assert_called_once_with(
            "the-model", "the-tokenizer"
        )

    def test_prefers_auto_train_student(self):
        controller = FeedbackController(repo_root=Path("/tmp/fb-wiring-test"))
        controller._workflow = MagicMock()
        at_state = _fake_module(
            state=_fake_module(
                student_net="student-net", student_tokenizer="student-tok"
            )
        )
        with patch.dict(
            sys.modules,
            {
                "routers.auto_train": at_state,
                "routers": _fake_module(),
            },
        ):
            controller._wire_model()
        controller._workflow.set_model.assert_called_once_with(
            "student-net", "student-tok"
        )

    def test_no_wire_without_server_model(self):
        controller = FeedbackController(repo_root=Path("/tmp/fb-wiring-test"))
        controller._workflow = MagicMock()
        server_state = _fake_module(model=None, tokenizer=None)
        at_state = _fake_module(state=_fake_module(student_net=None))
        with patch.dict(
            sys.modules,
            {
                "state": server_state,
                "routers.auto_train": at_state,
                "routers": _fake_module(),
            },
        ):
            controller._wire_model()
        controller._workflow.set_model.assert_not_called()

    def test_no_workflow_noop(self):
        controller = FeedbackController(repo_root=Path("/tmp/fb-wiring-test"))
        controller._workflow = None
        at_state = _fake_module(state=_fake_module(student_net=None))
        with patch.dict(
            sys.modules,
            {
                "routers.auto_train": at_state,
                "routers": _fake_module(),
            },
        ):
            controller._wire_model()
        assert True
