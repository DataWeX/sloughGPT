import threading
from domains.training.runtime_protocol import (
    set_training_runtime,
    get_training_runtime,
    _NoOpRuntime,
)


class TestNoOpRuntime:
    def test_register_noop(self):
        rt = _NoOpRuntime()
        rt.register("j1", {"name": "test"})

    def test_get_returns_none(self):
        rt = _NoOpRuntime()
        assert rt.get("j1") is None

    def test_sync_noop(self):
        rt = _NoOpRuntime()
        rt.sync("j1")

    def test_register_with_optional_args(self):
        rt = _NoOpRuntime()
        rt.register("j2", {}, cancel_event=threading.Event(), config={"lr": 0.001})


class TestTrainingRuntimeSingleton:
    def setup_method(self):
        set_training_runtime(None)

    def teardown_method(self):
        set_training_runtime(None)

    def test_default_is_noop(self):
        rt = get_training_runtime()
        assert isinstance(rt, _NoOpRuntime)

    def test_set_and_get(self):
        class FakeRuntime:
            def register(self, job_id, job, cancel_event=None, config=None):
                pass
            def get(self, job_id):
                return {"id": job_id}
            def sync(self, job_id):
                pass

        fake = FakeRuntime()
        set_training_runtime(fake)
        assert get_training_runtime() is fake

    def test_get_after_none_returns_noop(self):
        set_training_runtime(None)
        rt = get_training_runtime()
        assert isinstance(rt, _NoOpRuntime)
