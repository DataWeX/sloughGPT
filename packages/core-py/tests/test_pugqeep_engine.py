"""Tests for PGQ Engine — dispatch mode, routing, callbacks, lifecycle."""

import threading
import time

import pytest

from domains.infrastructure.pugqeep.engine import (
    Engine,
    Process,
    ProcessStatus,
    Stem,
    StemStatus,
    Tree,
    TreeStatus,
)


# ── Helpers ──────────────────────────────────────────────────────

def _noop():
    return "ok"


def _sleep_and_return(secs, value):
    time.sleep(secs)
    return value


def _fail():
    raise RuntimeError("boom")


# ── Process lifecycle ────────────────────────────────────────────

class TestProcess:
    def test_created_by_default(self):
        p = Process(fn=_noop)
        assert p.status == ProcessStatus.CREATED
        assert not p.is_done

    def test_ready(self):
        p = Process(fn=_noop)
        p.ready()
        assert p.status == ProcessStatus.READY

    def test_running(self):
        p = Process(fn=_noop)
        p.running()
        assert p.status == ProcessStatus.RUNNING
        assert p.started_at is not None

    def test_complete(self):
        p = Process(fn=_noop)
        p.running()
        p.complete("result")
        assert p.status == ProcessStatus.COMPLETED
        assert p.result == "result"
        assert p.is_done
        assert p.elapsed is not None

    def test_fail(self):
        p = Process(fn=_noop)
        p.running()
        p.fail("error msg")
        assert p.status == ProcessStatus.FAILED
        assert p.error == "error msg"
        assert p.is_done

    def test_cancel(self):
        p = Process(fn=_noop)
        p.cancel()
        assert p.status == ProcessStatus.CANCELLED
        assert p.is_done

    def test_to_dict(self):
        p = Process(fn=_noop, name="test")
        d = p.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "created"
        assert "id" in d


# ── Stem lifecycle ───────────────────────────────────────────────

class TestStem:
    def test_all_done_when_empty(self):
        s = Stem()
        assert s.all_done

    def test_all_done_when_all_complete(self):
        p1 = Process(fn=_noop)
        p2 = Process(fn=_noop)
        p1.complete()
        p2.complete()
        s = Stem(processes=[p1, p2])
        assert s.all_done

    def test_not_all_done_when_one_running(self):
        p1 = Process(fn=_noop)
        p2 = Process(fn=_noop)
        p1.complete()
        p2.running()
        s = Stem(processes=[p1, p2])
        assert not s.all_done

    def test_results_collects_completed(self):
        p1 = Process(fn=_noop)
        p2 = Process(fn=_noop)
        p1.complete("a")
        p2.fail("err")
        s = Stem(processes=[p1, p2])
        assert s.results() == ["a"]

    def test_errors_collects_failed(self):
        p1 = Process(fn=_noop)
        p2 = Process(fn=_noop)
        p1.complete()
        p2.fail("err msg")
        s = Stem(processes=[p1, p2])
        assert s.errors() == ["err msg"]


# ── Tree ─────────────────────────────────────────────────────────

class TestTree:
    def test_branch_executes_processes(self):
        tree = Tree("test", pool_workers=2)
        p1 = Process(fn=_sleep_and_return, args=(0.01, "a"))
        p2 = Process(fn=_sleep_and_return, args=(0.01, "b"))
        stem = tree.branch([p1, p2])
        tree.wait_stem(stem, timeout=5)
        assert p1.result == "a"
        assert p2.result == "b"
        tree.shutdown()

    def test_branch_marks_stem_complete(self):
        tree = Tree("test", pool_workers=2)
        p = Process(fn=_noop)
        stem = tree.branch([p])
        tree.wait_stem(stem, timeout=5)
        assert stem.status == StemStatus.COMPLETED
        tree.shutdown()

    def test_branch_handles_failure(self):
        tree = Tree("test", pool_workers=2)
        p = Process(fn=_fail)
        stem = tree.branch([p])
        tree.wait_stem(stem, timeout=5)
        assert stem.status == StemStatus.FAILED
        assert p.status == ProcessStatus.FAILED
        tree.shutdown()

    def test_store_and_recall(self):
        tree = Tree("test")
        tree.store("key", "value")
        assert tree.recall("key") == "value"
        assert tree.recall("missing") is None
        tree.shutdown()

    def test_max_stems_limit(self):
        tree = Tree("test", max_stems=1, pool_workers=1)
        p1 = Process(fn=_sleep_and_return, args=(1.0, None))
        tree.branch([p1])
        with pytest.raises(RuntimeError, match="max stems"):
            p2 = Process(fn=_noop)
            tree.branch([p2])
        tree.shutdown()


# ── Engine: direct mode ──────────────────────────────────────────

class TestEngineDirect:
    def test_spawn_and_branch(self):
        engine = Engine("test")
        engine.tree("t1")
        p = engine.spawn(_sleep_and_return, 0.01, "done")
        stem = engine.branch("t1", [p])
        engine.get_tree("t1").wait_stem(stem, timeout=5)
        assert p.result == "done"
        engine.stop()

    def test_spawn_with_explicit_tree(self):
        engine = Engine("test")
        engine.tree("data")
        engine.tree("compute")
        p = engine.spawn(_noop, tree="compute")
        assert p._tree_name == "compute"
        engine.stop()

    def test_branch_on_missing_tree_raises(self):
        engine = Engine("test")
        with pytest.raises(ValueError, match="not found"):
            engine.branch("nope", [])

    def test_list_processes(self):
        engine = Engine("test")
        engine.tree("t")
        p1 = engine.spawn(_noop)
        p2 = engine.spawn(_noop)
        all_procs = engine.list_processes()
        assert len(all_procs) == 2
        engine.stop()

    def test_get_process(self):
        engine = Engine("test")
        p = engine.spawn(_noop)
        assert engine.get_process(p.id) is p
        assert engine.get_process("nope") is None
        engine.stop()


# ── Engine: dispatch mode ────────────────────────────────────────

class TestEngineDispatch:
    def test_dispatch_routes_by_name(self):
        engine = Engine("test")
        engine.tree("data")
        engine.tree("compute")
        engine.route("load", "data")
        engine.route("run", "compute")

        p1 = engine.spawn(_noop, name="load")
        p2 = engine.spawn(_noop, name="run")

        dispatched = engine.dispatch()
        assert dispatched == 2
        assert p1._tree_name == "data"
        assert p2._tree_name == "compute"
        engine.stop()

    def test_dispatch_round_robin_ungrouped(self):
        engine = Engine("test")
        engine.tree("a")
        engine.tree("b")

        p1 = engine.spawn(_noop, name="unrouted")
        p2 = engine.spawn(_noop, name="unrouted")

        engine.dispatch()
        # Round-robin: first → "a", second → "b"
        trees = {p1._tree_name, p2._tree_name}
        assert trees == {"a", "b"}
        engine.stop()

    def test_dispatch_explicit_tree_overrides_routing(self):
        engine = Engine("test")
        engine.tree("default")
        engine.tree("special")
        engine.route("task", "default")

        p = engine.spawn(_noop, name="task", tree="special")
        engine.dispatch()
        assert p._tree_name == "special"
        engine.stop()

    def test_dispatch_empty_returns_zero(self):
        engine = Engine("test")
        assert engine.dispatch() == 0
        engine.stop()

    def test_dispatch_batches_large_groups(self):
        engine = Engine("test")
        engine.tree("t", pool_workers=1)
        engine.route("work", "t")
        engine._dispatch_batch_size = 2

        procs = [engine.spawn(_noop, name="work") for _ in range(5)]
        dispatched = engine.dispatch()
        assert dispatched == 5
        engine.stop()


# ── Engine: run loop ─────────────────────────────────────────────

class TestEngineRun:
    def test_run_dispatches_pending(self):
        engine = Engine("test")
        engine.tree("t")
        engine.route("work", "t")

        p = engine.spawn(_sleep_and_return, 0.01, "done", name="work")

        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()

        assert p.result == "done"
        assert p.status == ProcessStatus.COMPLETED

    def test_run_background_is_non_blocking(self):
        engine = Engine("test")
        engine.tree("t")
        thread = engine.run_background(poll_interval=0.05)
        assert thread.is_alive()
        engine.stop()
        thread.join(timeout=2)
        assert not thread.is_alive()


# ── Engine: on_complete callbacks ────────────────────────────────

class TestEngineCallbacks:
    def test_on_complete_fires(self):
        engine = Engine("test")
        engine.tree("t")
        engine.route("work", "t")

        completed = []
        engine.on_complete(lambda p: completed.append(p.id))

        p = engine.spawn(_sleep_and_return, 0.01, "x", name="work")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()

        assert p.id in completed

    def test_on_complete_fires_for_failure(self):
        engine = Engine("test")
        engine.tree("t")
        engine.route("fail", "t")

        results = []
        engine.on_complete(lambda p: results.append(p.status))

        p = engine.spawn(_fail, name="fail")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()

        assert ProcessStatus.FAILED in results

    def test_on_complete_callback_error_does_not_crash(self):
        engine = Engine("test")
        engine.tree("t")
        engine.route("work", "t")

        def bad_callback(p):
            raise ValueError("callback error")

        engine.on_complete(bad_callback)
        p = engine.spawn(_noop, name="work")

        # Should not raise
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()
        assert p.is_done


# ── Engine: to_dict ──────────────────────────────────────────────

class TestEngineToDict:
    def test_to_dict_structure(self):
        engine = Engine("test")
        engine.tree("t1")
        engine.route("work", "t1")
        engine.spawn(_noop, name="work")

        d = engine.to_dict()
        assert d["name"] == "test"
        assert "trees" in d
        assert d["processes"] == 1
        assert d["pending"] == 1
        assert d["routing"] == {"work": "t1"}
        engine.stop()

    def test_max_trees_limit(self):
        engine = Engine("test", max_trees=2)
        engine.tree("a")
        engine.tree("b")
        with pytest.raises(RuntimeError, match="max trees"):
            engine.tree("c")
        engine.stop()


# ── Integration: model loading + training scenario ───────────────

class TestEngineIntegration:
    def test_model_load_then_train(self):
        """Simulate: load model, then train on completion."""
        engine = Engine("sim")
        engine.tree("data", pool_workers=2)
        engine.tree("train", pool_workers=2)
        engine.route("load", "data")
        engine.route("epoch", "train")

        model = {"loaded": False, "loss": 5.0}

        def do_load():
            time.sleep(0.05)
            model["loaded"] = True
            return {"loaded": True}

        def do_epoch():
            model["loss"] *= 0.9
            return {"loss": model["loss"]}

        load_proc = engine.spawn(do_load, name="load")
        epoch_procs = [engine.spawn(do_epoch, name="epoch") for _ in range(3)]

        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=10)
        engine.stop()

        assert load_proc.result == {"loaded": True}
        assert model["loaded"]
        for p in epoch_procs:
            assert p.status == ProcessStatus.COMPLETED
            assert p.result["loss"] < 5.0

    def test_parallel_inference_during_training(self):
        """Inference runs alongside training on separate trees."""
        engine = Engine("sim")
        engine.tree("train", pool_workers=2)
        engine.tree("infer", pool_workers=2)
        engine.route("train", "train")
        engine.route("infer", "infer")

        results = {"train": [], "infer": []}

        def train_step():
            time.sleep(0.02)
            results["train"].append(1)
            return "trained"

        def infer_step():
            time.sleep(0.02)
            results["infer"].append(1)
            return "inferred"

        for _ in range(3):
            engine.spawn(train_step, name="train")
        for _ in range(3):
            engine.spawn(infer_step, name="infer")

        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=10)
        engine.stop()

        assert len(results["train"]) == 3
        assert len(results["infer"]) == 3
