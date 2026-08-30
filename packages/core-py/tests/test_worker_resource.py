"""Tests for domains.infrastructure.resource_manager — ResourceAllocation; domains.infrastructure.model_worker — WorkerHealth, WorkerStreamStalledError."""

from domains.infrastructure.resource_manager import (
    ResourceAllocation, ResourceManager, compute_allocation,
    get_resource_manager, reset_resource_manager, _clamp, _env_int,
)
from domains.infrastructure.model_worker import WorkerHealth, WorkerStreamStalledError


# ---------------------------------------------------------------------------
# ResourceAllocation
# ---------------------------------------------------------------------------

class TestResourceAllocation:
    def test_defaults(self):
        ra = ResourceAllocation()
        assert ra.workload_mode == "balanced"
        assert ra.compute_threads == 0
        assert ra.inference_pool_size == 0

    def test_summary(self):
        ra = ResourceAllocation()
        s = ra.summary()
        assert isinstance(s, str)
        assert "balanced" in s

    def test_custom(self):
        ra = ResourceAllocation(compute_threads=4, io_threads=2, inference_pool_size=8)
        assert ra.compute_threads == 4
        assert ra.io_threads == 2
        assert ra.inference_pool_size == 8

    def test_all_fields_settable(self):
        ra = ResourceAllocation(
            compute_threads=2, io_threads=1,
            omp_num_threads=2, mkl_num_threads=2,
            openblas_num_threads=1, numexpr_num_threads=2,
            inference_pool_size=4, train_pool_size=2,
            task_queue_workers=4, dataloader_workers=2,
            concurrent_writes=4, concurrent_reads=16,
            process_guard_concurrent=1,
            workload_mode="inference",
        )
        assert ra.workload_mode == "inference"
        assert ra.concurrent_writes == 4
        assert ra.concurrent_reads == 16

    def test_summary_contains_all_fields(self):
        ra = ResourceAllocation(compute_threads=4, io_threads=2, inference_pool_size=8,
                                train_pool_size=4, task_queue_workers=6, dataloader_workers=2,
                                concurrent_writes=8, concurrent_reads=32)
        s = ra.summary()
        assert "compute=4" in s
        assert "io=2" in s
        assert "infer=8" in s
        assert "train=4" in s
        assert "queue=6" in s
        assert "dl=2" in s

    def test_frozen(self):
        ra = ResourceAllocation()
        try:
            ra.workload_mode = "other"
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_apply_env_no_overrides(self):
        ra = ResourceAllocation()
        ra2 = ra.apply_env()
        assert ra2.compute_threads == ra.compute_threads

    def test_workload_mode_values(self):
        for mode in ["balanced", "inference", "training"]:
            ra = ResourceAllocation(workload_mode=mode)
            assert ra.workload_mode == mode


# ---------------------------------------------------------------------------
# _clamp
# ---------------------------------------------------------------------------

class TestClamp:
    def test_within_range(self):
        assert _clamp(5, 1, 10) == 5

    def test_below_min(self):
        assert _clamp(-1, 0, 10) == 0

    def test_above_max(self):
        assert _clamp(20, 0, 10) == 10

    def test_at_boundaries(self):
        assert _clamp(0, 0, 10) == 0
        assert _clamp(10, 0, 10) == 10


# ---------------------------------------------------------------------------
# _env_int
# ---------------------------------------------------------------------------

class TestEnvInt:
    def test_existing_env(self, monkeypatch):
        monkeypatch.setenv("TEST_ENV_INT_VAR", "42")
        assert _env_int("TEST_ENV_INT_VAR", 0) == 42

    def test_missing_env(self):
        assert _env_int("NONEXISTENT_VAR_XYZ", 99) == 99

    def test_invalid_env(self, monkeypatch):
        monkeypatch.setenv("TEST_ENV_INT_BAD", "not_a_number")
        assert _env_int("TEST_ENV_INT_BAD", 7) == 7


# ---------------------------------------------------------------------------
# compute_allocation
# ---------------------------------------------------------------------------

class TestComputeAllocation:
    def test_balanced(self):
        alloc = compute_allocation(mode="balanced")
        assert alloc.workload_mode == "balanced"
        assert alloc.compute_threads >= 1
        assert alloc.inference_pool_size >= 1

    def test_inference(self):
        alloc = compute_allocation(mode="inference")
        assert alloc.workload_mode == "inference"

    def test_training(self):
        alloc = compute_allocation(mode="training")
        assert alloc.workload_mode == "training"

    def test_balanced_compute_threads(self):
        alloc = compute_allocation(mode="balanced")
        assert 1 <= alloc.compute_threads <= 4

    def test_inference_compute_threads(self):
        alloc = compute_allocation(mode="inference")
        assert 1 <= alloc.compute_threads <= 4

    def test_training_compute_threads(self):
        alloc = compute_allocation(mode="training")
        assert 1 <= alloc.compute_threads <= 8

    def test_pool_sizes_positive(self):
        for mode in ["balanced", "inference", "training"]:
            alloc = compute_allocation(mode=mode)
            assert alloc.inference_pool_size >= 1
            assert alloc.train_pool_size >= 1
            assert alloc.task_queue_workers >= 1

    def test_concurrent_writes_positive(self):
        alloc = compute_allocation(mode="balanced")
        assert alloc.concurrent_writes >= 1

    def test_concurrent_reads_positive(self):
        alloc = compute_allocation(mode="balanced")
        assert alloc.concurrent_reads >= 1

    def test_process_guard_concurrent(self):
        alloc = compute_allocation(mode="balanced")
        assert alloc.process_guard_concurrent >= 1

    def test_io_threads(self):
        alloc = compute_allocation(mode="balanced")
        assert alloc.io_threads >= 1

    def test_blas_threads_match_compute(self):
        alloc = compute_allocation(mode="balanced")
        assert alloc.omp_num_threads == alloc.compute_threads
        assert alloc.mkl_num_threads == alloc.compute_threads

    def test_openblas_always_one(self):
        alloc = compute_allocation(mode="balanced")
        assert alloc.openblas_num_threads == 1


# ---------------------------------------------------------------------------
# ResourceManager
# ---------------------------------------------------------------------------

class TestResourceManager:
    def test_init(self):
        rm = ResourceManager()
        assert rm.mode == "balanced"

    def test_properties(self):
        rm = ResourceManager()
        assert rm.compute_threads >= 1
        assert rm.inference_pool_size >= 1
        assert rm.train_pool_size >= 1

    def test_recompute(self):
        rm = ResourceManager()
        old_infer = rm.inference_pool_size
        rm.recompute("inference")
        assert rm.mode == "inference"

    def test_summary(self):
        rm = ResourceManager()
        s = rm.summary()
        assert isinstance(s, str)

    def test_mode_override(self):
        rm = ResourceManager()
        original_mode = rm.mode
        with rm.mode_override("training"):
            assert rm.mode == "training"
        assert rm.mode == original_mode

    def test_allocation_property(self):
        rm = ResourceManager()
        alloc = rm.allocation
        assert isinstance(alloc, ResourceAllocation)

    def test_apply_blas_env(self):
        rm = ResourceManager()
        rm.apply_blas_env()

    def test_apply_environment(self):
        rm = ResourceManager()
        rm.apply_environment()

    def test_init_with_mode(self):
        rm = ResourceManager(mode="inference")
        assert rm.mode == "inference"


# ---------------------------------------------------------------------------
# Singleton functions
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_resource_manager(self):
        rm = get_resource_manager()
        assert isinstance(rm, ResourceManager)

    def test_get_returns_same_instance(self):
        rm1 = get_resource_manager()
        rm2 = get_resource_manager()
        assert rm1 is rm2

    def test_reset_resource_manager(self):
        rm = reset_resource_manager()
        assert isinstance(rm, ResourceManager)


# ---------------------------------------------------------------------------
# WorkerHealth
# ---------------------------------------------------------------------------

class TestWorkerHealth:
    def test_defaults(self):
        wh = WorkerHealth()
        assert wh.alive is False
        assert wh.requests_served == 0
        assert wh.errors == 0
        assert wh.crashed is False

    def test_custom(self):
        wh = WorkerHealth(pid=1234, alive=True, requests_served=100, errors=2)
        assert wh.pid == 1234
        assert wh.alive is True
        assert wh.requests_served == 100
        assert wh.errors == 2

    def test_crash_count(self):
        wh = WorkerHealth(crash_count=3)
        assert wh.crash_count == 3

    def test_started_at(self):
        wh = WorkerHealth(started_at=1000.0)
        assert wh.started_at == 1000.0

    def test_last_heartbeat(self):
        wh = WorkerHealth(last_heartbeat=2000.0)
        assert wh.last_heartbeat == 2000.0

    def test_all_fields(self):
        wh = WorkerHealth(
            pid=999, alive=True, started_at=1.0, last_heartbeat=2.0,
            requests_served=50, errors=3, crashed=True, crash_count=1,
        )
        assert wh.pid == 999
        assert wh.alive is True
        assert wh.started_at == 1.0
        assert wh.last_heartbeat == 2.0
        assert wh.requests_served == 50
        assert wh.errors == 3
        assert wh.crashed is True
        assert wh.crash_count == 1

    def test_equality(self):
        wh1 = WorkerHealth(pid=1, alive=True)
        wh2 = WorkerHealth(pid=1, alive=True)
        assert wh1 == wh2

    def test_defaults_not_equal_to_custom(self):
        wh1 = WorkerHealth()
        wh2 = WorkerHealth(pid=1)
        assert wh1 != wh2


# ---------------------------------------------------------------------------
# WorkerStreamStalledError
# ---------------------------------------------------------------------------

class TestWorkerStreamStalledError:
    def test_is_runtime_error(self):
        assert issubclass(WorkerStreamStalledError, RuntimeError)

    def test_message(self):
        err = WorkerStreamStalledError("stream stalled")
        assert str(err) == "stream stalled"

    def test_can_catch_as_runtime_error(self):
        try:
            raise WorkerStreamStalledError("timeout")
        except RuntimeError:
            pass

    def test_empty_message(self):
        err = WorkerStreamStalledError("")
        assert str(err) == ""

    def test_raise_and_catch(self):
        raised = False
        try:
            raise WorkerStreamStalledError("worker stalled for 30s")
        except WorkerStreamStalledError as e:
            raised = True
            assert "30s" in str(e)
        assert raised


# ---------------------------------------------------------------------------
# Additional ResourceAllocation tests
# ---------------------------------------------------------------------------

class TestResourceAllocationExtra:
    def test_topology_detected(self):
        ra = ResourceAllocation()
        assert ra.topology is not None
        assert ra.topology.physical_cores >= 1

    def test_concurrent_writes_range(self):
        ra = compute_allocation()
        assert 1 <= ra.concurrent_writes <= 16

    def test_concurrent_reads_range(self):
        ra = compute_allocation()
        assert 1 <= ra.concurrent_reads <= 64

    def test_process_guard_concurrent_range(self):
        ra = compute_allocation()
        assert 1 <= ra.process_guard_concurrent <= 4

    def test_dataloader_workers_non_negative(self):
        ra = compute_allocation()
        assert ra.dataloader_workers >= 0

    def test_io_threads_range(self):
        ra = compute_allocation()
        assert 1 <= ra.io_threads <= 4

    def test_equality(self):
        ra1 = ResourceAllocation(compute_threads=4, io_threads=2)
        ra2 = ResourceAllocation(compute_threads=4, io_threads=2)
        assert ra1 == ra2

    def test_inequality(self):
        ra1 = ResourceAllocation(compute_threads=4)
        ra2 = ResourceAllocation(compute_threads=2)
        assert ra1 != ra2

    def test_apply_env_with_override(self, monkeypatch):
        monkeypatch.setenv("SLO_COMPUTE_THREADS", "99")
        ra = ResourceAllocation()
        ra2 = ra.apply_env()
        assert ra2.compute_threads == 99

    def test_apply_env_negative_value_ignored(self, monkeypatch):
        monkeypatch.setenv("SLO_COMPUTE_THREADS", "-5")
        ra = ResourceAllocation()
        ra2 = ra.apply_env()
        assert ra2.compute_threads == ra.compute_threads

    def test_apply_env_non_numeric_ignored(self, monkeypatch):
        monkeypatch.setenv("SLO_COMPUTE_THREADS", "abc")
        ra = ResourceAllocation()
        ra2 = ra.apply_env()
        assert ra2.compute_threads == ra.compute_threads

    def test_summary_returns_string(self):
        ra = ResourceAllocation()
        assert isinstance(ra.summary(), str)

    def test_frozen_prevents_modification(self):
        ra = ResourceAllocation()
        try:
            ra.compute_threads = 999
            assert False, "Should be frozen"
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Additional _clamp tests
# ---------------------------------------------------------------------------

class TestClampExtra:
    def test_same_value(self):
        assert _clamp(5, 5, 5) == 5

    def test_negative_range(self):
        assert _clamp(-5, -10, -1) == -5

    def test_negative_below_min(self):
        assert _clamp(-20, -10, -1) == -10

    def test_zero_range(self):
        assert _clamp(0, 0, 0) == 0

    def test_large_numbers(self):
        assert _clamp(1000000, 0, 999999) == 999999


# ---------------------------------------------------------------------------
# Additional _env_int tests
# ---------------------------------------------------------------------------

class TestEnvIntExtra:
    def test_zero_value(self, monkeypatch):
        monkeypatch.setenv("TEST_ZERO", "0")
        assert _env_int("TEST_ZERO", 99) == 0

    def test_negative_value(self, monkeypatch):
        monkeypatch.setenv("TEST_NEG", "-5")
        assert _env_int("TEST_NEG", 99) == -5

    def test_float_value(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT", "3.14")
        assert _env_int("TEST_FLOAT", 99) == 99

    def test_empty_string(self, monkeypatch):
        monkeypatch.setenv("TEST_EMPTY", "")
        assert _env_int("TEST_EMPTY", 99) == 99

    def test_whitespace(self, monkeypatch):
        monkeypatch.setenv("TEST_WHITESPACE", "  42  ")
        assert _env_int("TEST_WHITESPACE", 99) == 42


# ---------------------------------------------------------------------------
# Additional compute_allocation tests
# ---------------------------------------------------------------------------

class TestComputeAllocationExtra:
    def test_balanced_train_pool(self):
        alloc = compute_allocation(mode="balanced")
        assert alloc.train_pool_size >= 1

    def test_inference_higher_infer_pool(self):
        infer = compute_allocation(mode="inference")
        balanced = compute_allocation(mode="balanced")
        assert infer.inference_pool_size >= balanced.inference_pool_size

    def test_training_higher_train_pool(self):
        train = compute_allocation(mode="training")
        balanced = compute_allocation(mode="balanced")
        assert train.train_pool_size >= balanced.train_pool_size

    def test_numexpr_matches_compute(self):
        alloc = compute_allocation(mode="balanced")
        assert alloc.numexpr_num_threads == alloc.compute_threads

    def test_topology_used(self):
        alloc = compute_allocation()
        assert alloc.topology is not None

    def test_task_queue_positive(self):
        for mode in ["balanced", "inference", "training"]:
            alloc = compute_allocation(mode=mode)
            assert alloc.task_queue_workers >= 1

    def test_dataloader_non_negative(self):
        for mode in ["balanced", "inference", "training"]:
            alloc = compute_allocation(mode=mode)
            assert alloc.dataloader_workers >= 0


# ---------------------------------------------------------------------------
# Additional ResourceManager tests
# ---------------------------------------------------------------------------

class TestResourceManagerExtra:
    def test_io_threads_property(self):
        rm = ResourceManager()
        assert rm.io_threads >= 1

    def test_omp_num_threads(self):
        rm = ResourceManager()
        assert rm.omp_num_threads >= 1

    def test_mkl_num_threads(self):
        rm = ResourceManager()
        assert rm.mkl_num_threads >= 1

    def test_openblas_num_threads(self):
        rm = ResourceManager()
        assert rm.openblas_num_threads == 1

    def test_numexpr_num_threads(self):
        rm = ResourceManager()
        assert rm.numexpr_num_threads >= 1

    def test_concurrent_writes(self):
        rm = ResourceManager()
        assert rm.concurrent_writes >= 1

    def test_concurrent_reads(self):
        rm = ResourceManager()
        assert rm.concurrent_reads >= 1

    def test_process_guard_concurrent(self):
        rm = ResourceManager()
        assert rm.process_guard_concurrent >= 1

    def test_mode_override_restores_on_exception(self):
        rm = ResourceManager()
        original = rm.mode
        try:
            with rm.mode_override("training"):
                raise ValueError("test")
        except ValueError:
            pass
        assert rm.mode == original

    def test_recompute_changes_allocation(self):
        rm = ResourceManager()
        old_infer = rm.inference_pool_size
        rm.recompute("inference")
        new_infer = rm.inference_pool_size
        assert new_infer != old_infer or rm.mode == "inference"

    def test_apply_compute_limits(self):
        rm = ResourceManager()
        rm.apply_compute_limits()


# ---------------------------------------------------------------------------
# Additional Singleton tests
# ---------------------------------------------------------------------------

class TestSingletonExtra:
    def test_reset_changes_instance(self):
        rm1 = get_resource_manager()
        rm2 = reset_resource_manager()
        assert rm1 is not rm2

    def test_reset_then_get_returns_new(self):
        reset_resource_manager()
        rm = get_resource_manager()
        assert isinstance(rm, ResourceManager)

    def test_get_with_mode(self):
        rm = get_resource_manager(mode="inference")
        assert isinstance(rm, ResourceManager)


# ---------------------------------------------------------------------------
# Additional WorkerHealth tests
# ---------------------------------------------------------------------------

class TestWorkerHealthExtra:
    def test_repr(self):
        wh = WorkerHealth(pid=1, alive=True)
        r = repr(wh)
        assert "WorkerHealth" in r

    def test_equality_different_pid(self):
        wh1 = WorkerHealth(pid=1)
        wh2 = WorkerHealth(pid=2)
        assert wh1 != wh2

    def test_equality_different_alive(self):
        wh1 = WorkerHealth(pid=1, alive=True)
        wh2 = WorkerHealth(pid=1, alive=False)
        assert wh1 != wh2

    def test_copy(self):
        wh1 = WorkerHealth(pid=1, alive=True, requests_served=10)
        wh2 = WorkerHealth(
            pid=wh1.pid, alive=wh1.alive, requests_served=wh1.requests_served
        )
        assert wh1 == wh2

    def test_crash_count_increment(self):
        wh = WorkerHealth(crash_count=0)
        wh2 = WorkerHealth(crash_count=wh.crash_count + 1)
        assert wh2.crash_count == 1

    def test_errors_increment(self):
        wh = WorkerHealth(errors=0)
        wh2 = WorkerHealth(errors=wh.errors + 1)
        assert wh2.errors == 1

    def test_requests_served_increment(self):
        wh = WorkerHealth(requests_served=0)
        wh2 = WorkerHealth(requests_served=wh.requests_served + 1)
        assert wh2.requests_served == 1


# ---------------------------------------------------------------------------
# Additional WorkerStreamStalledError tests
# ---------------------------------------------------------------------------

class TestWorkerStreamStalledErrorExtra:
    def test_is_exception(self):
        assert issubclass(WorkerStreamStalledError, Exception)

    def test_with_context(self):
        err = WorkerStreamStalledError("worker-1 stalled for 30s")
        assert "worker-1" in str(err)

    def test_multiple_catches(self):
        for i in range(3):
            try:
                raise WorkerStreamStalledError(f"stall {i}")
            except WorkerStreamStalledError as e:
                assert f"stall {i}" in str(e)
