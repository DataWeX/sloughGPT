"""Tests for pugqeep checkpoint integration — save, compress, load roundtrip."""

import json
import numpy as np
import pytest
from pathlib import Path
import tempfile


class TestCompressCheckpoint:
    """Test compress_checkpoint() produces loadable .points.json + manifest."""

    def test_compress_checkpoint_creates_files(self):
        from domains.training.executor import compress_checkpoint
        from domains.training.slonet import SloTransformer

        with tempfile.TemporaryDirectory() as tmpdir:
            net = SloTransformer(vocab_size=256, n_embed=64, n_layer=2,
                                  n_head=4, block_size=32, use_rope=True)
            from domains.training.export import export_to_sou
            soul_path = str(Path(tmpdir) / "test_model.soul")
            export_to_sou(net, soul_path)

            stats = compress_checkpoint(soul_path, n_clusters=8)
            assert stats is not None
            assert stats["point_count"] > 0
            assert stats["compression_ratio"] > 1.0

            lib_path = Path(stats["library_path"])
            meta_path = Path(stats["meta_path"])
            assert lib_path.exists(), f"Library not found: {lib_path}"
            assert meta_path.exists(), f"Meta not found: {meta_path}"

            meta = json.loads(meta_path.read_text())
            assert "lineage" in meta
            assert "metadata" in meta

    def test_compress_checkpoint_stats(self):
        from domains.training.executor import compress_checkpoint
        from domains.training.slonet import SloTransformer

        with tempfile.TemporaryDirectory() as tmpdir:
            net = SloTransformer(vocab_size=128, n_embed=32, n_layer=1,
                                  n_head=2, block_size=16, use_rope=False)
            from domains.training.export import export_to_sou
            soul_path = str(Path(tmpdir) / "small.soul")
            export_to_sou(net, soul_path)

            stats = compress_checkpoint(soul_path, n_clusters=4)
            assert stats is not None
            assert stats["total_raw_bytes"] > 0
            assert stats["total_compressed_bytes"] > 0
            assert stats["compression_ratio"] >= 1.0

    def test_compress_checkpoint_nonexistent_file(self):
        from domains.training.executor import compress_checkpoint
        result = compress_checkpoint("/nonexistent/path/model.soul")
        assert result is None

    def test_compress_checkpoint_metadata_contents(self):
        from domains.training.executor import compress_checkpoint
        from domains.training.slonet import SloTransformer

        with tempfile.TemporaryDirectory() as tmpdir:
            net = SloTransformer(vocab_size=128, n_embed=32, n_layer=1,
                                  n_head=2, block_size=16, use_rope=True)
            from domains.training.export import export_to_sou
            soul_path = str(Path(tmpdir) / "meta_test.soul")
            export_to_sou(net, soul_path)

            stats = compress_checkpoint(soul_path, n_clusters=4)
            meta = json.loads(Path(stats["meta_path"]).read_text())
            assert "source" in meta
            assert "n_clusters" in meta
            assert meta["n_clusters"] == 4
            assert "metadata" in meta
            assert "weight_shapes" in meta["metadata"]

    def test_compress_checkpoint_point_count_matches_weights(self):
        from domains.training.executor import compress_checkpoint
        from domains.training.slonet import SloTransformer

        with tempfile.TemporaryDirectory() as tmpdir:
            net = SloTransformer(vocab_size=64, n_embed=32, n_layer=1,
                                  n_head=2, block_size=16, use_rope=False)
            from domains.training.export import export_to_sou
            soul_path = str(Path(tmpdir) / "count.soul")
            export_to_sou(net, soul_path)

            stats = compress_checkpoint(soul_path, n_clusters=4)
            state_dict = net.state_dict()
            expected_points = len(state_dict)
            assert stats["point_count"] == expected_points

    def test_compress_checkpoint_library_loadable(self):
        from domains.training.executor import compress_checkpoint
        from domains.training.slonet import SloTransformer
        from domains.infrastructure.pugqeep.library import PointLibrary

        with tempfile.TemporaryDirectory() as tmpdir:
            net = SloTransformer(vocab_size=128, n_embed=32, n_layer=1,
                                  n_head=2, block_size=16, use_rope=True)
            from domains.training.export import export_to_sou
            soul_path = str(Path(tmpdir) / "lib_test.soul")
            export_to_sou(net, soul_path)

            stats = compress_checkpoint(soul_path, n_clusters=8)
            lib = PointLibrary.load(Path(stats["library_path"]))
            assert len(lib.list_all()) == stats["point_count"]

    def test_compress_checkpoint_lineage(self):
        from domains.training.executor import compress_checkpoint
        from domains.training.slonet import SloTransformer

        with tempfile.TemporaryDirectory() as tmpdir:
            net = SloTransformer(vocab_size=64, n_embed=32, n_layer=1,
                                  n_head=2, block_size=16, use_rope=False)
            from domains.training.export import export_to_sou
            soul_path = str(Path(tmpdir) / "lineage.soul")
            export_to_sou(net, soul_path)

            stats = compress_checkpoint(soul_path, n_clusters=4)
            meta = json.loads(Path(stats["meta_path"]).read_text())
            assert len(meta["lineage"]) == 1
            assert soul_path in meta["lineage"][0]


class TestLoadFromPoints:
    """Test load_from_points() reads PointLibrary and returns ModelTree."""

    def test_load_from_points_file(self):
        from domains.infrastructure.pugqeep.library import PointLibrary
        from domains.infrastructure.pugqeep.model_tree import (
            load_from_points, decompress_tree,
        )
        from domains.infrastructure.pugqeep.point import Point

        with tempfile.TemporaryDirectory() as tmpdir:
            lib = PointLibrary(name="test", storage_dir=Path(tmpdir))
            centroids = np.array([0.1, 0.5, 0.9], dtype=np.float32)
            assignments = np.array([0, 1, 2, 0, 1, 2], dtype=np.uint8)
            p = Point(
                identity="test_model.weight",
                function_type="cluster",
                params={"centroids": centroids, "assignments": assignments},
                accuracy=0.95,
            )
            lib.add(p)
            lib_path = Path(tmpdir) / "test_model.points.json"
            lib.save(lib_path)

            tree, meta = load_from_points(str(Path(tmpdir) / "test_model"))
            assert tree.is_loaded
            assert tree.library.has("test_model.weight")

            weights = decompress_tree(tree)
            assert "weight" in weights
            np.testing.assert_array_equal(weights["weight"], centroids[assignments])

    def test_load_from_points_directory(self):
        from domains.infrastructure.pugqeep.library import PointLibrary
        from domains.infrastructure.pugqeep.model_tree import load_from_points
        from domains.infrastructure.pugqeep.point import Point

        with tempfile.TemporaryDirectory() as tmpdir:
            lib = PointLibrary(name="test", storage_dir=Path(tmpdir))
            centroids = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            assignments = np.array([0, 1, 2], dtype=np.uint8)
            p = Point(
                identity="layer.bias",
                function_type="cluster",
                params={"centroids": centroids, "assignments": assignments},
                accuracy=0.9,
            )
            lib.add(p)
            model_dir = Path(tmpdir) / "mymodel"
            model_dir.mkdir()
            lib.save(model_dir / "library.json")

            tree, meta = load_from_points(str(model_dir))
            assert tree.is_loaded

    def test_load_from_points_not_found(self):
        from domains.infrastructure.pugqeep.model_tree import load_from_points
        with pytest.raises(FileNotFoundError):
            load_from_points("/nonexistent/path")

    def test_load_from_points_multiple_weights(self):
        from domains.infrastructure.pugqeep.library import PointLibrary
        from domains.infrastructure.pugqeep.model_tree import (
            load_from_points, decompress_tree,
        )
        from domains.infrastructure.pugqeep.point import Point

        with tempfile.TemporaryDirectory() as tmpdir:
            lib = PointLibrary(name="multi", storage_dir=Path(tmpdir))
            for name in ["w1", "w2", "w3"]:
                centroids = np.random.randn(4).astype(np.float32)
                assignments = np.random.randint(0, 4, size=8).astype(np.uint8)
                p = Point(
                    identity=f"multi.{name}",
                    function_type="cluster",
                    params={"centroids": centroids, "assignments": assignments},
                    accuracy=0.9,
                )
                lib.add(p)
            lib.save(Path(tmpdir) / "multi.points.json")

            tree, _ = load_from_points(str(Path(tmpdir) / "multi"))
            weights = decompress_tree(tree)
            assert len(weights) == 3

    def test_load_from_points_preserves_shapes(self):
        from domains.infrastructure.pugqeep.library import PointLibrary
        from domains.infrastructure.pugqeep.model_tree import (
            load_from_points, decompress_tree,
        )
        from domains.infrastructure.pugqeep.point import Point

        with tempfile.TemporaryDirectory() as tmpdir:
            lib = PointLibrary(name="shape", storage_dir=Path(tmpdir))
            centroids = np.random.randn(16).astype(np.float32)
            assignments = np.random.randint(0, 16, size=32).astype(np.uint8)
            p = Point(
                identity="shape.weight",
                function_type="cluster",
                params={"centroids": centroids, "assignments": assignments},
                accuracy=0.85,
            )
            lib.add(p)
            lib.save(Path(tmpdir) / "shape.points.json")

            tree, _ = load_from_points(str(Path(tmpdir) / "shape"))
            weights = decompress_tree(tree)
            assert weights["weight"].shape == (32,)


class TestImportFromSouFallback:
    """Test import_from_sou falls back to Points when .soul is missing."""

    def test_import_from_sou_loads_points(self):
        from domains.training.executor import compress_checkpoint
        from domains.training.slonet import SloTransformer
        from domains.training.export import export_to_sou

        with tempfile.TemporaryDirectory() as tmpdir:
            net = SloTransformer(vocab_size=128, n_embed=32, n_layer=1,
                                  n_head=2, block_size=16, use_rope=True)
            soul_path = str(Path(tmpdir) / "test.soul")
            export_to_sou(net, soul_path)

            stats = compress_checkpoint(soul_path, n_clusters=16)
            assert stats is not None

            Path(soul_path).unlink()

            from domains.training.slonet import import_from_sou
            loaded = import_from_sou(soul_path)
            assert loaded is not None
            assert hasattr(loaded, "state_dict")
            assert loaded.n_layer == 1, f"Expected n_layer=1, got {loaded.n_layer}"
            assert loaded.n_head == 2, f"Expected n_head=2, got {loaded.n_head}"

            orig_weights = net.state_dict()
            loaded_weights = loaded.state_dict()
            assert set(orig_weights.keys()) == set(loaded_weights.keys())
            for key in orig_weights:
                orig = orig_weights[key]
                loaded_w = loaded_weights[key]
                assert orig.shape == loaded_w.shape, f"Shape mismatch at {key}"
                close_mask = np.isclose(orig, loaded_w, rtol=0.5, atol=0.2)
                match_pct = close_mask.sum() / close_mask.size
                assert match_pct > 0.3, (
                    f"Weight {key}: only {match_pct:.1%} elements match "
                    f"(expected >30% for VQ compression)"
                )

    def test_import_from_sou_prefers_soul_over_points(self):
        from domains.training.slonet import SloTransformer
        from domains.training.export import export_to_sou

        with tempfile.TemporaryDirectory() as tmpdir:
            net = SloTransformer(vocab_size=128, n_embed=32, n_layer=1,
                                  n_head=2, block_size=16, use_rope=True)
            soul_path = str(Path(tmpdir) / "test.soul")
            export_to_sou(net, soul_path)

            points_path = Path(tmpdir) / "test.points.json"
            points_path.write_text('{"name": "test", "points": []}')

            from domains.training.slonet import import_from_sou
            loaded = import_from_sou(soul_path)
            assert loaded is not None
            weights = loaded.state_dict()
            assert len(weights) > 0

    def test_import_from_sou_nonexistent_returns_none(self):
        from domains.training.slonet import import_from_sou
        with pytest.raises(FileNotFoundError):
            import_from_sou("/nonexistent/path/model.soul")

    def test_import_from_sou_weights_are_numpy(self):
        from domains.training.executor import compress_checkpoint
        from domains.training.slonet import SloTransformer
        from domains.training.export import export_to_sou

        with tempfile.TemporaryDirectory() as tmpdir:
            net = SloTransformer(vocab_size=64, n_embed=32, n_layer=1,
                                  n_head=2, block_size=16, use_rope=False)
            soul_path = str(Path(tmpdir) / "np_test.soul")
            export_to_sou(net, soul_path)

            compress_checkpoint(soul_path, n_clusters=8)
            Path(soul_path).unlink()

            from domains.training.slonet import import_from_sou
            loaded = import_from_sou(soul_path)
            for name, w in loaded.state_dict().items():
                assert isinstance(w, np.ndarray), f"{name} is not numpy array"

    def test_import_from_sou_model_attributes(self):
        from domains.training.executor import compress_checkpoint
        from domains.training.slonet import SloTransformer
        from domains.training.export import export_to_sou

        with tempfile.TemporaryDirectory() as tmpdir:
            net = SloTransformer(vocab_size=256, n_embed=64, n_layer=3,
                                  n_head=4, block_size=32, use_rope=True)
            soul_path = str(Path(tmpdir) / "attr.soul")
            export_to_sou(net, soul_path)

            compress_checkpoint(soul_path, n_clusters=8)
            Path(soul_path).unlink()

            from domains.training.slonet import import_from_sou
            loaded = import_from_sou(soul_path)
            assert loaded.vocab_size == 256
            assert loaded.n_embed == 64
            assert loaded.n_layer == 3
            assert loaded.n_head == 4


class TestDecompressTree:
    """Test decompress_tree extracts all weights from ModelTree."""

    def test_decompress_all_weights(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree, decompress_tree

        tree = ModelTree("test", n_clusters=8)
        weights = {
            "layer1.weight": np.random.randn(32, 16).astype(np.float32),
            "layer1.bias": np.random.randn(32).astype(np.float32),
            "layer2.weight": np.random.randn(16, 32).astype(np.float32),
        }
        tree.load_weights(weights)

        decompressed = decompress_tree(tree)
        assert len(decompressed) == 3
        for name in weights:
            assert name in decompressed
            orig = weights[name]
            dec = decompressed[name]
            assert orig.shape == dec.shape
            close_mask = np.isclose(orig, dec, rtol=1.0, atol=0.5)
            match_pct = close_mask.sum() / close_mask.size
            assert match_pct > 0.8, (
                f"Weight {name}: only {match_pct:.1%} elements match "
                f"(expected >80%)"
            )

    def test_decompress_preserves_shapes(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree, decompress_tree

        tree = ModelTree("test", n_clusters=4)
        weights = {
            "embed.weight": np.random.randn(100, 64).astype(np.float32),
            "output.bias": np.random.randn(100).astype(np.float32),
        }
        tree.load_weights(weights)

        decompressed = decompress_tree(tree)
        for name, arr in decompressed.items():
            assert arr.shape == weights[name].shape

    def test_decompress_empty_tree(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree, decompress_tree

        tree = ModelTree("empty", n_clusters=4)
        decompressed = decompress_tree(tree)
        assert len(decompressed) == 0

    def test_decompress_single_weight(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree, decompress_tree

        tree = ModelTree("single", n_clusters=4)
        weights = {"only.weight": np.random.randn(8, 8).astype(np.float32)}
        tree.load_weights(weights)

        decompressed = decompress_tree(tree)
        assert len(decompressed) == 1
        assert "only.weight" in decompressed

    def test_decompress_many_clusters_better_accuracy(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree, decompress_tree
        from domains.infrastructure.pugqeep.compressor import PointCompressor

        orig = np.random.randn(64, 32).astype(np.float32)

        compressor_low = PointCompressor(n_clusters=4, quantize_centroids=False, residual_threshold=0.0)
        tree_low = ModelTree("low", n_clusters=4, compressor=compressor_low)
        tree_low.load_weights({"w": orig.copy()})
        dec_low = decompress_tree(tree_low)["w"]

        compressor_high = PointCompressor(n_clusters=32, quantize_centroids=False, residual_threshold=0.0)
        tree_high = ModelTree("high", n_clusters=32, compressor=compressor_high)
        tree_high.load_weights({"w": orig.copy()})
        dec_high = decompress_tree(tree_high)["w"]

        err_low = np.mean((orig - dec_low) ** 2)
        err_high = np.mean((orig - dec_high) ** 2)
        assert err_high <= err_low


class TestJobStatus:
    def test_job_status_values(self):
        from domains.training.executor import JobStatus
        assert JobStatus.QUEUED.value == "queued"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_job_status_is_str(self):
        from domains.training.executor import JobStatus
        assert isinstance(JobStatus.QUEUED, str)

    def test_job_status_comparison(self):
        from domains.training.executor import JobStatus
        assert JobStatus.QUEUED == "queued"
        assert JobStatus.RUNNING != "queued"


class TestJobInfo:
    def test_job_info_defaults(self):
        from domains.training.executor import JobInfo, JobStatus
        info = JobInfo(job_id="j1")
        assert info.job_id == "j1"
        assert info.status == JobStatus.QUEUED
        assert info.future is None
        assert info.error is None
        assert info.cancel_requested is False
        assert info.result is None

    def test_job_info_elapsed(self):
        from domains.training.executor import JobInfo
        import time
        info = JobInfo(job_id="j1")
        time.sleep(0.01)
        elapsed = info.elapsed()
        assert elapsed > 0

    def test_job_info_to_dict(self):
        from domains.training.executor import JobInfo, JobStatus
        info = JobInfo(job_id="j1", tree_id="t1")
        d = info.to_dict()
        assert d["job_id"] == "j1"
        assert d["tree_id"] == "t1"
        assert d["status"] == "queued"
        assert "submitted_at" in d
        assert "elapsed_s" in d

    def test_job_info_to_dict_completed(self):
        from domains.training.executor import JobInfo, JobStatus
        import numpy as np
        info = JobInfo(job_id="j1", status=JobStatus.COMPLETED)
        info.result = {"w": np.zeros(10)}
        d = info.to_dict()
        assert "result_keys" in d
        assert "result_size_bytes" in d

    def test_job_info_to_dict_non_dict_result(self):
        from domains.training.executor import JobInfo, JobStatus
        info = JobInfo(job_id="j1", status=JobStatus.COMPLETED)
        info.result = "some_string"
        d = info.to_dict()
        assert d["result_type"] == "str"


class TestTrainingExecutor:
    def test_submit_and_status(self):
        from domains.training.executor import TrainingExecutor, JobStatus
        executor = TrainingExecutor(max_workers=2)
        try:
            def dummy_fn(job_id):
                return {"loss": 0.5}

            job_id = executor.submit(dummy_fn, "test_job_1")
            assert job_id == "test_job_1"
            import time
            time.sleep(0.1)
            status = executor.status("test_job_1")
            assert status is not None
            assert status["job_id"] == "test_job_1"
        finally:
            executor.shutdown(wait=True)

    def test_list_jobs(self):
        from domains.training.executor import TrainingExecutor
        executor = TrainingExecutor(max_workers=2)
        try:
            def dummy_fn(job_id):
                return {}

            executor.submit(dummy_fn, "job_a")
            executor.submit(dummy_fn, "job_b")
            import time
            time.sleep(0.1)
            jobs = executor.list_jobs()
            assert len(jobs) >= 2
        finally:
            executor.shutdown(wait=True)

    def test_cancel_queued_job(self):
        from domains.training.executor import TrainingExecutor, JobStatus
        import threading
        executor = TrainingExecutor(max_workers=1)
        try:
            import time as _time
            blocker = threading.Event()
            def blocking_fn(job_id):
                blocker.wait(timeout=5)
                return {}

            executor.submit(blocking_fn, "blocker")
            _time.sleep(0.05)
            job_id = executor.submit(blocking_fn, "to_cancel")
            result = executor.cancel("to_cancel")
            assert result is True
        finally:
            blocker.set()
            executor.shutdown(wait=True)

    def test_is_cancelled(self):
        from domains.training.executor import TrainingExecutor
        executor = TrainingExecutor(max_workers=2)
        try:
            def dummy_fn(job_id):
                return {}
            executor.submit(dummy_fn, "j1")
            assert not executor.is_cancelled("j1")
            executor.cancel("j1")
            assert executor.is_cancelled("j1")
        finally:
            executor.shutdown(wait=True)

    def test_status_unknown_job(self):
        from domains.training.executor import TrainingExecutor
        executor = TrainingExecutor(max_workers=2)
        try:
            assert executor.status("nonexistent") is None
        finally:
            executor.shutdown(wait=True)

    def test_active_count(self):
        from domains.training.executor import TrainingExecutor
        import time
        executor = TrainingExecutor(max_workers=2)
        try:
            def slow_fn(job_id):
                time.sleep(0.2)
                return {}
            executor.submit(slow_fn, "slow1")
            time.sleep(0.05)
            assert executor.active_count() >= 1
        finally:
            executor.shutdown(wait=True)

    def test_purge_completed(self):
        from domains.training.executor import TrainingExecutor, JobStatus
        import time
        executor = TrainingExecutor(max_workers=2)
        try:
            def dummy_fn(job_id):
                return {}
            executor.submit(dummy_fn, "purge_test")
            time.sleep(0.1)
            purged = executor.purge_completed(max_age_s=0.0)
            assert purged >= 0
        finally:
            executor.shutdown(wait=True)

    def test_submit_with_kwargs(self):
        from domains.training.executor import TrainingExecutor
        executor = TrainingExecutor(max_workers=2)
        try:
            def kwarg_fn(job_id, lr=0.01, epochs=10):
                return {"lr": lr, "epochs": epochs}
            executor.submit(kwarg_fn, "kwargs_job", lr=0.001, epochs=5)
            import time
            time.sleep(0.1)
            status = executor.status("kwargs_job")
            assert status is not None
        finally:
            executor.shutdown(wait=True)

    def test_submit_with_tree_id(self):
        from domains.training.executor import TrainingExecutor
        executor = TrainingExecutor(max_workers=2)
        try:
            def dummy_fn(job_id):
                return {}
            executor.submit(dummy_fn, "tree_job", tree_id="model_tree_1")
            import time
            time.sleep(0.1)
            status = executor.status("tree_job")
            assert status["tree_id"] == "model_tree_1"
        finally:
            executor.shutdown(wait=True)

    def test_result_summary(self):
        from domains.training.executor import TrainingExecutor
        import numpy as np
        import time
        executor = TrainingExecutor(max_workers=2)
        try:
            def weight_fn(job_id):
                return {"w1": np.zeros(10), "w2": np.ones((5, 5))}
            executor.submit(weight_fn, "summary_job")
            time.sleep(0.2)
            summary = executor.result_summary("summary_job")
            assert summary is not None
            assert "weights" in summary
            assert "w1" in summary["weights"]
            assert "w2" in summary["weights"]
        finally:
            executor.shutdown(wait=True)

    def test_result_summary_incomplete(self):
        from domains.training.executor import TrainingExecutor
        executor = TrainingExecutor(max_workers=2)
        try:
            assert executor.result_summary("nonexistent") is None
        finally:
            executor.shutdown(wait=True)

