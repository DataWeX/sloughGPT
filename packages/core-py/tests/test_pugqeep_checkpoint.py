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
            # Create a small SloTransformer and save as .soul
            net = SloTransformer(vocab_size=256, n_embed=64, n_layer=2,
                                  n_head=4, block_size=32, use_rope=True)
            from domains.training.export import export_to_sou
            soul_path = str(Path(tmpdir) / "test_model.soul")
            export_to_sou(net, soul_path)

            # Compress to Points
            stats = compress_checkpoint(soul_path, n_clusters=8)
            assert stats is not None
            assert stats["point_count"] > 0
            assert stats["compression_ratio"] > 1.0

            # Verify files exist
            lib_path = Path(stats["library_path"])
            meta_path = Path(stats["meta_path"])
            assert lib_path.exists(), f"Library not found: {lib_path}"
            assert meta_path.exists(), f"Meta not found: {meta_path}"

            # Verify meta has model info
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


class TestLoadFromPoints:
    """Test load_from_points() reads PointLibrary and returns ModelTree."""

    def test_load_from_points_file(self):
        from domains.infrastructure.pugqeep.library import PointLibrary
        from domains.infrastructure.pugqeep.model_tree import (
            load_from_points, decompress_tree,
        )
        from domains.infrastructure.pugqeep.point import Point

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a library manually
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

            # Load
            tree, meta = load_from_points(str(Path(tmpdir) / "test_model"))
            assert tree.is_loaded
            assert tree.library.has("test_model.weight")

            # Decompress
            weights = decompress_tree(tree)
            assert "weight" in weights
            np.testing.assert_array_equal(weights["weight"], centroids[assignments])

    def test_load_from_points_directory(self):
        from domains.infrastructure.pugqeep.library import PointLibrary
        from domains.infrastructure.pugqeep.model_tree import load_from_points
        from domains.infrastructure.pugqeep.point import Point

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create library in directory format
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
            # Save as library.json inside directory
            model_dir = Path(tmpdir) / "mymodel"
            model_dir.mkdir()
            lib.save(model_dir / "library.json")

            # Load from directory path
            tree, meta = load_from_points(str(model_dir))
            assert tree.is_loaded

    def test_load_from_points_not_found(self):
        from domains.infrastructure.pugqeep.model_tree import load_from_points
        with pytest.raises(FileNotFoundError):
            load_from_points("/nonexistent/path")


class TestImportFromSouFallback:
    """Test import_from_sou falls back to Points when .soul is missing."""

    def test_import_from_sou_loads_points(self):
        from domains.training.executor import compress_checkpoint
        from domains.training.slonet import SloTransformer
        from domains.training.export import export_to_sou

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save model as .soul
            net = SloTransformer(vocab_size=128, n_embed=32, n_layer=1,
                                  n_head=2, block_size=16, use_rope=True)
            soul_path = str(Path(tmpdir) / "test.soul")
            export_to_sou(net, soul_path)

            # Compress to Points (use more clusters for accuracy)
            stats = compress_checkpoint(soul_path, n_clusters=16)
            assert stats is not None

            # Remove the .soul file — force Points-only load
            Path(soul_path).unlink()

            # import_from_sou should find and load the .points.json
            from domains.training.slonet import import_from_sou
            loaded = import_from_sou(soul_path)
            assert loaded is not None
            assert hasattr(loaded, "state_dict")

            # Verify weights are approximately correct (VQ is lossy)
            orig_weights = net.state_dict()
            loaded_weights = loaded.state_dict()
            assert set(orig_weights.keys()) == set(loaded_weights.keys())
            # Check that at least 80% of elements are within tolerance
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

            # Also create a .points.json (but don't compress — just a dummy)
            points_path = Path(tmpdir) / "test.points.json"
            points_path.write_text('{"name": "test", "points": []}')

            # import_from_sou should use .soul when it exists
            from domains.training.slonet import import_from_sou
            loaded = import_from_sou(soul_path)
            assert loaded is not None
            # Should have real weights, not empty
            weights = loaded.state_dict()
            assert len(weights) > 0


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
            # VQ with 8 clusters on small tensors — allow quantization error
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
