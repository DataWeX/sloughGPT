"""Tests for multimodal.video — TemporalEncoder and VideoProcessor."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from domains.multimodal.video import TemporalEncoder, VideoProcessor


# ── TemporalEncoder ───────────────────────────────────────────────────────


class TestTemporalEncoder:

    def test_init_defaults(self):
        enc = TemporalEncoder()
        assert enc.embed_dim == 256
        assert enc.max_frames == 16

    def test_init_custom(self):
        enc = TemporalEncoder(embed_dim=128, n_heads=2, n_layers=1, max_frames=8)
        assert enc.embed_dim == 128
        assert enc.max_frames == 8

    def test_forward_shape(self):
        enc = TemporalEncoder(embed_dim=64, max_frames=4)
        x = np.random.randn(1, 4, 64).astype(np.float32)
        result = enc.forward(x)
        assert result.data.shape == (1, 4, 64)

    def test_forward_different_seq_length(self):
        enc = TemporalEncoder(embed_dim=64, max_frames=8)
        x = np.random.randn(1, 3, 64).astype(np.float32)
        result = enc.forward(x)
        assert result.data.shape == (1, 3, 64)

    def test_parameters_count(self):
        enc = TemporalEncoder(embed_dim=32, n_layers=1, max_frames=4)
        params = enc.parameters()
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


# ── VideoProcessor ────────────────────────────────────────────────────────


class TestVideoProcessor:

    def test_init_defaults(self):
        vp = VideoProcessor()
        assert vp.embed_dim == 256
        assert vp.max_frames == 16

    def test_init_custom(self):
        vp = VideoProcessor(embed_dim=128, max_frames=8)
        assert vp.embed_dim == 128
        assert vp.max_frames == 8

    def test_encode_video(self):
        vp = VideoProcessor(embed_dim=64, max_frames=4)
        frames = [np.random.randn(224, 224, 3).astype(np.float32) for _ in range(4)]

        mock_encoder = MagicMock()
        mock_result = MagicMock()
        mock_result.data = np.random.randn(1, 1, 64).astype(np.float32)
        mock_encoder.forward.return_value = mock_result

        result = vp.encode_video(frames, mock_encoder)
        assert result.data.shape[2] == 64
        assert mock_encoder.forward.call_count == 4

    def test_parameters(self):
        vp = VideoProcessor(embed_dim=32, max_frames=4)
        params = vp.parameters()
        assert len(params) > 0


# ── extract_frames (mocked cv2) ───────────────────────────────────────────


class TestExtractFrames:

    def test_import_error(self):
        vp = VideoProcessor(max_frames=4)
        with patch.dict("sys.modules", {"cv2": None}):
            with pytest.raises(ImportError, match="opencv"):
                vp.extract_frames("video.mp4")

    def test_file_not_found(self):
        vp = VideoProcessor(max_frames=4)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch.dict("sys.modules", {"cv2": MagicMock()}):
            import cv2
            cv2.VideoCapture.return_value = mock_cap
            with pytest.raises(RuntimeError, match="Cannot open"):
                vp.extract_frames("nonexistent.mp4")

    def test_no_frames(self):
        vp = VideoProcessor(max_frames=4)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0

        with patch.dict("sys.modules", {"cv2": MagicMock()}):
            import cv2
            cv2.VideoCapture.return_value = mock_cap
            with pytest.raises(RuntimeError, match="no frames"):
                vp.extract_frames("empty.mp4")
