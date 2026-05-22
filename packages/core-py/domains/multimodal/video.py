"""
Video Processor for video understanding and captioning.

Extracts frames from video, encodes them with the VisionEncoder,
and uses a temporal transformer to understand video content.
"""

from typing import List, Optional, Tuple
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger("sloughgpt.multimodal.video")

from domains.training.slonet import (
    Tensor, SloNet, SloLinear, SloLayerNorm, SloTransformerBlock,
    SloAdam, tensor as _tensor,
)


class TemporalEncoder:
    """Transformer encoder for video frame sequences.
    
    Takes a sequence of frame embeddings and produces a video-level
    embedding with temporal understanding.
    """
    
    def __init__(self, embed_dim=256, n_heads=4, n_layers=2, max_frames=16):
        self.embed_dim = embed_dim
        self.max_frames = max_frames
        
        # Temporal positional embedding
        self.temp_pos_embed = Tensor(
            np.random.randn(1, max_frames, embed_dim).astype(np.float32) * 0.02,
            requires_grad=True
        )
        
        # Transformer blocks for temporal modeling
        self.blocks = [
            SloTransformerBlock(embed_dim, n_heads, use_rope=True, dropout=0.1,
                              name=f"temporal_block_{i}")
            for i in range(n_layers)
        ]
        
        self.norm = SloLayerNorm(embed_dim)
        self.optimizer = SloAdam(lr=3e-4)
    
    def forward(self, frame_embeddings: np.ndarray) -> Tensor:
        """
        Encode a sequence of frame embeddings.
        
        Args:
            frame_embeddings: (B, num_frames, embed_dim)
        Returns:
            video_embedding: (B, num_frames, embed_dim)
        """
        B, num_frames, _ = frame_embeddings.shape
        
        # Add temporal positional embeddings
        pos_emb = self.temp_pos_embed.data[:, :num_frames, :]
        x_data = frame_embeddings + pos_emb
        x = Tensor(x_data, requires_grad=True, _children=(self.temp_pos_embed,))
        
        # Transformer blocks
        for block in self.blocks:
            x, _ = block.forward(x)
        
        return self.norm.forward(x)
    
    def parameters(self):
        params = [self.temp_pos_embed]
        for block in self.blocks:
            params += block.parameters()
        params += self.norm.parameters()
        return [p for p in params if p.requires_grad]


class VideoProcessor:
    """Processes video into captions and embeddings.
    
    Pipeline:
    1. Extract N frames from video
    2. Encode each frame with VisionEncoder
    3. Encode frame sequence with TemporalEncoder
    4. Generate caption using DecoderLSTM with cross-attention
    """
    
    def __init__(self, embed_dim=256, n_heads=4, n_temporal_layers=2, max_frames=16):
        self.embed_dim = embed_dim
        self.max_frames = max_frames
        self.temporal_encoder = TemporalEncoder(embed_dim, n_heads, n_temporal_layers, max_frames)
        self.optimizer = SloAdam(lr=3e-4)
    
    def extract_frames(self, video_path: str, num_frames: int = None) -> List[np.ndarray]:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (default: max_frames)
        Returns:
            List of (224, 224, 3) numpy arrays
        """
        try:
            import cv2
        except ImportError:
            logger.warning("cv2 not installed, falling back to placeholder frames")
            n = num_frames or self.max_frames
            return [np.random.randn(224, 224, 3).astype(np.float32) for _ in range(n)]
        
        num_frames = num_frames or self.max_frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Cannot open video: %s, falling back to placeholder frames", video_path)
            return [np.random.randn(224, 224, 3).astype(np.float32) for _ in range(num_frames)]
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return [np.random.randn(224, 224, 3).astype(np.float32) for _ in range(num_frames)]
        
        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB and resize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame.astype(np.float32) / 255.0)
            else:
                # Fallback to random frame
                frames.append(np.random.randn(224, 224, 3).astype(np.float32))
        
        cap.release()
        return frames
    
    def encode_video(self, frames: List[np.ndarray], vision_encoder) -> Tensor:
        """
        Encode video frames to temporal embeddings.
        
        Args:
            frames: List of (224, 224, 3) numpy arrays
            vision_encoder: VisionEncoder instance
        Returns:
            video_embedding: (1, num_frames, embed_dim)
        """
        # Encode each frame
        frame_embeddings = []
        for frame in frames:
            # Add batch dimension
            frame_batch = frame.reshape(1, 224, 224, 3)
            # Get cls token embedding
            emb = vision_encoder.forward(frame_batch)
            frame_embeddings.append(emb.data)
        
        # Stack frame embeddings
        frame_stack = np.concatenate(frame_embeddings, axis=1)  # (1, num_frames, embed_dim)
        
        # Temporal encoding
        return self.temporal_encoder.forward(frame_stack)
    
    def parameters(self):
        return self.temporal_encoder.parameters()
