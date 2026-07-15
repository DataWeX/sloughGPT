"""
Latent Diffusion Model for text-to-image generation.

Operates in the VAE latent space (64x7x7) rather than pixel space.
Text conditioning via cross-attention with text embeddings.
"""

from typing import Optional, Tuple
import numpy as np
import logging
import math

logger = logging.getLogger("slo.multimodal.diffusion")

from domains.training.slonet import (
    Tensor, SloNet, SloConv2D, SloLinear, SloCrossAttention,
    SloAdam, relu as _relu, silu as _silu,
    tensor as _tensor, zeros, ones,
    sigmoid as _sigmoid,
)


def _group_norm(x: Tensor, num_groups: int = 32, eps: float = 1e-5) -> Tensor:
    """Group normalization for 4D tensors (B, C, H, W)."""
    B, C, H, W = x.data.shape
    if C < num_groups:
        num_groups = max(1, C // 2)
    if C % num_groups != 0:
        num_groups = 1

    data = x.data.reshape(B, num_groups, C // num_groups, H, W)
    mean = data.mean(axis=(2, 3, 4), keepdims=True)
    var = data.var(axis=(2, 3, 4), keepdims=True)
    normalized = (data - mean) / np.sqrt(var + eps)
    normalized = normalized.reshape(B, C, H, W)

    out = Tensor(normalized, requires_grad=True, _children=(x,))
    return out


def _timestep_embedding(timesteps: np.ndarray, dim: int, max_period: int = 10000) -> np.ndarray:
    """Create sinusoidal timestep embeddings (like transformer positional encoding)."""
    half = dim // 2
    freqs = np.exp(-math.log(max_period) * np.arange(0, half, dtype=np.float32) / half)
    args = timesteps[:, None] * freqs[None, :]
    embedding = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
    if dim % 2:
        embedding = np.concatenate([embedding, np.zeros((embedding.shape[0], 1))], axis=-1)
    return embedding


class TimestepEmbedder:
    """Embeds diffusion timesteps into a vector for conditioning."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.proj1 = SloLinear(dim, dim * 4)
        self.proj2 = SloLinear(dim * 4, dim)

    def forward(self, timesteps: np.ndarray) -> Tensor:
        emb_data = _timestep_embedding(timesteps, self.dim)
        emb = _tensor(emb_data, requires_grad=False)
        h = _silu(self.proj1.forward(emb))
        return self.proj2.forward(h)

    def parameters(self):
        return self.proj1.parameters() + self.proj2.parameters()


class ResBlock:
    """Residual block with timestep conditioning."""

    def __init__(self, in_channels: int, out_channels: int, temb_dim: int = 256):
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = SloConv2D(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = SloConv2D(out_channels, out_channels, kernel_size=3, padding=1)

        # Timestep conditioning
        self.temb_proj = SloLinear(temb_dim, out_channels)

        # Skip connection
        if in_channels != out_channels:
            self.skip = SloConv2D(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = None

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        h = _group_norm(x)
        h = _silu(h)
        h = self.conv1.forward(h)

        # Add timestep conditioning
        # temb is (B, out_channels), need to reshape to (B, out_channels, 1, 1)
        temb_data = self.temb_proj.forward(temb).data
        if temb_data.ndim == 2:
            temb_data = temb_data[:, :, None, None]
        temb_tensor = Tensor(temb_data, requires_grad=True, _children=(temb,))
        h = h + temb_tensor

        h = _group_norm(h)
        h = _silu(h)
        h = self.conv2.forward(h)

        # Skip connection
        if self.skip:
            x = self.skip.forward(x)
        return x + h

    def parameters(self):
        params = self.conv1.parameters() + self.conv2.parameters()
        params += self.temb_proj.parameters()
        if self.skip:
            params += self.skip.parameters()
        return params


class UNetBlock:
    """Single UNet block with ResBlocks and optional cross-attention."""

    def __init__(self, in_channels: int, out_channels: int, temb_dim: int = 256,
                 context_dim: int = 256, n_heads: int = 4, has_cross_attn: bool = False):
        self.res1 = ResBlock(in_channels, out_channels, temb_dim)
        self.res2 = ResBlock(out_channels, out_channels, temb_dim)
        self.has_cross_attn = has_cross_attn

        if has_cross_attn:
            # Cross-attention for text conditioning
            self.cross_attn = SloCrossAttention(out_channels, n_heads)

    def forward(self, x: Tensor, temb: Tensor, context: Optional[Tensor] = None) -> Tensor:
        x = self.res1.forward(x, temb)
        x = self.res2.forward(x, temb)

        if self.has_cross_attn and context is not None:
            # x is (B, C, H, W), need (B, H*W, C) for cross-attention
            B, C_out, H, W = x.data.shape
            x_flat = Tensor(x.data.reshape(B, C_out, H * W).transpose(0, 2, 1),
                           requires_grad=True, _children=(x,))
            # Project context to match channel dim if needed
            ctx = context
            if context.data.shape[-1] != C_out:
                proj = SloLinear(context.data.shape[-1], C_out)
                ctx = proj.forward(context)
            x_flat = self.cross_attn.forward(x_flat, ctx)
            x = Tensor(x_flat.data.transpose(0, 2, 1).reshape(B, C_out, H, W),
                      requires_grad=True, _children=(x_flat,))

        return x

    def parameters(self):
        params = self.res1.parameters() + self.res2.parameters()
        if self.has_cross_attn:
            params += self.cross_attn.parameters()
        return params


class LatentUNet:
    """UNet for latent diffusion with text conditioning.

    Architecture:
    - Encoder: 4 downsampling blocks
    - Middle: 2 ResBlocks
    - Decoder: 4 upsampling blocks with skip connections
    - Cross-attention at lowest resolution for text conditioning
    """

    def __init__(self, in_channels=64, model_channels=128, out_channels=64,
                 temb_dim=256, context_dim=256, n_heads=4):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels

        # Timestep embedding
        self.timestep_embedder = TimestepEmbedder(temb_dim)

        # Initial convolution
        self.init_conv = SloConv2D(in_channels, model_channels, kernel_size=3, padding=1)

        # Encoder (downsampling)
        ch_mult = [1, 2, 4, 8]
        self.down_blocks = []
        in_ch = model_channels
        for i, mult in enumerate(ch_mult):
            out_ch = model_channels * mult
            has_cross_attn = (i == len(ch_mult) - 1)  # Cross-attn at lowest resolution
            block = UNetBlock(in_ch, out_ch, temb_dim, context_dim, n_heads, has_cross_attn)
            self.down_blocks.append(block)
            in_ch = out_ch

        # Middle blocks
        self.middle_block1 = ResBlock(in_ch, in_ch, temb_dim)
        self.middle_block2 = ResBlock(in_ch, in_ch, temb_dim)

        # Decoder (upsampling) with skip connections
        # Track encoder skip channel counts for correct concatenation
        skip_channels = [model_channels * mult for mult in ch_mult]
        self.up_blocks = []
        in_ch = model_channels * ch_mult[-1]  # middle block output channels
        for i, mult in enumerate(reversed(ch_mult)):
            out_ch = model_channels * mult
            skip_ch = skip_channels.pop(-1)
            block = UNetBlock(in_ch + skip_ch, out_ch, temb_dim, context_dim, n_heads,
                            has_cross_attn=(i == 0))
            self.up_blocks.append(block)
            in_ch = out_ch

        # Output
        self.out_conv = SloConv2D(in_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor, timesteps: np.ndarray, context: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: (B, in_channels, H, W) noisy latents
            timesteps: (B,) diffusion timesteps
            context: (B, seq_len, context_dim) text embeddings
        Returns:
            noise_pred: (B, out_channels, H, W) predicted noise
        """
        B = x.data.shape[0]

        # Timestep embedding
        temb = self.timestep_embedder.forward(timesteps)

        # Initial conv
        h = self.init_conv.forward(x)

        # Encoder
        skip_connections = []
        for block in self.down_blocks:
            h = block.forward(h, temb, context)
            skip_connections.append(h)

        # Middle
        h = self.middle_block1.forward(h, temb)
        h = self.middle_block2.forward(h, temb)

        # Decoder with skip connections
        for i, block in enumerate(self.up_blocks):
            skip = skip_connections.pop(-1)
            # Concatenate skip connection
            h_data = np.concatenate([h.data, skip.data], axis=1)
            h = Tensor(h_data, requires_grad=True, _children=(h, skip))
            h = block.forward(h, temb, context)

        # Output
        h = _group_norm(h)
        h = _silu(h)
        return self.out_conv.forward(h)

    def parameters(self):
        params = self.timestep_embedder.parameters()
        params += self.init_conv.parameters()
        for block in self.down_blocks:
            params += block.parameters()
        params += self.middle_block1.parameters() + self.middle_block2.parameters()
        for block in self.up_blocks:
            params += block.parameters()
        params += self.out_conv.parameters()
        return params


class LatentDiffusionModel:
    """Complete latent diffusion model for text-to-image generation.

    Uses a VAE to compress images to latent space, then trains a UNet
    to predict noise in the latent space conditioned on text embeddings.
    """

    def __init__(self, latent_dim=64, model_channels=128, temb_dim=256,
                 context_dim=256, n_heads=4, num_timesteps=1000):
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps
        self.unet = LatentUNet(latent_dim, model_channels, latent_dim,
                              temb_dim, context_dim, n_heads)
        self.optimizer = SloAdam(lr=1e-4)

        # Noise schedule (linear)
        self.betas = np.linspace(1e-4, 0.02, num_timesteps, dtype=np.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)

    def _get_sqrt_alpha_bar(self, t: np.ndarray) -> np.ndarray:
        """Get sqrt(alpha_cumprod) for timestep t."""
        return np.sqrt(self.alphas_cumprod[t])[:, None, None, None]

    def _get_sqrt_one_minus_alpha_bar(self, t: np.ndarray) -> np.ndarray:
        """Get sqrt(1 - alpha_cumprod) for timestep t."""
        return np.sqrt(1 - self.alphas_cumprod[t])[:, None, None, None]

    def add_noise(self, latents: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Add noise to latents at timestep t."""
        noise = np.random.randn(*latents.shape).astype(np.float32)
        sqrt_alpha_bar = self._get_sqrt_alpha_bar(t)
        sqrt_one_minus = self._get_sqrt_one_minus_alpha_bar(t)
        noisy = sqrt_alpha_bar * latents + sqrt_one_minus * noise
        return noisy, noise

    def train_step(self, latents: np.ndarray, text_embeddings: np.ndarray) -> float:
        """
        Single training step.

        Args:
            latents: (B, latent_dim, 7, 7) from VAE encoder
            text_embeddings: (B, seq_len, context_dim) from text encoder
        Returns:
            loss: MSE between predicted and actual noise
        """
        B = latents.shape[0]
        t = np.random.randint(0, self.num_timesteps, size=B)

        noisy, noise = self.add_noise(latents, t)

        noisy_tensor = _tensor(noisy, requires_grad=False)
        t_tensor = t
        context_tensor = _tensor(text_embeddings, requires_grad=False)

        noise_pred = self.unet.forward(noisy_tensor, t_tensor, context_tensor)

        # MSE loss
        loss_data = ((noise_pred.data - noise) ** 2).mean()
        loss = Tensor(loss_data, requires_grad=True, _children=(noise_pred,))

        loss.backward()
        self.optimizer.step(self.unet.parameters())
        for p in self.unet.parameters():
            p.grad = None

        return float(loss.data)

    @np.errstate(over='ignore')
    def sample(self, text_embeddings: np.ndarray, num_steps=50, guidance_scale=7.5) -> np.ndarray:
        """
        Generate image latents from text embeddings using DDIM sampling.

        Args:
            text_embeddings: (1, seq_len, context_dim)
            num_steps: number of denoising steps
            guidance_scale: classifier-free guidance scale
        Returns:
            latents: (1, latent_dim, 7, 7) generated latents
        """
        B = 1
        latent_shape = (B, self.latent_dim, 7, 7)

        # Start from random noise
        x = np.random.randn(*latent_shape).astype(np.float32)

        # Sampling timesteps
        timesteps = np.linspace(self.num_timesteps - 1, 0, num_steps, dtype=np.int32)

        for i, t in enumerate(timesteps):
            t_array = np.array([t])

            # Predict noise
            x_tensor = _tensor(x, requires_grad=False)
            context_tensor = _tensor(text_embeddings, requires_grad=False)
            noise_pred = self.unet.forward(x_tensor, t_array, context_tensor)

            # Classifier-free guidance (if we had unconditional model)
            # For now, just use conditional prediction
            noise_pred_np = noise_pred.data

            # DDIM update
            alpha_bar_t = self.alphas_cumprod[t]
            if t > 0:
                alpha_bar_prev = self.alphas_cumprod[t - 1]
            else:
                alpha_bar_prev = 1.0

            sigma = 0.0  # DDIM deterministic

            # Predict x_0
            pred_x0 = (x - np.sqrt(1 - alpha_bar_t) * noise_pred_np) / np.sqrt(alpha_bar_t)
            pred_x0 = np.clip(pred_x0, -1.0, 1.0)

            # Direction pointing to x_t
            dir_xt = np.sqrt(1 - alpha_bar_prev - sigma ** 2) * noise_pred_np

            # Random noise (none for DDIM)
            noise = sigma * np.random.randn(*latent_shape).astype(np.float32)

            x = np.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + noise

        return x

    def parameters(self):
        return self.unet.parameters()
