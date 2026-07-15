"""
Variational Autoencoder (VAE) for image compression and generation.

Encodes images to a compact latent space, then decodes back to pixels.
Used as the foundation for text-to-image generation (diffusion operates
in the latent space, not pixel space).
"""

from typing import Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger("slo.multimodal.vae")

from domains.training.slonet import (
    Tensor, SloNet, SloConv2D, SloLinear,
    SloAdam, relu as _relu, sigmoid as _sigmoid,
    tensor as _tensor, zeros, ones,
    _get_accelerator,
)


def _group_norm(x: Tensor, num_groups: int = 32, eps: float = 1e-5) -> Tensor:
    """Group normalization for 4D tensors (B, C, H, W)."""
    B, C, H, W = x.data.shape
    assert C % num_groups == 0, f"Channels {C} not divisible by groups {num_groups}"

    data = x.data.reshape(B, num_groups, C // num_groups, H, W)
    mean = data.mean(axis=(2, 3, 4), keepdims=True)
    var = data.var(axis=(2, 3, 4), keepdims=True)
    normalized = (data - mean) / np.sqrt(var + eps)
    normalized = normalized.reshape(B, C, H, W)

    out = Tensor(normalized, requires_grad=True, _children=(x,))
    return out


class SloVAEEncoder:
    """CNN encoder that compresses images to latent space.

    Architecture: 3x224x224 -> Conv blocks -> (latent_dim, 8, 8)
    Outputs mean and log_variance for reparameterization trick.
    """

    def __init__(self, latent_dim=64):
        self.latent_dim = latent_dim
        # Encoder: 3x224x224 -> 64x8x8 (32x compression)
        self.model = SloNet(layers=[
            # Block 1: 224 -> 112
            SloConv2D(3, 32, kernel_size=4, stride=2, padding=1),
            _relu,

            # Block 2: 112 -> 56
            SloConv2D(32, 64, kernel_size=4, stride=2, padding=1),
            _relu,

            # Block 3: 56 -> 28
            SloConv2D(64, 128, kernel_size=4, stride=2, padding=1),
            _relu,

            # Block 4: 28 -> 14
            SloConv2D(128, 256, kernel_size=4, stride=2, padding=1),
            _relu,

            # Block 5: 14 -> 7
            SloConv2D(256, 512, kernel_size=4, stride=2, padding=1),
            _relu,

            # Project to latent (mean + log_var)
            SloConv2D(512, latent_dim * 2, kernel_size=3, padding=1),
        ])
        self.optimizer = SloAdam(lr=1e-4)

    def forward(self, images_np: np.ndarray) -> Tuple[Tensor, Tensor]:
        """
        Args:
            images_np: (B, C, H, W) images
        Returns:
            mean: (B, latent_dim, 7, 7)
            log_var: (B, latent_dim, 7, 7)
        """
        x = _tensor(images_np, requires_grad=False)
        for layer in self.model.layers:
            if callable(layer):
                x = layer(x)
            else:
                x = layer.forward(x)

        # Split into mean and log_var
        out_data = x.data
        B = out_data.shape[0]
        mean_data = out_data[:, :self.latent_dim, :, :]
        log_var_data = out_data[:, self.latent_dim:, :, :]

        mean = Tensor(mean_data, requires_grad=True, _children=(x,))
        log_var = Tensor(log_var_data, requires_grad=True, _children=(x,))
        return mean, log_var

    def sample(self, mean: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization trick: sample from N(mean, exp(log_var))."""
        std = Tensor(np.exp(0.5 * log_var.data), requires_grad=True, _children=(log_var,))
        noise_data = np.random.randn(*mean.data.shape).astype(np.float32)
        noise = Tensor(noise_data, requires_grad=False)
        return mean + std * noise

    def parameters(self):
        return self.model.parameters()


class SloVAEDecoder:
    """CNN decoder that reconstructs images from latent space.

    Architecture: (latent_dim, 7, 7) -> Conv transpose -> 3x224x224
    """

    def __init__(self, latent_dim=64):
        self.latent_dim = latent_dim
        # Decoder: 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
        # Pattern: conv(keep size) -> upsample(2x) -> conv(keep size) -> upsample -> ... -> final conv(keep size)
        self.model = SloNet(layers=[
            # Block 1: 7x7 -> conv -> upsample -> 14x14
            SloConv2D(latent_dim, 256, kernel_size=3, padding=1),
            _relu,

            # Block 2: 14x14 -> conv -> upsample -> 28x28
            SloConv2D(256, 128, kernel_size=3, padding=1),
            _relu,

            # Block 3: 28x28 -> conv -> upsample -> 56x56
            SloConv2D(128, 64, kernel_size=3, padding=1),
            _relu,

            # Block 4: 56x56 -> conv -> upsample -> 112x112
            SloConv2D(64, 32, kernel_size=3, padding=1),
            _relu,

            # Block 5: 112x112 -> conv -> upsample -> 224x224
            SloConv2D(32, 16, kernel_size=3, padding=1),
            _relu,

            # Final: 224x224 -> conv -> 224x224
            SloConv2D(16, 3, kernel_size=3, padding=1),
            _sigmoid,  # Output in [0, 1]
        ])
        self.optimizer = SloAdam(lr=1e-4)

    def _upsample(self, x: Tensor) -> Tensor:
        """Nearest-neighbor 2x upsample."""
        data = x.data
        B, C, H, W = data.shape
        # Repeat each element 2x in both spatial dimensions
        upsampled = np.repeat(np.repeat(data, 2, axis=2), 2, axis=3)
        return Tensor(upsampled, requires_grad=True, _children=(x,))

    def forward(self, latents: Tensor) -> Tensor:
        """
        Args:
            latents: (B, latent_dim, 7, 7)
        Returns:
            reconstructed: (B, 3, 224, 224) in [0, 1]
        """
        x = latents
        # Pattern: conv -> upsample -> conv -> upsample -> ... -> conv
        # Total convs = 6, upsamples between conv 0-4 (5 upsamples total)
        # Layer indices: 0=conv, 1=conv, 2=conv, 3=conv, 4=conv, 5=conv, 6=conv
        # Wait, that's 7 layers with 6 convs... let me recount

        # Layers: [conv0, relu0, conv1, relu1, conv2, relu2, conv3, relu3, conv4, relu4, conv5, relu5, conv6, sigmoid]
        #conv0(7), conv1(14), conv2(28), conv3(56), conv4(112), conv5(224) -> output 224
        # Actually simpler: conv(7) -> upsample -> conv(14) -> upsample -> conv(28) -> upsample -> conv(56) -> upsample -> conv(112) -> upsample -> conv(224)
        # conv indices: 0, 2, 4, 6, 8, 10
        # upsample after conv 0,1,2,3,4 (indices 0,2,4,6,8)

        for i, layer in enumerate(self.model.layers):
            if callable(layer):
                x = layer(x)
            else:
                x = layer.forward(x)
            # Upsample after conv layers 0,1,2,3,4 (indices 0,2,4,6,8)
            if i in [0, 2, 4, 6, 8]:
                x = self._upsample(x)
        return x

    def parameters(self):
        return self.model.parameters()


class SloVAE:
    """Complete VAE: encoder + decoder with KL divergence loss.

    Trained to reconstruct images while keeping latent space well-behaved.
    """

    def __init__(self, latent_dim=64):
        self.latent_dim = latent_dim
        self.encoder = SloVAEEncoder(latent_dim)
        self.decoder = SloVAEDecoder(latent_dim)
        self.optimizer = SloAdam(lr=1e-4)

    def forward(self, images_np: np.ndarray) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            images_np: (B, C, H, W) images in [0, 1]
        Returns:
            reconstructed: (B, C, H, W)
            mean: (B, latent_dim, 7, 7)
            log_var: (B, latent_dim, 7, 7)
        """
        mean, log_var = self.encoder.forward(images_np)
        latent = self.encoder.sample(mean, log_var)
        reconstructed = self.decoder.forward(latent)
        return reconstructed, mean, log_var

    def loss(self, images_np: np.ndarray) -> Tensor:
        """VAE loss = reconstruction loss + KL divergence."""
        reconstructed, mean, log_var = self.forward(images_np)

        # Reconstruction loss (MSE)
        recon_loss = ((reconstructed.data - images_np) ** 2).mean()
        recon_tensor = Tensor(recon_loss, requires_grad=True, _children=(reconstructed,))

        # KL divergence: -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        kl_loss = -0.5 * np.mean(1 + log_var.data - mean.data ** 2 - np.exp(log_var.data))
        kl_tensor = Tensor(kl_loss, requires_grad=True, _children=(mean, log_var))

        total_loss = recon_tensor + kl_tensor * 0.001  # KL weight
        return total_loss

    def train_step(self, images_np: np.ndarray) -> float:
        """Single training step."""
        loss = self.loss(images_np)
        loss.backward()
        self.optimizer.step(self.encoder.parameters() + self.decoder.parameters())
        for p in self.encoder.parameters() + self.decoder.parameters():
            p.grad = None
        return float(loss.data)

    def encode(self, images_np: np.ndarray) -> np.ndarray:
        """Encode images to latent space (no gradient)."""
        mean, _ = self.encoder.forward(images_np)
        return mean.data

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Decode latents to images (no gradient)."""
        latent_tensor = _tensor(latents, requires_grad=False)
        out = self.decoder.forward(latent_tensor)
        return out.data

    def parameters(self):
        return self.encoder.parameters() + self.decoder.parameters()
