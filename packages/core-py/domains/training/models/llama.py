"""
LLaMA Style Model Implementation
Modern architecture with RMSNorm, SwiGLU, RoPE (Rotary Position Embeddings)
"""

from domains.training.slonet_compat import torch
F = torch.F
nn = torch.nn
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 256, base: int = 10000):
        super().__init__()
        self.dim = dim

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0),
            self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims (DEPRECATED - use domains.models.rotate_half)."""
    import warnings
    warnings.warn(
        "rotate_half in training.models.llama is deprecated. Use domains.models.rotate_half",
        DeprecationWarning,
        stacklevel=2
    )
    from domains.models import rotate_half as _rotate_half
    return _rotate_half(x)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings (DEPRECATED - use domains.models.apply_rotary_pos_emb)."""
    import warnings
    warnings.warn(
        "apply_rotary_pos_emb in training.models.llama is deprecated. Use domains.models.apply_rotary_pos_emb",
        DeprecationWarning,
        stacklevel=2
    )
    from domains.models import apply_rotary_pos_emb as _apply_rotary_pos_emb
    return _apply_rotary_pos_emb(q, k, cos, sin)


class LLaMAAttention(nn.Module):
    """LLaMA style attention with RoPE and SDPA support."""

    def __init__(self, n_embed: int, n_head: int, dropout: float = 0.0, use_sdpa: bool = True):
        super().__init__()

        self.n_head = n_head
        self.head_dim = n_embed // n_head
        self.use_sdpa = use_sdpa and self._sdpa_available()

        self.wq = nn.Linear(n_embed, n_embed, bias=False)
        self.wk = nn.Linear(n_embed, n_embed, bias=False)
        self.wv = nn.Linear(n_embed, n_embed, bias=False)
        self.wo = nn.Linear(n_embed, n_embed, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(dropout)

        if self.use_sdpa:
            self._attention_backend = "sdpa"
        else:
            self._attention_backend = "standard"

    def _sdpa_available(self) -> bool:
        """Check if SDPA is available."""
        return hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()

        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(x, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.use_sdpa:
            y = self._forward_sdpa(q, k, v, B, T)
        else:
            y = self._forward_standard(q, k, v, B, T, mask)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)

    def _forward_sdpa(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, B: int, T: int
    ) -> torch.Tensor:
        """Forward pass using SDPA."""
        causal_mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=causal_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        return y

    def _forward_standard(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        B: int,
        T: int,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Standard attention forward pass."""
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is None:
            mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
        att = att.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        return y


class SwiGLU(nn.Module):
    """SwiGLU activation function for feed-forward network."""

    def __init__(self, n_embed: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or int(8 * n_embed / 3)
        hidden_dim = int((hidden_dim + 255) // 256 * 256)

        self.w1 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_embed, bias=False)
        self.w3 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class LLaMABlock(nn.Module):
    """LLaMA transformer block with RMSNorm and SwiGLU."""

    def __init__(self, n_embed: int, n_head: int, dropout: float = 0.0, use_sdpa: bool = True):
        super().__init__()
        self.attention = LLaMAAttention(n_embed, n_head, dropout, use_sdpa=use_sdpa)
        self.feed_forward = SwiGLU(n_embed, dropout=dropout)
        self.attention_norm = RMSNorm(n_embed)
        self.ffn_norm = RMSNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class LLaMA(nn.Module):
    """LLaMA Style Language Model with SDPA support."""

    def __init__(
        self,
        vocab_size: int,
        n_embed: int = 512,
        n_layer: int = 8,
        n_head: int = 8,
        dropout: float = 0.0,
        block_size: int = 256,
        use_sdpa: bool = True,
    ):
        super().__init__()

        self.block_size = block_size
        self.vocab_size = vocab_size

        self.tok_embeddings = nn.Embedding(vocab_size, n_embed)
        self.layers = nn.ModuleList(
            [LLaMABlock(n_embed, n_head, dropout, use_sdpa=use_sdpa) for _ in range(n_layer)]
        )
        self.norm = RMSNorm(n_embed)
        self.output = nn.Linear(n_embed, vocab_size, bias=False)

        self.tok_embeddings.weight = self.output.weight
        self.use_sdpa = use_sdpa

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()

        assert T <= self.block_size, f"Sequence {T} > block_size {self.block_size}"

        x = self.tok_embeddings(idx)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.output(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


DEFAULT_CONFIG = {
    "vocab_size": 1000,
    "n_embed": 256,
    "n_layer": 4,
    "n_head": 4,
    "dropout": 0.0,
    "block_size": 256,
}

__all__ = ["LLaMA", "LLaMABlock", "LLaMAAttention", "SwiGLU", "RMSNorm"]
