"""
GPT-2 Style Model Implementation
Standard GPT-2 architecture with LayerNorm, GELU, learned position embeddings
With SDPA/Flash Attention support
"""

from domains.training.slonet_compat import torch
F = torch.F
nn = torch.nn
import math
from typing import Optional, Tuple


class GPT2Attention(nn.Module):
    """GPT-2 style multi-head attention with causal mask and SDPA support."""

    def __init__(self, n_embed: int, n_head: int, dropout: float = 0.1, use_sdpa: bool = True):
        super().__init__()
        assert n_embed % n_head == 0

        self.n_embed = n_embed
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        self.use_sdpa = use_sdpa and self._sdpa_available()

        self.c_attn = nn.Linear(n_embed, 3 * n_embed)
        self.c_proj = nn.Linear(n_embed, n_embed)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer("bias", torch.tril(torch.ones(256, 256)).view(1, 1, 256, 256))

        if self.use_sdpa:
            self._attention_backend = "sdpa"
        else:
            self._attention_backend = "standard"

    def _sdpa_available(self) -> bool:
        """Check if SDPA is available."""
        return hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_sdpa:
            y = self._forward_sdpa(q, k, v, B, T)
        else:
            y = self._forward_standard(q, k, v, B, T)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

    def _forward_sdpa(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, B: int, T: int
    ) -> torch.Tensor:
        """Forward pass using SDPA (Scaled Dot Product Attention)."""
        # Create causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)

        # Use SDPA with causal mask
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=causal_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        return y

    def _forward_standard(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, B: int, T: int
    ) -> torch.Tensor:
        """Standard attention forward pass."""
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        return y


class GPT2MLP(nn.Module):
    """GPT-2 style MLP with GELU activation."""

    def __init__(self, n_embed: int, dropout: float = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(n_embed, 4 * n_embed)
        self.c_proj = nn.Linear(4 * n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    """GPT-2 transformer block with pre-norm."""

    def __init__(self, n_embed: int, n_head: int, dropout: float = 0.1, use_sdpa: bool = True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed)
        self.attn = GPT2Attention(n_embed, n_head, dropout, use_sdpa=use_sdpa)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.mlp = GPT2MLP(n_embed, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    """GPT-2 Style Language Model with SDPA support."""

    def __init__(
        self,
        vocab_size: int,
        n_embed: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        dropout: float = 0.1,
        block_size: int = 256,
        use_sdpa: bool = True,
    ):
        super().__init__()

        self.block_size = block_size
        self.vocab_size = vocab_size

        self.wte = nn.Embedding(vocab_size, n_embed)
        self.wpe = nn.Embedding(block_size, n_embed)

        self.blocks = nn.ModuleList(
            [GPT2Block(n_embed, n_head, dropout, use_sdpa=use_sdpa) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        self.wte.weight = self.lm_head.weight
        self.use_sdpa = use_sdpa

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        device = idx.device

        assert T <= self.block_size, f"Sequence {T} > block_size {self.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

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
    "dropout": 0.1,
    "block_size": 256,
}

__all__ = ["GPT2", "GPT2Block", "GPT2Attention", "GPT2MLP"]
