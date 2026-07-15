"""
Text Encoder for multimodal generation.

Transforms tokenized text into embeddings for conditioning the diffusion model.
Uses a simple transformer encoder with the BPE tokenizer.
"""

from typing import List
import numpy as np
import logging

logger = logging.getLogger("slo.multimodal.text_encoder")

from domains.training.slonet import (
    Tensor, SloEmbedding, SloTransformerBlock, SloLayerNorm, SloLinear,
    SloAdam, tensor as _tensor,
)
from domains.multimodal.bpe_tokenizer import BPETokenizer


class TextEncoder:
    """Transformer-based text encoder for diffusion conditioning.

    Encodes tokenized text into a sequence of embeddings that can be
    used as cross-attention context for the diffusion UNet.
    """

    def __init__(self, vocab_size=4096, embed_dim=256, n_heads=4, n_layers=4, max_seq_len=77):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = SloEmbedding(vocab_size, embed_dim)

        # Positional embedding (learned)
        self.pos_embedding = Tensor(
            np.random.randn(1, max_seq_len, embed_dim).astype(np.float32) * 0.02,
            requires_grad=True
        )

        # Transformer blocks
        self.blocks = [
            SloTransformerBlock(embed_dim, n_heads, use_rope=True, dropout=0.1,
                              name=f"text_block_{i}")
            for i in range(n_layers)
        ]

        # Final layer norm
        self.norm = SloLayerNorm(embed_dim)

        # Projection to context_dim (for diffusion model)
        self.context_proj = SloLinear(embed_dim, embed_dim)

        self.tokenizer = BPETokenizer(vocab_size=vocab_size)
        self.optimizer = SloAdam(lr=1e-4)

    def encode_tokens(self, token_ids: np.ndarray) -> Tensor:
        """
        Encode token IDs to embeddings.

        Args:
            token_ids: (B, seq_len) token IDs
        Returns:
            embeddings: (B, seq_len, embed_dim)
        """
        B, seq_len = token_ids.shape

        # Token embeddings
        tok_emb = self.token_embedding.forward(_tensor(token_ids, requires_grad=False))

        # Add positional embeddings
        pos_emb = self.pos_embedding.data[:, :seq_len, :]
        x_data = tok_emb.data + pos_emb
        x = Tensor(x_data, requires_grad=True, _children=(tok_emb, self.pos_embedding))

        # Transformer blocks
        for block in self.blocks:
            x, _ = block.forward(x)

        # Final norm and projection
        x = self.norm.forward(x)
        return self.context_proj.forward(x)

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode raw text strings to embeddings.

        Auto-trains the tokenizer on first use with a default vocabulary
        so manual train_tokenizer() is never needed.

        Args:
            texts: List of text strings
        Returns:
            embeddings: (B, seq_len, embed_dim) numpy array
        """
        if not self.tokenizer._built:
            default_texts = [
                "the cat sat on the mat", "a quick brown fox jumps over the lazy dog",
                "hello world how are you today", "this is a test sentence for training",
                "the sky is blue and the grass is green", "i love to learn new things",
                "what is the meaning of life", "the sun rises in the east",
                "one two three four five six seven eight nine ten",
                "machine learning is fun and interesting", "please generate an image of",
                "a beautiful landscape with mountains and trees",
                "portrait of a person with blue eyes", "still life with fruit and flowers",
                "abstract art with geometric shapes and bright colors",
            ]
            self.tokenizer.train(default_texts)

        # Tokenize and pad
        token_lists = [self.tokenizer.encode(t) for t in texts]
        max_len = min(max(len(t) for t in token_lists), self.max_seq_len)

        token_ids = np.zeros((len(texts), max_len), dtype=np.int32)
        for i, tokens in enumerate(token_lists):
            token_ids[i, :min(len(tokens), max_len)] = tokens[:max_len]

        # Encode
        embeddings = self.encode_tokens(token_ids)
        return embeddings.data

    def train_tokenizer(self, texts: List[str]):
        """Train the BPE tokenizer on text corpus."""
        self.tokenizer.train(texts)

    def parameters(self):
        params = self.token_embedding.parameters()
        params.append(self.pos_embedding)
        for block in self.blocks:
            params += block.parameters()
        params += self.norm.parameters()
        params += self.context_proj.parameters()
        return params
