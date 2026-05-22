"""
Unified Multimodal Model — understands images, video, audio, and text jointly.

Architecture:
  Each modality produces feature tokens in a shared embedding space.
  Tokens are concatenated with modality markers and processed by
  shared transformer blocks. A text decoder generates output from
  the unified representation.

  Image  → VisionEncoder  → img_patches (196 tokens)
  Video  → VisionEncoder per-frame → TemporalEncoder → vid_tokens (N tokens)
  Audio  → mel spectrogram → AudioProjector → aud_tokens (M tokens)
  Text   → BPETokenizer → SloEmbedding → text_tokens

  All tokens concatenated → SharedTransformer → DecoderLSTM → text
"""

from typing import List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger("sloughgpt.multimodal.unified")

from domains.training.slonet import (
    Tensor, SloLinear, SloLayerNorm, SloTransformerBlock,
    SloCrossAttention, SloEmbedding, SloLSTM,
    SloAdam, softmax as _softmax, tensor as _tensor,
    cross_entropy as _cross_entropy, sigmoid as _sigmoid,
)
from .engine import VisionEncoder, DecoderLSTM, ReplayBuffer, augment_image, contrastive_loss
from .video import VideoProcessor, TemporalEncoder
from .bpe_tokenizer import BPETokenizer


class AudioProjector:
    """Projects mel spectrogram features into shared embedding space.

    Takes (B, n_mels, n_frames) mel spectrograms and produces
    (B, n_frames, shared_dim) audio tokens.
    """

    def __init__(self, n_mels=80, max_frames=128, shared_dim=256):
        self.n_mels = n_mels
        self.max_frames = max_frames
        self.shared_dim = shared_dim
        self.proj = SloLinear(n_mels, shared_dim)
        self.pos_embed = Tensor(
            np.random.randn(1, max_frames, shared_dim).astype(np.float32) * 0.02,
            requires_grad=True,
        )
        self.norm = SloLayerNorm(shared_dim)
        self.optimizer = SloAdam(lr=3e-4)

    def forward(self, mel_spec: np.ndarray) -> Tensor:
        """Project mel spectrogram to shared embedding.

        Args:
            mel_spec: (B, n_mels, n_frames) or (B, n_frames, n_mels)
        Returns:
            (B, n_frames, shared_dim) audio tokens
        """
        if mel_spec.ndim == 2:
            mel_spec = mel_spec[np.newaxis, :, :]
        if mel_spec.shape[1] == self.n_mels and mel_spec.ndim == 3:
            mel_spec = mel_spec.transpose(0, 2, 1)
        B, n_frames, _ = mel_spec.shape
        x = self.proj.forward(_tensor(mel_spec, requires_grad=False))
        pos = self.pos_embed.data[:, :n_frames, :]
        x_data = x.data + pos
        x = Tensor(x_data, requires_grad=True, _children=(x, self.pos_embed))
        x = self.norm.forward(x)
        return x

    def parameters(self):
        return self.proj.parameters() + [self.pos_embed] + self.norm.parameters()


class VideoProjector:
    """Projects video frame embeddings into shared space.

    Takes (B, n_frames, embed_dim) from TemporalEncoder and
    applies a linear projection + temporal position.
    """

    def __init__(self, embed_dim=256, shared_dim=256, max_frames=16):
        self.embed_dim = embed_dim
        self.shared_dim = shared_dim
        self.max_frames = max_frames
        self.proj = SloLinear(embed_dim, shared_dim) if embed_dim != shared_dim else None
        self.optimizer = SloAdam(lr=3e-4)

    def forward(self, video_embed: Tensor) -> Tensor:
        """Project video embedding to shared space.

        Args:
            video_embed: (B, n_frames, embed_dim) from TemporalEncoder
        Returns:
            (B, n_frames, shared_dim)
        """
        if self.proj is not None:
            return self.proj.forward(video_embed)
        return video_embed

    def parameters(self):
        if self.proj is not None:
            return self.proj.parameters()
        return []


class ModalityProcessor:
    """Processes tokens from a single modality into the shared space.

    Each modality gets:
    - A start token (<IMG>, <VID>, <AUD>, <TXT>)
    - Feature tokens projected to shared_dim
    - Optional positional embeddings
    """

    def __init__(self, shared_dim=256):
        self.shared_dim = shared_dim

    def make_modality_tokens(
        self,
        feature_tokens: np.ndarray,
        start_token: np.ndarray,
    ) -> np.ndarray:
        """Prepend start token to feature tokens.

        Args:
            feature_tokens: (B, N, shared_dim) feature sequence
            start_token: (1, 1, shared_dim) learned start token
        Returns:
            (B, N+1, shared_dim) with start token prepended
        """
        B = feature_tokens.shape[0]
        start = start_token.reshape(1, 1, self.shared_dim).repeat(B, axis=0)
        return np.concatenate([start, feature_tokens], axis=1)


class UnifiedMultimodalModel:
    """Unified model that understands images, video, audio, and text.

    All modalities produce tokens in a shared embedding space.
    They are concatenated into a single sequence and processed
    by shared transformer blocks followed by a text decoder.

    Training is multi-task: images, video clips, and audio segments
    all produce text outputs via the same decoder.
    """

    MODALITY_IMG = "image"
    MODALITY_VID = "video"
    MODALITY_AUD = "audio"
    MODALITY_TXT = "text"

    def __init__(
        self,
        shared_dim=256,
        hidden_dim=512,
        n_heads=8,
        n_vit_layers=4,
        n_shared_layers=2,
        n_mels=80,
        vocab_size=4096,
    ):
        self.shared_dim = shared_dim
        self.hidden_dim = hidden_dim

        # Modality start tokens (learned)
        self.img_token = Tensor(np.random.randn(1, 1, shared_dim).astype(np.float32) * 0.02, requires_grad=True)
        self.vid_token = Tensor(np.random.randn(1, 1, shared_dim).astype(np.float32) * 0.02, requires_grad=True)
        self.aud_token = Tensor(np.random.randn(1, 1, shared_dim).astype(np.float32) * 0.02, requires_grad=True)
        self.txt_token = Tensor(np.random.randn(1, 1, shared_dim).astype(np.float32) * 0.02, requires_grad=True)

        # Encoders
        self.vision = VisionEncoder(shared_dim, n_heads, n_vit_layers)
        self.temporal_encoder = TemporalEncoder(shared_dim, n_heads // 2, n_layers=2, max_frames=16)

        # Projectors
        self.audio_projector = AudioProjector(n_mels=n_mels, max_frames=128, shared_dim=shared_dim)
        self.video_projector = VideoProjector(embed_dim=shared_dim, shared_dim=shared_dim)

        # Shared reasoning transformer
        self.shared_blocks = [
            SloTransformerBlock(shared_dim, n_heads, use_rope=True, dropout=0.1, name=f"shared_block_{i}")
            for i in range(n_shared_layers)
        ]
        self.shared_norm = SloLayerNorm(shared_dim)

        # Text decoder — generates text from unified representation
        self.decoder = DecoderLSTM(
            vocab_size=vocab_size,
            embed_dim=shared_dim,
            hidden_dim=hidden_dim,
            n_heads=max(1, n_heads // 2),
        )

        # Text tokenizer
        self.text = self.decoder  # reuse decoder's embedding/vocab
        self.bpe = BPETokenizer(vocab_size=vocab_size)
        self.vocab: List[str] = []
        self.stoi: dict = {}
        self.itos: dict = {}

        self._trained = False
        self._learning_count = 0

    # ── Token Management ──

    def build_vocab(self, texts: List[str]):
        """Build BPE vocabulary from training texts."""
        self.bpe.train(texts)
        vocab_set = set()
        for t in texts:
            for w in t.lower().split():
                vocab_set.add(w)
        vocab_set.update(["<BOS>", "<EOS>", "<PAD>", "<IMG>", "<VID>", "<AUD>"])
        self.vocab = sorted(vocab_set)
        self.stoi = {w: i for i, w in enumerate(self.vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}
        # Rebuild decoder with new vocab size
        self.decoder = DecoderLSTM(
            vocab_size=len(self.vocab),
            embed_dim=self.shared_dim,
            hidden_dim=self.hidden_dim,
            n_heads=max(1, 8 // 2),
        )
        self.text = self.decoder

    def encode_text(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.bpe._built:
            return self.bpe.encode(text)
        tokens = [self.stoi.get("<BOS>", 0)]
        for w in text.lower().split():
            tokens.append(self.stoi.get(w, self.stoi.get("<PAD>", 0)))
        tokens.append(self.stoi.get("<EOS>", 0))
        return tokens

    def decode_text(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self.bpe._built and hasattr(self.bpe, 'decode'):
            return self.bpe.decode(token_ids)
        words = []
        for tid in token_ids:
            w = self.itos.get(tid, "")
            if w in ("<BOS>", "<EOS>", "<PAD>", "<IMG>", "<VID>", "<AUD>", ""):
                continue
            words.append(w)
        return " ".join(words)

    # ── Modality Encoding ──

    def encode_image(self, images_np: np.ndarray) -> Tuple[Tensor, Tensor]:
        """Encode image to (cls_token, patch_tokens).

        Returns:
            cls_embed: (B, 1, shared_dim)
            patch_tokens: (B, num_patches+1, shared_dim)
        """
        cls_embed = self.vision.forward(images_np)
        patch_tokens = self.vision.get_patch_embeddings(images_np)
        return cls_embed, patch_tokens

    def encode_video(self, frames: List[np.ndarray]) -> Tensor:
        """Encode video frames to temporal tokens.

        Args:
            frames: list of (224, 224, 3) numpy arrays
        Returns:
            (1, n_frames, shared_dim) video tokens
        """
        frame_embeddings = []
        for frame in frames:
            frame_batch = frame.reshape(1, 224, 224, 3)
            emb = self.vision.forward(frame_batch)
            frame_embeddings.append(emb.data)
        frame_stack = np.concatenate(frame_embeddings, axis=1)
        vid_embed = self.temporal_encoder.forward(frame_stack)
        return self.video_projector.forward(vid_embed)

    def encode_audio(self, mel_spec: np.ndarray) -> Tensor:
        """Encode mel spectrogram to audio tokens.

        Args:
            mel_spec: (n_mels, n_frames) or (B, n_mels, n_frames)
        Returns:
            (B, n_frames, shared_dim) audio tokens
        """
        return self.audio_projector.forward(mel_spec)

    def make_unified_input(
        self,
        image_tokens: Optional[Tensor] = None,
        video_tokens: Optional[Tensor] = None,
        audio_tokens: Optional[Tensor] = None,
        text_tokens: Optional[np.ndarray] = None,
    ) -> Tensor:
        """Concatenate all modality tokens into a single sequence.

        Each modality is preceded by its start token.
        The sequence is: [IMG] img_tokens [VID] vid_tokens [AUD] aud_tokens [TXT] text_tokens

        Returns:
            (B, total_seq_len, shared_dim) unified token sequence
        """
        token_list = []
        children = []

        if image_tokens is not None:
            B = image_tokens.data.shape[0]
            img_start = self.img_token.data.repeat(B, axis=0)
            img_seq = np.concatenate([img_start, image_tokens.data], axis=1)
            token_list.append(img_seq)
            children += [self.img_token]

        if video_tokens is not None:
            B = video_tokens.data.shape[0]
            vid_start = self.vid_token.data.repeat(B, axis=0)
            vid_seq = np.concatenate([vid_start, video_tokens.data], axis=1)
            token_list.append(vid_seq)
            children += [self.vid_token]

        if audio_tokens is not None:
            B = audio_tokens.data.shape[0]
            aud_start = self.aud_token.data.repeat(B, axis=0)
            aud_seq = np.concatenate([aud_start, audio_tokens.data], axis=1)
            token_list.append(aud_seq)
            children += [self.aud_token]

        if text_tokens is not None:
            B = text_tokens.shape[0]
            txt_start = self.txt_token.data.repeat(B, axis=0)
            txt_feats = self.decoder.embedding.forward(_tensor(text_tokens, requires_grad=False))
            txt_seq = np.concatenate([txt_start, txt_feats.data], axis=1)
            token_list.append(txt_seq)
            children += [self.txt_token]

        if not token_list:
            raise ValueError("At least one modality must be provided")

        unified_data = np.concatenate(token_list, axis=1)
        unified = Tensor(unified_data, requires_grad=True, _children=tuple(children))
        return unified

    # ── Forward / Training ──

    def forward_unified(self, unified_input: Tensor) -> Tensor:
        """Process unified token sequence through shared transformer blocks.

        Args:
            unified_input: (B, seq_len, shared_dim)
        Returns:
            (B, seq_len, shared_dim) processed sequence
        """
        x = unified_input
        for block in self.shared_blocks:
            x, _ = block.forward(x)
        x = self.shared_norm.forward(x)
        return x

    def train_on_image_caption(
        self,
        images_np: np.ndarray,
        text_tokens: np.ndarray,
    ) -> float:
        """Train the unified model on an image-caption pair.

        Flow: image → patches → unified → shared transformer → decoder → cross-entropy
        """
        _, patch_tokens = self.encode_image(images_np)
        unified_input = self.make_unified_input(image_tokens=patch_tokens)
        unified_out = self.forward_unified(unified_input)

        # Use the last unified token as decoder context
        last_token = Tensor(unified_out.data[:, -1:, :], requires_grad=True, _children=(unified_out,))
        decoder_h = self.decoder.proj_h.forward(last_token)
        decoder_c = self.decoder.proj_c.forward(last_token)
        if decoder_h.data.ndim == 3:
            decoder_h = Tensor(decoder_h.data.reshape(decoder_h.data.shape[0], -1), requires_grad=True, _children=(decoder_h,))
        if decoder_c.data.ndim == 3:
            decoder_c = Tensor(decoder_c.data.reshape(decoder_c.data.shape[0], -1), requires_grad=True, _children=(decoder_c,))

        logits, _ = self._decode_from_state(decoder_h, decoder_c, text_tokens[:, :-1], unified_out)
        targets = _tensor(text_tokens[:, 1:].reshape(-1), requires_grad=False)
        loss = _cross_entropy(logits, targets)
        loss.backward()

        self.decoder.optimizer.step(self.decoder.parameters())
        self.decoder.optimizer.step(self.vision.parameters())
        self.decoder.optimizer.step(self.audio_projector.parameters())
        for block in self.shared_blocks:
            self.decoder.optimizer.step(block.parameters())
        self.decoder.optimizer.step(self.shared_norm.parameters())

        for p in self._all_params():
            p.grad = None
        self._trained = True
        return float(loss.data)

    def _decode_from_state(
        self,
        h: Tensor,
        c: Tensor,
        token_ids: np.ndarray,
        context: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Decode text tokens given initial LSTM state and cross-attention context.

        Args:
            h: (B, hidden_dim) initial hidden state
            c: (B, hidden_dim) initial cell state
            token_ids: (B, seq_len) input token IDs
            context: (B, ctx_len, shared_dim) unified context for cross-attention
        Returns:
            logits: (seq_len, vocab_size)
            h: final hidden state
        """
        all_logits = []
        for t in range(token_ids.shape[1]):
            tok_t = int(np.clip(token_ids[0, t], 0, max(1, self.decoder.vocab_size - 1)))
            idx = _tensor(np.array([[tok_t]]), requires_grad=False)
            emb_t = self.decoder.embedding.forward(idx)
            if emb_t.data.ndim == 3:
                emb_t = Tensor(emb_t.data.reshape(emb_t.data.shape[0], -1), requires_grad=False)

            gates_ih = self.decoder.W_ih.forward(emb_t)
            gates_hh = self.decoder.W_hh.forward(h)
            gates_data = gates_ih.data + gates_hh.data

            hd = self.decoder.hidden_dim
            gi = _sigmoid(_tensor(gates_data[:, :hd], requires_grad=False))
            gf = _sigmoid(_tensor(gates_data[:, hd:2*hd], requires_grad=False))
            gg = _tensor(np.tanh(gates_data[:, 2*hd:3*hd]), requires_grad=False)
            go = _sigmoid(_tensor(gates_data[:, 3*hd:], requires_grad=False))

            c_new = gf * c + gi * gg
            h = go * Tensor(np.tanh(c_new.data), requires_grad=False)
            c = c_new

            # Cross-attention to unified context
            img_ctx = self.decoder.img_proj.forward(context)
            h_3d = Tensor(h.data.reshape(h.data.shape[0], 1, -1), requires_grad=True, _children=(h,))
            h_3d = self.decoder.cross_attn.forward(h_3d, img_ctx)
            h = Tensor(h_3d.data.reshape(h_3d.data.shape[0], -1), requires_grad=True, _children=(h_3d,))

            h_for_fc = h if h.data.ndim == 2 else Tensor(h.data.reshape(h.data.shape[0], -1), requires_grad=True, _children=(h,))
            all_logits.append(self.decoder.fc_out.forward(h_for_fc))

        logits_data = np.concatenate([l.data for l in all_logits], axis=0)
        logits = _tensor(logits_data, requires_grad=True)
        return logits, h

    def generate(
        self,
        images_np: Optional[np.ndarray] = None,
        frames: Optional[List[np.ndarray]] = None,
        mel_spec: Optional[np.ndarray] = None,
        max_len: int = 20,
        temperature: float = 1.0,
    ) -> str:
        """Generate text from any input modality.

        Args:
            images_np: (1, 224, 224, 3) image
            frames: list of (224, 224, 3) video frames
            mel_spec: (n_mels, n_frames) mel spectrogram
            max_len: maximum tokens to generate
            temperature: sampling temperature
        Returns:
            generated text string
        """
        modality_tokens = None
        if images_np is not None:
            _, patch_tokens = self.encode_image(images_np)
            modality_tokens = patch_tokens
            unified_input = self.make_unified_input(image_tokens=modality_tokens)
        elif frames is not None:
            vid_tokens = self.encode_video(frames)
            modality_tokens = vid_tokens
            unified_input = self.make_unified_input(video_tokens=modality_tokens)
        elif mel_spec is not None:
            aud_tokens = self.encode_audio(mel_spec)
            modality_tokens = aud_tokens
            unified_input = self.make_unified_input(audio_tokens=modality_tokens)
        else:
            return "no input provided"

        unified_out = self.forward_unified(unified_input)
        last_token = Tensor(unified_out.data[:, -1:, :], requires_grad=True, _children=(unified_out,))
        h = self.decoder.proj_h.forward(last_token)
        c = self.decoder.proj_c.forward(last_token)
        if h.data.ndim == 3:
            h = Tensor(h.data.reshape(h.data.shape[0], -1), requires_grad=True, _children=(h,))
        if c.data.ndim == 3:
            c = Tensor(c.data.reshape(c.data.shape[0], -1), requires_grad=True, _children=(c,))

        bos = self.stoi.get("<BOS>", 0)
        eos = self.stoi.get("<EOS>", 0)
        token_ids = [bos]

        for _ in range(max_len):
            inp = _tensor(np.array([token_ids]), requires_grad=False)
            logits, (h, c) = self._decode_from_state(h, c, inp, unified_out)
            if temperature > 0 and self._trained:
                probs = _softmax(logits[-1:] / temperature)
                probs_np = probs.data.flatten()
                probs_np = np.maximum(probs_np, 1e-8)
                for t in token_ids[1:]:
                    if 0 <= t < len(probs_np):
                        probs_np[t] *= 0.4
                probs_np /= probs_np.sum()
                next_tok = int(np.random.choice(len(probs_np), p=probs_np))
            else:
                scores = logits.data[-1].copy()
                for t in token_ids[1:]:
                    if 0 <= t < len(scores):
                        scores[t] -= 5.0
                next_tok = int(np.argmax(scores))
            if next_tok == eos:
                break
            token_ids.append(next_tok)

        return self.decode_text(token_ids)

    def _all_params(self):
        """Return all trainable parameters across all sub-modules."""
        params = []
        params += self.vision.parameters()
        params += self.temporal_encoder.parameters()
        params += self.audio_projector.parameters()
        params += self.video_projector.parameters()
        for block in self.shared_blocks:
            params += block.parameters()
        params += self.shared_norm.parameters()
        params += self.decoder.parameters()
        params += [self.img_token, self.vid_token, self.aud_token, self.txt_token]
        return [p for p in params if p.requires_grad]

    def save(self, path: str = "") -> str:
        """Save model weights to .npz file."""
        import os
        if not path:
            os.makedirs("data/multimodal", exist_ok=True)
            path = "data/multimodal/unified_engine.npz"
        weights = {}
        for i, p in enumerate(self._all_params()):
            weights[f"param_{i}"] = p.data
        np.savez_compressed(path, **weights)
        meta = {
            "shared_dim": self.shared_dim,
            "hidden_dim": self.hidden_dim,
            "trained": self._trained,
            "vocab": self.vocab,
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
        }
        import json
        with open(path + ".json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Unified model saved to %s", path)
        return path

    @classmethod
    def load(cls, path: str = "") -> "UnifiedMultimodalModel":
        """Load model weights from .npz file."""
        if not path:
            path = "data/multimodal/unified_engine.npz"
        import json
        with open(path + ".json") as f:
            meta = json.load(f)
        model = cls(
            shared_dim=meta["shared_dim"],
            hidden_dim=meta["hidden_dim"],
        )
        model.vocab = list(meta["vocab"])
        model.stoi = meta["stoi"]
        model.itos = {int(k): v for k, v in meta["itos"].items()}
        model._trained = meta.get("trained", False)
        data = np.load(path)
        params = model._all_params()
        for i, p in enumerate(params):
            key = f"param_{i}"
            if key in data:
                p.data = data[key]
        logger.info("Unified model loaded from %s", path)
        return model
