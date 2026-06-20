"""
VLM Inference Engine: loads a trained SigLIP + MLP connector + Qwen LoRA model
for image-conditioned text generation.

Architecture:
  Image → SigLIP encoder → MLP connector → vision embeddings
                                         ↕  concatenated
  Text  → Qwen tokenizer  → Qwen embed    →
                                  ↓
                          Qwen LM head → generated tokens

Usage:
    vlm = VLMInference("models/vlm-finetuned")
    text = vlm.generate(image, "Describe this image")
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("man.vlm_inference")


from domains.training.multimodal import MLPConnector


class VLMInference:
    """Vision-Language Model inference engine.

    Loads a trained checkpoint from ``VLMTrainer`` and provides
    image-conditioned text generation.

    Args:
        model_dir: Path to the ``VLMTrainer`` output directory containing
            ``vlm_config.json``, ``connector.pt``, and the ``final/`` subdirectory
            with LoRA adapter weights.
        device: Torch device (default: auto-detect)
        dtype: Torch dtype (default: float32)
    """

    def __init__(
        self,
        model_dir: str = "models/vlm-finetuned",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Validate checkpoint exists
        config_path = self.model_dir / "vlm_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"No VLM checkpoint found at {self.model_dir} (missing vlm_config.json). "
                "Train a VLM model first via the VLM trainer."
            )

        with open(config_path) as f:
            cfg = json.load(f)

        vision_encoder = cfg.get("vision_encoder", "google/siglip-base-patch16-224")
        llm_name = cfg.get("llm", "Qwen/Qwen2.5-0.5B-Instruct")
        final_dir = self.model_dir / "final"

        logger.info(
            "Loading VLM: vision=%s llm=%s final=%s",
            vision_encoder, llm_name, final_dir,
        )

        # ── Load vision encoder ──────────────────────────────────
        raw_vision = AutoModel.from_pretrained(
            vision_encoder, trust_remote_code=True,
        ).to(self.device).eval()
        if hasattr(raw_vision, "vision_model"):
            self.vision_model = raw_vision.vision_model
        else:
            self.vision_model = raw_vision
        vcfg = self.vision_model.config
        vision_dim = getattr(vcfg, "hidden_size", None) or getattr(vcfg, "d_model", 768)

        # ── Load LLM (Qwen) with LoRA adapters ───────────────────
        # Load base model first, then merge LoRA adapters
        self.lm = AutoModelForCausalLM.from_pretrained(
            str(final_dir) if final_dir.exists() else llm_name,
            trust_remote_code=True,
            dtype=self.dtype,
        ).to(self.device).eval()
        llm_dim = self.lm.config.hidden_size

        # ── Load connector ───────────────────────────────────────
        connector_path = self.model_dir / "connector.pt"
        self.connector = MLPConnector(vision_dim, llm_dim).to(self.device).eval()
        if connector_path.exists():
            self.connector.load_state_dict(
                torch.load(str(connector_path), map_location=self.device, weights_only=True),
            )
            logger.info("Loaded connector weights from %s", connector_path)
        else:
            logger.warning("No connector weights found at %s — using random init", connector_path)

        # ── Tokenizer ───────────────────────────────────────────
        tokenizer_path = final_dir if final_dir.exists() else llm_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Number of vision tokens (CLS + patch tokens for SigLIP 224px)
        self._n_vision = 1 + (224 // 16) ** 2  # 1 CLS + 196 patches

        logger.info(
            "VLM loaded: vision_dim=%d llm_dim=%d device=%s",
            vision_dim, llm_dim, self.device,
        )

    def _process_image(self, image) -> torch.Tensor:
        """Process a PIL image into vision embeddings.

        Args:
            image: PIL Image or numpy array (H, W, 3)

        Returns:
            Vision embeddings tensor (1, n_vision, llm_dim)
        """
        from PIL import Image
        import torchvision.transforms as T

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        pixel_values = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            vision_out = self.vision_model(pixel_values=pixel_values)
            vision_embeds = vision_out.last_hidden_state  # (1, 197, vision_dim)
            vision_embeds = self.connector(vision_embeds)  # (1, 197, llm_dim)

        return vision_embeds

    @torch.no_grad()
    def generate(
        self,
        image,
        text: str = "Describe this image in detail.",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generate text conditioned on an image.

        Args:
            image: PIL Image or numpy array
            text: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeated tokens (>1 = more penalty)

        Returns:
            Generated text string
        """
        # Process image
        vision_embeds = self._process_image(image)
        n_vision = vision_embeds.shape[1]

        # Tokenize text
        messages = [{"role": "user", "content": text}]
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        text_ids = self.tokenizer(
            chat_text, return_tensors="pt", truncation=True,
            max_length=512,
        ).input_ids.to(self.device)

        # Get text embeddings
        text_embeds = self.lm.get_input_embeddings()(text_ids)  # (1, seq_len, llm_dim)

        # Combine: prepend vision embeddings
        combined = torch.cat([vision_embeds, text_embeds], dim=1)  # (1, n_vision+seq_len, llm_dim)

        # Generate autoregressively
        generated = text_ids
        past_len = combined.shape[1]

        for _ in range(max_new_tokens):
            # Full forward pass with combined embeddings
            outputs = self.lm(
                inputs_embeds=combined,
                use_cache=False,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    logits[:, token_id] /= repetition_penalty

            # Top-p (nucleus) sampling
            if top_p < 1.0 and temperature > 0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove,
                )
                logits[indices_to_remove] = float("-inf")

            # Sample
            if temperature > 0:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            next_embed = self.lm.get_input_embeddings()(next_token)
            combined = torch.cat([combined, next_embed], dim=1)

            # Stop at EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Safety limit
            if combined.shape[1] > 1024:
                break

        # Decode, skipping input tokens
        input_len = text_ids.shape[1]
        output_ids = generated[0, input_len:].tolist()
        try:
            eos_pos = output_ids.index(self.tokenizer.eos_token_id)
            output_ids = output_ids[:eos_pos]
        except ValueError:
            pass

        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def generate_batch(
        self,
        images: list,
        texts: list[str],
        max_new_tokens: int = 64,
        temperature: float = 0.7,
    ) -> list[str]:
        """Generate for multiple image-text pairs sequentially."""
        return [
            self.generate(img, txt, max_new_tokens=max_new_tokens, temperature=temperature)
            for img, txt in zip(images, texts)
        ]
