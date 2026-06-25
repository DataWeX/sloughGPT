"""VLMInference — load and run trained VLM models for image-text generation."""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("man.vlm_inference")


class VisionConnector(nn.Module):
    """Linear projection from vision encoder hidden dim to LLM hidden dim."""

    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.proj = nn.Linear(vision_dim, llm_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class VLMInference:
    """Loaded VLM model ready for image-conditioned text generation.

    Loads the vision encoder, trained connector, and LoRA-tuned LLM
    from a VLM training output directory.
    """

    def __init__(self, model_dir: str, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)

        config_path = self.model_dir / "vlm_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"VLM config not found: {config_path}")

        with open(config_path) as f:
            self.config = json.load(f)

        self._load_models()

    def _load_models(self):
        cfg = self.config
        logger.info("Loading vision encoder: %s", cfg["vision_encoder"])
        self.vision_encoder = AutoModel.from_pretrained(
            cfg["vision_encoder"],
            trust_remote_code=True,
        ).to(self.device).eval()

        logger.info("Loading LLM: %s", cfg["llm"])
        self.llm = AutoModelForCausalLM.from_pretrained(
            cfg["llm"],
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["llm"],
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Build connector
        vision_dim = self.vision_encoder.config.hidden_size
        llm_dim = self.llm.config.hidden_size
        self.connector = VisionConnector(vision_dim, llm_dim)

        connector_path = self.model_dir / "connector.pt"
        if connector_path.exists():
            self.connector.load_state_dict(torch.load(connector_path, map_location=self.device, weights_only=True))
            logger.info("Loaded connector weights from %s", connector_path)

        self.connector.to(self.device).eval()

        # Load LoRA adapter
        lora_path = self.model_dir / "lora"
        if lora_path.exists():
            from peft import PeftModel
            self.llm = PeftModel.from_pretrained(self.llm, str(lora_path))
            logger.info("Loaded LoRA adapter from %s", lora_path)

        self.llm.eval()

    @torch.no_grad()
    def generate(
        self,
        image_base64: str,
        prompt: str = "Describe this image in detail.",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> dict:
        """Generate text conditioned on an image.

        Args:
            image_base64: Base64-encoded JPEG/PNG image.
            prompt: Text prompt describing what to generate.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            dict with ``text``, ``tokens_generated``, ``elapsed_ms``.
        """
        import base64
        import io
        from PIL import Image
        from transformers import AutoImageProcessor

        t0 = time.time()

        # Decode image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Get vision embedding
        processor = AutoImageProcessor.from_pretrained(self.config["vision_encoder"])
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.vision_encoder(**inputs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            vision_emb = outputs.pooler_output
        else:
            vision_emb = outputs.last_hidden_state[:, 0, :]

        projected = self.connector(vision_emb.to(torch.float32))

        # Tokenize prompt
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_embeds = self.llm.get_input_embeddings()(prompt_ids["input_ids"])
        vision_token = projected.unsqueeze(1)
        combined = torch.cat([vision_token, input_embeds], dim=1)

        attention_mask = torch.cat([
            torch.ones((1, 1), device=self.device),
            prompt_ids["attention_mask"],
        ], dim=1)

        # Generate
        generated = self.llm.generate(
            inputs_embeds=combined,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode (skip the vision token)
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        elapsed_ms = int((time.time() - t0) * 1000)

        return {
            "text": text,
            "tokens_generated": len(generated[0]),
            "elapsed_ms": elapsed_ms,
        }
