"""
SloNet Chat Provider — pure NumPy inference via SloTransformer.

Loads a HuggingFace model's weights into SloTransformer and runs inference
entirely through NumPy ops. No PyTorch dependency at inference time.
"""
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncIterator
import numpy as np

logger = logging.getLogger(__name__)

QWEN_KEYS = {
    "model.embed_tokens.weight": "tok_emb.weight",
    "model.layers.{i}.input_layernorm.weight": "blocks.{i}.attn_norm.weight",
    "model.layers.{i}.self_attn.q_proj.weight": "blocks.{i}.attn.q_proj.weight",
    "model.layers.{i}.self_attn.q_proj.bias": "blocks.{i}.attn.q_proj.bias",
    "model.layers.{i}.self_attn.k_proj.weight": "blocks.{i}.attn.k_proj.weight",
    "model.layers.{i}.self_attn.k_proj.bias": "blocks.{i}.attn.k_proj.bias",
    "model.layers.{i}.self_attn.v_proj.weight": "blocks.{i}.attn.v_proj.weight",
    "model.layers.{i}.self_attn.v_proj.bias": "blocks.{i}.attn.v_proj.bias",
    "model.layers.{i}.self_attn.o_proj.weight": "blocks.{i}.attn.o_proj.weight",
    "model.layers.{i}.post_attention_layernorm.weight": "blocks.{i}.ff_norm.weight",
    "model.layers.{i}.mlp.gate_proj.weight": "blocks.{i}.ff.w1.weight",
    "model.layers.{i}.mlp.down_proj.weight": "blocks.{i}.ff.w2.weight",
    "model.layers.{i}.mlp.up_proj.weight": "blocks.{i}.ff.w3.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "lm_head.weight",
}


def convert_hf_to_slonet(hf_state_dict: Dict[str, np.ndarray], n_layer: int) -> Dict[str, np.ndarray]:
    """Map HuggingFace Qwen2.5 state dict keys to SloTransformer keys."""
    result = {}
    for hf_key, arr in hf_state_dict.items():
        mapped = False
        for pattern, target in QWEN_KEYS.items():
            if "{i}" in pattern:
                for i in range(n_layer):
                    concrete = pattern.replace("{i}", str(i))
                    if hf_key == concrete:
                        result[target.replace("{i}", str(i))] = arr
                        mapped = True
                        break
            else:
                if hf_key == pattern:
                    result[target] = arr
                    mapped = True
                    break
        if not mapped and "weight" in hf_key:
            pass
    return result


def _silu_np(x: np.ndarray) -> np.ndarray:
    return x * (1 / (1 + np.exp(-x)))


class SloNetChatProvider:
    """ModelProvider backed by SloTransformer (pure NumPy, no PyTorch).

    Args:
        hf_model_id: HuggingFace model ID (e.g. 'Qwen/Qwen2.5-0.5B-Instruct')
        device: ignored (always CPU / NumPy)
    """

    def __init__(self, hf_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "cpu"):
        from domains.training.slonet import SloTransformer
        import json

        self._model_id = hf_model_id
        self._tokenizer = None
        self._model = None

        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{hf_model_id.replace('/', '--')}"
        snapshots = list(cache_dir.glob("snapshots/*")) if cache_dir.exists() else []
        if not snapshots:
            raise RuntimeError(f"Model {hf_model_id} not found in HF cache. Run the server with PyTorch once to download it.")

        snapshot = snapshots[0]
        config = json.loads((snapshot / "config.json").read_text())

        n_layer = config["num_hidden_layers"]
        n_embed = config["hidden_size"]
        n_head = config["num_attention_heads"]
        n_kv_head = config.get("num_key_value_heads", n_head)
        vocab_size = config["vocab_size"]
        intermediate_size = config["intermediate_size"]
        rope_base = config.get("rope_theta", 10000.0)
        rms_norm_eps = config.get("rms_norm_eps", 1e-6)

        logger.info("Building SloTransformer from %s config (n_embed=%d, n_layer=%d, n_head=%d, n_kv_head=%d, vocab=%d)",
                     hf_model_id, n_embed, n_layer, n_head, n_kv_head, vocab_size)

        self._model = SloTransformer(
            vocab_size=vocab_size,
            n_embed=n_embed,
            n_layer=n_layer,
            n_head=n_head,
            n_kv_head=n_kv_head,
            block_size=2048,
            max_seq_len=2048,
            dropout=0.0,
            eps=rms_norm_eps,
            use_rope=True,
            rope_base=float(rope_base),
            tie_weights=True,
            intermediate_size=intermediate_size,
            soul_name=hf_model_id,
        )

        self._load_weights(snapshot, n_layer)

        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id or 0

        logger.info("SloNetChatProvider ready: %s (NumPy weights loaded)", hf_model_id)

    def _load_weights(self, snapshot: Path, n_layer: int):
        """Load PyTorch weights from HF snapshot and feed into SloTransformer."""
        import torch
        weight_files = sorted(snapshot.glob("*.safetensors"))
        if not weight_files:
            weight_files = sorted(snapshot.glob("pytorch_model*.bin"))
        if not weight_files:
            weight_files = sorted(snapshot.glob("model*.safetensors"))

        logger.info("Loading weights from %d file(s) in %s", len(weight_files), snapshot)

        hf_dict = {}
        for f in weight_files:
            try:
                w = torch.load(f, map_location="cpu", weights_only=True, mmap=True)
                for k, v in w.items():
                    hf_dict[k] = v.to(dtype=torch.float32)
            except Exception:
                try:
                    from safetensors import safe_open
                    with safe_open(str(f), framework="pt", device="cpu") as sf:
                        for k in sf.keys():
                            t = sf.get_tensor(k)
                            hf_dict[k] = t.to(dtype=torch.float32)
                except Exception as e2:
                    raise RuntimeError(f"Failed to load {f}: {e2}")

        mapped = convert_hf_to_slonet(hf_dict, n_layer)
        logger.info("Mapped %d / %d HF weights to SloTransformer keys", len(mapped), len(hf_dict))

        missing = self._model.load_state_dict(mapped, strict=False)
        if missing:
            logger.warning("SloTransformer.load_state_dict had %d unmatched keys", len(missing))

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def capabilities(self):
        from domains.models.provider import ModelCapabilities
        return ModelCapabilities(chat=True, streaming=True, embedding=False, vision=False)

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "model_id": self._model_id,
            "type": "slonet",
            "n_embed": self._model.n_embed if self._model else 0,
            "n_layer": self._model.n_layer if self._model else 0,
            "n_head": self._model.n_head if self._model else 0,
            "vocab_size": self._model.vocab_size if self._model else 0,
        }

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Apply Qwen's chat template."""
        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    async def chat(self, messages: List[Dict], max_tokens: int = 512, temperature: float = 0.8, **kwargs) -> str:
        chunks = []
        async for chunk in self.chat_stream(messages, max_tokens, temperature, **kwargs):
            chunks.append(chunk)
        return "".join(chunks)

    async def chat_stream(self, messages: List[Dict], max_tokens: int = 512, temperature: float = 0.8, **kwargs) -> AsyncIterator[str]:
        prompt = self._messages_to_prompt(messages)
        input_ids = self._tokenizer(prompt, return_tensors="np")["input_ids"]
        eos_id = self._tokenizer.eos_token_id or 0

        tokens = self._model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=kwargs.get("top_k", 40),
            top_p=kwargs.get("top_p", 0.9),
            eos_token=eos_id,
        )

        new_ids = tokens.data[0, input_ids.shape[1]:]
        decoded = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        yield decoded

    def embed(self, text: str) -> List[float]:
        return []
