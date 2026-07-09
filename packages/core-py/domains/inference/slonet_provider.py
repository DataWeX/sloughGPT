"""
SloNet Chat Provider — pure NumPy inference via SloTransformer.

Loads a HuggingFace model's weights into SloTransformer and runs inference
entirely through NumPy ops. No PyTorch dependency at inference time.
"""
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncIterator
import numpy as np

logger = logging.getLogger("man.inference.slonet_provider")

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

# GPT-2 weight key mapping
GPT2_KEYS = {
    "wte.weight": "tok_emb.weight",
    "wpe.weight": "pos_emb.weight",
    "h.{i}.ln_1.weight": "blocks.{i}.attn_norm.weight",
    "h.{i}.ln_1.bias": "blocks.{i}.attn_norm.bias",
    "h.{i}.attn.c_attn.weight": "blocks.{i}.attn.cat.weight",
    "h.{i}.attn.c_attn.bias": "blocks.{i}.attn.cat.bias",
    "h.{i}.attn.c_proj.weight": "blocks.{i}.attn.o_proj.weight",
    "h.{i}.attn.c_proj.bias": "blocks.{i}.attn.o_proj.bias",
    "h.{i}.ln_2.weight": "blocks.{i}.ff_norm.weight",
    "h.{i}.ln_2.bias": "blocks.{i}.ff_norm.bias",
    "h.{i}.mlp.c_fc.weight": "blocks.{i}.ff.w1.weight",
    "h.{i}.mlp.c_fc.bias": "blocks.{i}.ff.w1.bias",
    "h.{i}.mlp.c_proj.weight": "blocks.{i}.ff.w2.weight",
    "h.{i}.mlp.c_proj.bias": "blocks.{i}.ff.w2.bias",
    "ln_f.weight": "norm.weight",
    "ln_f.bias": "norm.bias",
}


def convert_hf_to_slonet(hf_state_dict: Dict[str, np.ndarray], n_layer: int) -> Dict[str, np.ndarray]:
    """Map HuggingFace state dict keys to SloTransformer keys.

    Auto-detects model type (Qwen2.5 vs GPT-2) based on weight key patterns.
    """
    # Auto-detect which key mapping to use
    sample_keys = list(hf_state_dict.keys())[:5]
    if any(k.startswith("model.layers.") for k in sample_keys):
        key_map = QWEN_KEYS
    elif any(k.startswith("h.") for k in sample_keys):
        key_map = GPT2_KEYS
    else:
        key_map = QWEN_KEYS  # default

    result = {}
    for hf_key, arr in hf_state_dict.items():
        mapped = False
        for pattern, target in key_map.items():
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

    .. deprecated::
        Use ``NumpyEngine`` instead. This engine will be removed in a future version.

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

        # Support both Qwen2.5 and GPT-2 config key conventions
        n_layer = config.get("num_hidden_layers") or config.get("n_layer")
        n_embed = config.get("hidden_size") or config.get("n_embd")
        n_head = config.get("num_attention_heads") or config.get("n_head")
        n_kv_head = config.get("num_key_value_heads", n_head)
        vocab_size = config["vocab_size"]
        intermediate_size = config.get("intermediate_size", n_embed * 4)
        rope_base = config.get("rope_theta", 10000.0)
        rms_norm_eps = config.get("rms_norm_eps") or config.get("layer_norm_epsilon", 1e-6)
        # GPT-2 uses gelu, not rope — detect and disable rope for non-RoPE models
        use_rope = config.get("rope_theta") is not None or config.get("position_embedding_type", "") == "rope"

        logger.info("Building SloTransformer from %s config (n_embed=%d, n_layer=%d, n_head=%d, n_kv_head=%d, vocab=%d)",
                     hf_model_id, n_embed, n_layer, n_head, n_kv_head, vocab_size)

        self._model = SloTransformer(
            vocab_size=vocab_size,
            n_embed=n_embed,
            n_layer=n_layer,
            n_head=n_head,
            n_kv_head=n_kv_head,
            block_size=config.get("n_ctx") or config.get("max_position_embeddings", 2048),
            max_seq_len=config.get("n_ctx") or config.get("max_position_embeddings", 2048),
            dropout=0.0,
            eps=rms_norm_eps,
            use_rope=use_rope,
            rope_base=float(rope_base),
            tie_weights=True,
            intermediate_size=intermediate_size,
            soul_name=hf_model_id,
        )

        self._load_weights(snapshot, n_layer)

        # Try MorphTokenizer (pure numpy) first, fall back to transformers
        try:
            from domains.infrastructure.morph_tokenizer import MorphTokenizer
            self._tokenizer = MorphTokenizer.from_pretrained(hf_model_id)
            logger.info("Using MorphTokenizer for %s", hf_model_id)
        except Exception:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
                logger.info("Using transformers AutoTokenizer for %s", hf_model_id)
            except Exception as e:
                logger.warning("No tokenizer available for %s: %s", hf_model_id, e)
                self._tokenizer = None
        if self._tokenizer is not None and hasattr(self._tokenizer, "pad_token_id") and self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = getattr(self._tokenizer, "eos_token_id", None) or 0

        logger.info("SloNetChatProvider ready: %s (NumPy weights loaded)", hf_model_id)

    def _load_weights(self, snapshot: Path, n_layer: int):
        """Load HF weights from safetensors (numpy) into SloTransformer.

        Prefers safetensors with numpy backend — no PyTorch required.
        Falls back to PyTorch ``torch.load`` only for legacy ``.bin`` files.
        """
        weight_files = sorted(snapshot.glob("*.safetensors"))
        if not weight_files:
            weight_files = sorted(snapshot.glob("model*.safetensors"))
        if not weight_files:
            weight_files = sorted(snapshot.glob("pytorch_model*.bin"))

        logger.info("Loading weights from %d file(s) in %s", len(weight_files), snapshot)

        hf_dict = {}
        for f in weight_files:
            if f.suffix == ".safetensors":
                try:
                    from safetensors import safe_open
                    with safe_open(str(f), framework="np") as sf:
                        for k in sf.keys():
                            arr = sf.get_tensor(k)
                            # Convert bfloat16/float16 to float32
                            if arr.dtype.name in ("bfloat16", "float16"):
                                if arr.dtype.name == "bfloat16":
                                    raw = arr.view(np.uint16).astype(np.uint32) << 16
                                    arr = raw.view(np.float32)
                                else:
                                    arr = arr.astype(np.float32)
                            hf_dict[k] = arr
                except Exception as e:
                    # If numpy framework not supported, fall back to pt
                    try:
                        import torch
                        from safetensors import safe_open
                        with safe_open(str(f), framework="pt", device="cpu") as sf:
                            for k in sf.keys():
                                t = sf.get_tensor(k)
                                hf_dict[k] = t.to(dtype=torch.float32)
                    except Exception as e2:
                        raise RuntimeError(f"Failed to load safetensors {f}: {e2}")
            else:
                try:
                    from domains.training.slonet_compat import torch
                    w = torch.load(f, map_location="cpu", weights_only=True, mmap=True)
                    for k, v in w.items():
                        hf_dict[k] = v.to(dtype=torch.float32)
                except Exception as e:
                    raise RuntimeError(f"Failed to load bin file {f}: {e}")

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

        # Handle both HF tokenizer (callable) and MorphTokenizer (encode method)
        if callable(self._tokenizer) and not hasattr(self._tokenizer, 'encode'):
            input_ids_arr = self._tokenizer(prompt, return_tensors="np")["input_ids"]
            eos_id = self._tokenizer.eos_token_id or 0
            decode_fn = self._tokenizer.decode
        else:
            ids = self._tokenizer.encode(prompt)
            input_ids_arr = np.array([ids], dtype=np.int64)
            eos_id = getattr(self._tokenizer, 'eos_token_id', None) or 0
            decode_fn = self._tokenizer.decode

        tokens = self._model.generate(
            input_ids_arr,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=kwargs.get("top_k", 40),
            top_p=kwargs.get("top_p", 0.9),
            eos_token=eos_id,
        )

        new_ids = tokens.data[0, input_ids_arr.shape[1]:]
        id_list = new_ids.tolist()
        try:
            decoded = decode_fn(id_list, skip_special_tokens=True)
        except TypeError:
            decoded = decode_fn(id_list)
        yield decoded

    def embed(self, text: str) -> List[float]:
        return []
