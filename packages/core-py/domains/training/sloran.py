"""
SloRAN — Slough Recurrent Attention Network

A pure-NumPy language model architecture replacing transformer self-attention
with a gated recurrent mixer + rotating associative memory bank.

No softmax attention. No CUDA. No PyTorch.
"""

from __future__ import annotations

import math
import numpy as np
from typing import List, Dict, Optional, Tuple

from .slonet import (
    Tensor, no_grad, randn, zeros,
    SloLayer, SloNet, SloEmbedding, SloDropout,
    SloLinear, SloLayerNorm, SloRMSNorm,
    _add, _mul, _matmul, _ensure,
    cross_entropy, silu,
)

# =========================================================================
# Helpers (Tensor-aware, autograd-compatible)
# =========================================================================

def _sigmoid_t(x: Tensor) -> Tensor:
    """Sigmoid with autograd support (wraps slonet's sigmoid)."""
    from .slonet import sigmoid as _s
    return _s(x)


def _ones_like_t(t: Tensor) -> Tensor:
    return Tensor(np.ones_like(t.data), requires_grad=False)


def _sub_one_minus(a: Tensor) -> Tensor:
    """Compute 1 - a element-wise as a Tensor operation."""
    return Tensor(np.ones_like(a.data), requires_grad=False) - a


# =========================================================================
# Gated Recurrent Mixer
# =========================================================================

class GatedRecurrentMixer(SloLayer):
    """Input-dependent gated recurrence — the core sequence mixer.

    Replaces softmax attention with a selective recurrence:

        gate = sigmoid(x_t @ W_g + h @ U_g + b_g)
        candidate = silu(x_t @ W_c + h @ U_c + b_c)
        h = gate * h + (1 - gate) * candidate
        out = h * silu(x_t @ W_o + b_o)
    """

    def __init__(self, d_model: int, d_state: int = 0, name: str = ""):
        super().__init__(name or "GRM")
        d_state = d_state or d_model
        self.d_model = d_model
        self.d_state = d_state

        scale = 1.0 / math.sqrt(d_model)
        self.W_g = randn((d_model, d_state)) * scale
        self.U_g = randn((d_state, d_state)) * scale
        self.b_g = zeros((d_state,))
        self.W_c = randn((d_model, d_state)) * scale
        self.U_c = randn((d_state, d_state)) * scale
        self.b_c = zeros((d_state,))
        self.W_o = randn((d_state, d_model)) * scale
        self.b_o = zeros((d_model,))

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        B, T, D = x.data.shape
        if h is None:
            h_np = np.zeros((B, self.d_state), dtype=np.float32)
            h = Tensor(h_np, requires_grad=False)
        else:
            h_np = h.data.copy()

        outputs_np = []
        h_np = h.data.copy()

        # Store intermediates for backward
        gates_np = np.zeros((B, T, self.d_state), dtype=np.float32)
        cands_np = np.zeros((B, T, self.d_state), dtype=np.float32)
        h_vals_np = np.zeros((B, T, self.d_state), dtype=np.float32)
        h_prev_np = h_np.copy()

        for t in range(T):
            xt_np = x.data[:, t, :]

            gate = 1.0 / (1.0 + np.exp(-(xt_np @ self.W_g.data + h_np @ self.U_g.data + self.b_g.data)))
            cand = silu(xt_np @ self.W_c.data + h_np @ self.U_c.data + self.b_c.data)
            h_np = gate * h_np + (1.0 - gate) * cand
            out = silu(h_np @ self.W_o.data + self.b_o.data)

            gates_np[:, t, :] = gate
            cands_np[:, t, :] = cand
            h_vals_np[:, t, :] = h_np
            outputs_np.append(out)

        y_np = np.stack(outputs_np, axis=1)
        # Include x in _children so topological sort visits it
        y = Tensor(y_np, requires_grad=True, _children=(x,) + tuple(self.parameters()))

        stored = dict(gates=gates_np, cands=cands_np, h_vals=h_vals_np,
                      h_prev=h_prev_np)

        def bk(g):
            g_np = g.reshape(B, T, D) if g.ndim > 2 else g.reshape(B, T, D)
            d_W_g = np.zeros_like(self.W_g.data)
            d_U_g = np.zeros_like(self.U_g.data)
            d_b_g = np.zeros_like(self.b_g.data)
            d_W_c = np.zeros_like(self.W_c.data)
            d_U_c = np.zeros_like(self.U_c.data)
            d_b_c = np.zeros_like(self.b_c.data)
            d_W_o = np.zeros_like(self.W_o.data)
            d_b_o = np.zeros_like(self.b_o.data)
            d_in = np.zeros_like(x.data)

            s = stored
            for t in range(T):
                g_t = g_np[:, t, :]
                h_t = s['h_vals'][:, t, :]
                xt = x.data[:, t, :]
                gate = s['gates'][:, t, :]
                cand = s['cands'][:, t, :]
                h_prev = s['h_prev'] if t == 0 else s['h_vals'][:, t - 1, :]

                # backward of out = silu(h_t @ W_o + b_o)
                pre_act = h_t @ self.W_o.data + self.b_o.data  # (B, d_model)
                sig_pre = 1.0 / (1.0 + np.exp(-pre_act))
                d_silu = sig_pre + pre_act * sig_pre * (1.0 - sig_pre)  # (B, d_model)
                d_out_t = g_t * d_silu  # (B, d_model)
                d_b_o += np.sum(d_out_t, axis=0)  # (d_model,)
                d_W_o += h_t.T @ d_out_t  # (d_state, d_model)
                d_h = d_out_t @ self.W_o.data.T  # (B, d_state)

                # backward of h_t = gate * h_prev + (1-gate) * cand
                d_cand = d_h * (1.0 - gate)
                d_gate = d_h * (h_prev - cand)

                # backward through gate = sigmoid(xt@W_g + h_prev@U_g + b_g)
                pre_g = xt @ self.W_g.data + h_prev @ self.U_g.data + self.b_g.data
                d_sig_g = gate * (1.0 - gate) * d_gate
                d_b_g += np.sum(d_sig_g, axis=0)
                d_W_g += xt.T @ d_sig_g
                d_U_g += h_prev.T @ d_sig_g

                # backward through cand = silu(xt@W_c + h_prev@U_c + b_c)
                pre_c = xt @ self.W_c.data + h_prev @ self.U_c.data + self.b_c.data
                sig_c = 1.0 / (1.0 + np.exp(-pre_c))
                d_silu_c = sig_c + pre_c * sig_c * (1.0 - sig_c)
                d_cand_in = d_cand * d_silu_c
                d_b_c += np.sum(d_cand_in, axis=0)
                d_W_c += xt.T @ d_cand_in
                d_U_c += h_prev.T @ d_cand_in
                d_in[:, t, :] += d_sig_g @ self.W_g.data.T
                d_in[:, t, :] += d_cand_in @ self.W_c.data.T

            # Accumulate gradients to weight parameters
            for param, grad in [(self.W_g, d_W_g), (self.U_g, d_U_g), (self.b_g, d_b_g),
                                 (self.W_c, d_W_c), (self.U_c, d_U_c), (self.b_c, d_b_c),
                                 (self.W_o, d_W_o), (self.b_o, d_b_o)]:
                if param.requires_grad:
                    if param.grad is None:
                        param.grad = Tensor(grad)
                    else:
                        param.grad.data[:] += grad

            # Accumulate gradient to input x so chain continues
            if x.requires_grad:
                if x.grad is None:
                    x.grad = Tensor(d_in)
                else:
                    x.grad.data[:] += d_in

        y._backward_fn = bk
        h_out = Tensor(h_np.copy(), requires_grad=False)
        return y, h_out

    def forward_numpy(self, x: np.ndarray, h: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        B, T, D = x.shape
        if h is None:
            h = np.zeros((B, self.d_state), dtype=np.float32)

        outputs = []
        for t in range(T):
            xt = x[:, t, :]
            gate = 1.0 / (1.0 + np.exp(-(xt @ self.W_g.data + h @ self.U_g.data + self.b_g.data)))
            cand = silu(xt @ self.W_c.data + h @ self.U_c.data + self.b_c.data)
            h = gate * h + (1.0 - gate) * cand
            out = silu(h @ self.W_o.data + self.b_o.data)
            outputs.append(out)
        return np.stack(outputs, axis=1), h

    def parameters(self) -> List[Tensor]:
        return [self.W_g, self.U_g, self.b_g,
                self.W_c, self.U_c, self.b_c,
                self.W_o, self.b_o]


# =========================================================================
# Rotating Memory Bank
# =========================================================================

class RotatingMemoryBank(SloLayer):
    """Content-addressable rotating memory with N fixed slots.

    Provides long-range context through differentiable read/write memory.
    Unlike transformer attention (O(T^2)), this is O(N) per timestep.

    Each timestep: query slots → soft-read → write → rotate.
    """

    def __init__(self, d_model: int, n_slots: int = 64, name: str = ""):
        super().__init__(name or "RMB")
        self.d_model = d_model
        self.n_slots = n_slots

        scale = 1.0 / math.sqrt(d_model)
        self.W_q = randn((d_model, d_model)) * scale
        self.W_k = randn((d_model, d_model)) * scale
        self.W_v = randn((d_model, d_model)) * scale
        self.W_out = randn((d_model, d_model)) * scale

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.data.shape
        mem = self._init_memory(B)

        # Batch compute Q, K, V for all timesteps
        q = _matmul(x, self.W_q).data  # (B, T, D)
        k = _matmul(x, self.W_k).data
        v = _matmul(x, self.W_v).data

        outputs_np = np.zeros((B, T, D), dtype=np.float32)
        mem_t = mem.copy()

        # Store intermediates for backward
        attn_store = np.zeros((B, T, self.n_slots), dtype=np.float32)
        read_store = np.zeros((B, T, D), dtype=np.float32)
        out_gate_store = np.zeros((B, T, D), dtype=np.float32)

        for t in range(T):
            qt = q[:, t, :]
            kt = k[:, t, :]
            vt = v[:, t, :]

            scores = mem_t @ qt[:, :, None]  # (B, N, 1)
            attn_t = self._softmax_np(scores.squeeze(-1))  # (B, N)
            read_t = np.sum(attn_t[:, :, None] * mem_t, axis=1)  # (B, D)

            write_gate = attn_t[:, :, None] * (kt[:, None, :] - mem_t)
            mem_t = mem_t + 0.1 * write_gate
            mem_t = np.concatenate([mem_t[:, 1:, :], mem_t[:, :1, :]], axis=1)

            pre_gate = read_t @ self.W_out.data
            out_gate_t = 1.0 / (1.0 + np.exp(-pre_gate))
            outputs_np[:, t, :] = vt * out_gate_t

            attn_store[:, t, :] = attn_t
            read_store[:, t, :] = read_t
            out_gate_store[:, t, :] = out_gate_t

        y = Tensor(outputs_np, requires_grad=True, _children=(x, self.W_q, self.W_k, self.W_v, self.W_out))

        def bk(g):
            g_np = g.reshape(B, T, D) if g.ndim > 2 else g.reshape(B, T, D)
            mem_c = mem.copy()
            d_W_q = np.zeros_like(self.W_q.data)
            d_W_k = np.zeros_like(self.W_k.data)
            d_W_v = np.zeros_like(self.W_v.data)
            d_W_out = np.zeros_like(self.W_out.data)
            d_in = np.zeros((B, T, D), dtype=np.float32)

            for t in range(T):
                g_t = g_np[:, t, :]
                qt = q[:, t, :]
                kt = k[:, t, :]
                vt = v[:, t, :]
                attn_t = attn_store[:, t, :]
                read_t = read_store[:, t, :]
                out_gate_t = out_gate_store[:, t, :]

                # backward: y_t = vt * out_gate_t
                d_vt = g_t * out_gate_t
                d_out_gate = g_t * vt

                # backward: out_gate_t = sigmoid(read_t @ W_out)
                pre = read_t @ self.W_out.data
                sig = 1.0 / (1.0 + np.exp(-pre))
                d_sig = d_out_gate * sig * (1.0 - sig)
                d_W_out += read_t.T @ d_sig
                d_read = d_sig @ self.W_out.data.T

                # backward: read_t = sum(attn * mem_c)
                d_attn = np.sum(mem_c * d_read[:, None, :], axis=2)

                # backward: attn_t = softmax(scores)
                scores = mem_c @ qt[:, :, None]
                s = scores.squeeze(-1)
                p = attn_t
                d_scores = p * (d_attn - np.sum(d_attn * p, axis=1, keepdims=True))
                d_qt = np.sum(mem_c * d_scores[:, :, None], axis=1)

                # backward: scores = mem_c @ qt
                # (already handled above — d_qt includes the mem_c term)

                # backward: qt = xt @ W_q
                xt = x.data[:, t, :]
                d_W_q += xt.T @ d_qt[:, None] if d_qt.ndim == 1 else xt.T @ d_qt
                d_in[:, t, :] += d_qt @ self.W_q.data.T

                # backward: vt = xt @ W_v
                d_W_v += xt.T @ d_vt if d_vt.ndim == 1 else xt.T @ d_vt
                d_in[:, t, :] += d_vt @ self.W_v.data.T

                # Recompute mem forward for this timestep to get write gate
                scores_t = mem_c @ qt[:, :, None]
                attn_t2 = self._softmax_np(scores_t.squeeze(-1))
                write_gate = attn_t2[:, :, None] * (kt[:, None, :] - mem_c)
                mem_c = mem_c + 0.1 * write_gate
                mem_c = np.concatenate([mem_c[:, 1:, :], mem_c[:, :1, :]], axis=1)

            # Accumulate gradients
            for param, grad in [(self.W_q, d_W_q), (self.W_k, d_W_k),
                                 (self.W_v, d_W_v), (self.W_out, d_W_out)]:
                if param.requires_grad:
                    if param.grad is None:
                        param.grad = Tensor(grad)
                    else:
                        param.grad.data[:] += grad

            # Pass gradient back to input x
            if x.requires_grad:
                d_in_t = Tensor(d_in)
                if x.grad is None:
                    x.grad = d_in_t
                else:
                    x.grad.data[:] += d_in_t.data

        y._backward_fn = bk
        return y

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        q = x @ self.W_q.data
        k = x @ self.W_k.data
        v = x @ self.W_v.data
        B, T, D = x.shape
        mem = self._init_memory(B)

        outputs = []
        for t in range(T):
            qt = q[:, t, :]
            kt = k[:, t, :]
            vt = v[:, t, :]

            scores = mem @ qt[:, :, None]
            attn = self._softmax_np(scores.squeeze(-1))
            read = np.sum(attn[:, :, None] * mem, axis=1)

            write_gate = attn[:, :, None] * (kt[:, None, :] - mem)
            mem = mem + 0.1 * write_gate
            mem = np.concatenate([mem[:, 1:, :], mem[:, :1, :]], axis=1)

            out_gate = 1.0 / (1.0 + np.exp(-(read @ self.W_out.data)))
            outputs.append(vt * out_gate)

        return np.stack(outputs, axis=1)

    def _init_memory(self, batch: int) -> np.ndarray:
        mem = np.zeros((batch, self.n_slots, self.d_model), dtype=np.float32)
        idx = np.arange(self.n_slots, dtype=np.float32) / self.n_slots
        mem += idx[None, :, None] * 0.01
        return mem

    @staticmethod
    def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        e = np.exp(np.clip(x - x_max, -500, 500))
        return e / (np.sum(e, axis=axis, keepdims=True) + 1e-10)

    def parameters(self) -> List[Tensor]:
        return [self.W_q, self.W_k, self.W_v, self.W_out]


# =========================================================================
# Feed-Forward Network (SwiGLU)
# =========================================================================

class SwiGLUFFN(SloLayer):
    """Feed-forward with SwiGLU activation."""

    def __init__(self, d_model: int, d_ff: int = 0, name: str = ""):
        super().__init__(name or "FFN")
        d_ff = d_ff or d_model * 4
        scale = 1.0 / math.sqrt(d_model)
        self.W1 = randn((d_model, d_ff)) * scale
        self.W2 = randn((d_model, d_ff)) * scale
        self.W3 = randn((d_ff, d_model)) * scale

    def forward(self, x: Tensor) -> Tensor:
        h = silu(_matmul(x, self.W1)) * _matmul(x, self.W2)
        return _matmul(h, self.W3)

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        h = silu(x @ self.W1.data) * (x @ self.W2.data)
        return h @ self.W3.data

    def parameters(self) -> List[Tensor]:
        return [self.W1, self.W2, self.W3]


# =========================================================================
# SloRAN Layer
# =========================================================================

class SloRANLayer(SloLayer):
    """One SloRAN block: Norm → GRM → Memory → FFN → Residual."""

    def __init__(self, d_model: int, n_slots: int = 64, d_state: int = 0,
                 d_ff: int = 0, dropout: float = 0.1, eps: float = 1e-5, name: str = ""):
        super().__init__(name or "SloRANLayer")
        self.norm1 = SloLayerNorm(d_model, eps, f"{name}.norm1")
        self.grm = GatedRecurrentMixer(d_model, d_state, f"{name}.grm")
        self.memory = RotatingMemoryBank(d_model, n_slots, f"{name}.memory")
        self.norm2 = SloLayerNorm(d_model, eps, f"{name}.norm2")
        self.ffn = SwiGLUFFN(d_model, d_ff, f"{name}.ffn")
        self.drop = SloDropout(dropout, f"{name}.drop") if dropout > 0 else None

    def forward(self, x: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        residual = x
        x = self.norm1.forward(x)
        x, new_state = self.grm.forward(x, state)
        x = self.memory.forward(x)
        x = residual + x
        if self.drop:
            x = self.drop.forward(x)

        residual = x
        x = self.norm2.forward(x)
        x = self.ffn.forward(x)
        x = residual + x
        if self.drop:
            x = self.drop.forward(x)

        return x, new_state

    def forward_numpy(self, x: np.ndarray, state: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        residual = x
        x = _layer_norm(x, self.norm1.weight.data, self.norm1.bias.data, self.norm1.eps)
        x, new_state = self.grm.forward_numpy(x, state)
        x = self.memory.forward_numpy(x)
        x = residual + x

        residual = x
        x = _layer_norm(x, self.norm2.weight.data, self.norm2.bias.data, self.norm2.eps)
        x = self.ffn.forward_numpy(x)
        x = residual + x
        return x, new_state

    def parameters(self) -> List[Tensor]:
        ps = self.grm.parameters() + self.memory.parameters() + self.ffn.parameters()
        ps += self.norm1.parameters() + self.norm2.parameters()
        return ps


# =========================================================================
# Full SloRAN Language Model
# =========================================================================

class SloRAN(SloNet):
    """SloRAN language model: Embed → SloRANLayer × N → Norm → LM Head.

    Proprietary attention-free architecture using gated recurrence and
    rotating associative memory instead of quadratic self-attention.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 256,
        n_layers: int = 4,
        n_slots: int = 64,
        d_state: int = 0,
        d_ff: int = 0,
        block_size: int = 128,
        dropout: float = 0.1,
        eps: float = 1e-5,
        tie_weights: bool = True,
        soul_name: str = "SloRAN",
        soul_traits: Optional[Dict[str, float]] = None,
    ):
        d_ff = d_ff or d_model * 4
        d_state = d_state or d_model

        layers: List[SloLayer] = [SloEmbedding(vocab_size, d_model, "tok_emb")]
        if dropout > 0:
            layers.append(SloDropout(dropout, "emb_drop"))
        for i in range(n_layers):
            layers.append(SloRANLayer(
                d_model, n_slots=n_slots, d_state=d_state,
                d_ff=d_ff, dropout=0, eps=eps, name=f"ran_block.{i}",
            ))
        layers.append(SloRMSNorm(d_model, eps, "norm"))
        layers.append(SloLinear(d_model, vocab_size, "lm_head"))

        super().__init__(
            layers=layers,
            soul_name=soul_name,
            soul_traits=soul_traits or {"warmth": 0.5, "creativity": 0.5, "curiosity": 0.5, "confidence": 0.5},
            system_prompt="",
            lineage="sloran",
            metadata={
                "vocab_size": vocab_size, "d_model": d_model,
                "n_layers": n_layers, "n_slots": n_slots,
                "d_state": d_state, "block_size": block_size,
                "dropout": dropout, "model_type": "sloran",
            },
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.block_size = block_size
        self._states: List[Optional[np.ndarray]] = [None] * n_layers

        if tie_weights:
            try:
                self.layers[-1].weight.data[:] = self.layers[0].weight.data.copy()
            except Exception:
                pass

    @property
    def ran_blocks(self) -> List[SloRANLayer]:
        start = 2 if isinstance(self.layers[1], SloDropout) else 1
        return [l for l in self.layers[start:-2] if isinstance(l, SloRANLayer)]

    def reset_states(self):
        self._states = [None] * self.n_layers

    def forward(self, input_ids, targets=None, use_cache=False, **kwargs):
        if isinstance(input_ids, np.ndarray):
            x = Tensor(input_ids.astype(np.int64))
        elif isinstance(input_ids, Tensor):
            x = input_ids
        else:
            x = Tensor(np.array(input_ids, dtype=np.int64))

        x = self.layers[0].forward(x)
        for i, l in enumerate(self.layers[1:-2]):
            if isinstance(l, SloDropout):
                x = l.forward(x)
            elif isinstance(l, SloRANLayer):
                state = Tensor(self._states[i], requires_grad=False) if use_cache and self._states[i] is not None else None
                x, new_state = l.forward(x, state=state)
                if use_cache:
                    self._states[i] = new_state.data.copy() if isinstance(new_state, Tensor) else new_state.copy()

        x = self.layers[-2].forward(x)
        logits = self.layers[-1].forward(x)

        if targets is not None:
            if isinstance(targets, np.ndarray):
                t = targets.astype(np.int64)
            elif isinstance(targets, Tensor):
                t = targets.data.astype(np.int64)
            else:
                t = np.array(targets, dtype=np.int64)
            loss = cross_entropy(logits.reshape(-1, self.vocab_size), Tensor(t.reshape(-1)))
            return logits, loss

        return logits

    def generate(self, input_ids: np.ndarray, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: Optional[int] = None) -> np.ndarray:
        """Generate tokens autoregressively."""
        self.reset_states()
        with no_grad():
            generated = list(input_ids.flatten())
            for _ in range(max_new_tokens):
                ctx = Tensor(np.array([generated[-self.block_size:]], dtype=np.int64))
                logits = self.forward(ctx)
                logits_np = logits.data[:, -1, :]

                if top_k is not None:
                    top_vals = np.sort(logits_np)[:, -top_k:]
                    logits_np[logits_np < top_vals[:, -1:]] = -1e9

                if temperature <= 0:
                    next_id = int(logits_np[0].argmax())
                else:
                    logits_np = logits_np / temperature
                    probs = self._softmax_np(logits_np, axis=-1)
                    next_id = int(np.random.choice(self.vocab_size, p=probs[0]))
                generated.append(int(next_id))

        return np.array(generated, dtype=np.int64)

    @staticmethod
    def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        e = np.exp(np.clip(x - x_max, -500, 500))
        return e / (np.sum(e, axis=axis, keepdims=True) + 1e-10)


# =========================================================================
# NumPy-only helpers (no Tensor wrapping)
# =========================================================================

def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return weight * (x - mean) / np.sqrt(var + eps) + bias
