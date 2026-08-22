/**
 * JS-based SloNet runtime for on-device inference.
 *
 * Loads flat float32 weights from the server's export-mobile endpoint and
 * runs the forward pass in pure JS (Hermes JIT).  This is viable because
 * the SloTransformer is a baby model:
 *   - vocab_size=256, n_embed=128, n_layer=4, n_head=4
 *   - ~2.5M parameters → ~10 MB Float32 weights
 *   - ~2-5 ms per token on modern phone CPUs
 *
 * Architecture (matches SloTransformer in slonet.py):
 *   Embed → {RMSNorm → QKV → RoPE → SDPA → O_proj → +residual →
 *            RMSNorm → SwiGLU FFN → +residual} × n_layer →
 *   RMSNorm → LM_head → logits
 */

import {getApiUrl} from './api-client';
import AsyncStorage from '@react-native-async-storage/async-storage';
import type {
  MobileExportResponse,
  MobileExportConfig,
  LocalGenerateResult,
  OnTokenCallback,
} from '../types/local-inference';

const CACHE_KEY_PREFIX = '@sloughgpt/slonet_';

// ── Tensor helpers (Float32Array wrappers) ──────────────────────────────

function matMul(A: Float32Array, B: Float32Array, M: number, K: number, N: number): Float32Array {
  const out = new Float32Array(M * N);
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += A[m * K + k] * B[k * N + n];
      }
      out[m * N + n] = sum;
    }
  }
  return out;
}

function matMulAdd(
  C: Float32Array, A: Float32Array, B: Float32Array,
  M: number, K: number, N: number,
): void {
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] += sum;
    }
  }
}

function rmsNorm(x: Float32Array, w: Float32Array, eps: number): Float32Array {
  const n = x.length;
  let sqSum = 0;
  for (let i = 0; i < n; i++) sqSum += x[i] * x[i];
  const rms = Math.sqrt(sqSum / n + eps);
  const inv = 1 / rms;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
  return out;
}

function silu(x: number): number {
  return x / (1 + Math.exp(-x));
}

function softmax(logits: Float32Array): Float32Array {
  const n = logits.length;
  let maxVal = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] > maxVal) maxVal = logits[i];
  let sum = 0;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.exp(logits[i] - maxVal);
    sum += out[i];
  }
  const inv = 1 / sum;
  for (let i = 0; i < n; i++) out[i] *= inv;
  return out;
}

// ── RoPE ────────────────────────────────────────────────────────────────

function precomputeFreqs(dim: number, maxSeq: number, base: number): [Float32Array, Float32Array] {
  const cos = new Float32Array(maxSeq * dim);
  const sin = new Float32Array(maxSeq * dim);
  for (let i = 0; i < maxSeq; i++) {
    for (let j = 0; j < dim; j += 2) {
      const freq = i / Math.pow(base, j / dim);
      cos[i * dim + j] = Math.cos(freq);
      cos[i * dim + j + 1] = Math.cos(freq);
      sin[i * dim + j] = Math.sin(freq);
      sin[i * dim + j + 1] = Math.sin(freq);
    }
  }
  return [cos, sin];
}

function applyRoPE(
  q: Float32Array, k: Float32Array,
  cos: Float32Array, sin: Float32Array, pos: number, nHeads: number, headDim: number,
): [Float32Array, Float32Array] {
  const nTokens = q.length / (nHeads * headDim);
  const qOut = new Float32Array(q.length);
  const kOut = new Float32Array(k.length);
  for (let t = 0; t < nTokens; t++) {
    const offset = (pos + t) * headDim;
    for (let h = 0; h < nHeads; h++) {
      const idx = (t * nHeads + h) * headDim;
      for (let d = 0; d < headDim; d++) {
        const c = cos[offset + d];
        const s = sin[offset + d];
        const half = headDim / 2;
        const qRotDim = d < half ? d : d - half;
        const qRot = d < half ? -q[idx + qRotDim + half] : q[idx + qRotDim];
        qOut[idx + d] = q[idx + d] * c + qRot * s;
        const kRot = d < half ? -k[idx + qRotDim + half] : k[idx + qRotDim];
        kOut[idx + d] = k[idx + d] * c + kRot * s;
      }
    }
  }
  return [qOut, kOut];
}

// ── Checkpoint loader ───────────────────────────────────────────────────

interface LoadedCheckpoint {
  config: MobileExportConfig;
  cos: Float32Array;
  sin: Float32Array;
  /** Flat Float32Array, structured per export-mobile layout */
  weights: Float32Array;
}

let _loaded: LoadedCheckpoint | null = null;

function _decodeWeights(b64: string): Float32Array {
  const raw = atob(b64);
  const buf = new ArrayBuffer(raw.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < raw.length; i++) view[i] = raw.charCodeAt(i);
  return new Float32Array(buf);
}

function _offsetWeights(cfg: MobileExportConfig): Map<string, [number, number]> {
  const map = new Map<string, [number, number]>();
  let off = 0;

  function push(name: string, n: number) {
    map.set(name, [off, n]);
    off += n;
  }

  push('tok_emb.weight', cfg.vocab_size * cfg.n_embed);

  for (let i = 0; i < cfg.n_layer; i++) {
    push(`blocks.${i}.attn_norm.weight`, cfg.n_embed);
    push(`blocks.${i}.attn.q_proj.weight`, cfg.n_embed * cfg.n_embed);
    push(`blocks.${i}.attn.k_proj.weight`, cfg.n_embed * cfg.n_embed);
    push(`blocks.${i}.attn.v_proj.weight`, cfg.n_embed * cfg.n_embed);
    push(`blocks.${i}.attn.o_proj.weight`, cfg.n_embed * cfg.n_embed);
    push(`blocks.${i}.ff_norm.weight`, cfg.n_embed);
    push(`blocks.${i}.ff.w1.weight`, cfg.n_embed * _nEmbedDimFF(cfg));
    push(`blocks.${i}.ff.w2.weight`, _nEmbedDimFF(cfg) * cfg.n_embed);
    push(`blocks.${i}.ff.w3.weight`, cfg.n_embed * _nEmbedDimFF(cfg));
  }

  push('norm.weight', cfg.n_embed);
  push('lm_head.weight', cfg.n_embed * cfg.vocab_size);

  return map;
}

function _nEmbedDimFF(cfg: MobileExportConfig): number {
  const raw = cfg.n_embed * 8 / 3;
  return Math.ceil(raw / 64) * 64;
}

// ── Forward pass ────────────────────────────────────────────────────────

function _slice(flat: Float32Array, off: number, n: number): Float32Array {
  return flat.slice(off, off + n);
}

function _reshape(v: Float32Array, rows: number, cols: number): Float32Array {
  // Already flat, just validate
  if (v.length !== rows * cols) throw new Error(`reshape ${v.length} → ${rows}×${cols}`);
  return v;
}

function _forwardSingle(
  tokens: Int32Array,
  cp: LoadedCheckpoint,
  startPos: number,
  temperature: number,
  topK: number,
  topP: number,
  maxNewTokens: number,
  eosToken: number,
  onToken?: OnTokenCallback,
  signal?: AbortSignal,
): string {
  const cfg = cp.config;
  const w = cp.weights;
  const offsets = _offsetWeights(cfg);

  const nEmb = cfg.n_embed;
  const nHead = cfg.n_head;
  const nLayer = cfg.n_layer;
  const headDim = nEmb / nHead;
  const dimFF = _nEmbedDimFF(cfg);
  const eps = 1e-5;

  const genTokens: number[] = [];
  let prompt = Array.from(tokens);

  for (let step = 0; step < maxNewTokens; step++) {
    if (signal?.aborted) break
    // Embedding lookup
    const seqLen = step === 0 ? prompt.length : 1;
    const emb = new Float32Array(seqLen * nEmb);
    const [tokOff] = offsets.get('tok_emb.weight')!;
    for (let t = 0; t < seqLen; t++) {
      const tokenId = step === 0 ? prompt[t] : prompt[prompt.length - 1];
      const row = w.slice(tokOff + tokenId * nEmb, tokOff + (tokenId + 1) * nEmb);
      emb.set(row, t * nEmb);
    }

    let x = emb;
    const pos = step === 0 ? 0 : prompt.length - 1;

    // Transformer blocks
    for (let i = 0; i < nLayer; i++) {
      // Attn norm
      const [anOff] = offsets.get(`blocks.${i}.attn_norm.weight`)!;
      const anW = _slice(w, anOff, nEmb);
      const normed = new Float32Array(seqLen * nEmb);
      for (let t = 0; t < seqLen; t++) {
        const slice = x.slice(t * nEmb, (t + 1) * nEmb);
        normed.set(rmsNorm(slice, anW, eps), t * nEmb);
      }

      // QKV projections
      const [qOff] = offsets.get(`blocks.${i}.attn.q_proj.weight`)!;
      const [kOff] = offsets.get(`blocks.${i}.attn.k_proj.weight`)!;
      const [vOff] = offsets.get(`blocks.${i}.attn.v_proj.weight`)!;
      const qW = _slice(w, qOff, nEmb * nEmb);
      const kW = _slice(w, kOff, nEmb * headDim);
      const vW = _slice(w, vOff, nEmb * headDim);

      const Q = matMul(normed, qW, seqLen, nEmb, nEmb);
      const K = matMul(normed, kW, seqLen, nEmb, headDim);
      const V = matMul(normed, vW, seqLen, nEmb, headDim);

      // RoPE
      const [cosT, sinT] = precomputeFreqs(headDim, pos + seqLen, 10000);
      const [Qr, Kr] = applyRoPE(Q, K, cosT, sinT, pos, nHead, headDim);

      // SDPA
      const scale = 1 / Math.sqrt(headDim);
      const scores = new Float32Array(nHead * seqLen * seqLen);
      for (let h = 0; h < nHead; h++) {
        for (let t1 = 0; t1 < seqLen; t1++) {
          for (let t2 = 0; t2 < seqLen; t2++) {
            let sum = 0;
            for (let d = 0; d < headDim; d++) {
              const qIdx = (t1 * nHead + h) * headDim + d;
              const kIdx = (t2 * nHead + h) * headDim + d;
              sum += Qr[qIdx] * Kr[kIdx];
            }
            scores[h * seqLen * seqLen + t1 * seqLen + t2] = sum * scale;
          }
        }
      }

      // Causal mask
      for (let t1 = 0; t1 < seqLen; t1++) {
        for (let t2 = t1 + 1; t2 < seqLen; t2++) {
          for (let h = 0; h < nHead; h++) {
            scores[h * seqLen * seqLen + t1 * seqLen + t2] = -1e9;
          }
        }
      }

      // Softmax per head per query position
      const attn = new Float32Array(nHead * seqLen * seqLen);
      for (let h = 0; h < nHead; h++) {
        for (let t1 = 0; t1 < seqLen; t1++) {
          const start = h * seqLen * seqLen + t1 * seqLen;
          const logits = scores.slice(start, start + seqLen);
          const probs = softmax(logits);
          attn.set(probs, start);
        }
      }

      // Attention output
      const attnOut = new Float32Array(seqLen * nEmb);
      for (let h = 0; h < nHead; h++) {
        for (let t1 = 0; t1 < seqLen; t1++) {
          for (let d = 0; d < headDim; d++) {
            let sum = 0;
            for (let t2 = 0; t2 < seqLen; t2++) {
              const aIdx = h * seqLen * seqLen + t1 * seqLen + t2;
              const vIdx = (t2 * nHead + h) * headDim + d;
              sum += attn[aIdx] * V[vIdx];
            }
            attnOut[t1 * nEmb + h * headDim + d] = sum;
          }
        }
      }

      // Output projection + residual
      const [oOff] = offsets.get(`blocks.${i}.attn.o_proj.weight`)!;
      const oW = _slice(w, oOff, nEmb * nEmb);
      const proj = matMul(attnOut, oW, seqLen, nEmb, nEmb);
      for (let t = 0; t < seqLen * nEmb; t++) x[t] += proj[t];

      // FFN norm
      const [fnOff] = offsets.get(`blocks.${i}.ff_norm.weight`)!;
      const fnW = _slice(w, fnOff, nEmb);
      const ffNormed = new Float32Array(seqLen * nEmb);
      for (let t = 0; t < seqLen; t++) {
        const slice = x.slice(t * nEmb, (t + 1) * nEmb);
        ffNormed.set(rmsNorm(slice, fnW, eps), t * nEmb);
      }

      // SwiGLU FFN
      const [w1Off] = offsets.get(`blocks.${i}.ff.w1.weight`)!;
      const [w2Off] = offsets.get(`blocks.${i}.ff.w2.weight`)!;
      const [w3Off] = offsets.get(`blocks.${i}.ff.w3.weight`)!;
      const w1W = _slice(w, w1Off, nEmb * dimFF);
      const w2W = _slice(w, w2Off, dimFF * nEmb);
      const w3W = _slice(w, w3Off, nEmb * dimFF);

      const h1 = matMul(ffNormed, w1W, seqLen, nEmb, dimFF);
      const h3 = matMul(ffNormed, w3W, seqLen, nEmb, dimFF);
      const gated = new Float32Array(seqLen * dimFF);
      for (let t = 0; t < seqLen * dimFF; t++) {
        gated[t] = silu(h1[t]) * h3[t];
      }
      const ffOut = matMul(gated, w2W, seqLen, dimFF, nEmb);
      for (let t = 0; t < seqLen * nEmb; t++) x[t] += ffOut[t];
    }

    // Final RMSNorm + LM head
    const [normOff] = offsets.get('norm.weight')!;
    const normW = _slice(w, normOff, nEmb);
    const finalNormed = new Float32Array(seqLen * nEmb);
    for (let t = 0; t < seqLen; t++) {
      const slice = x.slice(t * nEmb, (t + 1) * nEmb);
      finalNormed.set(rmsNorm(slice, normW, eps), t * nEmb);
    }

    const [lmOff] = offsets.get('lm_head.weight')!;
    const lmW = _slice(w, lmOff, nEmb * cfg.vocab_size);
    const logits = matMul(finalNormed, lmW, seqLen, nEmb, cfg.vocab_size);
    const lastLogits = logits.slice((seqLen - 1) * cfg.vocab_size);

    // Temperature
    if (temperature > 0 && temperature !== 1) {
      const invT = 1 / temperature;
      for (let i = 0; i < lastLogits.length; i++) lastLogits[i] *= invT;
    }

    // Top-K
    if (topK > 0) {
      const threshold = _kthLargest(lastLogits, Math.min(topK, lastLogits.length));
      for (let i = 0; i < lastLogits.length; i++) {
        if (lastLogits[i] < threshold) lastLogits[i] = -Infinity;
      }
    }

    // Top-P (nucleus sampling)
    if (topP < 1) {
      const indexed: Array<{v: number; i: number}> = [];
      for (let i = 0; i < lastLogits.length; i++) indexed.push({v: lastLogits[i], i});
      indexed.sort((a, b) => b.v - a.v);
      const probsArr = new Float32Array(indexed.length);
      for (let i = 0; i < indexed.length; i++) probsArr[i] = indexed[i].v;
      const probs = softmax(probsArr);
      let cum = 0;
      let cutoff = false;
      for (let pos = 0; pos < indexed.length; pos++) {
        if (!cutoff) {
          cum += probs[pos];
          if (cum > topP) cutoff = true;
        }
        if (cutoff) {
          lastLogits[indexed[pos].i] = -Infinity;
        }
      }
    }

    // Sample
    const probs = softmax(lastLogits);
    let nextToken: number;
    if (temperature <= 0) {
      let maxIdx = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[maxIdx]) maxIdx = i;
      nextToken = maxIdx;
    } else {
      const r = Math.random();
      let cum = 0;
      nextToken = probs.length - 1;
      for (let i = 0; i < probs.length; i++) {
        cum += probs[i];
        if (r < cum) { nextToken = i; break; }
      }
    }

    genTokens.push(nextToken);
    if (onToken) onToken(_decodeTokens([nextToken]));
    if (nextToken === eosToken) break;
    prompt.push(nextToken);
  }

  return _decodeTokens(genTokens);
}

function _kthLargest(arr: Float32Array, k: number): number {
  if (k >= arr.length) return -Infinity;
  const sorted = Array.from(arr).sort((a, b) => b - a);
  return sorted[k - 1] ?? -Infinity;
}

function _decodeTokens(tokens: number[]): string {
  const maxVocab = 256;
  const bytes = new Uint8Array(tokens.filter(t => t < maxVocab));
  return new TextDecoder('utf-8', {fatal: false}).decode(bytes);
}

// ── Public API ──────────────────────────────────────────────────────────

/** Load a checkpoint from the server or local cache. */
export async function loadCheckpoint(name: string): Promise<void> {
  // Check cache first
  const cacheKey = CACHE_KEY_PREFIX + name;
  const cached = await AsyncStorage.getItem(cacheKey);
  if (cached) {
    const parsed: MobileExportResponse = JSON.parse(cached);
    _loadFromResponse(parsed);
    return;
  }

  // Download from server
  const baseUrl = await getApiUrl();
  const res = await fetch(`${baseUrl}/auto-train/checkpoints/${encodeURIComponent(name)}/export-mobile`);
  if (!res.ok) throw new Error(`Failed to load checkpoint: ${res.status}`);
  const data: MobileExportResponse = await res.json();

  // Cache
  await AsyncStorage.setItem(cacheKey, JSON.stringify(data));

  _loadFromResponse(data);
}

function _loadFromResponse(data: MobileExportResponse): void {
  const cfg = data.config as MobileExportConfig & {nEmbedDimFF?: () => number};
  const weights = _decodeWeights(data.weights_b64);
  const [cos, sin] = precomputeFreqs(cfg.n_embed / cfg.n_head, cfg.block_size, 10000);
  _loaded = {config: data.config, weights, cos, sin};
}

/** Check if a checkpoint is currently loaded. */
export function isLoaded(): boolean {
  return _loaded !== null;
}

/** Unload the current checkpoint. */
export function unload(): void {
  _loaded = null;
}

/** Load a pre-computed flat weight array with config (used by sou-loader). */
export function loadFlatWeights(
  config: MobileExportConfig,
  weights: Float32Array,
): void {
  const [cos, sin] = precomputeFreqs(config.n_embed / config.n_head, config.block_size, 10000);
  _loaded = { config, weights, cos, sin };
}

/** Generate text using the loaded SloNet model. */
export async function generate(
  prompt: string,
  maxNewTokens = 64,
  temperature = 0.8,
  topK = 40,
  topP = 0.9,
  eosToken = 0,
  onToken?: OnTokenCallback,
  signal?: AbortSignal,
): Promise<LocalGenerateResult> {
  if (!_loaded) throw new Error('No checkpoint loaded');

  const tokens = _encodePrompt(prompt, _loaded.config.vocab_size);
  const t0 = Date.now();

  const text = _forwardSingle(
    tokens,
    _loaded,
    0,
    temperature,
    topK,
    topP,
    maxNewTokens,
    eosToken,
    onToken,
    signal,
  );

  return {
    text,
    tokens_generated: text.length,
    elapsed_ms: Date.now() - t0,
  };
}

function _encodePrompt(prompt: string, vocabSize: number): Int32Array {
  const bytes = new TextEncoder().encode(prompt);
  const tokens = new Int32Array(bytes.length + 2);
  tokens[0] = 1;
  for (let i = 0; i < bytes.length; i++) {
    tokens[i + 1] = bytes[i] < vocabSize ? bytes[i] : 0;
  }
  tokens[bytes.length + 1] = 2;
  return tokens;
}
