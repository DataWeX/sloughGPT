/**
 * SoulTransformerWebGPU — browser-side Transformer inference engine.
 *
 * Architecture: decoder-only Transformer with RoPE, RMSNorm, SwiGLU FFN, KV cache
 * (see trained baby_step_X models: 384 embed, 8 heads, 6 layers, 1024 FFN, vocab=45).
 *
 * CPU: embedding, RMSNorm, matmuls (QKV proj, FFN, LM head), RoPE, sampling
 * GPU: fused multi-head attention (Q×K^T → softmax → ×V) — the O(n²) bottleneck
 */

import { parseSou, type SoulCheckpoint, type SoulMetadata, type SoulTransformerArch } from './weights'

const ATTENTION_SHADER = `
struct P { n_heads: u32, n_kv_head: u32, head_dim: u32, cache_len: u32, scale: f32, _pad: u32, __pad: u32 };
@group(0) @binding(0) var<storage, read> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k_cache: array<f32>;
@group(0) @binding(3) var<storage, read> v_cache: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> scores: array<f32, 2048>;
var<workgroup> max_s: array<f32, 1>;
var<workgroup> sum_exp: array<f32, 1>;

@compute @workgroup_size(8, 1, 1)
fn attn_fused(@builtin(global_invocation_id) id: vec3<u32>) {
  let h = id.x;
  let H = p.n_heads;
  let KH = p.n_kv_head;
  let D = p.head_dim;
  let T = p.cache_len;
  let rep = H / KH;
  let kh = h / rep;
  let scale = p.scale;
  if (h >= H || T == 0u) { return; }

  // 1. Compute scores: Q[h] · K_cache[t][kh] / sqrt(D)
  var m = -1e30;
  for (var t = 0u; t < T; t++) {
    var s = 0.0;
    for (var d = 0u; d < D; d++) {
      let qd = q[h * D + d];
      let kd = k_cache[t * KH * D + kh * D + d];
      s += qd * kd;
    }
    s = s * scale;
    if (s > m) { m = s; }
    scores[t] = s;
  }
  max_s[0] = m;
  workgroupBarrier();

  // 2. Softmax: exp(s - max), sum
  var esum = 0.0;
  for (var t = 0u; t < T; t++) {
    let e = exp(scores[t] - max_s[0]);
    scores[t] = e;
    esum += e;
  }
  sum_exp[0] = esum;
  workgroupBarrier();

  // 3. Weighted sum: out[h][d] = sum(attn[t] * V_cache[t][kh][d])
  let inv = 1.0 / max(sum_exp[0], 1e-10);
  for (var d = 0u; d < D; d++) {
    var o = 0.0;
    for (var t = 0u; t < T; t++) {
      let vt = v_cache[t * KH * D + kh * D + d];
      o += (scores[t] * inv) * vt;
    }
    out[h * D + d] = o;
  }
}
`

function silu(x: number): number {
  return x / (1 + Math.exp(-x))
}

function rmsnorm(x: Float32Array, w: Float32Array, eps: number): Float32Array {
  let sumSq = 0
  for (let i = 0; i < x.length; i++) sumSq += x[i] * x[i]
  const rms = Math.sqrt(sumSq / x.length + eps)
  const y = new Float32Array(x.length)
  for (let i = 0; i < x.length; i++) y[i] = (x[i] / rms) * w[i]
  return y
}

function matMul(x: Float32Array, w: Float32Array, outDim: number, inDim: number): Float32Array {
  const y = new Float32Array(outDim)
  for (let i = 0; i < outDim; i++) {
    let s = 0
    for (let j = 0; j < inDim; j++) s += x[j] * w[i * inDim + j]
    y[i] = s
  }
  return y
}

function applyRoPE(q: Float32Array, k: Float32Array, pos: number, headDim: number): [Float32Array, Float32Array] {
  const nHeads = q.length / headDim
  const qr = new Float32Array(q.length)
  const kr = new Float32Array(k.length)
  for (let h = 0; h < nHeads; h++) {
    for (let d = 0; d < headDim; d += 2) {
      const freq = 1 / Math.pow(10000, d / headDim)
      const theta = pos * freq
      const cos = Math.cos(theta)
      const sin = Math.sin(theta)
      const i0 = h * headDim + d
      const i1 = h * headDim + d + 1
      qr[i0] = q[i0] * cos - q[i1] * sin
      qr[i1] = q[i0] * sin + q[i1] * cos
      kr[i0] = k[i0] * cos - k[i1] * sin
      kr[i1] = k[i0] * sin + k[i1] * cos
    }
  }
  return [qr, kr]
}

export class SoulTransformerWebGPU {
  private device: GPUDevice | null = null
  private pipeline: GPUComputePipeline | null = null
  private bgLayout: GPUBindGroupLayout | null = null

  private arch: SoulTransformerArch | null = null
  metadata: SoulMetadata | null = null
  ready = false
  loading = false

  // Parameters stored as flat Float32Arrays indexed by layer
  private params: {
    embWeight: Float32Array    // [vocab, embed] or [vocab, embed] T
    normWeight: Float32Array   // final norm weight [embed]
    lmWeight: Float32Array     // [vocab, embed] (tied to emb)
    layers: Array<{
      attnNorm: Float32Array   // [embed]
      ffNorm: Float32Array     // [embed]
      wQ: Float32Array         // [embed, embed]
      wK: Float32Array         // [embed, embed]
      wV: Float32Array         // [embed, embed]
      wO: Float32Array         // [embed, embed]
      w1: Float32Array         // [dimFF, embed]
      w2: Float32Array         // [embed, dimFF]
      w3: Float32Array         // [dimFF, embed]
    }>
  } | null = null

  // KV cache: per-layer Float32Arrays for K and V states [maxSeqLen * n_kv_head * head_dim]
  private kCache: Float32Array[] = []
  private vCache: Float32Array[] = []
  private cacheLen = 0
  private maxSeqLen = 2048

  // GPU attention buffers
  private gpuQ: GPUBuffer | null = null
  private gpuKCache: GPUBuffer | null = null
  private gpuVCache: GPUBuffer | null = null
  private gpuOut: GPUBuffer | null = null
  private gpuParams: GPUBuffer | null = null
  private gpuRead: GPUBuffer | null = null

  // Character vocabulary for char-level models (vocab=45)
  private stoi: Record<string, number> = {}
  private itos: string[] = []
  private charset = ' abcdefghijklmnopqrstuvwxyz0123456789.,!?-\''

  /** Initialize WebGPU device and compile attention shader. */
  async init(): Promise<void> {
    if (this.device) return
    const a = await navigator.gpu?.requestAdapter()
    if (!a) throw new Error('WebGPU unavailable')
    this.device = await a.requestDevice()
    const mod = this.device.createShaderModule({ code: ATTENTION_SHADER })
    this.bgLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    })
    this.pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bgLayout] }),
      compute: { module: mod, entryPoint: 'attn_fused' },
    })
  }

  /** Load a .sou Transformer checkpoint.
   *
   * Expects flat p{i} parameters. The architecture must be provided
   * from inferArch() output.
   */
  async load(urlOrBuffer: string | ArrayBuffer, arch: SoulTransformerArch): Promise<SoulCheckpoint> {
    this.loading = true
    try {
      const raw = typeof urlOrBuffer === 'string'
        ? await fetch(urlOrBuffer).then(r => r.arrayBuffer())
        : urlOrBuffer
      const cp = parseSou(raw)
      this.metadata = cp.metadata
      this.arch = arch
      const { embedDim: e, numHeads: H, numKVHeads: KH, numLayers: L, dimFF: ff, vocabSize: v, maxSeqLen, eps } = arch
      this.maxSeqLen = maxSeqLen
      const headDim = e / H

      // Build char vocab from checkpoint metadata or default
      const cs = (cp.metadata?.charset as string) || this.charset.slice(0, v)
      this.stoi = {}
      this.itos = []
      for (let i = 0; i < Math.min(cs.length, v); i++) {
        this.stoi[cs[i]] = i
        this.itos[i] = cs[i]
      }
      // Fill any missing indices
      for (let i = cs.length; i < v; i++) {
        this.itos[i] = '\ufffd'
      }

      const w = cp.weights

      // Helper to get param by index
      const param = (i: number) => {
        const key = `p${i}` as const
        if (!w[key]) throw new Error(`Missing p${i} in checkpoint`)
        return w[key]!
      }

      const N = Object.keys(w).length
      // p0 = tok_emb.weight [vocab, embed]
      this.params = {
        embWeight: param(0),
        lmWeight: param(N - 1), // tied to tok_emb
        normWeight: param(N - 2), // final norm.weight
        layers: [],
      }

      // Each layer: p(1 + li*9) .. p(9 + li*9)
      // attn_norm(1), q(2), k(3), v(4), o(5), ff_norm(6), w1(7), w2(8), w3(9)
      let pi = 1
      for (let li = 0; li < L; li++) {
        this.params.layers.push({
          attnNorm: param(pi),
          wQ: param(pi + 1),
          wK: param(pi + 2),
          wV: param(pi + 3),
          wO: param(pi + 4),
          ffNorm: param(pi + 5),
          w1: param(pi + 6),
          w2: param(pi + 7),
          w3: param(pi + 8),
        })
        pi += 9
      }

      // Allocate KV cache (one contiguous buffer per layer)
      const cacheSize = maxSeqLen * KH * headDim
      for (let li = 0; li < L; li++) {
        this.kCache.push(new Float32Array(cacheSize))
        this.vCache.push(new Float32Array(cacheSize))
      }
      this.cacheLen = 0

      // GPU attention buffers
      const d = this.device!
      const maxCacheBytes = this.kCache[0].byteLength
      const headSize = e // n_heads * head_dim
      const mkBuf = (size: number) =>
        d.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC })

      this.gpuQ = mkBuf(headSize * 4)
      this.gpuKCache = mkBuf(maxCacheBytes)
      this.gpuVCache = mkBuf(maxCacheBytes)
      this.gpuOut = mkBuf(headSize * 4)
      this.gpuParams = mkBuf(32) // params struct: 8 × u32/f32
      this.gpuRead = d.createBuffer({
        size: headSize * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      })

      this.ready = true
      return cp
    } finally {
      this.loading = false
    }
  }

  /** Reset KV cache (call before starting a new sequence). */
  resetState(): void {
    this.cacheLen = 0
  }

  /** Run one transformer forward pass for a single token.
   *
   * CPU: embed → (rmsnorm, matmuls, RoPE, FFN, LM head)
   * GPU: fused multi-head attention
   */
  async forward(tokenId: number): Promise<Float32Array> {
    if (!this.arch || !this.params) throw new Error('Engine not ready')
    const { embedDim: e, numHeads: H, numKVHeads: KH, numLayers: L, dimFF: ff, vocabSize: v, eps } = this.arch
    const headDim = e / H
    const d = this.device!
    const params = this.params
    const pos = this.cacheLen

    // 1. Embedding gather
    let x: Float32Array = new Float32Array(e)
    const embOff = tokenId * e
    for (let i = 0; i < e; i++) x[i] = params.embWeight[embOff + i]

    // 2. Process each layer
    for (let li = 0; li < L; li++) {
      const layer = params.layers[li]

      // a. Pre-attention RMSNorm
      const xNorm = rmsnorm(x, layer.attnNorm, eps)

      // b. Q, K, V projections (CPU matmul)
      const Qraw = matMul(xNorm, layer.wQ, e, e)   // W_q: [e, e] × [e]
      const Kraw = matMul(xNorm, layer.wK, e, e)   // W_k: [e, e] × [e]
      const Vraw = matMul(xNorm, layer.wV, e, e)   // W_v: [e, e] × [e]

      // c. Apply RoPE to Q and K
      const Q = new Float32Array(e)
      const K = new Float32Array(e)
      // Reshape to [n_heads, head_dim] then apply per-dim RoPE pair
      for (let h = 0; h < H; h++) {
        for (let d2 = 0; d2 < headDim; d2 += 2) {
          const freq = 1 / Math.pow(10000, d2 / headDim)
          const theta = pos * freq
          const cos = Math.cos(theta)
          const sin = Math.sin(theta)
          const i0 = h * headDim + d2
          const i1 = h * headDim + d2 + 1
          Q[i0] = Qraw[i0] * cos - Qraw[i1] * sin
          Q[i1] = Qraw[i0] * sin + Qraw[i1] * cos
          // K: use same RoPE but with kv_head grouping
          const kh = Math.min(h, KH - 1) // GQA: group
          const ki0 = kh * headDim + d2
          const ki1 = kh * headDim + d2 + 1
          K[ki0] = Kraw[ki0] * cos - Kraw[ki1] * sin
          K[ki1] = Kraw[ki0] * sin + Kraw[ki1] * cos
        }
      }

      // d. Store K, V in cache at position pos
      const kOff = pos * KH * headDim
      const vOff = pos * KH * headDim
      for (let i = 0; i < KH * headDim; i++) {
        this.kCache[li][kOff + i] = K[i]
        this.vCache[li][vOff + i] = Vraw[i]
      }

      // e. GPU: fused attention
      const T = pos + 1 // cache length including current token
      const cacheBytes = this.kCache[li].byteLength

      // Upload Q, K_cache, V_cache to GPU (only upload current slice)
      const qSlice = Q.slice(0, e)
      d.queue.writeBuffer(this.gpuQ!, 0, qSlice.buffer, qSlice.byteOffset, qSlice.byteLength)
      d.queue.writeBuffer(this.gpuKCache!, 0, this.kCache[li].buffer, 0, cacheBytes)
      d.queue.writeBuffer(this.gpuVCache!, 0, this.vCache[li].buffer, 0, cacheBytes)

      // Write params struct
      const paramData = new Uint32Array([H, KH, headDim, T, 0, 0, 0])
      const paramF32 = new Float32Array(paramData.buffer)
      paramF32[4] = 1.0 / Math.sqrt(headDim) // scale
      d.queue.writeBuffer(this.gpuParams!, 0, paramF32.buffer)

      const bg = d.createBindGroup({
        layout: this.bgLayout!,
        entries: [
          { binding: 0, resource: { buffer: this.gpuParams! } },
          { binding: 1, resource: { buffer: this.gpuQ! } },
          { binding: 2, resource: { buffer: this.gpuKCache! } },
          { binding: 3, resource: { buffer: this.gpuVCache! } },
          { binding: 4, resource: { buffer: this.gpuOut! } },
        ],
      })

      const enc = d.createCommandEncoder()
      const pass = enc.beginComputePass()
      pass.setPipeline(this.pipeline!)
      pass.setBindGroup(0, bg)
      pass.dispatchWorkgroups(1) // 8 threads in workgroup, one per head
      pass.end()
      d.queue.submit([enc.finish()])
      await d.queue.onSubmittedWorkDone()

      // Read back attention output
      const renc = d.createCommandEncoder()
      renc.copyBufferToBuffer(this.gpuOut!, 0, this.gpuRead!, 0, e * 4)
      d.queue.submit([renc.finish()])
      await d.queue.onSubmittedWorkDone()
      await this.gpuRead!.mapAsync(GPUMapMode.READ)
      const attnArr = new Float32Array(this.gpuRead!.getMappedRange())
      const attnOut = new Float32Array(attnArr)
      this.gpuRead!.unmap()

      // f. Output projection
      const mhaOut = matMul(attnOut, layer.wO, e, e)

      // g. Residual
      for (let i = 0; i < e; i++) x[i] += mhaOut[i]

      // h. Pre-FFN RMSNorm
      const xNorm2 = rmsnorm(x, layer.ffNorm, eps)

      // i. SwiGLU FFN
      const w1_out = matMul(xNorm2, layer.w1, ff, e) // [dimFF]
      const w3_out = matMul(xNorm2, layer.w3, ff, e) // [dimFF]
      const gated = new Float32Array(ff)
      for (let i = 0; i < ff; i++) gated[i] = silu(w1_out[i]) * w3_out[i]
      const ffOut = matMul(gated, layer.w2, e, ff) // [embed]

      // j. Residual
      for (let i = 0; i < e; i++) x[i] += ffOut[i]
    }

    // 3. Final RMSNorm
    x = rmsnorm(x, params.normWeight, eps)

    // 4. LM head (tied weights) — logits = x @ embWeight^T
    const logits = new Float32Array(v)
    for (let i = 0; i < v; i++) {
      let s = 0
      for (let j = 0; j < e; j++) s += x[j] * params.lmWeight[i * e + j]
      logits[i] = s
    }

    // Advance cache
    this.cacheLen = pos + 1

    return logits
  }

  /** Generate text autoregressively. */
  async *generate(prompt: string, maxTokens: number, temperature = 1.0): AsyncGenerator<string, void, unknown> {
    const ids: number[] = []
    for (const ch of prompt.toLowerCase()) {
      if (ch in this.stoi) ids.push(this.stoi[ch])
    }
    if (ids.length === 0) ids.push(0)
    this.resetState()

    // Consume prompt silently
    for (const id of ids) await this.forward(id)

    // Generate new tokens
    for (let t = 0; t < maxTokens; t++) {
      const logits = await this.forward(ids[ids.length - 1])
      const nextId = temperature > 0 ? sampleMultinomial(logits, temperature) : sampleArgmax(logits)
      const token = this.itos[nextId] ?? '\ufffd'
      ids.push(nextId)
      yield token
    }
  }

  destroy(): void {
    for (const b of [this.gpuQ, this.gpuKCache, this.gpuVCache, this.gpuOut, this.gpuParams, this.gpuRead]) {
      b?.destroy()
    }
    this.device = null
    this.ready = false
  }
}

function sampleArgmax(l: Float32Array): number {
  let b = 0
  for (let i = 1; i < l.length; i++) if (l[i] > l[b]) b = i
  return b
}

function sampleMultinomial(l: Float32Array, t: number): number {
  let max = -Infinity
  for (let i = 0; i < l.length; i++) if (l[i] > max) max = l[i]
  let sum = 0
  const s = new Float32Array(l.length)
  for (let i = 0; i < l.length; i++) { s[i] = Math.exp((l[i] - max) / Math.max(t, 0.001)); sum += s[i] }
  let r = Math.random() * sum
  for (let i = 0; i < l.length; i++) { r -= s[i]; if (r <= 0) return i }
  return l.length - 1
}
