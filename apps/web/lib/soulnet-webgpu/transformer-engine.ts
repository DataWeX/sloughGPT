/**
 * SoulTransformerWebGPU — browser-side Transformer inference engine.
 *
 * Architecture: decoder-only Transformer with RoPE, RMSNorm, SwiGLU FFN, KV cache
 * (see trained baby_step_X models: 384 embed, 8 heads, 6 layers, 1024 FFN, vocab=45).
 *
 * CPU: embedding, RMSNorm, matmuls (QKV proj, FFN, LM head), RoPE, sampling
 * GPU: fused multi-head attention (Q×K^T → softmax → ×V), fused SwiGLU FFN
 */

import { parseSou, type SoulCheckpoint, type SoulMetadata, type SoulTransformerArch } from './weights'
import { WeightCache } from './cache'
import { logger } from '@/lib/dev-log'

const _weightCache = new WeightCache()

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

const FFN_GATED_SHADER = `
struct P { embed_dim: u32, dim_ff: u32, _pad: u32, __pad: u32 };
@group(0) @binding(0) var<storage, read> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w1: array<f32>;
@group(0) @binding(3) var<storage, read> w3: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

fn silu_gpu(v: f32) -> f32 { return v / (1.0 + exp(-v)); }

@compute @workgroup_size(256)
fn ffn_gated(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  let E = p.embed_dim;
  let F = p.dim_ff;
  if (i >= F) { return; }
  var h1 = 0.0;
  var h3 = 0.0;
  for (var j = 0u; j < E; j++) {
    h1 += w1[i * E + j] * x[j];
    h3 += w3[i * E + j] * x[j];
  }
  out[i] = silu_gpu(h1) * h3;
}
`

const FFN_OUT_SHADER = `
struct P { embed_dim: u32, dim_ff: u32, _pad: u32, __pad: u32 };
@group(0) @binding(0) var<storage, read> p: P;
@group(0) @binding(1) var<storage, read> h: array<f32>;
@group(0) @binding(2) var<storage, read> w2: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn ffn_out(@builtin(global_invocation_id) id: vec3<u32>) {
  let j = id.x;
  let E = p.embed_dim;
  let F = p.dim_ff;
  if (j >= E) { return; }
  var s = 0.0;
  for (var i = 0u; i < F; i++) {
    s += w2[j * F + i] * h[i];
  }
  out[j] = s;
}
`

const MATMUL_SHADER = `
struct P { out_dim: u32, in_dim: u32, _pad: u32, __pad: u32 };
@group(0) @binding(0) var<storage, read> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn matmul_kernel(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  let M = p.out_dim;
  let K = p.in_dim;
  if (i >= M) { return; }
  var s = 0.0;
  for (var j = 0u; j < K; j++) {
    s += x[j] * w[i * K + j];
  }
  out[i] = s;
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
  private ffnGatedPipeline: GPUComputePipeline | null = null
  private ffnOutPipeline: GPUComputePipeline | null = null
  private ffnBgLayout: GPUBindGroupLayout | null = null
  private matmulPipeline: GPUComputePipeline | null = null
  private matmulBgLayout: GPUBindGroupLayout | null = null

  private arch: SoulTransformerArch | null = null
  metadata: SoulMetadata | null = null
  ready = false
  loading = false
  cpuOnly = false

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

  // GPU FFN buffers (per-layer, allocated in load)
  private gpuW1: GPUBuffer[] = []
  private gpuW3: GPUBuffer[] = []
  private gpuW2: GPUBuffer[] = []
  private gpuFfnH: GPUBuffer | null = null
  private gpuFfnOut: GPUBuffer | null = null
  private gpuFfnParams: GPUBuffer | null = null

  // GPU matmul buffers (reused across QKV/output/LM head)
  private gpuMatmulIn: GPUBuffer | null = null
  private gpuMatmulW: GPUBuffer | null = null
  private gpuMatmulOut: GPUBuffer | null = null
  private gpuMatmulParams: GPUBuffer | null = null

  // Character vocabulary for char-level models (vocab=45)
  private stoi: Record<string, number> = {}
  private itos: string[] = []
  private charset = ' abcdefghijklmnopqrstuvwxyz0123456789.,!?-\''

  /** Initialize WebGPU device and compile attention shader.
   *  Falls back to CPU-only mode if WebGPU is unavailable. */
  async init(): Promise<void> {
    if (this.device || this.cpuOnly) return
    const a = await navigator.gpu?.requestAdapter()
    if (!a) {
      this.cpuOnly = true
      return
    }
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

    // FFN gated pipeline (W1 + W3 + silu fused)
    const ffnGatedMod = this.device.createShaderModule({ code: FFN_GATED_SHADER })
    this.ffnBgLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    })
    this.ffnGatedPipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.ffnBgLayout] }),
      compute: { module: ffnGatedMod, entryPoint: 'ffn_gated' },
    })

    // FFN output projection pipeline (W2 matmul)
    const ffnOutMod = this.device.createShaderModule({ code: FFN_OUT_SHADER })
    const ffnOutBgLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    })
    this.ffnOutPipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [ffnOutBgLayout] }),
      compute: { module: ffnOutMod, entryPoint: 'ffn_out' },
    })

    // Matmul pipeline (QKV projections, output projection, LM head)
    const matmulMod = this.device.createShaderModule({ code: MATMUL_SHADER })
    this.matmulBgLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    })
    this.matmulPipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.matmulBgLayout] }),
      compute: { module: matmulMod, entryPoint: 'matmul_kernel' },
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
      let raw: ArrayBuffer
      if (typeof urlOrBuffer === 'string') {
        const cached = await _weightCache.get(urlOrBuffer)
        if (cached) {
          raw = cached
        } else {
          const resp = await fetch(urlOrBuffer)
          if (!resp.ok) throw new Error(`HTTP ${resp.status} from ${urlOrBuffer}`)
          raw = await resp.arrayBuffer()
          _weightCache.put(urlOrBuffer, raw).catch(e => { logger.debug('weight cache put failed', { url: String(urlOrBuffer), exception: String(e) }) })
        }
      } else {
        raw = urlOrBuffer
      }
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

      if (!this.cpuOnly && this.device) {
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

        // GPU FFN buffers (per-layer W1, W3, W2 + shared intermediates)
        const ffnParamsSize = 16 // 2 × u32 + padding
        this.gpuFfnParams = mkBuf(ffnParamsSize)
        this.gpuFfnH = mkBuf(ff * 4)     // [dimFF] intermediate
        this.gpuFfnOut = mkBuf(e * 4)    // [embed] output
      for (let li = 0; li < L; li++) {
        const layer = this.params.layers[li]
        this.gpuW1.push(mkBuf(layer.w1.byteLength))
        this.gpuW3.push(mkBuf(layer.w3.byteLength))
        this.gpuW2.push(mkBuf(layer.w2.byteLength))
        d.queue.writeBuffer(this.gpuW1[li], 0, layer.w1.buffer, layer.w1.byteOffset, layer.w1.byteLength)
        d.queue.writeBuffer(this.gpuW3[li], 0, layer.w3.buffer, layer.w3.byteOffset, layer.w3.byteLength)
        d.queue.writeBuffer(this.gpuW2[li], 0, layer.w2.buffer, layer.w2.byteOffset, layer.w2.byteLength)
      }

      // GPU matmul buffers (reused across QKV/output/LM head)
      this.gpuMatmulIn = mkBuf(e * 4)           // [embed] input vector
      this.gpuMatmulW = mkBuf(e * e * 4)        // [embed, embed] weight (max size for QKV/output)
      this.gpuMatmulOut = mkBuf(Math.max(e, v) * 4) // [max(embed, vocab)] output
      this.gpuMatmulParams = mkBuf(16)          // params struct: [out_dim, in_dim, _pad, __pad]
      } // end GPU buffer allocation

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

  /** CPU matmul: out = x @ W^T. */
  private cpuMatMul(x: Float32Array, w: Float32Array, outDim: number, inDim: number): Float32Array {
    const out = new Float32Array(outDim)
    for (let i = 0; i < outDim; i++) {
      let s = 0
      for (let j = 0; j < inDim; j++) s += x[j] * w[i * inDim + j]
      out[i] = s
    }
    return out
  }

  /** CPU multi-head attention. */
  private cpuAttention(Q: Float32Array, K: Float32Array, V: Float32Array,
    H: number, KH: number, headDim: number, T: number): Float32Array {
    const e = H * headDim
    const scale = 1 / Math.sqrt(headDim)
    const out = new Float32Array(e)
    // For each head
    for (let h = 0; h < H; h++) {
      const kh = Math.min(h, KH - 1)
      // Compute attention scores: Q[h,:] @ K[all T, kh]^T * scale
      const scores = new Float32Array(T)
      for (let t = 0; t < T; t++) {
        let s = 0
        for (let d = 0; d < headDim; d++) {
          s += Q[h * headDim + d] * K[t * KH * headDim + kh * headDim + d]
        }
        scores[t] = s * scale
      }
      // Softmax
      let maxVal = -Infinity
      for (let t = 0; t < T; t++) if (scores[t] > maxVal) maxVal = scores[t]
      let sumExp = 0
      for (let t = 0; t < T; t++) { scores[t] = Math.exp(scores[t] - maxVal); sumExp += scores[t] }
      for (let t = 0; t < T; t++) scores[t] /= sumExp
      // Weighted sum: out[h,:] = sum_t scores[t] * V[t, kh, :]
      for (let d = 0; d < headDim; d++) {
        let s = 0
        for (let t = 0; t < T; t++) s += scores[t] * V[t * KH * headDim + kh * headDim + d]
        out[h * headDim + d] = s
      }
    }
    return out
  }

  /** CPU SwiGLU FFN: h = silu(W1 @ x) * (W3 @ x); out = W2 @ h. */
  private cpuSwiGLU(x: Float32Array, w1: Float32Array, w2: Float32Array, w3: Float32Array,
    e: number, ff: number): Float32Array {
    const h = new Float32Array(ff)
    for (let i = 0; i < ff; i++) {
      let s1 = 0, s3 = 0
      for (let j = 0; j < e; j++) {
        s1 += x[j] * w1[i * e + j]
        s3 += x[j] * w3[i * e + j]
      }
      h[i] = (s1 / (1 + Math.exp(-Math.min(Math.max(s1, -500), 500)))) * s3
    }
    const out = new Float32Array(e)
    for (let j = 0; j < e; j++) {
      let s = 0
      for (let i = 0; i < ff; i++) s += h[i] * w2[j * ff + i]
      out[j] = s
    }
    return out
  }

  /** GPU matmul: out = x @ W^T. Reuses pre-allocated buffers. */
  private async gpuMatmul(x: Float32Array, w: Float32Array, outDim: number, inDim: number): Promise<Float32Array> {
    const d = this.device!
    // Write input and weight
    d.queue.writeBuffer(this.gpuMatmulIn!, 0, x.buffer, x.byteOffset, x.byteLength)
    d.queue.writeBuffer(this.gpuMatmulW!, 0, w.buffer, w.byteOffset, w.byteLength)
    // Write params: [out_dim, in_dim, _pad, __pad]
    const paramData = new Uint32Array([outDim, inDim, 0, 0])
    d.queue.writeBuffer(this.gpuMatmulParams!, 0, paramData.buffer)
    // Dispatch
    const bg = d.createBindGroup({
      layout: this.matmulBgLayout!,
      entries: [
        { binding: 0, resource: { buffer: this.gpuMatmulParams! } },
        { binding: 1, resource: { buffer: this.gpuMatmulIn! } },
        { binding: 2, resource: { buffer: this.gpuMatmulW! } },
        { binding: 3, resource: { buffer: this.gpuMatmulOut! } },
      ],
    })
    const enc = d.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(this.matmulPipeline!)
    pass.setBindGroup(0, bg)
    pass.dispatchWorkgroups(Math.ceil(outDim / 256))
    pass.end()
    d.queue.submit([enc.finish()])
    await d.queue.onSubmittedWorkDone()
    // Read back
    const renc = d.createCommandEncoder()
    renc.copyBufferToBuffer(this.gpuMatmulOut!, 0, this.gpuRead!, 0, outDim * 4)
    d.queue.submit([renc.finish()])
    await d.queue.onSubmittedWorkDone()
    await this.gpuRead!.mapAsync(GPUMapMode.READ)
    const arr = new Float32Array(this.gpuRead!.getMappedRange())
    const result = new Float32Array(outDim)
    for (let i = 0; i < outDim; i++) result[i] = arr[i]
    this.gpuRead!.unmap()
    return result
  }

  /** Run one transformer forward pass for a single token.
   *
   * CPU-only path when no GPU available: all matmuls, attention, FFN on CPU.
   * GPU path: attention + FFN fused, matmuls via GPU compute shaders.
   */
  async forward(tokenId: number): Promise<Float32Array> {
    if (!this.arch || !this.params) throw new Error('Engine not ready')
    const { embedDim: e, numHeads: H, numKVHeads: KH, numLayers: L, dimFF: ff, vocabSize: v, eps } = this.arch
    const headDim = e / H
    const params = this.params
    const pos = this.cacheLen

    // 1. Embedding gather
    let x: Float32Array = new Float32Array(e)
    const embOff = tokenId * e
    for (let i = 0; i < e; i++) x[i] = params.embWeight[embOff + i]

    if (this.cpuOnly) {
      // CPU-only forward pass
      for (let li = 0; li < L; li++) {
        const layer = params.layers[li]
        const xNorm = rmsnorm(x, layer.attnNorm, eps)
        const Qraw = this.cpuMatMul(xNorm, layer.wQ, e, e)
        const Kraw = this.cpuMatMul(xNorm, layer.wK, e, e)
        const Vraw = this.cpuMatMul(xNorm, layer.wV, e, e)
        const Q = new Float32Array(e)
        const K = new Float32Array(e)
        for (let h = 0; h < H; h++) {
          for (let d = 0; d < headDim; d += 2) {
            const freq = 1 / Math.pow(10000, d / headDim)
            const theta = pos * freq
            const cosv = Math.cos(theta), sinv = Math.sin(theta)
            const i0 = h * headDim + d, i1 = h * headDim + d + 1
            Q[i0] = Qraw[i0] * cosv - Qraw[i1] * sinv
            Q[i1] = Qraw[i0] * sinv + Qraw[i1] * cosv
            const kh = Math.min(h, KH - 1)
            const ki0 = kh * headDim + d, ki1 = kh * headDim + d + 1
            K[ki0] = Kraw[ki0] * cosv - Kraw[ki1] * sinv
            K[ki1] = Kraw[ki0] * sinv + Kraw[ki1] * cosv
          }
        }
        const kOff = pos * KH * headDim
        for (let i = 0; i < KH * headDim; i++) {
          this.kCache[li][kOff + i] = K[i]
          this.vCache[li][kOff + i] = Vraw[i]
        }
        const T = pos + 1
        const attnOut = this.cpuAttention(Q, this.kCache[li], this.vCache[li], H, KH, headDim, T)
        const mhaOut = this.cpuMatMul(attnOut, layer.wO, e, e)
        for (let i = 0; i < e; i++) x[i] += mhaOut[i]
        const xNorm2 = rmsnorm(x, layer.ffNorm, eps)
        const ffOut = this.cpuSwiGLU(xNorm2, layer.w1, layer.w2, layer.w3, e, ff)
        for (let i = 0; i < e; i++) x[i] += ffOut[i]
      }
      x = rmsnorm(x, params.normWeight, eps)
      const logits = this.cpuMatMul(x, params.lmWeight, v, e)
      this.cacheLen = pos + 1
      return logits
    }

    // GPU forward pass
    const d = this.device!

    // 2. Process each layer
    for (let li = 0; li < L; li++) {
      const layer = params.layers[li]

      // a. Pre-attention RMSNorm
      const xNorm = rmsnorm(x, layer.attnNorm, eps)

      // b. Q, K, V projections (GPU matmul)
      const Qraw = await this.gpuMatmul(xNorm, layer.wQ, e, e)
      const Kraw = await this.gpuMatmul(xNorm, layer.wK, e, e)
      const Vraw = await this.gpuMatmul(xNorm, layer.wV, e, e)

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

      // f. Output projection (GPU matmul)
      const mhaOut = await this.gpuMatmul(attnOut, layer.wO, e, e)

      // g. Residual
      for (let i = 0; i < e; i++) x[i] += mhaOut[i]

      // h. Pre-FFN RMSNorm
      const xNorm2 = rmsnorm(x, layer.ffNorm, eps)

      // i. GPU: fused SwiGLU FFN
      // Upload xNorm2 for gated kernel
      d.queue.writeBuffer(this.gpuFfnH!, 0, xNorm2.buffer, xNorm2.byteOffset, xNorm2.byteLength)
      // Write FFN params struct: [embed_dim, dim_ff, _pad, __pad]
      const ffnParamData = new Uint32Array([e, ff, 0, 0])
      d.queue.writeBuffer(this.gpuFfnParams!, 0, ffnParamData.buffer)

      // Stage 1: gated activation — h[i] = silu(W1[i,:] @ x) * (W3[i,:] @ x)
      const ffnGatedBg = d.createBindGroup({
        layout: this.ffnBgLayout!,
        entries: [
          { binding: 0, resource: { buffer: this.gpuFfnParams! } },
          { binding: 1, resource: { buffer: this.gpuFfnH! } },
          { binding: 2, resource: { buffer: this.gpuW1[li] } },
          { binding: 3, resource: { buffer: this.gpuW3[li] } },
          { binding: 4, resource: { buffer: this.gpuFfnH! } },
        ],
      })
      const ffnEnc = d.createCommandEncoder()
      const ffnPass = ffnEnc.beginComputePass()
      ffnPass.setPipeline(this.ffnGatedPipeline!)
      ffnPass.setBindGroup(0, ffnGatedBg)
      ffnPass.dispatchWorkgroups(Math.ceil(ff / 256))
      ffnPass.end()
      d.queue.submit([ffnEnc.finish()])
      await d.queue.onSubmittedWorkDone()

      // Stage 2: output projection — out[j] = W2[j,:] @ h
      d.queue.writeBuffer(this.gpuFfnOut!, 0, new Float32Array(e).buffer)
      const ffnOutBg = d.createBindGroup({
        layout: this.ffnBgLayout!,
        entries: [
          { binding: 0, resource: { buffer: this.gpuFfnParams! } },
          { binding: 1, resource: { buffer: this.gpuFfnH! } },
          { binding: 2, resource: { buffer: this.gpuW2[li] } },
          { binding: 3, resource: { buffer: this.gpuFfnOut! } },
        ],
      })
      const ffnOutEnc = d.createCommandEncoder()
      const ffnOutPass = ffnOutEnc.beginComputePass()
      ffnOutPass.setPipeline(this.ffnOutPipeline!)
      ffnOutPass.setBindGroup(0, ffnOutBg)
      ffnOutPass.dispatchWorkgroups(Math.ceil(e / 256))
      ffnOutPass.end()
      d.queue.submit([ffnOutEnc.finish()])
      await d.queue.onSubmittedWorkDone()

      // Read back FFN output
      const ffnReadEnc = d.createCommandEncoder()
      ffnReadEnc.copyBufferToBuffer(this.gpuFfnOut!, 0, this.gpuRead!, 0, e * 4)
      d.queue.submit([ffnReadEnc.finish()])
      await d.queue.onSubmittedWorkDone()
      await this.gpuRead!.mapAsync(GPUMapMode.READ)
      const ffnArr = new Float32Array(this.gpuRead!.getMappedRange())
      const ffOut = new Float32Array(ffnArr)
      this.gpuRead!.unmap()

      // j. Residual
      for (let i = 0; i < e; i++) x[i] += ffOut[i]
    }

    // 3. Final RMSNorm
    x = rmsnorm(x, params.normWeight, eps)

    // 4. LM head (tied weights) — logits = x @ embWeight^T (GPU matmul)
    const logits = await this.gpuMatmul(x, params.lmWeight, v, e)

    // Advance cache
    this.cacheLen = pos + 1

    return logits
  }

  /** Generate text autoregressively.
   *  @param eosToken - stop generating when this token ID is produced (0 = no stop)
   */
  async *generate(prompt: string, maxTokens: number, temperature = 1.0, eosToken = 0): AsyncGenerator<string, void, unknown> {
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
      if (eosToken > 0 && nextId === eosToken) return
      const token = this.itos[nextId] ?? '\ufffd'
      ids.push(nextId)
      yield token
    }
  }

  destroy(): void {
    for (const b of [this.gpuQ, this.gpuKCache, this.gpuVCache, this.gpuOut, this.gpuParams, this.gpuRead,
      this.gpuFfnParams, this.gpuFfnH, this.gpuFfnOut, ...this.gpuW1, ...this.gpuW3, ...this.gpuW2,
      this.gpuMatmulIn, this.gpuMatmulW, this.gpuMatmulOut, this.gpuMatmulParams]) {
      b?.destroy()
    }
    this.device?.destroy()
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
