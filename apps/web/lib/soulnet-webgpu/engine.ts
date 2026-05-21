/**
 * SoulNetWebGPU — hybrid CPU/GPU inference engine for SoulNet LSTM.
 *
 * CPU: embedding, fc_out, softmax
 * GPU: LSTM cell forward (the heavy matmul)
 *
 * Weight file format: v3 binary .sou (see weights.ts for the canonical spec).
 *
 * Usage:
 *   const engine = new SoulNetWebGPU()
 *   await engine.init()
 *   await engine.load('/models/friendly.sou', {
 *     embedDim: 256, hiddenDim: 512, vocabSize: 50, numLayers: 2
 *   })
 *   for await (const chunk of engine.generate('hello', 100)) {
 *     text += chunk
 *   }
 */

import { parseSou, SoulCheckpoint, SoulMetadata } from './weights'

const LSTM_SHADER = `
struct P { embed_dim: u32, hidden_dim: u32, input_dim: u32, _pad1: u32; };

@group(0) @binding(0) var<storage, read> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> h_prev: array<f32>;
@group(0) @binding(3) var<storage, read> c_prev: array<f32>;
@group(0) @binding(4) var<storage, read> W_ih: array<f32>;
@group(0) @binding(5) var<storage, read> W_hh: array<f32>;
@group(0) @binding(6) var<storage, read_write> h_new: array<f32>;
@group(0) @binding(7) var<storage, read_write> c_new: array<f32>;

fn sig(v: f32) -> f32 { return 1.0 / (1.0 + exp(-clamp(v, -500.0, 500.0))); }

@compute @workgroup_size(256)
fn lstm_cell(@builtin(global_invocation_id) id: vec3<u32>) {
  let j = id.x; let h = p.hidden_dim; let d = p.input_dim;
  if (j >= h) { return; }
  var gi = 0.0; var gf = 0.0; var gg = 0.0; var go = 0.0;
  for (var k = 0u; k < d; k++) {
    let xk = x[k];
    gi += W_ih[j * d + k] * xk;
    gf += W_ih[(h + j) * d + k] * xk;
    gg += W_ih[(2*h + j) * d + k] * xk;
    go += W_ih[(3*h + j) * d + k] * xk;
  }
  for (var k = 0u; k < h; k++) {
    let hpk = h_prev[k];
    gi += W_hh[j * h + k] * hpk;
    gf += W_hh[(h + j) * h + k] * hpk;
    gg += W_hh[(2*h + j) * h + k] * hpk;
    go += W_hh[(3*h + j) * h + k] * hpk;
  }
  let c = sig(gf) * c_prev[j] + sig(gi) * tanh(gg);
  c_new[j] = c;
  h_new[j] = sig(go) * tanh(c);
}
`

export interface SoulNetConfig {
  embedDim: number
  hiddenDim: number
  vocabSize: number
  numLayers: number
  charset?: string
}

const DEFAULT_CHARSET = " abcdefghijklmnopqrstuvwxyz0123456789.,!?-'"
function buildVocab(charset: string): { stoi: Record<string, number>; itos: string[] } {
  const stoi: Record<string, number> = {}
  const itos: string[] = []
  for (let i = 0; i < charset.length; i++) { stoi[charset[i]] = i; itos[i] = charset[i] }
  return { stoi, itos }
}

export class SoulNetWebGPU {
  private device: GPUDevice | null = null
  private pipeline: GPUComputePipeline | null = null

  // GPU weight buffers (persistent)
  private wIh: GPUBuffer[] = []            // per layer: [4*h, e]
  private wHh: GPUBuffer[] = []            // per layer: [4*h, h]
  private wFc: GPUBuffer | null = null     // [vocab, h]
  private bFc: GPUBuffer | null = null     // [vocab]

  // GPU state buffers (updated each step)
  private stateH: GPUBuffer[] = []         // per layer
  private stateC: GPUBuffer[] = []
  private bufIn: GPUBuffer | null = null   // input to current layer
  private bufHOut: GPUBuffer | null = null // output from current layer
  private bufCOut: GPUBuffer | null = null
  private bufParams: GPUBuffer | null = null
  private bufRead: GPUBuffer | null = null   // reusable staging buffer for readback

  // CPU-side copies (for embedding + fc_out ops done on CPU)
  private cpuEmb: Float32Array | null = null
  private cpuFcW: Float32Array | null = null
  private cpuFcB: Float32Array | null = null

  private cfg: SoulNetConfig | null = null
  metadata: SoulMetadata | null = null
  ready = false
  loading = false
  private stoi: Record<string, number> = {}
  private itos: string[] = []

  /** Initialize WebGPU adapter, device, and compute pipeline.

      Idempotent — safe to call multiple times.

      @throws if WebGPU is unavailable
  */
  async init(): Promise<void> {
    if (this.device) return
    const a = await navigator.gpu?.requestAdapter()
    if (!a) throw new Error('WebGPU unavailable')
    this.device = await a.requestDevice()
    const m = this.device.createShaderModule({ code: LSTM_SHADER })
    const gl = this.device.createBindGroupLayout({
      entries: Array.from({ length: 8 }, (_, i) => ({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: i >= 6 ? 'storage' : 'read-only-storage' },
      })),
    })
    this.pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [gl] }),
      compute: { module: m, entryPoint: 'lstm_cell' },
    })
  }

  /** Load a .sou checkpoint and configure for inference.

      Accepts either a URL (fetched automatically) or pre-loaded ArrayBuffer.
      Auto-detects old (7-param) vs new (8/12-param) format and w model type.

      @param urlOrBuffer - model URL or raw .sou bytes
      @param cfg - model architecture (embedDim, hiddenDim, vocabSize, numLayers)
      @returns parsed checkpoint metadata
      @throws if weights are incompatible with the engine
  */
  async load(urlOrBuffer: string | ArrayBuffer, cfg: SoulNetConfig): Promise<SoulCheckpoint> {
    this.loading = true
    try {
      const raw = typeof urlOrBuffer === 'string' ? await fetch(urlOrBuffer).then(r => r.arrayBuffer()) : urlOrBuffer
      const cp = parseSou(raw)
      this.metadata = cp.metadata; this.cfg = cfg
      const { stoi, itos } = buildVocab(cfg.charset ?? DEFAULT_CHARSET)
      this.stoi = stoi; this.itos = itos
      const w = cp.weights
      const p = (i: number) => w[`p${i}` as const]
      const d = this.device!
      const { embedDim: e, hiddenDim: h, vocabSize: v, numLayers: nl } = cfg

      // CPU copies: embed, fc_weight, fc_bias
      const wKeys = Object.keys(w)
      const N = wKeys.length
      // Embedding: use p0, but if p1 has the same size (new format with lstm_embed),
      // the LSTM forward pass uses p1 internally — so prefer p1
      const sameAsP0 = p(1)?.length === p(0).length
      this.cpuEmb = sameAsP0 ? p(1) : p(0)
      this.cpuFcW = p(N - 2)
      this.cpuFcB = p(N - 1)

      const mk = (data: Float32Array, usage = GPUBufferUsage.STORAGE): GPUBuffer => {
        const b = d.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC })
        d.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength); return b
      }

      // Persist LSTM weights (handles new format with biases + old format without)
      const isNewFormat = (N - 4) % 4 === 0 && N > 4
      for (let li = 0; li < nl; li++) {
        const ihIdx = isNewFormat ? 2 + li * 4 : 1 + li * 2
        const hhIdx = isNewFormat ? 4 + li * 4 : 2 + li * 2
        this.wIh.push(mk(p(ihIdx)))
        this.wHh.push(mk(p(hhIdx)))
      }

      this.wFc = mk(this.cpuFcW)
      this.bFc = mk(this.cpuFcB)

      // State buffers (one pair per layer)
      for (let li = 0; li < nl; li++) {
        this.stateH.push(mk(new Float32Array(h)))
        this.stateC.push(mk(new Float32Array(h)))
      }

      // Temp in/out buffers
      this.bufIn = mk(new Float32Array(e))
      this.bufHOut = mk(new Float32Array(h))
      this.bufCOut = mk(new Float32Array(h))

      // Params: [embed_dim, hidden_dim, input_dim, _pad1]
      this.bufParams = mk(new Float32Array(new Uint32Array([e, h, e, 0]).buffer))

      this.ready = true
      return cp
    } finally { this.loading = false }
  }

  /** Reset LSTM hidden and cell state to zero. */
  resetState(): void {
    if (!this.cfg) return
    const d = this.device!; const h = this.cfg.hiddenDim; const zero = new Float32Array(h)
    for (let li = 0; li < this.cfg.numLayers; li++) {
      d.queue.writeBuffer(this.stateH[li], 0, zero.buffer, zero.byteOffset, zero.byteLength)
      d.queue.writeBuffer(this.stateC[li], 0, zero.buffer, zero.byteOffset, zero.byteLength)
    }
  }

  private _makeBG(inBuf: GPUBuffer, hPrev: GPUBuffer, cPrev: GPUBuffer,
    wIh: GPUBuffer, wHh: GPUBuffer, hOut: GPUBuffer, cOut: GPUBuffer): GPUBindGroup {
    return this.device!.createBindGroup({
      layout: this.pipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.bufParams! } },
        { binding: 1, resource: { buffer: inBuf } },
        { binding: 2, resource: { buffer: hPrev } },
        { binding: 3, resource: { buffer: cPrev } },
        { binding: 4, resource: { buffer: wIh } },
        { binding: 5, resource: { buffer: wHh } },
        { binding: 6, resource: { buffer: hOut } },
        { binding: 7, resource: { buffer: cOut } },
      ],
    })
  }

  /** Forward: tokenId → logits (on CPU Float32Array). */
  /** Run one forward pass through the LSTM.

      CPU: embedding lookup → GPU: LSTM cell → CPU: fc_out + softmax.

      @param tokenId - vocabulary ID of the input token
      @returns logits (vocab-sized Float32Array) for sampling
  */
  async forward(tokenId: number): Promise<Float32Array> {
    if (!this.device || !this.cfg) throw new Error('Engine not ready')
    const d = this.device!; const cfg = this.cfg

    // 1. CPU: embedding gather
    const e = cfg.embedDim; const h = cfg.hiddenDim
    const embed = new Float32Array(e)
    const off = tokenId * e
    for (let i = 0; i < e; i++) embed[i] = this.cpuEmb![off + i]
    d.queue.writeBuffer(this.bufIn!, 0, embed.buffer, embed.byteOffset, embed.byteLength)

    // 2. For each LSTM layer: dispatch GPU shader + copy state in one encoder
    for (let li = 0; li < cfg.numLayers; li++) {
      const inputDim = li === 0 ? e : h
      d.queue.writeBuffer(this.bufParams!, 8, new Uint32Array([inputDim]).buffer)
      const bg = this._makeBG(
        li === 0 ? this.bufIn! : this.bufHOut!,
        this.stateH[li], this.stateC[li],
        this.wIh[li], this.wHh[li],
        this.bufHOut!, this.bufCOut!,
      )
      const enc = d.createCommandEncoder()
      const p = enc.beginComputePass()
      p.setPipeline(this.pipeline!)
      p.setBindGroup(0, bg)
      p.dispatchWorkgroups(Math.ceil(h / 256))
      p.end()
      enc.copyBufferToBuffer(this.bufHOut!, 0, this.stateH[li], 0, h * 4)
      enc.copyBufferToBuffer(this.bufCOut!, 0, this.stateC[li], 0, h * 4)
      d.queue.submit([enc.finish()])
      await d.queue.onSubmittedWorkDone()
    }

    // 3. Read back final h_out (reusable staging buffer)
    if (!this.bufRead) {
      this.bufRead = d.createBuffer({ size: h * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    }
    const renc = d.createCommandEncoder()
    renc.copyBufferToBuffer(this.bufHOut!, 0, this.bufRead, 0, h * 4)
    d.queue.submit([renc.finish()])
    await this.bufRead.mapAsync(GPUMapMode.READ)
    const hArr = new Float32Array(this.bufRead.getMappedRange())
    const hVec = new Float32Array(hArr)
    this.bufRead.unmap()

    // 4. CPU: fc_out = h @ W_fc^T + b_fc
    const v = cfg.vocabSize
    const logits = new Float32Array(v)
    for (let i = 0; i < v; i++) {
      let s = this.cpuFcB![i]
      for (let j = 0; j < h; j++) s += hVec[j] * this.cpuFcW![i * h + j]
      logits[i] = s
    }
    return logits
  }

  /** Generate text token-by-token via autoregressive sampling.

      Consumes prompt silently (no output), then yields new tokens.
      State is maintained across the full generation.

      @param prompt - input string
      @param maxTokens - max tokens to generate
      @param temperature - sampling temperature (0 = argmax, higher = more random)
      @yields each generated character
  */
  async *generate(prompt: string, maxTokens: number, temperature = 1.0): AsyncGenerator<string, void, unknown> {
    const ids: number[] = []
    for (const ch of prompt.toLowerCase()) { if (ch in this.stoi) ids.push(this.stoi[ch]) }
    if (ids.length === 0) ids.push(this.stoi[' '] ?? 0)
    this.resetState()
    // Consume prompt silently to initialize LSTM state
    for (const id of ids) await this.forward(id)
    // Generate new tokens autoregressively
    for (let t = 0; t < maxTokens; t++) {
      const logits = await this.forward(ids[ids.length - 1])
      const nextId = temperature > 0 ? sampleMultinomial(logits, temperature) : sampleArgmax(logits)
      const token = this.itos[nextId] ?? '?'
      ids.push(nextId)
      yield token
    }
  }

  destroy(): void {
    for (const b of [this.bufIn, this.bufHOut, this.bufCOut, this.bufParams, this.bufRead,
      ...this.wIh, ...this.wHh, this.wFc, this.bFc, ...this.stateH, ...this.stateC])
      b?.destroy()
    this.device = null; this.ready = false
  }
}

function sampleArgmax(l: Float32Array): number { let b = 0; for (let i = 1; i < l.length; i++) if (l[i] > l[b]) b = i; return b }

function sampleMultinomial(l: Float32Array, t: number): number {
  let max = -Infinity; for (let i = 0; i < l.length; i++) if (l[i] > max) max = l[i]
  let sum = 0; const s = new Float32Array(l.length)
  for (let i = 0; i < l.length; i++) { s[i] = Math.exp((l[i] - max) / Math.max(t, 0.001)); sum += s[i] }
  let r = Math.random() * sum; for (let i = 0; i < l.length; i++) { r -= s[i]; if (r <= 0) return i }
  return l.length - 1
}
