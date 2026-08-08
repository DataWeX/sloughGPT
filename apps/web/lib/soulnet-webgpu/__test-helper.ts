/**
 * Shared test helper for lib/soulnet-webgpu tests.
 *
 * Provides:
 * - makeSou / makeSouV2 — binary .sou builders (v3 + v2)
 * - makeLstmSou / makeTransformerSou — standard-format checkpoints
 * - makeWebGPU — stubs navigator.gpu + WebGPU constants with a fake device
 * - FakeWorker + stubWorker — for SoulEngineWorker protocol tests
 * - installFakeIndexedDB — in-memory IndexedDB for cache.ts
 */

import { vi } from 'vitest'

export interface SouParam {
  name: string
  shape: number[]
  data: Float32Array
}

const SOU_MAGIC_BYTES = new Uint8Array([0x53, 0x4f, 0x55, 0x4c])

export function makeSou(metadata: Record<string, unknown>, params: SouParam[]): ArrayBuffer {
  const enc = new TextEncoder()
  const metaBytes = enc.encode(JSON.stringify(metadata))
  const chunks: Uint8Array[] = []
  chunks.push(SOU_MAGIC_BYTES)
  const header = new Uint8Array(8)
  const dv = new DataView(header.buffer)
  dv.setUint32(0, 3, true)
  dv.setUint32(4, metaBytes.length, true)
  chunks.push(header, metaBytes)
  const nBuf = new Uint8Array(4)
  new DataView(nBuf.buffer).setUint32(0, params.length, true)
  chunks.push(nBuf)
  for (const p of params) {
    const nameBytes = enc.encode(p.name)
    const nl = new Uint8Array(4)
    new DataView(nl.buffer).setUint32(0, nameBytes.length, true)
    chunks.push(nl, nameBytes)
    const ndim = new Uint8Array(4)
    new DataView(ndim.buffer).setUint32(0, p.shape.length, true)
    chunks.push(ndim)
    for (const s of p.shape) {
      const sb = new Uint8Array(4)
      new DataView(sb.buffer).setUint32(0, s, true)
      chunks.push(sb)
    }
    const soFar = chunks.reduce((a, c) => a + c.byteLength, 0)
    const pad = (4 - (soFar % 4)) % 4
    if (pad) chunks.push(new Uint8Array(pad))
    chunks.push(new Uint8Array(p.data.buffer, p.data.byteOffset, p.data.byteLength))
  }
  const total = chunks.reduce((a, c) => a + c.byteLength, 0)
  const out = new Uint8Array(total)
  let off = 0
  for (const c of chunks) {
    out.set(c, off)
    off += c.byteLength
  }
  return out.buffer
}

export function makeSouV2(metadata: Record<string, unknown>, weightsJson: string): ArrayBuffer {
  const enc = new TextEncoder()
  const meta = enc.encode(JSON.stringify(metadata))
  const wjson = enc.encode(weightsJson)
  const out = new Uint8Array(4 + 8 + meta.length + 4 + wjson.length)
  const dv = new DataView(out.buffer)
  out.set(SOU_MAGIC_BYTES, 0)
  dv.setUint32(4, 2, true)
  dv.setUint32(8, meta.length, true)
  out.set(meta, 12)
  let off = 12 + meta.length
  dv.setUint32(off, wjson.length, true)
  off += 4
  out.set(wjson, off)
  return out.buffer
}

const META = { version: 3, soul_name: 'test-soul', soul_traits: {}, system_prompt: '', lineage: 'unit' }

/** New-format LSTM checkpoint: p0 embed, p1 lstm_embed, 4 params/layer, fc_w, fc_b. */
export function makeLstmSou(
  opts: { e: number; h: number; v: number; nl: number },
  overrides: Record<string, Float32Array> = {},
): ArrayBuffer {
  const { e, h, v, nl } = opts
  const params: SouParam[] = [
    { name: 'p0', shape: [v, e], data: new Float32Array(v * e) },
    { name: 'p1', shape: [v, e], data: new Float32Array(v * e) },
  ]
  for (let li = 0; li < nl; li++) {
    const base = 2 + li * 4
    params.push({ name: `p${base}`, shape: [4 * h, e], data: new Float32Array(4 * h * e) })
    params.push({ name: `p${base + 1}`, shape: [4 * h], data: new Float32Array(4 * h) })
    params.push({ name: `p${base + 2}`, shape: [4 * h, h], data: new Float32Array(4 * h * h) })
    params.push({ name: `p${base + 3}`, shape: [4 * h], data: new Float32Array(4 * h) })
  }
  const fcBase = 2 + nl * 4
  params.push({ name: `p${fcBase}`, shape: [v, h], data: new Float32Array(v * h) })
  params.push({ name: `p${fcBase + 1}`, shape: [v], data: new Float32Array(v) })
  for (const [key, data] of Object.entries(overrides)) {
    const p = params.find(x => x.name === key)
    if (p) p.data = data
  }
  return makeSou(META, params)
}

/** Standard transformer checkpoint: p0 emb, 9 params/layer, final norm, lm. */
export function makeTransformerSou(
  opts: { e: number; L: number; ff: number; v: number },
  overrides: Record<string, Float32Array> = {},
): ArrayBuffer {
  const { e, L, ff, v } = opts
  const params: SouParam[] = [{ name: 'p0', shape: [v, e], data: new Float32Array(v * e) }]
  let pi = 1
  for (let li = 0; li < L; li++) {
    params.push({ name: `p${pi}`, shape: [e], data: new Float32Array(e) })
    params.push({ name: `p${pi + 1}`, shape: [e, e], data: new Float32Array(e * e) })
    params.push({ name: `p${pi + 2}`, shape: [e, e], data: new Float32Array(e * e) })
    params.push({ name: `p${pi + 3}`, shape: [e, e], data: new Float32Array(e * e) })
    params.push({ name: `p${pi + 4}`, shape: [e, e], data: new Float32Array(e * e) })
    params.push({ name: `p${pi + 5}`, shape: [e], data: new Float32Array(e) })
    params.push({ name: `p${pi + 6}`, shape: [ff, e], data: new Float32Array(ff * e) })
    params.push({ name: `p${pi + 7}`, shape: [e, ff], data: new Float32Array(e * ff) })
    params.push({ name: `p${pi + 8}`, shape: [ff, e], data: new Float32Array(ff * e) })
    pi += 9
  }
  params.push({ name: `p${pi}`, shape: [e], data: new Float32Array(e) })
  params.push({ name: `p${pi + 1}`, shape: [v, e], data: new Float32Array(v * e) })
  for (const [key, data] of Object.entries(overrides)) {
    const p = params.find(x => x.name === key)
    if (p) p.data = data
  }
  return makeSou(META, params)
}

export interface FakeDevice {
  device: {
    createShaderModule: ReturnType<typeof vi.fn>
    createBindGroupLayout: ReturnType<typeof vi.fn>
    createPipelineLayout: ReturnType<typeof vi.fn>
    createComputePipeline: ReturnType<typeof vi.fn>
    createBindGroup: ReturnType<typeof vi.fn>
    createBuffer: ReturnType<typeof vi.fn>
    createCommandEncoder: ReturnType<typeof vi.fn>
    queue: {
      writeBuffer: ReturnType<typeof vi.fn>
      submit: ReturnType<typeof vi.fn>
      onSubmittedWorkDone: ReturnType<typeof vi.fn>
    }
  }
  buffers: Array<{
    size: number
    _data: ArrayBuffer
    destroy: ReturnType<typeof vi.fn>
    mapAsync: ReturnType<typeof vi.fn>
    getMappedRange: ReturnType<typeof vi.fn>
    unmap: ReturnType<typeof vi.fn>
  }>
  writeBuffer: ReturnType<typeof vi.fn>
}

/** Stub navigator.gpu + WebGPU globals with a fake device. readback = GPU output bytes. */
export function makeWebGPU(readback?: Float32Array): FakeDevice {
  const buffers: FakeDevice['buffers'] = []
  const writeBuffer = vi.fn(
    (buf: { _data: ArrayBuffer }, offset: number, data: ArrayBuffer, dataOffset = 0, size = data.byteLength) => {
      new Uint8Array(buf._data).set(new Uint8Array(data, dataOffset, size), offset)
    },
  )
  const device: FakeDevice['device'] = {
    createShaderModule: vi.fn(() => ({})),
    createBindGroupLayout: vi.fn(() => ({})),
    createPipelineLayout: vi.fn(() => ({})),
    createComputePipeline: vi.fn(() => ({ getBindGroupLayout: vi.fn(() => ({})) })),
    createBindGroup: vi.fn(() => ({})),
    createBuffer: vi.fn(({ size }: { size: number }) => {
      const buf: FakeDevice['buffers'][number] = {
        size,
        _data: new ArrayBuffer(size),
        destroy: vi.fn(),
        mapAsync: vi.fn(async () => {}),
        getMappedRange: vi.fn(() => {
          new Uint8Array(buf._data).fill(0)
          if (readback) new Float32Array(buf._data).set(readback)
          return buf._data
        }),
        unmap: vi.fn(),
      }
      buffers.push(buf)
      return buf
    }),
    createCommandEncoder: vi.fn(() => ({
      beginComputePass: vi.fn(() => ({
        setPipeline: vi.fn(),
        setBindGroup: vi.fn(),
        dispatchWorkgroups: vi.fn(),
        end: vi.fn(),
      })),
      copyBufferToBuffer: vi.fn(),
      finish: vi.fn(() => ({})),
    })),
    queue: {
      writeBuffer,
      submit: vi.fn(),
      onSubmittedWorkDone: vi.fn(async () => {}),
    },
  }
  const adapter = { requestDevice: vi.fn(async () => device) }
  vi.stubGlobal('navigator', { gpu: { requestAdapter: vi.fn(async () => adapter) } })
  vi.stubGlobal('GPUShaderStage', { COMPUTE: 4 })
  vi.stubGlobal('GPUBufferUsage', { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, MAP_READ: 8 })
  vi.stubGlobal('GPUMapMode', { READ: 1 })
  return { device, buffers, writeBuffer }
}

export class FakeWorker {
  onmessage: ((e: { data: Record<string, unknown> }) => void) | null = null
  postMessage = vi.fn((msg: Record<string, unknown>) => {
    this.messages.push(msg)
  })
  terminate = vi.fn(() => {
    this.terminated = true
  })
  terminated = false
  messages: Record<string, unknown>[] = []

  emit(data: Record<string, unknown>): void {
    this.onmessage?.({ data })
  }
}

/** Stub the global Worker constructor; returns a getter for the latest instance. */
export function stubWorker(): () => FakeWorker | null {
  let current: FakeWorker | null = null
  vi.stubGlobal(
    'Worker',
    vi.fn(() => {
      current = new FakeWorker()
      return current
    }),
  )
  return () => current
}

function makeIdbRequest(result: unknown) {
  const req: {
    result: unknown
    error: unknown
    onsuccess: (() => void) | null
    onerror: (() => void) | null
    onupgradeneeded: (() => void) | null
  } = { result, error: null, onsuccess: null, onerror: null, onupgradeneeded: null }
  setTimeout(() => {
    req.onsuccess?.()
  }, 0)
  return req
}

/** In-memory IndexedDB with the subset of APIs cache.ts uses. */
export function installFakeIndexedDB(): {
  open: ReturnType<typeof vi.fn>
  store: Map<string, unknown>
} {
  const store = new Map<string, unknown>()
  const db: Record<string, unknown> = {
    createObjectStore: vi.fn(() => ({})),
    transaction: vi.fn(() => {
      const tx: {
        objectStore: ReturnType<typeof vi.fn>
        oncomplete: (() => void) | null
        onerror: (() => void) | null
      } = {
        objectStore: vi.fn(() => ({
          get: vi.fn((key: string) => makeIdbRequest(store.get(key) ?? null)),
          put: vi.fn((value: unknown, key: string) => {
            store.set(key, value)
            return makeIdbRequest(undefined)
          }),
          clear: vi.fn(() => {
            store.clear()
            return makeIdbRequest(undefined)
          }),
        })),
        oncomplete: null,
        onerror: null,
      }
      setTimeout(() => {
        tx.oncomplete?.()
      }, 0)
      return tx
    }),
  }
  const open = vi.fn(() => {
    const req = makeIdbRequest(db)
    return req
  })
  vi.stubGlobal('indexedDB', { open })
  return { open, store }
}
