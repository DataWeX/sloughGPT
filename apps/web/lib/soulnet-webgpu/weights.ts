/**
 * SoulNet .sou weight file parser.
 *
 * ## Canonical v3 Binary Format (source of truth)
 *
 * ```
 * [4 bytes] SOU_MAGIC = b"SOUL"
 * [4 bytes] version (uint32 LE, =3)
 * [4 bytes] json_len (uint32 LE)
 * [json_len bytes] JSON metadata (UTF-8)
 * [4 bytes] num_params (uint32 LE) = N
 * For each param i in 0..N-1:
 *   [4 bytes] name_len (uint32 LE)
 *   [name_len bytes] key name (UTF-8, e.g. "p0")
 *   [4 bytes] ndim (uint32 LE)
 *   [ndim * 4 bytes] shape dimensions (uint32 LE each)
 *   [product(shape) * 4 bytes] raw float32 data (little-endian)
 * ```
 *
 * Parameter order convention (LSTM):
 *   Old format (7 params, no biases): p0=embed, p1+p2=(W_ih,W_hh) per layer, pN-2=fc_w, pN-1=fc_b
 *   New format (8/12 params, with biases): p0=embed, p1=lstm_embed, then per layer:
 *     p2+li*4=W_ih_w, p3+li*4=W_ih_b, p4+li*4=W_hh_w, p5+li*4=W_hh_b
 *     pN-2=fc_w, pN-1=fc_b
 *
 * Invariant: last param is always fc_bias (vocab size),
 * second-to-last is always fc_weight (vocab × hidden).
 *
 * ## v2 Format (legacy, JSON weights — no longer produced)
 * ```
 * [4 bytes] SOU_MAGIC
 * [4 bytes] version (uint32 LE, =2)
 * [4 bytes] json_len
 * [json_len bytes] JSON metadata
 * [4 bytes] weights_json_len
 * [weights_json_len bytes] JSON weights { "p0": [...], "p1": [...], ... }
 * ```
 */

const SOU_MAGIC = new Uint8Array([0x53, 0x4f, 0x55, 0x4c])

export interface SoulMetadata {
  version: number
  soul_name: string
  soul_traits: Record<string, number>
  system_prompt: string
  lineage: string
  step?: number
  [key: string]: unknown
}

export interface SoulWeights {
  [key: `p${number}`]: Float32Array
}

export interface SoulCheckpoint {
  metadata: SoulMetadata
  weights: SoulWeights
  totalElements: number
}

export interface SoulNetArch {
  embedDim: number
  hiddenDim: number
  vocabSize: number
  numLayers: number
  archType: 'lstm' | 'transformer'
}

export interface SoulTransformerArch {
  archType: 'transformer'
  embedDim: number
  numHeads: number
  numKVHeads: number
  numLayers: number
  dimFF: number
  vocabSize: number
  maxSeqLen: number
  eps: number
}

/** Infer SoulNet architecture from .sou weight shapes — no config needed.

    Heuristic: vocab size from last param length, hidden dim from second-last,
    embed dim from first param. LSTM vs transformer detected by param count.

    @param buffer - raw .sou file bytes
    @returns architecture details (embedDim, hiddenDim, vocabSize, numLayers, archType)
*/
export function inferArch(buffer: ArrayBuffer): SoulNetArch {
  const view = new DataView(buffer)
  const magic = new Uint8Array(buffer, 0, 4)
  for (let i = 0; i < 4; i++) {
    if (magic[i] !== SOU_MAGIC[i]) throw new Error('Invalid .sou magic')
  }
  const version = view.getUint32(4, true)
  const jsonLen = view.getUint32(8, true)
  // Data starts immediately after JSON metadata
  let offset = 12 + jsonLen

  function readSizes(): Record<string, number> {
    const sizes: Record<string, number> = {}
    if (version >= 3) {
      const n = view.getUint32(offset, true); offset += 4
      for (let i = 0; i < n; i++) {
        const nl = view.getUint32(offset, true); offset += 4
        const name = new TextDecoder().decode(new Uint8Array(buffer, offset, nl))
        offset += nl
        const ndim = view.getUint32(offset, true); offset += 4
        const shape: number[] = []
        for (let d = 0; d < ndim; d++) {
          shape.push(view.getUint32(offset, true)); offset += 4
        }
        const count = shape.reduce((a, b) => a * b, 1)
        sizes[name] = count
        offset += count * 4
      }
    } else {
      const wl = view.getUint32(offset, true); offset += 4
      const raw = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, offset, wl)))
      for (const [k, v] of Object.entries(raw)) sizes[k] = (v as number[]).length
    }
    return sizes
  }

  const sizes = readSizes()
  const keys = Object.keys(sizes)
  const N = keys.length

  if (N > 14) {
    const embedDim = Math.round(Math.sqrt(sizes['p2']!))
    const vocabSize = sizes['p0']! / embedDim
    const numLayers = Math.floor((N - 3) / 9)
    return { embedDim, hiddenDim: embedDim, vocabSize, numLayers, archType: 'transformer' }
  }

  const vocabSize = sizes[`p${N - 1}`]!
  const hiddenDim = sizes[`p${N - 2}`]! / vocabSize
  const embedDim = sizes['p0']! / vocabSize

  let numLayers: number
  if ((N - 4) % 4 === 0 && N > 4) {
    numLayers = (N - 4) / 4
  } else {
    numLayers = Math.max(1, Math.round((N - 3) / 2))
  }

  return { embedDim, hiddenDim, vocabSize, numLayers, archType: 'lstm' }
}

/** Parse a .sou file into metadata + flat weight arrays.

    Supports v2 (JSON weights) and v3 (binary float32) formats.
    Handles 4-byte alignment for Float32Array offsets (WebGPU requirement).

    @param buffer - raw .sou file bytes
    @returns parsed checkpoint with metadata and weights dict
*/
export function parseSou(buffer: ArrayBuffer): SoulCheckpoint {
  const view = new DataView(buffer)
  const magic = new Uint8Array(buffer, 0, 4)
  for (let i = 0; i < 4; i++) {
    if (magic[i] !== SOU_MAGIC[i]) {
      throw new Error(`Invalid .sou magic: expected SOUL, got ${String.fromCharCode(...magic)}`)
    }
  }

  const version = view.getUint32(4, true)
  const jsonLen = view.getUint32(8, true)
  const metaStr = new TextDecoder().decode(new Uint8Array(buffer, 12, jsonLen))
  const metadata: SoulMetadata = JSON.parse(metaStr)

  // Data starts immediately after JSON metadata (no padding in v3 format).
  // Individual Float32Array views handle alignment internally.
  let offset = 12 + jsonLen

  const weights: SoulWeights = {}
  let totalElements = 0

  if (version >= 3) {
    const numParams = view.getUint32(offset, true); offset += 4
    for (let i = 0; i < numParams; i++) {
      const nameLen = view.getUint32(offset, true); offset += 4
      const name = new TextDecoder().decode(new Uint8Array(buffer, offset, nameLen))
      offset += nameLen
      const ndim = view.getUint32(offset, true); offset += 4
      const shape: number[] = []
      for (let d = 0; d < ndim; d++) {
        shape.push(view.getUint32(offset, true)); offset += 4
      }
      const count = shape.reduce((a, b) => a * b, 1)
      const arr = new Float32Array(buffer, offset, count)
      weights[name as `p${number}`] = new Float32Array(arr)
      totalElements += count
      offset += count * 4
    }
  } else {
    const weightsJsonLen = view.getUint32(offset, true); offset += 4
    if (weightsJsonLen > 0) {
      const weightsStr = new TextDecoder().decode(new Uint8Array(buffer, offset, weightsJsonLen))
      const rawWeights: Record<string, number[]> = JSON.parse(weightsStr)
      for (const [key, arr] of Object.entries(rawWeights)) {
        weights[key as `p${number}`] = new Float32Array(arr)
        totalElements += arr.length
      }
    }
  }

  return { metadata, weights, totalElements }
}

export function guessShapes(
  weights: SoulWeights,
  embedDim: number,
  hiddenDim: number,
  vocabSize: number,
  numLayers: number,
): Record<string, { shape: number[]; data: Float32Array }> {
  /** Map flat param arrays (p0..pN) to named shapes for engine consumption.

      Known keys: embed.weight, lstm.W_ih, lstm.W_hh, fc_out.weight, fc_out.bias.

      @param weights - flat param dict from parseSou
      @param embedDim - embedding dimension
      @param hiddenDim - LSTM hidden dimension
      @param vocabSize - vocabulary size
      @param numLayers - number of LSTM layers
      @returns record of { shape, data } for each known layer
  */
  const result: Record<string, { shape: number[]; data: Float32Array }> = {}
  let idx = 0

  const param = (i: number) => weights[`p${i}` as const]

  if (param(0)) {
    result['embed.weight'] = { shape: [vocabSize, embedDim], data: param(0) }
    idx++
  }

  if (param(idx)) {
    result['lstm.W_ih'] = { shape: [4 * hiddenDim, embedDim], data: param(idx) }
    idx++
  }

  if (param(idx)) {
    result['lstm.W_hh'] = { shape: [4 * hiddenDim, hiddenDim], data: param(idx) }
    idx++
  }

  if (numLayers > 1 && param(idx)) {
    result['lstm.W_ih2'] = { shape: [4 * hiddenDim, embedDim], data: param(idx) }
    idx++
  }

  if (numLayers > 1 && param(idx)) {
    result['lstm.W_hh2'] = { shape: [4 * hiddenDim, hiddenDim], data: param(idx) }
    idx++
  }

  if (param(idx)) {
    result['fc_out.weight'] = { shape: [vocabSize, hiddenDim], data: param(idx) }
    idx++
  }

  if (param(idx)) {
    result['fc_out.bias'] = { shape: [vocabSize], data: param(idx) }
    idx++
  }

  return result
}
