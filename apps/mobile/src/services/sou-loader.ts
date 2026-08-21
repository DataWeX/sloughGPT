/**
 * .sou file loader for mobile — pick, parse, and cache local .sou checkpoint files.
 *
 * Reuses the same binary .sou parser as the web engine, converts indexed params
 * (p0, p1, ...) to the flat layout expected by the onnx-inference-service forward pass,
 * and caches in AsyncStorage for offline reload.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

const SOU_MAGIC = new Uint8Array([0x53, 0x4f, 0x55, 0x4c]); // "SOUL"
const CACHE_KEY = '@sloughgpt/sou_file_cache';
const META_CACHE_KEY = '@sloughgpt/sou_file_meta';

// ── .sou binary parser (v3) ─────────────────────────────────────────────

interface SoulMetadata {
  version: number;
  soul_name: string;
  soul_traits: Record<string, number>;
  system_prompt: string;
  lineage: string;
  [key: string]: unknown;
}

interface SoulWeights {
  [key: `p${number}`]: Float32Array;
}

interface SoulCheckpoint {
  metadata: SoulMetadata;
  weights: SoulWeights;
  totalElements: number;
}

export interface SouArch {
  archType: 'lstm' | 'transformer';
  vocabSize: number;
  nEmbed: number;
  nLayer: number;
  nHead: number;
  dimFF: number;
}

function parseSou(buffer: ArrayBuffer): SoulCheckpoint {
  const view = new DataView(buffer);
  const magic = new Uint8Array(buffer, 0, 4);
  for (let i = 0; i < 4; i++) {
    if (magic[i] !== SOU_MAGIC[i]) {
      throw new Error('Invalid .sou magic');
    }
  }

  const version = view.getUint32(4, true);
  const jsonLen = view.getUint32(8, true);
  const metaStr = new TextDecoder().decode(new Uint8Array(buffer, 12, jsonLen));
  const metadata: SoulMetadata = JSON.parse(metaStr);

  let offset = 12 + jsonLen;
  const weights: SoulWeights = {};
  let totalElements = 0;

  if (version >= 3) {
    const numParams = view.getUint32(offset, true); offset += 4;
    for (let i = 0; i < numParams; i++) {
      const nameLen = view.getUint32(offset, true); offset += 4;
      const name = new TextDecoder().decode(new Uint8Array(buffer, offset, nameLen));
      offset += nameLen;
      const ndim = view.getUint32(offset, true); offset += 4;
      const shape: number[] = [];
      for (let d = 0; d < ndim; d++) {
        shape.push(view.getUint32(offset, true)); offset += 4;
      }
      const count = shape.reduce((a, b) => a * b, 1);
      while (offset % 4 !== 0) offset++;
      const arr = new Float32Array(buffer, offset, count);
      weights[name as `p${number}`] = new Float32Array(arr);
      totalElements += count;
      offset += count * 4;
    }
  } else {
    throw new Error(`Unsupported .sou version: ${version}`);
  }

  return { metadata, weights, totalElements };
}

function inferArch(buffer: ArrayBuffer): SouArch {
  const view = new DataView(buffer);
  const magic = new Uint8Array(buffer, 0, 4);
  for (let i = 0; i < 4; i++) {
    if (magic[i] !== SOU_MAGIC[i]) throw new Error('Invalid .sou magic');
  }
  const version = view.getUint32(4, true);
  const jsonLen = view.getUint32(8, true);
  let offset = 12 + jsonLen;

  const sizes: Record<string, number> = {};
  if (version >= 3) {
    const n = view.getUint32(offset, true); offset += 4;
    for (let i = 0; i < n; i++) {
      const nl = view.getUint32(offset, true); offset += 4;
      const name = new TextDecoder().decode(new Uint8Array(buffer, offset, nl));
      offset += nl;
      const ndim = view.getUint32(offset, true); offset += 4;
      const shape: number[] = [];
      for (let d = 0; d < ndim; d++) {
        shape.push(view.getUint32(offset, true)); offset += 4;
      }
      const count = shape.reduce((a, b) => a * b, 1);
      sizes[name] = count;
      while (offset % 4 !== 0) offset++;
      offset += count * 4;
    }
  } else {
    throw new Error(`Unsupported .sou version: ${version}`);
  }

  const N = Object.keys(sizes).length;
  if (N > 14) {
    const embedDim = Math.round(Math.sqrt(sizes['p2']!));
    const vocabSize = sizes['p0']! / embedDim;
    const numLayers = Math.floor((N - 3) / 9);
    return { archType: 'transformer', vocabSize, nEmbed: embedDim, nLayer: numLayers, nHead: embedDim / 64, dimFF: Math.ceil(embedDim * 8 / 3 / 64) * 64 };
  }
  const vocabSize = sizes[`p${N - 1}`]!;
  const hiddenDim = sizes[`p${N - 2}`]! / vocabSize;
  const embedDim = sizes['p0']! / vocabSize;
  let numLayers: number;
  if ((N - 4) % 4 === 0 && N > 4) {
    numLayers = (N - 4) / 4;
  } else {
    numLayers = Math.max(1, Math.round((N - 3) / 2));
  }
  return { archType: 'lstm', vocabSize, nEmbed: embedDim, nLayer: numLayers, nHead: 0, dimFF: 0 };
}

// ── Weight conversion: indexed p0..pN → flat layout for mobile engine ───

function convertToFlatLayout(cp: SoulCheckpoint, arch: SouArch): Float32Array {
  const w = cp.weights;
  const param = (i: number) => w[`p${i}` as const]!;

  const nEmb = arch.nEmbed;
  const nHead = arch.nHead || Math.min(8, nEmb);
  const dimFF = arch.dimFF || Math.ceil(nEmb * 8 / 3 / 64) * 64;
  const v = arch.vocabSize;
  const L = arch.nLayer;
  const N = Object.keys(w).length;

  if (arch.archType === 'transformer') {
    // Transformer layout mirrors _offsetWeights():
    // p0 = tok_emb, then per layer 9 params, then norm, lm_head
    const pieces: Float32Array[] = [];
    pieces.push(param(0)); // tok_emb.weight
    let pi = 1;
    for (let li = 0; li < L; li++) {
      pieces.push(param(pi));     // attn_norm
      pieces.push(param(pi + 1)); // q_proj
      pieces.push(param(pi + 2)); // k_proj
      pieces.push(param(pi + 3)); // v_proj
      pieces.push(param(pi + 4)); // o_proj
      pieces.push(param(pi + 5)); // ff_norm
      pieces.push(param(pi + 6)); // w1
      pieces.push(param(pi + 7)); // w2
      pieces.push(param(pi + 8)); // w3
      pi += 9;
    }
    pieces.push(param(N - 2)); // norm.weight
    pieces.push(param(N - 1)); // lm_head.weight

    let totalLen = 0;
    for (const p of pieces) totalLen += p.length;
    const flat = new Float32Array(totalLen);
    let off = 0;
    for (const p of pieces) {
      flat.set(p, off);
      off += p.length;
    }
    return flat;
  }

  // LSTM layout — handle all param-count variants
  const pieces: Float32Array[] = [];
  const isNewFormat = (N - 4) % 4 === 0 && N > 4;

  // tok_emb.weight
  pieces.push(param(0));

  if (isNewFormat) {
    // New format: p0=emb, p1=lstm_embed (skip), then per-layer W_ih/W_hh (+biases), then fc
    for (let li = 0; li < L; li++) {
      pieces.push(param(2 + li * 4)); // W_ih
      pieces.push(param(4 + li * 4)); // W_hh
    }
  } else if (N > 3) {
    // Old format without biases: p0=emb, p1=lstm_embed (skip), then per-layer W_ih/W_hh
    for (let li = 0; li < L; li++) {
      pieces.push(param(1 + li * 2)); // W_ih
      pieces.push(param(2 + li * 2)); // W_hh
    }
  }
  // else N <= 3: no LSTM layers (old format without LSTM, just embed → fc)

  // fc_out.weight, fc_out.bias
  pieces.push(param(N - 2));
  pieces.push(param(N - 1));

  let totalLen = 0;
  for (const p of pieces) totalLen += p.length;
  const flat = new Float32Array(totalLen);
  let off = 0;
  for (const p of pieces) {
    flat.set(p, off);
    off += p.length;
  }
  return flat;
}

function buildMobileConfig(arch: SouArch): {
  vocab_size: number; n_embed: number; n_layer: number; n_head: number; block_size: number;
} {
  return {
    vocab_size: arch.vocabSize,
    n_embed: arch.nEmbed,
    n_layer: arch.nLayer,
    n_head: arch.nHead || Math.min(8, arch.nEmbed),
    block_size: 128,
  };
}

// ── State ────────────────────────────────────────────────────────────────

let _loaded = false;
let _flatWeights: Float32Array | null = null;
let _config: { vocab_size: number; n_embed: number; n_layer: number; n_head: number; block_size: number } | null = null;
let _metadata: SoulMetadata | null = null;

// ── Public API ───────────────────────────────────────────────────────────

/** Load a .sou file from an ArrayBuffer (binary content). */
export async function loadFromSouFile(buffer?: ArrayBuffer): Promise<void> {
  if (!buffer) {
    // Try restoring from AsyncStorage cache
    const cached = await AsyncStorage.getItem(CACHE_KEY);
    const cachedMeta = await AsyncStorage.getItem(META_CACHE_KEY);
    if (cached && cachedMeta) {
      const raw = atob(cached);
      const arr = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i);
      const buf = arr.buffer;
      const arch = inferArch(buf);
      const cp = parseSou(buf);
      _flatWeights = convertToFlatLayout(cp, arch);
      _config = buildMobileConfig(arch);
      _metadata = JSON.parse(cachedMeta);
      _loaded = true;
      return;
    }
    throw new Error('No .sou buffer provided and no cached checkpoint available');
  }

  const arch = inferArch(buffer);
  const cp = parseSou(buffer);
  _flatWeights = convertToFlatLayout(cp, arch);
  _config = buildMobileConfig(arch);
  _metadata = cp.metadata;
  _loaded = true;

  // Cache to AsyncStorage (base64 encode the buffer)
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  await AsyncStorage.setItem(CACHE_KEY, btoa(binary));
  await AsyncStorage.setItem(META_CACHE_KEY, JSON.stringify(cp.metadata));
}

/** Pick a .sou file from device using expo-document-picker and load it. */
export async function pickAndLoadSou(): Promise<{ name: string; config: SouArch } | null> {
  let DocumentPicker: any;
  try {
    DocumentPicker = require('expo-document-picker');
  } catch {
    return null;
  }

  let FS: any;
  try {
    FS = require('expo-file-system');
  } catch {
    FS = null;
  }

  try {
    const result = await DocumentPicker.getDocumentAsync({
      type: '*/*',
      copyToCacheDirectory: true,
      multiple: false,
    });

    if (result.canceled || !result.assets?.[0]) return null;

    const asset = result.assets[0];
    if (!asset.name.endsWith('.sou')) return null;

    let raw: string;
    if (FS?.readAsStringAsync) {
      raw = await FS.readAsStringAsync(asset.uri, { encoding: 'base64' as any });
    } else {
      // Fallback: fetch the file URI as a blob, convert to base64
      const resp = await fetch(asset.uri);
      const blob = await resp.blob();
      const arrayBuf = await blob.arrayBuffer();
      const bytes = new Uint8Array(arrayBuf);
      let binary = '';
      for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
      raw = btoa(binary);
    }

    const binaryStr = atob(raw);
    const arr = new Uint8Array(binaryStr.length);
    for (let i = 0; i < binaryStr.length; i++) arr[i] = binaryStr.charCodeAt(i);
    const buffer = arr.buffer;

    const arch = inferArch(buffer);
    const cp = parseSou(buffer);
    _flatWeights = convertToFlatLayout(cp, arch);
    _config = buildMobileConfig(arch);
    _metadata = cp.metadata;
    _loaded = true;

    // Cache to AsyncStorage
    await AsyncStorage.setItem(CACHE_KEY, raw);
    await AsyncStorage.setItem(META_CACHE_KEY, JSON.stringify(cp.metadata));

    return { name: asset.name, config: arch };
  } catch {
    return null;
  }
}

/** Check if a .sou checkpoint is loaded. */
export function isLoaded(): boolean {
  return _loaded;
}

/** Unload the current .sou checkpoint. */
export function unloadSou(): void {
  _loaded = false;
  _flatWeights = null;
  _config = null;
  _metadata = null;
}

/** Get the loaded .sou metadata, or null if not loaded. */
export function souMetadata(): SoulMetadata | null {
  return _metadata;
}

/** Generate text using the loaded .sou model. */
export async function generateFromSou(
  prompt: string,
  maxNewTokens = 64,
  temperature = 0.8,
  topK = 40,
  topP = 0.9,
  eosToken = 0,
  onToken?: (token: string) => void,
): Promise<{ text: string; tokens_generated: number; elapsed_ms: number }> {
  if (!_loaded || !_flatWeights || !_config) throw new Error('No .sou checkpoint loaded');

  // Delegate to the existing onnx-inference-service forward pass
  // by importing it and loading our converted flat weights
  const onnx = require('./onnx-inference-service');

  // Inject our flat weights via the exported function
  onnx.loadFlatWeights(_config, _flatWeights);

  // Generate
  const result = await onnx.generate(prompt, maxNewTokens, temperature, topK, topP, eosToken, onToken);
  return result;
}
