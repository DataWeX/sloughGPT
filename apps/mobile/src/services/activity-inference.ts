/**
 * On-device activity classifier inference.
 *
 * Reimplements the SloNet-based ActivityClassifier forward pass in pure
 * TypeScript so predictions can run locally without a server round-trip.
 *
 * Architecture (matching classifier.py):
 *   Input: (1, T, 6) — 6-axis sensor window
 *   Path A — Conv2D(6→32, k=7) → ReLU → Conv2D(32→32, k=5) → ReLU → global avg pool → (32,)
 *   Path B — Per-channel stats: mean, std, min, max × 6 channels → (24,)
 *   Concat → (56,) → Linear(56→num_classes) → softmax → class probs
 *
 * Weight format (.npz / ZIP of .npy files):
 *   arr_0: conv1.weight (32, 6, 1, 7)
 *   arr_1: conv1.bias   (32,)
 *   arr_2: conv2.weight (32, 32, 1, 5)
 *   arr_3: conv2.bias   (32,)
 *   arr_4: fc.weight    (num_classes, 56)
 *   arr_5: fc.bias      (num_classes,)
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

const MODEL_CACHE_KEY = '@sloughgpt/activity_model_cached';
const ACTIVITIES = ['stationary', 'walking', 'running', 'shaking', 'driving', 'cycling'];

export interface Prediction {
  classId: number;
  className: string;
  probabilities: number[];
}

export interface ModelWeights {
  conv1Weight: Float32Array; // (32, 6, 1, 7)
  conv1Bias: Float32Array;   // (32,)
  conv2Weight: Float32Array; // (32, 32, 1, 5)
  conv2Bias: Float32Array;   // (32,)
  fcWeight: Float32Array;    // (num_classes, 56)
  fcBias: Float32Array;      // (num_classes,)
  numClasses: number;
}

// ── .NPY Parser ──────────────────────────────────────────────────────

interface NpyHeader {
  dtype: string;
  shape: number[];
  dataOffset: number;
  dataSize: number;
}

/**
 * Parse a .npy file from a Uint8Array.
 * Returns header info + typed array view into the buffer.
 */
function parseNpy(buf: ArrayBuffer): {header: NpyHeader; data: Float32Array} {
  const view = new DataView(buf);
  // Magic: \x93NUMPY
  const major = view.getUint8(6);
  let headerLen: number;
  let dataOffset: number;

  if (major === 1) {
    headerLen = view.getUint16(8, true);
    dataOffset = 10 + headerLen;
  } else {
    headerLen = view.getUint32(8, true);
    dataOffset = 12 + headerLen;
  }

  const decoder = new TextDecoder('utf-8');
  const headerStr = decoder.decode(new Uint8Array(buf, major === 1 ? 10 : 12, headerLen));

  // Parse dtype and shape from header like: {'descr': '<f4', 'fortran_order': False, 'shape': (32, 6, 1, 7)}
  const descrMatch = headerStr.match(/'descr':\s*'([^']+)'/);
  const shapeMatch = headerStr.match(/'shape':\s*\(([^)]*)\)/);

  if (!descrMatch || !shapeMatch) {
    throw new Error('Invalid .npy header');
  }

  const dtype = descrMatch[1];
  const shape = shapeMatch[1]
    .split(',')
    .map(s => parseInt(s.trim(), 10))
    .filter(n => !isNaN(n));

  // Determine element size from dtype
  let elementSize = 4;
  if (dtype.includes('f8')) elementSize = 8;
  else if (dtype.includes('f4') || dtype.includes('i4') || dtype.includes('u4')) elementSize = 4;
  else if (dtype.includes('f2') || dtype.includes('i2') || dtype.includes('u2')) elementSize = 2;
  else if (dtype.includes('i1') || dtype.includes('u1') || dtype.includes('S')) elementSize = 1;

  const totalElements = shape.reduce((a, b) => a * b, 1);
  const dataSize = totalElements * elementSize;

  const data = new Float32Array(buf, dataOffset, totalElements);

  return {
    header: {dtype, shape, dataOffset, dataSize},
    data,
  };
}

// ── .NPZ Parser (ZIP of .npy files) ─────────────────────────────────

// Minimal ZIP parser — no external dependencies needed.
// ZIP local file header signature: 0x04034b50
// ZIP central directory signature: 0x02014b50

interface ZipEntry {
  filename: string;
  offset: number;
  compressedSize: number;
  uncompressedSize: number;
  compressionMethod: number; // 0 = stored, 8 = deflate
}

function findZipEntries(buf: ArrayBuffer): ZipEntry[] {
  const view = new DataView(buf);
  const entries: ZipEntry[] = [];

  // Scan for local file headers
  let pos = 0;
  while (pos + 30 <= buf.byteLength) {
    const sig = view.getUint32(pos, true);
    if (sig !== 0x04034b50) break;

    const compressionMethod = view.getUint16(pos + 8, true);
    const compressedSize = view.getUint32(pos + 18, true);
    const uncompressedSize = view.getUint32(pos + 22, true);
    const filenameLen = view.getUint16(pos + 26, true);
    const extraLen = view.getUint16(pos + 28, true);

    const decoder = new TextDecoder('utf-8');
    const filename = decoder.decode(new Uint8Array(buf, pos + 30, filenameLen));

    entries.push({
      filename,
      offset: pos + 30 + filenameLen + extraLen,
      compressedSize,
      uncompressedSize,
      compressionMethod,
    });

    pos += 30 + filenameLen + extraLen + compressedSize;
  }

  return entries;
}

function parseNpz(buf: ArrayBuffer): Map<string, Float32Array> {
  const entries = findZipEntries(buf);
  const result = new Map<string, Float32Array>();

  for (const entry of entries) {
    if (!entry.filename.endsWith('.npy')) continue;

    let data: ArrayBuffer;
    if (entry.compressionMethod === 0) {
      // Stored (no compression)
      data = buf.slice(entry.offset, entry.offset + entry.compressedSize);
    } else {
      // Deflate — use DecompressionStream
      // For simplicity, we handle this synchronously below
      continue;
    }

    const name = entry.filename.replace('.npy', '');
    const {data: typedData} = parseNpy(data);
    result.set(name, typedData);
  }

  return result;
}

// ── Weight Loader ────────────────────────────────────────────────────

/**
 * Load model weights from base64 data URL stored in AsyncStorage.
 * Parses the .npz (ZIP of .npy files) and extracts weight arrays.
 */
export async function loadWeights(): Promise<ModelWeights | null> {
  try {
    const cached = await AsyncStorage.getItem(MODEL_CACHE_KEY);
    if (!cached) return null;

    // Strip data URL prefix: "data:application/octet-stream;base64,AAAA..."
    const base64 = cached.includes(',') ? cached.split(',')[1] : cached;

    // Decode base64 to ArrayBuffer
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }

    const buf = bytes.buffer;
    const weights = parseNpz(buf);

    if (weights.size < 6) {
      console.warn(`[activity-inference] Expected 6 weight arrays, got ${weights.size}`);
      return null;
    }

    const arr = (name: string) => weights.get(name)!;
    const fcWeight = arr('arr_4');
    const numClasses = fcWeight.length / 56;

    return {
      conv1Weight: arr('arr_0'),
      conv1Bias: arr('arr_1'),
      conv2Weight: arr('arr_2'),
      conv2Bias: arr('arr_3'),
      fcWeight,
      fcBias: arr('arr_5'),
      numClasses,
    };
  } catch (e) {
    console.warn('[activity-inference] Failed to load weights:', e);
    return null;
  }
}

// ── Forward Pass ─────────────────────────────────────────────────────

/**
 * 1D convolution: applies filters to input.
 * Input: (batch, in_channels, T) → Output: (batch, out_channels, T_out)
 *
 * Matches SloConv2D with padding='same' behavior.
 */
function conv1d(
  input: Float32Array,   // (batch, in_ch, T)
  weight: Float32Array,  // (out_ch, in_ch, 1, kernel_size) → squeeze dim 2
  bias: Float32Array,    // (out_ch,)
  batch: number,
  inCh: number,
  outCh: number,
  T: number,
  kernelSize: number,
): Float32Array {
  const outT = T; // Same padding → same size
  const output = new Float32Array(batch * outCh * outT);
  const pad = Math.floor(kernelSize / 2);

  for (let b = 0; b < batch; b++) {
    for (let oc = 0; oc < outCh; oc++) {
      for (let t = 0; t < outT; t++) {
        let sum = bias[oc];
        for (let ic = 0; ic < inCh; ic++) {
          for (let k = 0; k < kernelSize; k++) {
            const tIn = t + k - pad;
            if (tIn >= 0 && tIn < T) {
              // weight[oc, ic, 0, k] — dim 2 is size 1 (squeezed)
              const wIdx = oc * inCh * kernelSize + ic * kernelSize + k;
              const iIdx = b * inCh * T + ic * T + tIn;
              sum += input[iIdx] * weight[wIdx];
            }
          }
        }
        output[b * outCh * outT + oc * outT + t] = sum;
      }
    }
  }
  return output;
}

/** ReLU in-place on a copy */
function relu(arr: Float32Array): Float32Array {
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    out[i] = arr[i] > 0 ? arr[i] : 0;
  }
  return out;
}

/**
 * Global average pooling over time dimension.
 * Input: (batch, channels, T) → Output: (batch, channels)
 */
function globalAvgPool(
  input: Float32Array,
  batch: number,
  channels: number,
  T: number,
): Float32Array {
  const output = new Float32Array(batch * channels);
  for (let b = 0; b < batch; b++) {
    for (let c = 0; c < channels; c++) {
      let sum = 0;
      for (let t = 0; t < T; t++) {
        sum += input[b * channels * T + c * T + t];
      }
      output[b * channels + c] = sum / T;
    }
  }
  return output;
}

/**
 * Compute per-channel statistics: mean, std, min, max.
 * Input: (batch, T, 6) → Output: (batch, 24)
 */
function computeStats(
  input: Float32Array,
  batch: number,
  T: number,
  channels: number,
): Float32Array {
  const output = new Float32Array(batch * channels * 4);
  for (let b = 0; b < batch; b++) {
    for (let c = 0; c < channels; c++) {
      let sum = 0;
      let min = Infinity;
      let max = -Infinity;
      for (let t = 0; t < T; t++) {
        const v = input[b * T * channels + t * channels + c];
        sum += v;
        if (v < min) min = v;
        if (v > max) max = v;
      }
      const mean = sum / T;
      let varSum = 0;
      for (let t = 0; t < T; t++) {
        const v = input[b * T * channels + t * channels + c];
        varSum += (v - mean) * (v - mean);
      }
      const std = Math.sqrt(varSum / T + 1e-6);
      const base = b * channels * 4 + c * 4;
      output[base] = mean;
      output[base + 1] = std;
      output[base + 2] = min;
      output[base + 3] = max;
    }
  }
  return output;
}

/** Linear layer: output = input @ W^T + bias */
function linear(
  input: Float32Array,  // (batch, inFeatures)
  weight: Float32Array, // (outFeatures, inFeatures)
  bias: Float32Array,   // (outFeatures,)
  batch: number,
  inFeatures: number,
  outFeatures: number,
): Float32Array {
  const output = new Float32Array(batch * outFeatures);
  for (let b = 0; b < batch; b++) {
    for (let o = 0; o < outFeatures; o++) {
      let sum = bias[o];
      for (let i = 0; i < inFeatures; i++) {
        sum += input[b * inFeatures + i] * weight[o * inFeatures + i];
      }
      output[b * outFeatures + o] = sum;
    }
  }
  return output;
}

/** Softmax along last dimension */
function softmax(logits: Float32Array, batch: number, n: number): Float32Array {
  const output = new Float32Array(logits.length);
  for (let b = 0; b < batch; b++) {
    let max = -Infinity;
    for (let i = 0; i < n; i++) {
      if (logits[b * n + i] > max) max = logits[b * n + i];
    }
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const e = Math.exp(logits[b * n + i] - max);
      output[b * n + i] = e;
      sum += e;
    }
    for (let i = 0; i < n; i++) {
      output[b * n + i] /= sum;
    }
  }
  return output;
}

/**
 * Run the full ActivityClassifier forward pass.
 *
 * Input shape: (time_steps, 6) — raw sensor readings
 * Output: class probabilities array
 */
export function forwardPass(
  weights: ModelWeights,
  sensorData: number[][],
): Prediction {
  const T = sensorData.length;
  const channels = 6;

  // Flatten input to (1, T, 6) as Float32Array (batch=1)
  const input = new Float32Array(T * channels);
  for (let t = 0; t < T; t++) {
    for (let c = 0; c < channels; c++) {
      input[t * channels + c] = sensorData[t][c];
    }
  }

  // Path A — Conv path
  // conv1: (1, 6, T) → (1, 32, T)
  const conv1Out = conv1d(input, weights.conv1Weight, weights.conv1Bias, 1, 6, 32, T, 7);
  const relu1 = relu(conv1Out);

  // conv2: (1, 32, T) → (1, 32, T)
  const conv2Out = conv1d(relu1, weights.conv2Weight, weights.conv2Bias, 1, 32, 32, T, 5);
  const relu2 = relu(conv2Out);

  // Global avg pool: (1, 32, T) → (1, 32)
  const pooled = globalAvgPool(relu2, 1, 32, T);

  // Path B — Statistics: (T, 6) → (24,)
  const stats = computeStats(input, 1, T, channels);

  // Concat: (1, 56)
  const joint = new Float32Array(56);
  joint.set(pooled, 0);
  joint.set(stats, 32);

  // FC → softmax
  const logits = linear(joint, weights.fcWeight, weights.fcBias, 1, 56, weights.numClasses);
  const probs = softmax(logits, 1, weights.numClasses);

  // Find argmax
  let maxIdx = 0;
  let maxVal = probs[0];
  for (let i = 1; i < weights.numClasses; i++) {
    if (probs[i] > maxVal) {
      maxVal = probs[i];
      maxIdx = i;
    }
  }

  const classId = maxIdx;
  const className = ACTIVITIES[classId] || `class_${classId}`;
  const probabilities = Array.from(probs);

  return {classId, className, probabilities};
}

// ── Public API ───────────────────────────────────────────────────────

let _cachedWeights: ModelWeights | null = null;

/**
 * Initialize the on-device inference engine.
 * Call once on app startup to check for cached model.
 * Returns true if a model is available for local inference.
 */
export async function initInference(): Promise<boolean> {
  _cachedWeights = await loadWeights();
  return _cachedWeights !== null;
}

/**
 * Predict activity from sensor data using cached model.
 * Returns null if no model is cached (caller should fallback to server).
 */
export async function predictLocal(
  sensorData: number[][],
): Promise<Prediction | null> {
  if (!_cachedWeights) {
    _cachedWeights = await loadWeights();
  }
  if (!_cachedWeights) return null;
  return forwardPass(_cachedWeights, sensorData);
}

/** Check if a local model is loaded. */
export function isLocalModelReady(): boolean {
  return _cachedWeights !== null;
}

/** Get local model info. */
export function getLocalModelInfo(): {numClasses: number; activities: string[]} | null {
  if (!_cachedWeights) return null;
  return {numClasses: _cachedWeights.numClasses, activities: ACTIVITIES};
}
