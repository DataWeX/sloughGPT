/**
 * llama.rn service — wraps the llama.rn native module for Qwen GGUF inference.
 *
 * llama.rn provides React Native bindings for llama.cpp with Metal GPU
 * acceleration.  The Qwen 500M Q4_K_M GGUF model (~250 MB) achieves
 * 15-30 tok/s on modern iPhones.
 *
 * This service handles:
 *   - Downloading the Qwen GGUF model (from server or HuggingFace)
 *   - Loading it into a LlamaContext
 *   - Chat completion with chat template
 *   - Streaming token generation
 */

import {Platform, NativeModules} from 'react-native';
import {getApiUrl} from './api-client';
import AsyncStorage from '@react-native-async-storage/async-storage';
import RNFS from 'react-native-fs';

const {LlamaContext} = NativeModules;

const MODEL_CACHE_KEY = '@sloughgpt/qwen_model_path';
const MODEL_URL_KEY = '@sloughgpt/qwen_model_url';
const GGUF_FILENAME = 'qwen2.5-0.5b-instruct-q4_k_m.gguf';
const GGUF_HF_REPO = 'Qwen/Qwen2.5-0.5B-Instruct-GGUF';
const GGUF_HF_FILE = 'qwen2.5-0.5b-instruct-q4_k_m.gguf';

export interface LlamaConfig {
  /** Path to local GGUF file */
  modelPath: string;
  /** Context size */
  nCtx: number;
  /** Number of GPU layers (0 = CPU only, -1 = all) */
  nGpuLayers: number;
}

export interface LlamaGenerateOptions {
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  repetitionPenalty?: number;
  stop?: string[];
}

export interface LlamaGenerateResult {
  text: string;
  tokensGenerated: number;
  elapsedMs: number;
}

let _context: any = null;
let _modelPath: string | null = null;

// ── Model download ──────────────────────────────────────────────────────

/** Get the local path where the GGUF model should be stored. */
function _getLocalPath(): string {
  const dir = Platform.OS === 'ios'
    ? RNFS.DocumentDirectoryPath
    : RNFS.CachesDirectoryPath;
  return `${dir}/${GGUF_FILENAME}`;
}

/**
 * Download the Qwen GGUF model.
 * Tries server first (GET /models/download/qwen-gguf), falls back to HF.
 */
export async function downloadModel(
  onProgress?: (fraction: number) => void,
): Promise<string> {
  const localPath = _getLocalPath();

  // Check if already downloaded
  const exists = await RNFS.exists(localPath);
  if (exists) {
    _modelPath = localPath;
    await AsyncStorage.setItem(MODEL_CACHE_KEY, localPath);
    return localPath;
  }

  const baseUrl = await getApiUrl();

  // Try server endpoint first
  const serverUrl = `${baseUrl}/models/download/qwen-gguf`;
  try {
    const serverRes = await fetch(serverUrl, {method: 'HEAD'});
    if (serverRes.ok) {
      const download = RNFS.downloadFile({
        fromUrl: serverUrl,
        toFile: localPath,
        progress: (res: {contentLength: number; bytesWritten: number}) => {
          if (res.contentLength > 0 && onProgress) {
            onProgress(res.bytesWritten / res.contentLength);
          }
        },
      });
      const result = await download.promise;
      if (result.statusCode === 200) {
        _modelPath = localPath;
        await AsyncStorage.setItem(MODEL_CACHE_KEY, localPath);
        return localPath;
      }
    }
  } catch {
    // Fall through to HuggingFace
  }

  // Fallback: download from HuggingFace
  const hfUrl = `https://huggingface.co/${GGUF_HF_REPO}/resolve/main/${GGUF_HF_FILE}`;
  const download = RNFS.downloadFile({
    fromUrl: hfUrl,
    toFile: localPath,
    progress: (res: {contentLength: number; bytesWritten: number}) => {
      if (res.contentLength > 0 && onProgress) {
        onProgress(res.bytesWritten / res.contentLength);
      }
    },
  });
  const result = await download.promise;
  if (result.statusCode !== 200) {
    throw new Error(`Failed to download GGUF model: HTTP ${result.statusCode}`);
  }

  _modelPath = localPath;
  await AsyncStorage.setItem(MODEL_CACHE_KEY, localPath);
  await AsyncStorage.setItem(MODEL_URL_KEY, hfUrl);
  return localPath;
}

/** Check if the GGUF model is downloaded. */
export async function isModelDownloaded(): Promise<boolean> {
  const path = _modelPath || await AsyncStorage.getItem(MODEL_CACHE_KEY);
  if (!path) return false;
  return RNFS.exists(path);
}

/** Get the local model path, or null if not downloaded. */
export async function getModelPath(): Promise<string | null> {
  return _modelPath || AsyncStorage.getItem(MODEL_CACHE_KEY);
}

// ── Context management ──────────────────────────────────────────────────

/** Load the GGUF model into a LlamaContext. */
export async function loadModel(config?: Partial<LlamaConfig>): Promise<void> {
  const modelPath = _modelPath || await AsyncStorage.getItem(MODEL_CACHE_KEY);
  if (!modelPath) throw new Error('Model not downloaded. Call downloadModel() first.');

  if (_context) {
    await _context.release();
    _context = null;
  }

  const cfg: LlamaConfig = {
    modelPath,
    nCtx: config?.nCtx ?? 2048,
    nGpuLayers: Platform.OS === 'ios' ? (config?.nGpuLayers ?? -1) : 0,
  };

  try {
    _context = await LlamaContext.create(cfg);
  } catch (e) {
    // Retry with CPU only if Metal fails
    if (cfg.nGpuLayers !== 0) {
      cfg.nGpuLayers = 0;
      _context = await LlamaContext.create(cfg);
    } else {
      throw e;
    }
  }
}

/** Unload the model and release memory. */
export async function unloadModel(): Promise<void> {
  if (_context) {
    try { await _context.release(); } catch {}
    _context = null;
  }
}

/** Check if a context is currently loaded. */
export function isLoaded(): boolean {
  return _context !== null;
}

// ── Inference ───────────────────────────────────────────────────────────

/** Chat completion via llama.rn (non-streaming). */
export async function chatCompletion(
  messages: Array<{role: string; content: string}>,
  opts?: LlamaGenerateOptions,
): Promise<LlamaGenerateResult> {
  if (!_context) throw new Error('Model not loaded');
  const t0 = Date.now();

  // Build prompt from messages using Qwen's chat template
  const prompt = _buildChatTemplate(messages);

  const result = await _context.completion({
    prompt,
    nPredict: opts?.maxTokens ?? 256,
    temperature: opts?.temperature ?? 0.7,
    topK: opts?.topK ?? 40,
    topP: opts?.topP ?? 0.9,
    repeatPenalty: opts?.repetitionPenalty ?? 1.1,
    stop: opts?.stop ?? ['<|im_end|>', '<|end|>', 'User:'],
  });

  return {
    text: result.text,
    tokensGenerated: result.tokens ?? 0,
    elapsedMs: Date.now() - t0,
  };
}

/** Streaming chat completion — yields tokens as they arrive. */
export async function* chatCompletionStream(
  messages: Array<{role: string; content: string}>,
  opts?: LlamaGenerateOptions,
): AsyncGenerator<string, LlamaGenerateResult, void> {
  if (!_context) throw new Error('Model not loaded');
  const t0 = Date.now();
  const prompt = _buildChatTemplate(messages);
  let fullText = '';
  let tokensGenerated = 0;

  const stream = await _context.completion({
    prompt,
    nPredict: opts?.maxTokens ?? 256,
    temperature: opts?.temperature ?? 0.7,
    topK: opts?.topK ?? 40,
    topP: opts?.topP ?? 0.9,
    repeatPenalty: opts?.repetitionPenalty ?? 1.1,
    stop: opts?.stop ?? ['<|im_end|>', '<|end|>', 'User:'],
  });

  for await (const token of stream) {
    fullText += token;
    tokensGenerated++;
    yield token;
  }

  return {
    text: fullText,
    tokensGenerated,
    elapsedMs: Date.now() - t0,
  };
}

// ── Helpers ─────────────────────────────────────────────────────────────

function _buildChatTemplate(messages: Array<{role: string; content: string}>): string {
  let prompt = '';
  for (const msg of messages) {
    if (msg.role === 'system') {
      prompt += `<|im_start|>system\n${msg.content}<|im_end|>\n`;
    } else if (msg.role === 'user') {
      prompt += `<|im_start|>user\n${msg.content}<|im_end|>\n`;
    } else if (msg.role === 'assistant') {
      prompt += `<|im_start|>assistant\n${msg.content}<|im_end|>\n`;
    }
  }
  prompt += '<|im_start|>assistant\n';
  return prompt;
}
