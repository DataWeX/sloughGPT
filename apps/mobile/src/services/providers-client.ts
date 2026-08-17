/**
 * Third-party inference client — handles API calls to OpenAI-compatible,
 * Anthropic, and Google Gemini endpoints.
 *
 * All three share a common streaming interface:
 *   for await (const event of streamProviderChat(config, messages, opts)) { ... }
 *
 * OpenAI-compatible providers (OpenAI, Groq, Together, DeepSeek, OpenRouter, Custom)
 * use the /chat/completions endpoint with SSE.
 *
 * Anthropic uses /messages with SSE.
 *
 * Google Gemini uses /models/{model}:streamGenerateContent with SSE.
 */

import type {
  ProviderConfig,
  ProviderId,
  ChatMessage,
  OnTokenCallback,
  StreamResult,
} from '../types/providers';
import {PROVIDER_REGISTRY} from '../types/providers';

const DEFAULT_TIMEOUT_MS = 60_000;

// ── Public API ──────────────────────────────────────────────────────────

/**
 * Stream a chat completion from any provider.
 * Yields tokens via onToken callback. Returns the full text when done.
 */
export async function streamProviderChat(
  config: ProviderConfig,
  messages: ChatMessage[],
  opts: {
    model?: string;
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    onToken?: OnTokenCallback;
    signal?: AbortSignal;
  } = {},
): Promise<StreamResult> {
  const model = opts.model || config.defaultModel;

  switch (config.id) {
    case 'anthropic':
      return _streamAnthropic(config, messages, model, opts);
    case 'google':
      return _streamGemini(config, messages, model, opts);
    default:
      // OpenAI-compatible (openai, groq, together, deepseek, openrouter, mistral, custom)
      return _streamOpenAICompatible(config, messages, model, opts);
  }
}

/**
 * Non-streaming chat completion. Returns the full response.
 */
export async function providerChat(
  config: ProviderConfig,
  messages: ChatMessage[],
  opts: {
    model?: string;
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    signal?: AbortSignal;
  } = {},
): Promise<StreamResult> {
  const model = opts.model || config.defaultModel;

  switch (config.id) {
    case 'anthropic':
      return _chatAnthropic(config, messages, model, opts);
    case 'google':
      return _chatGemini(config, messages, model, opts);
    default:
      return _chatOpenAICompatible(config, messages, model, opts);
  }
}

// ── OpenAI-compatible (streaming) ───────────────────────────────────────

async function _streamOpenAICompatible(
  config: ProviderConfig,
  messages: ChatMessage[],
  model: string,
  opts: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    onToken?: OnTokenCallback;
    signal?: AbortSignal;
  },
): Promise<StreamResult> {
  const url = `${config.baseUrl.replace(/\/$/, '')}/chat/completions`;
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${config.apiKey}`,
    ...config.headers,
  };

  const body = {
    model,
    messages: messages.map(m => ({role: m.role, content: m.content})),
    max_tokens: opts.maxTokens ?? 2048,
    temperature: opts.temperature ?? 0.7,
    top_p: opts.topP,
    stream: true,
  };

  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), DEFAULT_TIMEOUT_MS);
  if (opts.signal) {
    opts.signal.addEventListener('abort', () => ac.abort());
  }

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: ac.signal,
    });

    if (!res.ok) {
      const text = await res.text();
      throw new ProviderError(config.id, res.status, _parseError(text, res.status));
    }

    const reader = res.body?.getReader();
    if (!reader) throw new ProviderError(config.id, 0, 'No response body');

    const decoder = new TextDecoder();
    let buffer = '';
    let accumulated = '';
    let finishReason: string | undefined;

    try {
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, {stream: true});
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trimEnd();
          if (!trimmed.startsWith('data:')) continue;

          const payload = trimmed.slice(5).trim();
          if (!payload || payload === '[DONE]') continue;

          try {
            const parsed = JSON.parse(payload);
            const delta = parsed.choices?.[0]?.delta;
            finishReason = parsed.choices?.[0]?.finish_reason || finishReason;

            if (delta?.content) {
              accumulated += delta.content;
              opts.onToken?.(delta.content);
            }
          } catch {
            // skip malformed JSON
          }
        }
      }
    } finally {
      reader.releaseLock();
    }

    return {
      text: accumulated,
      model,
      finishReason,
    };
  } finally {
    clearTimeout(timer);
  }
}

// ── OpenAI-compatible (non-streaming) ──────────────────────────────────

async function _chatOpenAICompatible(
  config: ProviderConfig,
  messages: ChatMessage[],
  model: string,
  opts: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    signal?: AbortSignal;
  },
): Promise<StreamResult> {
  const url = `${config.baseUrl.replace(/\/$/, '')}/chat/completions`;
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${config.apiKey}`,
    ...config.headers,
  };

  const body = {
    model,
    messages: messages.map(m => ({role: m.role, content: m.content})),
    max_tokens: opts.maxTokens ?? 2048,
    temperature: opts.temperature ?? 0.7,
    top_p: opts.topP,
    stream: false,
  };

  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), DEFAULT_TIMEOUT_MS);
  if (opts.signal) opts.signal.addEventListener('abort', () => ac.abort());

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: ac.signal,
    });

    if (!res.ok) {
      const text = await res.text();
      throw new ProviderError(config.id, res.status, _parseError(text, res.status));
    }

    const data = await res.json();
    const text = data.choices?.[0]?.message?.content || '';

    return {
      text,
      model,
      tokensUsed: data.usage?.total_tokens,
      finishReason: data.choices?.[0]?.finish_reason,
    };
  } finally {
    clearTimeout(timer);
  }
}

// ── Anthropic (streaming) ──────────────────────────────────────────────

async function _streamAnthropic(
  config: ProviderConfig,
  messages: ChatMessage[],
  model: string,
  opts: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    onToken?: OnTokenCallback;
    signal?: AbortSignal;
  },
): Promise<StreamResult> {
  const url = `${config.baseUrl.replace(/\/$/, '')}/messages`;
  const {system, chatMessages} = _extractSystemMessage(messages);

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'x-api-key': config.apiKey,
    'anthropic-version': '2023-06-01',
    ...config.headers,
  };

  const body: Record<string, unknown> = {
    model,
    messages: chatMessages.map(m => ({role: m.role, content: m.content})),
    max_tokens: opts.maxTokens ?? 2048,
    stream: true,
  };
  if (system) body.system = system;
  if (opts.temperature != null) body.temperature = opts.temperature;
  if (opts.topP != null) body.top_p = opts.topP;

  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), DEFAULT_TIMEOUT_MS);
  if (opts.signal) opts.signal.addEventListener('abort', () => ac.abort());

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: ac.signal,
    });

    if (!res.ok) {
      const text = await res.text();
      throw new ProviderError(config.id, res.status, _parseError(text, res.status));
    }

    const reader = res.body?.getReader();
    if (!reader) throw new ProviderError(config.id, 0, 'No response body');

    const decoder = new TextDecoder();
    let buffer = '';
    let accumulated = '';
    let finishReason: string | undefined;

    try {
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, {stream: true});
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trimEnd();
          if (!trimmed.startsWith('data:')) continue;

          const payload = trimmed.slice(5).trim();
          if (!payload) continue;

          try {
            const parsed = JSON.parse(payload);

            if (parsed.type === 'content_block_delta') {
              const text = parsed.delta?.text;
              if (text) {
                accumulated += text;
                opts.onToken?.(text);
              }
            } else if (parsed.type === 'message_delta') {
              finishReason = parsed.delta?.stop_reason || finishReason;
            }
          } catch {
            // skip malformed JSON
          }
        }
      }
    } finally {
      reader.releaseLock();
    }

    return {text: accumulated, model, finishReason};
  } finally {
    clearTimeout(timer);
  }
}

// ── Anthropic (non-streaming) ─────────────────────────────────────────

async function _chatAnthropic(
  config: ProviderConfig,
  messages: ChatMessage[],
  model: string,
  opts: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    signal?: AbortSignal;
  },
): Promise<StreamResult> {
  const url = `${config.baseUrl.replace(/\/$/, '')}/messages`;
  const {system, chatMessages} = _extractSystemMessage(messages);

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'x-api-key': config.apiKey,
    'anthropic-version': '2023-06-01',
    ...config.headers,
  };

  const body: Record<string, unknown> = {
    model,
    messages: chatMessages.map(m => ({role: m.role, content: m.content})),
    max_tokens: opts.maxTokens ?? 2048,
    stream: false,
  };
  if (system) body.system = system;
  if (opts.temperature != null) body.temperature = opts.temperature;
  if (opts.topP != null) body.top_p = opts.topP;

  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), DEFAULT_TIMEOUT_MS);
  if (opts.signal) opts.signal.addEventListener('abort', () => ac.abort());

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: ac.signal,
    });

    if (!res.ok) {
      const text = await res.text();
      throw new ProviderError(config.id, res.status, _parseError(text, res.status));
    }

    const data = await res.json();
    const text = data.content?.[0]?.text || '';

    return {
      text,
      model,
      tokensUsed: data.usage?.input_tokens != null && data.usage?.output_tokens != null
        ? data.usage.input_tokens + data.usage.output_tokens
        : undefined,
      finishReason: data.stop_reason,
    };
  } finally {
    clearTimeout(timer);
  }
}

// ── Google Gemini (streaming) ──────────────────────────────────────────

async function _streamGemini(
  config: ProviderConfig,
  messages: ChatMessage[],
  model: string,
  opts: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    onToken?: OnTokenCallback;
    signal?: AbortSignal;
  },
): Promise<StreamResult> {
  const {system, chatMessages} = _extractSystemMessage(messages);

  const url = `${config.baseUrl.replace(/\/$/, '')}/models/${model}:streamGenerateContent?alt=sse&key=${config.apiKey}`;

  const contents = chatMessages.map(m => ({
    role: m.role === 'assistant' ? 'model' : 'user',
    parts: [{text: m.content}],
  }));

  const body: Record<string, unknown> = {contents};
  const generationConfig: Record<string, unknown> = {};
  if (opts.maxTokens != null) generationConfig.maxOutputTokens = opts.maxTokens;
  if (opts.temperature != null) generationConfig.temperature = opts.temperature;
  if (opts.topP != null) generationConfig.topP = opts.topP;
  if (Object.keys(generationConfig).length > 0) body.generationConfig = generationConfig;
  if (system) body.systemInstruction = {parts: [{text: system}]};

  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), DEFAULT_TIMEOUT_MS);
  if (opts.signal) opts.signal.addEventListener('abort', () => ac.abort());

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json', ...config.headers},
      body: JSON.stringify(body),
      signal: ac.signal,
    });

    if (!res.ok) {
      const text = await res.text();
      throw new ProviderError(config.id, res.status, _parseError(text, res.status));
    }

    const reader = res.body?.getReader();
    if (!reader) throw new ProviderError(config.id, 0, 'No response body');

    const decoder = new TextDecoder();
    let buffer = '';
    let accumulated = '';

    try {
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, {stream: true});
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trimEnd();
          if (!trimmed.startsWith('data:')) continue;

          const payload = trimmed.slice(5).trim();
          if (!payload) continue;

          try {
            const parsed = JSON.parse(payload);
            const candidates = parsed.candidates;
            if (candidates?.[0]?.content?.parts) {
              for (const part of candidates[0].content.parts) {
                if (part.text) {
                  accumulated += part.text;
                  opts.onToken?.(part.text);
                }
              }
            }
          } catch {
            // skip malformed JSON
          }
        }
      }
    } finally {
      reader.releaseLock();
    }

    return {text: accumulated, model};
  } finally {
    clearTimeout(timer);
  }
}

// ── Google Gemini (non-streaming) ─────────────────────────────────────

async function _chatGemini(
  config: ProviderConfig,
  messages: ChatMessage[],
  model: string,
  opts: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    signal?: AbortSignal;
  },
): Promise<StreamResult> {
  const {system, chatMessages} = _extractSystemMessage(messages);

  const url = `${config.baseUrl.replace(/\/$/, '')}/models/${model}:generateContent?key=${config.apiKey}`;

  const contents = chatMessages.map(m => ({
    role: m.role === 'assistant' ? 'model' : 'user',
    parts: [{text: m.content}],
  }));

  const body: Record<string, unknown> = {contents};
  const generationConfig: Record<string, unknown> = {};
  if (opts.maxTokens != null) generationConfig.maxOutputTokens = opts.maxTokens;
  if (opts.temperature != null) generationConfig.temperature = opts.temperature;
  if (opts.topP != null) generationConfig.topP = opts.topP;
  if (Object.keys(generationConfig).length > 0) body.generationConfig = generationConfig;
  if (system) body.systemInstruction = {parts: [{text: system}]};

  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), DEFAULT_TIMEOUT_MS);
  if (opts.signal) opts.signal.addEventListener('abort', () => ac.abort());

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json', ...config.headers},
      body: JSON.stringify(body),
      signal: ac.signal,
    });

    if (!res.ok) {
      const text = await res.text();
      throw new ProviderError(config.id, res.status, _parseError(text, res.status));
    }

    const data = await res.json();
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text || '';

    return {text, model};
  } finally {
    clearTimeout(timer);
  }
}

// ── Helpers ────────────────────────────────────────────────────────────

/** Extract system message from the messages array (Anthropic/Gemini separate it). */
function _extractSystemMessage(messages: ChatMessage[]): {
  system: string | null;
  chatMessages: ChatMessage[];
} {
  const systemMsgs = messages.filter(m => m.role === 'system');
  const chatMessages = messages.filter(m => m.role !== 'system');
  const system = systemMsgs.length > 0 ? systemMsgs.map(m => m.content).join('\n') : null;
  return {system, chatMessages};
}

/** Parse error response body into a human-readable message. */
function _parseError(text: string, status: number): string {
  try {
    const j = JSON.parse(text);
    const detail = j.error?.message || j.error?.detail || j.message || j.detail;
    if (detail) return `${status}: ${detail}`;
  } catch {}
  return `Request failed (${status})`;
}

/** Provider-specific error with status code. */
export class ProviderError extends Error {
  constructor(
    public provider: ProviderId,
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = 'ProviderError';
  }
}
