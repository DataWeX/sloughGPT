import {getApiUrl} from './api-client';

export interface SSEEvent {
  stream?: string;
  phase?: string;
  status?: string;
  data?: Record<string, unknown>;
  meta?: Record<string, unknown>;
  message?: string;
}

/** HTTP error from SSE (server is reachable but returned an error status). */
export class SSEHttpError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = 'SSEHttpError';
  }
}

export async function* streamSSE(
  path: string,
  body: unknown,
  signal?: AbortSignal,
): AsyncGenerator<SSEEvent> {
  const baseUrl = await getApiUrl();
  const res = await fetch(`${baseUrl}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      'Cache-Control': 'no-cache',
    },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new SSEHttpError(res.status, `SSE error ${res.status}: ${text}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';

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
          yield JSON.parse(payload) as SSEEvent;
        } catch {
          // skip malformed JSON lines
        }
      }
    }

    // Drain remaining buffer
    if (buffer.startsWith('data:')) {
      const payload = buffer.slice(5).trim();
      if (payload && payload !== '[DONE]') {
        try {
          yield JSON.parse(payload) as SSEEvent;
        } catch {}
      }
    }
  } finally {
    reader.releaseLock();
  }
}
