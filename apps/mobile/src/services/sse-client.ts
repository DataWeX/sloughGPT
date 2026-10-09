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

const BASE_RECONNECT_MS = 1000;
const MAX_RECONNECT_MS = 16000;
const MAX_RECONNECTS = 5;

async function _connectSSE(
  path: string,
  body: unknown,
  signal?: AbortSignal,
): Promise<ReadableStreamDefaultReader<Uint8Array>> {
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
  return reader;
}

export async function* streamSSE(
  path: string,
  body: unknown,
  signal?: AbortSignal,
): AsyncGenerator<SSEEvent> {
  let reconnectCount = 0;
  let reconnectMs = BASE_RECONNECT_MS;

  while (true) {
    let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
    let buffer = '';

    try {
      reader = await _connectSSE(path, body, signal);
      reconnectCount = 0;
      reconnectMs = BASE_RECONNECT_MS;

      const decoder = new TextDecoder();

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

      // Normal completion — no reconnect needed
      return;
    } catch (err: any) {
      if (signal?.aborted) return;

      // Don't retry HTTP errors (4xx/5xx are permanent)
      if (err instanceof SSEHttpError) {
        throw err;
      }

      reconnectCount++;
      if (reconnectCount > MAX_RECONNECTS) {
        throw new Error(`SSE connection lost after ${MAX_RECONNECTS} reconnect attempts`);
      }

      console.warn(`[SSE] Connection lost, reconnecting in ${reconnectMs}ms (${reconnectCount}/${MAX_RECONNECTS})...`);
      await new Promise(resolve => setTimeout(resolve, reconnectMs));
      reconnectMs = Math.min(reconnectMs * 2, MAX_RECONNECT_MS);
    } finally {
      reader?.releaseLock();
    }
  }
}
