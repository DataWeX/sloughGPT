import {getApiUrl} from './api-client';

export interface SSEEvent {
  token?: string;
  done?: boolean;
  error?: string;
  meta?: Record<string, unknown>;
  raw?: Record<string, unknown>;
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
    headers: {'Content-Type': 'application/json'},
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
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith(':')) continue;

        if (trimmed.startsWith('data: ')) {
          try {
            const data = JSON.parse(trimmed.slice(6));
            const event: SSEEvent = {raw: data};

            if (data.data?.token !== undefined) {
              event.token = data.data.token;
            }
            if (data.status === 'complete' || data.status === 'error') {
              event.done = true;
              if (data.status === 'error') {
                event.error = data.message || data.data?.error || 'Stream error';
              }
            }
            if (data.meta) {
              event.meta = data.meta;
            }

            yield event;
          } catch {
            // skip malformed JSON lines
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
