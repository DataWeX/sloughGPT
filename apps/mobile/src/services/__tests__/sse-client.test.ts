import {streamSSE} from '../sse-client';
import AsyncStorage from '@react-native-async-storage/async-storage';

function mockSSEResponse(lines: string[]): Response {
  const encoder = new TextEncoder();
  const chunks = lines.map(l => encoder.encode(l + '\n'));
  return {
    ok: true,
    status: 200,
    body: {
      getReader: () => {
        let i = 0;
        return {
          read: async () => {
            if (i >= chunks.length) return {done: true, value: undefined as any};
            return {done: false, value: chunks[i++]};
          },
          releaseLock: jest.fn(),
          cancel: jest.fn(),
          closed: Promise.resolve(undefined),
        };
      },
    },
  } as unknown as Response;
}

function mockErrorResponse(status: number, text: string): Response {
  return {
    ok: false,
    status,
    text: () => Promise.resolve(text),
  } as unknown as Response;
}

beforeEach(async () => {
  jest.clearAllMocks();
  (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
  // Spy on fetch
  jest.spyOn(global, 'fetch').mockImplementation();
});

afterEach(() => {
  jest.restoreAllMocks();
});

describe('streamSSE', () => {
  it('yields token events from SSE stream', async () => {
    const events = [
      'data: {"data":{"token":"Hel"}}',
      'data: {"data":{"token":"lo"}}',
      'data: {"status":"complete"}',
    ];
    (global.fetch as jest.Mock).mockResolvedValue(mockSSEResponse(events));

    const tokens: string[] = [];
    for await (const event of streamSSE('/chat/stream', {messages: []})) {
      if (event.data?.token) tokens.push(event.data.token as string);
    }
    expect(tokens).toEqual(['Hel', 'lo']);
  });

  it('yields complete status', async () => {
    const events = [
      'data: {"data":{"token":"ok"}}',
      'data: {"status":"complete"}',
    ];
    (global.fetch as jest.Mock).mockResolvedValue(mockSSEResponse(events));

    const statuses: string[] = [];
    for await (const event of streamSSE('/chat/stream', {messages: []})) {
      if (event.status === 'complete') statuses.push('complete');
    }
    expect(statuses).toEqual(['complete']);
  });

  it('yields error status with message', async () => {
    const events = [
      'data: {"status":"error","message":"Server error"}',
    ];
    (global.fetch as jest.Mock).mockResolvedValue(mockSSEResponse(events));

    const errors: Array<{status: string; message?: string}> = [];
    for await (const event of streamSSE('/chat/stream', {messages: []})) {
      if (event.status === 'error') {
        errors.push({status: event.status, message: event.message});
      }
    }
    expect(errors).toEqual([{status: 'error', message: 'Server error'}]);
  });

  it('passes meta from event', async () => {
    const events = [
      'data: {"meta":{"tokens":5,"elapsed_ms":100},"status":"complete"}',
    ];
    (global.fetch as jest.Mock).mockResolvedValue(mockSSEResponse(events));

    for await (const event of streamSSE('/chat/stream', {messages: []})) {
      expect(event.meta).toEqual({tokens: 5, elapsed_ms: 100});
    }
  });

  it('skips comment lines and empty lines', async () => {
    const events = [
      ': this is a comment',
      '',
      'data: {"data":{"token":"hi"}}',
      '',
      ': another comment',
      'data: {"status":"complete"}',
    ];
    (global.fetch as jest.Mock).mockResolvedValue(mockSSEResponse(events));

    const tokens: string[] = [];
    for await (const event of streamSSE('/chat/stream', {messages: []})) {
      if (event.data?.token) tokens.push(event.data.token as string);
    }
    expect(tokens).toEqual(['hi']);
  });

  it('skips malformed JSON lines gracefully', async () => {
    const events = [
      'data: {invalid json}',
      'data: {"data":{"token":"ok"}}',
      'data: {"status":"complete"}',
    ];
    (global.fetch as jest.Mock).mockResolvedValue(mockSSEResponse(events));

    const tokens: string[] = [];
    for await (const event of streamSSE('/chat/stream', {messages: []})) {
      if (event.data?.token) tokens.push(event.data.token as string);
    }
    expect(tokens).toEqual(['ok']);
  });

  it('handles buffered chunks split across read() calls', async () => {
    const encoder = new TextEncoder();
    const chunks = [
      encoder.encode('data: {"data":{"token":"Hel'),
      encoder.encode('lo"}}\ndata: {"status":"complete"}\n'),
    ];
    let idx = 0;
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      status: 200,
      body: {
        getReader: () => ({
          read: async () => {
            if (idx >= chunks.length) return {done: true, value: undefined as any};
            return {done: false, value: chunks[idx++]};
          },
          releaseLock: jest.fn(),
        }),
      },
    } as unknown as Response);

    const tokens: string[] = [];
    for await (const event of streamSSE('/chat/stream', {messages: []})) {
      if (event.data?.token) tokens.push(event.data.token as string);
    }
    expect(tokens).toEqual(['Hello']);
  });

  it('throws on non-ok response', async () => {
    (global.fetch as jest.Mock).mockResolvedValue(mockErrorResponse(404, 'Not found'));
    const gen = streamSSE('/bad', {});
    await expect(gen.next()).rejects.toThrow('SSE error 404');
  });

  it('throws when response has no body', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      status: 200,
      body: null,
    } as unknown as Response);
    const gen = streamSSE('/empty', {});
    await expect(gen.next()).rejects.toThrow('No response body');
  });

  it('releases reader lock in finally block', async () => {
    const releaseLock = jest.fn();
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      status: 200,
      body: {
        getReader: () => ({
          read: async () => ({done: true, value: undefined}),
          releaseLock,
        }),
      },
    } as unknown as Response);

    for await (const _ of streamSSE('/test', {})) {
      // consume
    }
    expect(releaseLock).toHaveBeenCalled();
  });

  it('yields raw envelope objects', async () => {
    const events = [
      'data: {"stream":"chat","phase":"STREAMING","status":"working","data":{"token":"a"}}',
      'data: {"stream":"chat","phase":"STREAMING","status":"complete"}',
    ];
    (global.fetch as jest.Mock).mockResolvedValue(mockSSEResponse(events));

    const envelopes: unknown[] = [];
    for await (const event of streamSSE('/chat/stream', {messages: []})) {
      envelopes.push(event);
    }
    expect(envelopes).toHaveLength(2);
    expect((envelopes[0] as any).stream).toBe('chat');
    expect((envelopes[0] as any).phase).toBe('STREAMING');
    expect((envelopes[0] as any).status).toBe('working');
    expect((envelopes[0] as any).data?.token).toBe('a');
  });

  it('includes error detail from data.error field', async () => {
    const events = [
      'data: {"status":"error","data":{"error":"Token limit exceeded"}}',
    ];
    (global.fetch as jest.Mock).mockResolvedValue(mockSSEResponse(events));

    for await (const event of streamSSE('/chat/stream', {messages: []})) {
      expect(event.status).toBe('error');
      expect(event.data?.error).toBe('Token limit exceeded');
    }
  });

  it('passes Accept and Cache-Control headers', async () => {
    (global.fetch as jest.Mock).mockResolvedValue(mockSSEResponse([]));

    for await (const _ of streamSSE('/chat/stream', {messages: []})) {}

    const [, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(init.headers).toMatchObject({
      Accept: 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Content-Type': 'application/json',
    });
  });
});
