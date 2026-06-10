import { describe, it, expect, vi } from 'vitest'
import { streamSSE } from '@/lib/sse-client'

describe('SSE Client', () => {
  it('should stream tokens from SSE response', async () => {
    const mockReader = {
      read: vi.fn()
        .mockResolvedValueOnce({
          done: false,
          value: new TextEncoder().encode('data: {"stream":"chat","phase":"STREAMING","status":"working","data":{"token":"Hello"}}\n\n'),
        })
        .mockResolvedValueOnce({
          done: false,
          value: new TextEncoder().encode('data: {"stream":"chat","phase":"STREAMING","status":"working","data":{"token":" world"}}\n\n'),
        })
        .mockResolvedValueOnce({
          done: false,
          value: new TextEncoder().encode('data: {"stream":"chat","phase":"STREAMING","status":"complete","data":{"token":""}}\n\n'),
        })
        .mockResolvedValueOnce({ done: true, value: undefined }),
      releaseLock: vi.fn(),
    }

    vi.mocked(global.fetch).mockResolvedValue({
      ok: true,
      body: {
        getReader: () => mockReader,
      },
    } as any)

    const tokens: string[] = []
    let done = false

    for await (const event of streamSSE('/chat/stream', { messages: [] })) {
      if (event.done) {
        done = true
        break
      }
      tokens.push(event.token)
    }

    expect(tokens).toEqual(['Hello', ' world'])
    expect(done).toBe(true)
    expect(mockReader.releaseLock).toHaveBeenCalled()
  })

  it('should handle SSE error', async () => {
    const mockReader = {
      read: vi.fn().mockResolvedValueOnce({
        done: false,
        value: new TextEncoder().encode('data: {"stream":"chat","phase":"ERROR","status":"error","data":{"error":"Stream failed"}}\n\n'),
      }),
      releaseLock: vi.fn(),
    }

    vi.mocked(global.fetch).mockResolvedValue({
      ok: true,
      body: {
        getReader: () => mockReader,
      },
    } as any)

    const events = []
    for await (const event of streamSSE('/chat/stream', {})) {
      events.push(event)
    }

    expect(events[0].done).toBe(true)
    expect(events[0].error).toBe('Stream failed')
  })

  it('should handle HTTP error response', async () => {
    vi.mocked(global.fetch).mockResolvedValue({
      ok: false,
      status: 500,
      text: () => Promise.resolve('Internal Server Error'),
    } as any)

    const events = []
    for await (const event of streamSSE('/chat/stream', {})) {
      events.push(event)
    }

    expect(events[0].done).toBe(true)
    expect(events[0].error).toBe('Internal Server Error')
  })

  it('should handle no response body', async () => {
    vi.mocked(global.fetch).mockResolvedValue({
      ok: true,
      body: null,
    } as any)

    const events = []
    for await (const event of streamSSE('/chat/stream', {})) {
      events.push(event)
    }

    expect(events[0].done).toBe(true)
    expect(events[0].error).toBe('No response body')
  })

  it('should skip malformed JSON lines', async () => {
    const mockReader = {
      read: vi.fn()
        .mockResolvedValueOnce({
          done: false,
          value: new TextEncoder().encode('data: {"stream":"chat","phase":"STREAMING","status":"working","data":{"token":"Good"}}\n\n'),
        })
        .mockResolvedValueOnce({
          done: false,
          value: new TextEncoder().encode('data: {invalid json}\n\n'),
        })
        .mockResolvedValueOnce({
          done: false,
          value: new TextEncoder().encode('data: {"stream":"chat","phase":"STREAMING","status":"complete","data":{"token":""}}\n\n'),
        })
        .mockResolvedValueOnce({ done: true, value: undefined }),
      releaseLock: vi.fn(),
    }

    vi.mocked(global.fetch).mockResolvedValue({
      ok: true,
      body: {
        getReader: () => mockReader,
      },
    } as any)

    const tokens: string[] = []
    for await (const event of streamSSE('/chat/stream', {})) {
      if (event.done) break
      tokens.push(event.token)
    }

    expect(tokens).toEqual(['Good'])
  })
})
