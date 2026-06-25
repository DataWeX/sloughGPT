// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { streamChatResponse } from './stream-chat-response'

type MockParams = {
  messages: { role: string; content: string }[]
  model: string
  systemPrompt: string
  maxTokens: number
  temperature: number
  userId: string
  sessionId: string
  onToken: ReturnType<typeof vi.fn>
  onComplete: ReturnType<typeof vi.fn>
  onError: ReturnType<typeof vi.fn>
  onKnowledge?: ReturnType<typeof vi.fn>
  onThinking?: ReturnType<typeof vi.fn>
  signal?: AbortSignal
  images?: string[]
}

function mockFetchStream(chunks: string[], status = 200) {
  const encoder = new TextEncoder()
  const stream = new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk))
      }
      controller.close()
    },
  })
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    text: () => Promise.resolve(''),
    body: { getReader: () => stream.getReader() },
  }))
}

describe('streamChatResponse', () => {
  const params = (): MockParams => ({
    messages: [{ role: 'user', content: 'hi' }],
    model: 'gpt2',
    systemPrompt: 'You are helpful',
    maxTokens: 100,
    temperature: 0.7,
    userId: 'u1',
    sessionId: 's1',
    onToken: vi.fn(),
    onComplete: vi.fn(),
    onError: vi.fn(),
  })

  beforeEach(() => { vi.clearAllMocks() })
  afterEach(() => { vi.unstubAllGlobals() })

  it('calls onToken for each token event', async () => {
    const p = params()
    mockFetchStream([
      `data: {"stream":"chat","phase":"STREAMING","status":"working","data":{"token":"Hello"}}\n`,
      `data: {"stream":"chat","phase":"STREAMING","status":"working","data":{"token":" world"}}\n`,
      `data: {"stream":"chat","phase":"STREAMING","status":"complete","data":{"token":"!"}}\n`,
    ])
    await streamChatResponse(p)
    expect(p.onToken).toHaveBeenNthCalledWith(1, 'Hello')
    expect(p.onToken).toHaveBeenNthCalledWith(2, ' world')
    expect(p.onToken).toHaveBeenNthCalledWith(3, '!')
  })

  it('calls onComplete when complete event received', async () => {
    const p = params()
    mockFetchStream([
      `data: {"status":"working","data":{"token":"ok"}}\n`,
      `data: {"status":"complete"}\n`,
    ])
    await streamChatResponse(p)
    expect(p.onComplete).toHaveBeenCalled()
  })

  it('calls onError on HTTP error', async () => {
    const p = params()
    mockFetchStream([], 500)
    await streamChatResponse(p)
    expect(p.onError).toHaveBeenCalledWith(500, '')
  })

  it('calls onError on SSE error status', async () => {
    const p = params()
    mockFetchStream([
      `data: {"status":"error","data":{"error":"OOM"}}\n`,
    ])
    await streamChatResponse(p)
    expect(p.onError).toHaveBeenCalledWith(500, 'OOM')
  })

  it('calls onError with message when data.error absent', async () => {
    const p = params()
    mockFetchStream([
      `data: {"status":"error","message":"Failed"}\n`,
    ])
    await streamChatResponse(p)
    expect(p.onError).toHaveBeenCalledWith(500, 'Failed')
  })

  it('calls onKnowledge when source present', async () => {
    const p = params()
    p.onKnowledge = vi.fn()
    mockFetchStream([
      `data: {"status":"working","data":{"token":"A","source":"docs","fact_count":3}}\n`,
      `data: {"status":"complete"}\n`,
    ])
    await streamChatResponse(p)
    expect(p.onKnowledge).toHaveBeenCalledWith('docs', 3)
  })

  it('calls onThinking on thinking status', async () => {
    const p = params()
    p.onThinking = vi.fn()
    mockFetchStream([
      `data: {"status":"thinking"}\n`,
      `data: {"status":"working","data":{"token":"result"}}\n`,
      `data: {"status":"complete"}\n`,
    ])
    await streamChatResponse(p)
    expect(p.onThinking).toHaveBeenCalled()
  })

  it('calls onToken("") for empty response', async () => {
    const p = params()
    mockFetchStream([`data: {"status":"complete"}\n`])
    await streamChatResponse(p)
    expect(p.onToken).toHaveBeenCalledWith('')
    expect(p.onComplete).toHaveBeenCalled()
  })

  it('handles split SSE chunks across boundaries', async () => {
    const p = params()
    mockFetchStream([
      `data: {"stream":"chat","phase":"STREAMIN`,
      `G","status":"working","data":{"token":"hi"}}\n`,
      `data: {"status":"complete"}\n`,
    ])
    await streamChatResponse(p)
    expect(p.onToken).toHaveBeenCalledWith('hi')
    expect(p.onComplete).toHaveBeenCalled()
  })

  it('skips malformed JSON lines', async () => {
    const p = params()
    mockFetchStream([
      `data: not-json\n`,
      `data: {"status":"working","data":{"token":"ok"}}\n`,
      `data: {"status":"complete"}\n`,
    ])
    await streamChatResponse(p)
    expect(p.onToken).toHaveBeenCalledWith('ok')
  })

  it('ignores [DONE] sentinel', async () => {
    const p = params()
    mockFetchStream([
      `data: [DONE]\n`,
      `data: {"status":"working","data":{"token":"ok"}}\n`,
      `data: {"status":"complete"}\n`,
    ])
    await streamChatResponse(p)
    expect(p.onToken).toHaveBeenCalledWith('ok')
  })

  it('handles buffer split in middle of data: prefix', async () => {
    const p = params()
    const payload = `{"status":"working","data":{"token":"y"}}`
    mockFetchStream([
      `dat`,
      `a: ${payload}\n`,
      `data: {"status":"complete"}\n`,
    ])
    await streamChatResponse(p)
    expect(p.onToken).toHaveBeenCalledWith('y')
  })

  it('passes AbortSignal to fetch', async () => {
    const p = params()
    const signal = new AbortController().signal
    p.signal = signal
    mockFetchStream([`data: {"status":"complete"}\n`])
    await streamChatResponse(p)
    expect(fetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ signal })
    )
  })
})
