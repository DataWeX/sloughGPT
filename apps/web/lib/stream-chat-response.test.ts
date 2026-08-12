import { describe, it, expect, vi, beforeEach } from 'vitest'
import { streamChatResponse } from './stream-chat-response'

describe('streamChatResponse', () => {
  const mockParams = {
    messages: [{ role: 'user', content: 'hello' }],
    model: 'gpt2',
    systemPrompt: '',
    maxTokens: 256,
    temperature: 0.8,
    userId: 'test',
    sessionId: 'test-session',
    onToken: vi.fn(),
    onComplete: vi.fn(),
    onError: vi.fn(),
    onToolCall: vi.fn(),
    onThinking: vi.fn(),
    onKnowledge: vi.fn(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal('fetch', vi.fn())
  })

  function mockSSE(lines: string[]) {
    const encoder = new TextEncoder()
    const stream = new ReadableStream({
      start(controller) {
        for (const line of lines) {
          controller.enqueue(encoder.encode(line + '\n'))
        }
        controller.close()
      },
    })
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      body: stream,
    } as Response)
  }

  it('dispatches tool_call executing event', async () => {
    mockSSE([
      'data: {"stream":"chat","phase":"TOOL","status":"working","data":{"tool":"calculator","args":{"expression":"2+2"},"status":"executing"},"message":"Running tool: calculator"}',
    ])
    await streamChatResponse(mockParams)
    expect(mockParams.onToolCall).toHaveBeenCalledWith(
      expect.objectContaining({ tool: 'calculator', status: 'executing' })
    )
  })

  it('dispatches tool_call success event', async () => {
    mockSSE([
      'data: {"stream":"chat","phase":"TOOL","status":"complete","data":{"tool":"calculator","status":"success","output":"4","duration_ms":12.5},"message":"Tool calculator completed"}',
    ])
    await streamChatResponse(mockParams)
    expect(mockParams.onToolCall).toHaveBeenCalledWith(
      expect.objectContaining({ tool: 'calculator', status: 'success', output: '4', duration_ms: 12.5 })
    )
  })

  it('dispatches tool_call error event', async () => {
    mockSSE([
      'data: {"stream":"chat","phase":"TOOL","status":"error","data":{"tool":"run_code","status":"error","error":"SyntaxError"},"message":"Tool run_code failed"}',
    ])
    await streamChatResponse(mockParams)
    expect(mockParams.onToolCall).toHaveBeenCalledWith(
      expect.objectContaining({ tool: 'run_code', status: 'error', error: 'SyntaxError' })
    )
  })

  it('handles multiple SSE events including tool calls', async () => {
    mockSSE([
      'data: {"stream":"chat","phase":"STREAMING","status":"thinking","data":{},"message":"Thinking..."}',
      'data: {"stream":"chat","phase":"TOOL","status":"working","data":{"tool":"calculator","status":"executing"},"message":"Running"}',
      'data: {"stream":"chat","phase":"TOOL","status":"complete","data":{"tool":"calculator","status":"success","output":"4"},"message":"Done"}',
      'data: {"stream":"chat","phase":"STREAMING","status":"working","data":{"token":"The"},"message":""}',
      'data: {"stream":"chat","phase":"STREAMING","status":"working","data":{"token":" answer"},"message":""}',
      'data: {"stream":"chat","phase":"STREAMING","status":"complete","data":{"token":""},"message":""}',
    ])
    await streamChatResponse(mockParams)
    expect(mockParams.onThinking).toHaveBeenCalled()
    expect(mockParams.onToolCall).toHaveBeenCalledTimes(2)
    expect(mockParams.onToolCall).toHaveBeenNthCalledWith(1, expect.objectContaining({ status: 'executing' }))
    expect(mockParams.onToolCall).toHaveBeenNthCalledWith(2, expect.objectContaining({ status: 'success' }))
    expect(mockParams.onToken).toHaveBeenCalledWith('The')
    expect(mockParams.onToken).toHaveBeenCalledWith(' answer')
    expect(mockParams.onComplete).toHaveBeenCalled()
  })

  it('does not dispatch tool_call for non-tool events', async () => {
    mockSSE(['data: {"stream":"chat","phase":"STREAMING","status":"working","data":{"token":"hello"},"message":""}'])
    await streamChatResponse(mockParams)
    expect(mockParams.onToolCall).not.toHaveBeenCalled()
  })

  it('dispatches knowledge event', async () => {
    mockSSE([
      'data: {"stream":"chat","phase":"STREAMING","status":"working","data":{"source":"rag","fact_count":3},"message":""}',
    ])
    await streamChatResponse(mockParams)
    expect(mockParams.onKnowledge).toHaveBeenCalledWith('rag', 3)
  })

  it('calls onComplete on stream finish', async () => {
    mockSSE([
      'data: {"stream":"chat","phase":"STREAMING","status":"complete","data":{"token":""},"message":""}',
    ])
    await streamChatResponse(mockParams)
    expect(mockParams.onComplete).toHaveBeenCalled()
  })

  it('calls onError on non-ok response', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 503,
      text: () => Promise.resolve('Service unavailable'),
    } as unknown as Response)
    await streamChatResponse(mockParams)
    expect(mockParams.onError).toHaveBeenCalledWith(503, 'Service unavailable')
  })
})
