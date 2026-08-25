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
    onKnowledge: vi.fn(),
    onThinking: vi.fn(),
    onToolCall: vi.fn(),
    onMemory: vi.fn(),
    onRagVerification: vi.fn(),
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

  it('dispatches memory event after complete', async () => {
    mockSSE([
      'data: {"stream":"chat","phase":"STREAMING","status":"complete","data":{"token":""},"message":""}',
      'data: {"stream":"chat","phase":"MEMORY","status":"success","data":{"stored":true,"fact":"The capital of France is Paris."},"message":"New fact remembered"}',
    ])
    await streamChatResponse(mockParams)
    expect(mockParams.onComplete).toHaveBeenCalledTimes(1)
    expect(mockParams.onMemory).toHaveBeenCalledWith({ stored: true, fact: 'The capital of France is Paris.' })
  })

  it('dispatches memory event without fact when payload omits it', async () => {
    mockSSE([
      'data: {"stream":"chat","phase":"MEMORY","status":"success","data":{"stored":true},"message":"New fact remembered"}',
    ])
    await streamChatResponse(mockParams)
    expect(mockParams.onMemory).toHaveBeenCalledWith({ stored: true, fact: undefined })
  })

  it('dispatches memory skipped event', async () => {
    mockSSE([
      'data: {"stream":"chat","phase":"MEMORY","status":"success","data":{"stored":false},"message":""}',
    ])
    await streamChatResponse(mockParams)
    expect(mockParams.onMemory).toHaveBeenCalledWith({ stored: false, fact: undefined })
  })

  it('dispatches RAG verification event', async () => {
    mockSSE([
      'data: {"stream":"chat","phase":"RAG_VERIFICATION","status":"success","data":{"confidence":0.92,"is_verified":true,"hallucination_rate":0.05,"citations":"[1] Source A","grounded_claims":3,"hallucinated_claims":0},"message":"RAG grounding verification complete"}',
    ])
    await streamChatResponse(mockParams)
    expect(mockParams.onRagVerification).toHaveBeenCalledWith({
      confidence: 0.92,
      is_verified: true,
      hallucination_rate: 0.05,
      citations: '[1] Source A',
      grounded_claims: 3,
      hallucinated_claims: 0,
    })
  })

  it('dispatches RAG verification with defaults on missing fields', async () => {
    mockSSE([
      'data: {"stream":"chat","phase":"RAG_VERIFICATION","status":"success","data":{},"message":""}',
    ])
    await streamChatResponse(mockParams)
    expect(mockParams.onRagVerification).toHaveBeenCalledWith({
      confidence: 0,
      is_verified: false,
      hallucination_rate: 0,
      citations: '',
      grounded_claims: 0,
      hallucinated_claims: 0,
    })
  })

  it('dispatches all stored facts when the payload carries a facts array', async () => {
    mockSSE([
      'data: {"stream":"chat","phase":"MEMORY","status":"success","data":{"stored":true,"fact":"A.","facts":["A.","B.","C."]},"message":"New fact remembered"}',
    ])
    await streamChatResponse(mockParams)
    expect(mockParams.onMemory).toHaveBeenCalledWith({ stored: true, fact: 'A.', facts: ['A.', 'B.', 'C.'] })
  })

  it('drops non-string entries from the facts array', async () => {
    mockSSE([
      'data: {"stream":"chat","phase":"MEMORY","status":"success","data":{"stored":true,"fact":"A.","facts":["A.",7,null,"B."]},"message":"New fact remembered"}',
    ])
    await streamChatResponse(mockParams)
    expect(mockParams.onMemory).toHaveBeenCalledWith({ stored: true, fact: 'A.', facts: ['A.', 'B.'] })
  })

  it('does not call onComplete twice when post-complete events follow', async () => {
    mockSSE([
      'data: {"stream":"chat","phase":"STREAMING","status":"complete","data":{"token":""},"message":""}',
      'data: {"stream":"chat","phase":"MEMORY","status":"success","data":{"stored":true,"fact":"Gradient descent is an optimizer."},"message":"New fact remembered"}',
      'data: {"stream":"chat","phase":"STREAMING","status":"working","data":{"token":"tail"},"message":""}',
    ])
    await streamChatResponse(mockParams)
    expect(mockParams.onComplete).toHaveBeenCalledTimes(1)
    expect(mockParams.onMemory).toHaveBeenCalledTimes(1)
  })

  it('calls onError on non-ok response', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 503,
      statusText: 'Service Unavailable',
      text: () => Promise.resolve('Service unavailable'),
    } as unknown as Response)
    await streamChatResponse(mockParams)
    expect(mockParams.onError).toHaveBeenCalledWith(503, expect.stringContaining('503'))
  })

  it('retries on transient 503 before calling onError', async () => {
    const encoder = new TextEncoder()
    const successStream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode('data: {"stream":"chat","phase":"STREAMING","status":"working","data":{"token":"Hi"},"message":""}\n'))
        controller.enqueue(encoder.encode('data: {"stream":"chat","phase":"STREAMING","status":"complete","data":{},"message":""}\n'))
        controller.close()
      },
    })
    const errorResponse = {
      ok: false,
      status: 503,
      statusText: 'Service Unavailable',
    } as unknown as Response
    const successResponse = {
      ok: true,
      body: successStream,
    } as unknown as Response

    vi.mocked(fetch)
      .mockResolvedValueOnce(errorResponse)
      .mockResolvedValueOnce(errorResponse)
      .mockResolvedValue(successResponse)

    await streamChatResponse(mockParams)

    expect(mockParams.onToken).toHaveBeenCalledWith('Hi')
    expect(mockParams.onComplete).toHaveBeenCalled()
    expect(mockParams.onError).not.toHaveBeenCalled()
  })

  it('calls onError after exhausting retries on persistent 503', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 503,
      statusText: 'Service Unavailable',
    } as unknown as Response)

    await streamChatResponse(mockParams)

    expect(mockParams.onError).toHaveBeenCalledTimes(1)
    expect(mockParams.onError).toHaveBeenCalledWith(503, expect.stringContaining('503'))
  })

  it('does not retry on non-retryable 400 error', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 400,
      statusText: 'Bad Request',
    } as unknown as Response)

    await streamChatResponse(mockParams)

    expect(mockParams.onError).toHaveBeenCalledTimes(1)
    expect(mockParams.onError).toHaveBeenCalledWith(400, expect.stringContaining('400'))
  })

  it('does not retry when signal is aborted', async () => {
    const ac = new AbortController()
    ac.abort()
    vi.mocked(fetch).mockImplementation(() => {
      const err = new DOMException('The operation was aborted.', 'AbortError')
      return Promise.reject(err)
    })

    await streamChatResponse({ ...mockParams, signal: ac.signal })

    expect(mockParams.onError).not.toHaveBeenCalled()
    expect(mockParams.onComplete).not.toHaveBeenCalled()
    expect(fetch).toHaveBeenCalledTimes(1)
  })
})
