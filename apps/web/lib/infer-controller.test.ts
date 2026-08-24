import { describe, it, expect, vi, beforeEach } from 'vitest'

const mockApiPost = vi.fn()
const mockApiGet = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiPost: (...args: unknown[]) => mockApiPost(...args),
  apiGet: (...args: unknown[]) => mockApiGet(...args),
}))

const { inferController } = await import('@/lib/infer-controller')

describe('inferController', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('generate calls POST /infer with correct params', async () => {
    mockApiPost.mockResolvedValue({ text: 'Hello', model: 'gpt2', tokens_generated: 1, elapsed_ms: 10 })
    const result = await inferController.generate({ prompt: 'Hi', max_new_tokens: 10, temperature: 0.7 })
    expect(mockApiPost).toHaveBeenCalledWith('/infer', { prompt: 'Hi', max_new_tokens: 10, temperature: 0.7 })
    expect(result).toEqual({ text: 'Hello', model: 'gpt2', tokens_generated: 1, elapsed_ms: 10 })
  })

  it('health calls GET /infer/health', async () => {
    mockApiGet.mockResolvedValue({ status: 'ok', model_loaded: true, has_streaming: true, has_embedding: false })
    const result = await inferController.health()
    expect(mockApiGet).toHaveBeenCalledWith('/infer/health')
    expect(result).toEqual({ status: 'ok', model_loaded: true, has_streaming: true, has_embedding: false })
  })

  it('info calls GET /infer/info', async () => {
    mockApiGet.mockResolvedValue({ model_id: 'gpt2', model_type: 'gpt2', num_parameters: 124000000, vocab_size: 50257, max_context: 1024, num_layers: 12, has_tokenizer: true, has_streaming: true, has_embedding: false, extra: {} })
    const result = await inferController.info()
    expect(mockApiGet).toHaveBeenCalledWith('/infer/info')
    expect(result.model_id).toBe('gpt2')
  })

  it('embed calls POST /infer/embed with text', async () => {
    mockApiPost.mockResolvedValue({ embedding: [0.1, 0.2], dimensions: 2, model: 'e5' })
    const result = await inferController.embed('hello')
    expect(mockApiPost).toHaveBeenCalledWith('/infer/embed', { text: 'hello', model: undefined })
    expect(result.dimensions).toBe(2)
  })

  it('tokenize calls POST /infer/tokenize', async () => {
    mockApiPost.mockResolvedValue({ tokens: ['hello'], ids: [1], count: 1 })
    const result = await inferController.tokenize('hello')
    expect(mockApiPost).toHaveBeenCalledWith('/infer/tokenize', { text: 'hello', model: undefined })
    expect(result.count).toBe(1)
  })

  it('detokenize calls POST /infer/detokenize', async () => {
    mockApiPost.mockResolvedValue({ text: 'hello', count: 1 })
    const result = await inferController.detokenize([1, 2])
    expect(mockApiPost).toHaveBeenCalledWith('/infer/detokenize', { ids: [1, 2], model: undefined })
    expect(result.text).toBe('hello')
  })

  it('propagates errors from apiPost', async () => {
    mockApiPost.mockRejectedValue(new Error('network error'))
    await expect(inferController.generate({ prompt: 'fail' })).rejects.toThrow('network error')
  })

  it('propagates errors from apiGet', async () => {
    mockApiGet.mockRejectedValue(new Error('timeout'))
    await expect(inferController.health()).rejects.toThrow('timeout')
  })
})
