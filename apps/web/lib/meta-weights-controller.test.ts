import { describe, it, expect, vi, beforeEach } from 'vitest'
import { metaWeightsController } from './meta-weights-controller'

const mockApiPost = vi.fn()
const mockApiGet = vi.fn()

vi.mock('./http-client', () => ({
  apiPost: (...args: unknown[]) => mockApiPost(...args),
  apiGet: (...args: unknown[]) => mockApiGet(...args),
}))

describe('metaWeightsController', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('getWeights calls POST /meta-weights/get with correct params', async () => {
    mockApiPost.mockResolvedValue({ temperature: 0.8, top_p: 0.9, repetition_penalty: 1.1, top_k: 40, style_bias: 0.5, confidence_boost: 0.6, based_on_samples: 5 })
    const result = await metaWeightsController.getWeights('hello world', 10, 'user1')
    expect(mockApiPost).toHaveBeenCalledWith('/meta-weights/get', { user_message: 'hello world', k: 10, user_id: 'user1' })
    expect(result).toEqual({ temperature: 0.8, top_p: 0.9, repetition_penalty: 1.1, top_k: 40, style_bias: 0.5, confidence_boost: 0.6, based_on_samples: 5 })
  })

  it('getWeights uses default k and userId', async () => {
    mockApiPost.mockResolvedValue({ temperature: 0.7 })
    await metaWeightsController.getWeights('test')
    expect(mockApiPost).toHaveBeenCalledWith('/meta-weights/get', { user_message: 'test', k: 5, user_id: 'default' })
  })

  it('getStats calls GET /meta-weights/stats', async () => {
    mockApiGet.mockResolvedValue({ history_length: 10, avg_temperature: 0.75 })
    const result = await metaWeightsController.getStats()
    expect(mockApiGet).toHaveBeenCalledWith('/meta-weights/stats')
    expect(result).toEqual({ history_length: 10, avg_temperature: 0.75 })
  })

  it('ping calls GET /meta-weights/ping', async () => {
    mockApiGet.mockResolvedValue({ status: 'ok' })
    const result = await metaWeightsController.ping()
    expect(mockApiGet).toHaveBeenCalledWith('/meta-weights/ping')
    expect(result).toEqual({ status: 'ok' })
  })

  it('propagates errors from apiPost', async () => {
    mockApiPost.mockRejectedValue(new Error('network error'))
    await expect(metaWeightsController.getWeights('fail')).rejects.toThrow('network error')
  })

  it('propagates errors from apiGet', async () => {
    mockApiGet.mockRejectedValue(new Error('timeout'))
    await expect(metaWeightsController.getStats()).rejects.toThrow('timeout')
  })
})
