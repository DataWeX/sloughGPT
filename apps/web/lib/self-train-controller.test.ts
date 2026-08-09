import { describe, it, expect, vi, beforeEach } from 'vitest'
import { selfTrainController } from './self-train-controller'

vi.mock('./http-client', () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
}))

import { apiGet, apiPost } from './http-client'

const mockGet = vi.mocked(apiGet)
const mockPost = vi.mocked(apiPost)

beforeEach(() => {
  vi.clearAllMocks()
})

describe('selfTrainController', () => {
  describe('getStatus', () => {
    it('returns status from nested data', async () => {
      mockGet.mockResolvedValue({ data: { status: 'running', pid: 123, history: ['line1'] } })
      const result = await selfTrainController.getStatus()
      expect(result.status).toBe('running')
      expect(result.pid).toBe(123)
      expect(mockGet).toHaveBeenCalledWith('/self-train/status')
    })

    it('returns flat status', async () => {
      mockGet.mockResolvedValue({ status: 'idle', history: [] })
      const result = await selfTrainController.getStatus()
      expect(result.status).toBe('idle')
    })
  })

  describe('start', () => {
    it('sends body with all options', async () => {
      mockPost.mockResolvedValue({ data: { status: 'started' } })
      await selfTrainController.start({ model: 'gpt2', temperature: 0.5, forever: true })
      expect(mockPost).toHaveBeenCalledWith('/self-train/start', {
        model: 'gpt2',
        temperature: 0.5,
        forever: true,
      })
    })

    it('sends empty body with no options', async () => {
      mockPost.mockResolvedValue({ data: { status: 'started' } })
      await selfTrainController.start()
      expect(mockPost).toHaveBeenCalledWith('/self-train/start', {})
    })

    it('returns error from response', async () => {
      mockPost.mockResolvedValue({ data: { status: 'error', error: 'model not found' } })
      const result = await selfTrainController.start({ model: 'missing' })
      expect(result.status).toBe('error')
      expect(result.error).toBe('model not found')
    })
  })

  describe('stop', () => {
    it('calls stop endpoint', async () => {
      mockPost.mockResolvedValue({})
      await selfTrainController.stop()
      expect(mockPost).toHaveBeenCalledWith('/self-train/stop', {})
    })
  })
})
