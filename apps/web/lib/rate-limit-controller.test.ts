import { describe, it, expect, vi, beforeEach } from 'vitest'
import { rateLimitController } from './rate-limit-controller'

vi.mock('./http-client', () => ({
  apiGet: vi.fn(),
}))

import { apiGet } from './http-client'

const mockGet = vi.mocked(apiGet)

beforeEach(() => {
  vi.clearAllMocks()
})

describe('rateLimitController', () => {
  describe('getStatus', () => {
    it('returns rate limit status', async () => {
      mockGet.mockResolvedValue({ requests_per_minute: 60, burst_size: 10, enabled: true })
      const result = await rateLimitController.getStatus()
      expect(result.requests_per_minute).toBe(60)
      expect(mockGet).toHaveBeenCalledWith('/rate-limit/status')
    })
  })

  describe('check', () => {
    it('returns check result', async () => {
      mockGet.mockResolvedValue({ allowed: true, wait_time: 0 })
      const result = await rateLimitController.check()
      expect(result.allowed).toBe(true)
      expect(mockGet).toHaveBeenCalledWith('/rate-limit/check')
    })
  })
})
