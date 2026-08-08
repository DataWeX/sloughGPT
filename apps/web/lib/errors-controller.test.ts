import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./auth', () => ({
  useAuthStore: {
    getState: () => ({ token: null as string | null }),
  },
}))

vi.mock('./config', () => ({
  PUBLIC_API_URL: 'http://127.0.0.1:9',
}))

import { setupApiMocks, apiClient } from './__test-helper'
setupApiMocks()

import { errorsController } from './errors-controller'

describe('errorsController.getGrouped', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /errors/grouped and returns groups', async () => {
    apiClient.apiGet.mockResolvedValue({ groups: [{ fingerprint: 'abc', message: 'err', source: 'chat', count: 5, latest: '2025-01-01', sample_id: '1', sample_url: '/chat', sample_line: 10 }] })

    const result = await errorsController.getGrouped()
    expect(result).toHaveLength(1)
    expect(result[0].fingerprint).toBe('abc')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/errors/grouped')
  })

  it('returns empty array on missing groups', async () => {
    apiClient.apiGet.mockResolvedValue({})
    const result = await errorsController.getGrouped()
    expect(result).toEqual([])
  })
})

describe('errorsController.getRecent', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /errors/recent with limit', async () => {
    apiClient.apiGet.mockResolvedValue({ errors: [{ id: '1', message: 'fail', source: 'web', timestamp: '2025-01-01', fingerprint: 'abc' }], total: 42 })

    const result = await errorsController.getRecent(50)
    expect(result.errors).toHaveLength(1)
    expect(result.total).toBe(42)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/errors/recent?limit=50')
  })

  it('returns defaults on missing data', async () => {
    apiClient.apiGet.mockResolvedValue({})
    const result = await errorsController.getRecent()
    expect(result.errors).toEqual([])
    expect(result.total).toBe(0)
  })
})

describe('errorsController.getTrends', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /errors/trends with hours', async () => {
    apiClient.apiGet.mockResolvedValue({ trends: [{ hour: '2025-01-01T12:00', count: 3 }] })

    const result = await errorsController.getTrends(48)
    expect(result).toHaveLength(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/errors/trends?hours=48')
  })

  it('returns empty array on missing trends', async () => {
    apiClient.apiGet.mockResolvedValue({ data: {} })
    const result = await errorsController.getTrends()
    expect(result).toEqual([])
  })
})

describe('errorsController.clear', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('DELETEs /errors/clear', async () => {
    apiClient.apiDelete.mockResolvedValue(undefined)

    await errorsController.clear()
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/errors/clear')
  })
})

describe('errorsController.export', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /errors/export', async () => {
    apiClient.apiGet.mockResolvedValue({ errors: [], total: 0 })

    const result = await errorsController.export()
    expect(result).toEqual({ errors: [], total: 0 })
    expect(apiClient.apiGet).toHaveBeenCalledWith('/errors/export')
  })
})
