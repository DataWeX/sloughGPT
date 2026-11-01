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

import { userAdaptersController } from './user-adapters-controller'

describe('userAdaptersController.list', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /user-adapters and returns stats + adapters', async () => {
    apiClient.apiGet.mockResolvedValue({
      adapters: [
        { user_id: 'u1', rank: 8, alpha: 16, model_dim: 768, created_at: '', updated_at: '', feedback_count: 5 },
      ],
      stats: {
        total_users: 5,
        total_size_bytes: 1024,
        total_size_mb: 1.0,
        adapter_rank: 8,
        model_dim: 768,
        avg_size_per_user_kb: 204.8,
      },
    })

    const result = await userAdaptersController.list()
    expect(result.stats.total_users).toBe(5)
    expect(result.stats.adapter_rank).toBe(8)
    expect(result.adapters).toHaveLength(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/user-adapters')
  })
})

describe('userAdaptersController.get', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /user-adapters/{userId}', async () => {
    apiClient.apiGet.mockResolvedValue({
      user_id: 'u1',
      rank: 8,
      alpha: 16,
      model_dim: 768,
      created_at: '2026-01-01',
      updated_at: '2026-01-02',
      feedback_count: 10,
    })

    const result = await userAdaptersController.get('u1')
    expect(result.user_id).toBe('u1')
    expect(result.feedback_count).toBe(10)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/user-adapters/u1')
  })
})

describe('userAdaptersController.getQuality', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /user-adapters/quality with default params', async () => {
    apiClient.apiGet.mockResolvedValue({ count: 2, adapters: [] })
    await userAdaptersController.getQuality()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/user-adapters/quality?min_feedback_count=3')
  })

  it('includes max_age_days when provided', async () => {
    apiClient.apiGet.mockResolvedValue({ count: 1, adapters: [] })
    await userAdaptersController.getQuality(5, 30)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/user-adapters/quality?min_feedback_count=5&max_age_days=30')
  })
})

describe('userAdaptersController.aggregateBest', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /user-adapters/aggregate-best with defaults', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', user_count: 3, output_path: '/out' })
    const result = await userAdaptersController.aggregateBest()
    expect(apiClient.apiPost).toHaveBeenCalledWith('/user-adapters/aggregate-best', {
      top_k: 10,
      min_feedback_count: 5,
      output_name: 'best_aggregated',
    })
    expect(result.status).toBe('ok')
  })

  it('passes custom params', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok' })
    await userAdaptersController.aggregateBest({
      top_k: 3,
      min_feedback_count: 2,
      output_name: 'custom',
    })
    expect(apiClient.apiPost).toHaveBeenCalledWith('/user-adapters/aggregate-best', {
      top_k: 3,
      min_feedback_count: 2,
      output_name: 'custom',
    })
  })
})

describe('userAdaptersController.prune', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /user-adapters/prune with defaults', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', deleted_count: 2, deleted_users: ['u1', 'u2'] })
    const result = await userAdaptersController.prune()
    expect(apiClient.apiPost).toHaveBeenCalledWith('/user-adapters/prune', {
      min_feedback_count: 1,
      max_age_days: 30,
    })
    expect(result.deleted_count).toBe(2)
  })

  it('passes custom prune params', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', deleted_count: 0, deleted_users: [] })
    await userAdaptersController.prune({ min_feedback_count: 5, max_age_days: 60 })
    expect(apiClient.apiPost).toHaveBeenCalledWith('/user-adapters/prune', {
      min_feedback_count: 5,
      max_age_days: 60,
    })
  })
})

describe('userAdaptersController.reset', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /user-adapters/{userId}/reset', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'reset', user_id: 'u1', feedback_count: 0 })
    const result = await userAdaptersController.reset('u1')
    expect(result.status).toBe('reset')
    expect(result.feedback_count).toBe(0)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/user-adapters/u1/reset')
  })
})
