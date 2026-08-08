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

import { experimentsController } from './experiments-controller'

describe('experimentsController.list', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /experiments and maps to objects', async () => {
    apiClient.apiGet.mockResolvedValue({ experiments: ['exp_1', 'exp_2'] })

    const result = await experimentsController.list()
    expect(result).toHaveLength(2)
    expect(result[0]).toEqual({ id: 'exp_1' })
    expect(result[1]).toEqual({ id: 'exp_2' })
    expect(apiClient.apiGet).toHaveBeenCalledWith('/experiments')
  })

  it('returns empty array on missing experiments', async () => {
    apiClient.apiGet.mockResolvedValue({})
    const result = await experimentsController.list()
    expect(result).toEqual([])
  })
})

describe('experimentsController.create', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /experiments with name', async () => {
    apiClient.apiPost.mockResolvedValue({ id: 'exp_test_2025', name: 'test', created: true })

    const result = await experimentsController.create('test')
    expect(result.id).toBe('exp_test_2025')
    expect(result.created).toBe(true)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/experiments', { name: 'test' })
  })
})

describe('experimentsController.delete', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('DELETEs /experiments/{id}', async () => {
    apiClient.apiDelete.mockResolvedValue({ id: 'exp_1', deleted: true })

    const result = await experimentsController.delete('exp_1')
    expect(result.deleted).toBe(true)
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/experiments/exp_1')
  })

  it('URL-encodes experiment ID', async () => {
    apiClient.apiDelete.mockResolvedValue({ id: 'exp/with/slash', deleted: true })

    await experimentsController.delete('exp/with/slash')
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/experiments/exp%2Fwith%2Fslash')
  })
})

describe('experimentsController.logMetric', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs metric to /experiments/{id}/log_metric', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'logged' })

    const result = await experimentsController.logMetric('exp_1', 'loss', 0.5)
    expect(result.status).toBe('logged')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/experiments/exp_1/log_metric', { metric_name: 'loss', value: 0.5 })
  })
})

describe('experimentsController.logParam', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs param to /experiments/{id}/log_param', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'logged' })

    const result = await experimentsController.logParam('exp_1', 'lr', '0.001')
    expect(result.status).toBe('logged')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/experiments/exp_1/log_param', { param_name: 'lr', value: '0.001' })
  })
})

describe('experimentsController.complete', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /experiments/{id}/complete', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'completed' })

    const result = await experimentsController.complete('exp_1')
    expect(result.status).toBe('completed')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/experiments/exp_1/complete', {})
  })
})
