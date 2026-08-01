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

import { datasetController } from './dataset-controller'

describe('datasetController.list', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GET /datasets and returns rows', async () => {
    const mockData = {
      datasets: [
        { id: 'ds1', name: 'shakespeare', source: 'local', size: 12345, samples: 100, type: 'text', created_at: '2026-01-01' },
      ],
    }
    apiClient.apiGet.mockResolvedValue(mockData)

    const rows = await datasetController.list()

    expect(apiClient.apiGet).toHaveBeenCalledWith('/datasets')
    expect(rows).toHaveLength(1)
    expect(rows[0].id).toBe('ds1')
    expect(rows[0].name).toBe('shakespeare')
  })

  it('handles empty datasets', async () => {
    apiClient.apiGet.mockResolvedValue({ datasets: [] })

    const rows = await datasetController.list()
    expect(rows).toEqual([])
  })

  it('throws when GET /datasets is not ok', async () => {
    apiClient.apiGet.mockRejectedValue(new Error('502'))

    await expect(datasetController.list()).rejects.toThrow('502')
  })
})

describe('datasetController versioning', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('creates a version via POST /datasets/{id}/versions', async () => {
    apiClient.apiPost.mockResolvedValue({ timestamp: '20260801120000', message: 'Version created' })

    const res = await datasetController.createVersion('ds1')

    expect(apiClient.apiPost).toHaveBeenCalledWith('/datasets/ds1/versions')
    expect(res.timestamp).toBe('20260801120000')
  })

  it('lists versions via GET /datasets/{id}/versions', async () => {
    apiClient.apiGet.mockResolvedValue({ versions: ['20260801120000', '20260801110000'], count: 2 })

    const res = await datasetController.listVersions('ds1')

    expect(apiClient.apiGet).toHaveBeenCalledWith('/datasets/ds1/versions')
    expect(res.versions).toHaveLength(2)
    expect(res.count).toBe(2)
  })

  it('restores a version via POST /datasets/{id}/versions/{timestamp}', async () => {
    apiClient.apiPost.mockResolvedValue({ success: true, message: 'Version restored' })

    const res = await datasetController.restoreVersion('ds1', '20260801120000')

    expect(apiClient.apiPost).toHaveBeenCalledWith('/datasets/ds1/versions/20260801120000')
    expect(res.success).toBe(true)
  })
})
