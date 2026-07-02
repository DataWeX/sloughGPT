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

import { activityController } from './activity-controller'

describe('activityController', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('recordData POSTs /activity/data', async () => {
    apiClient.apiPost.mockResolvedValue({ id: 1, path: '/data/rec_1.npy', samples: 128 })
    const body = { data: [[0.1, 0.2, 0.3, 0.0, 0.0, 9.81]], label: 0 }
    const result = await activityController.recordData(body)
    expect(result.id).toBe(1)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/activity/data', { data: [[0.1, 0.2, 0.3, 0.0, 0.0, 9.81]], label: 0 })
  })

  it('train POSTs /activity/train with defaults', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', epochs: 5, final_loss: 1.2, val_accuracy: 0.75, num_samples: 100, message: 'done' })
    const result = await activityController.train()
    expect(result.status).toBe('ok')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/activity/train', {})
  })

  it('train POSTs /activity/train with custom params', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', epochs: 10, final_loss: 0.5, val_accuracy: 0.88, num_samples: 200, message: 'done' })
    const result = await activityController.train({ epochs: 10, lr: 0.001, batch_size: 32 })
    expect(result.epochs).toBe(10)
    expect(result.val_accuracy).toBe(0.88)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/activity/train', { epochs: 10, lr: 0.001, batch_size: 32 })
  })

  it('predict POSTs /activity/predict', async () => {
    apiClient.apiPost.mockResolvedValue({ activity: 'walking', class_id: 1, confidence: 0.92, probabilities: [0.02, 0.92, 0.03, 0.01, 0.01, 0.01] })
    const result = await activityController.predict({ data: [[0.1] as unknown as number[]] })
    expect(result.activity).toBe('walking')
    expect(result.confidence).toBe(0.92)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/activity/predict', { data: [[0.1]] })
  })

  it('status GETs /activity/status', async () => {
    apiClient.apiGet.mockResolvedValue({ model_loaded: true, num_recordings: 50, num_labels: 6, activities: ['walking', 'running'], device: 'cpu' })
    const result = await activityController.status()
    expect(result.model_loaded).toBe(true)
    expect(result.device).toBe('cpu')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/activity/status')
  })

  it('dataset GETs /activity/dataset', async () => {
    apiClient.apiGet.mockResolvedValue({ recordings: [{ id: 1, path: '/data/rec_1.npy', samples: 128, label: 0, activity: 'stationary' }], total: 1 })
    const result = await activityController.dataset()
    expect(result.recordings.length).toBe(1)
    expect(result.total).toBe(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/activity/dataset')
  })

  it('deleteAll DELETEs /activity/data', async () => {
    apiClient.apiDelete.mockResolvedValue({ deleted: 42 })
    const result = await activityController.deleteAll()
    expect(result.deleted).toBe(42)
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/activity/data')
  })

  it('handles 404 from status', async () => {
    apiClient.apiGet.mockRejectedValue(new Error('Not Found'))
    await expect(activityController.status()).rejects.toThrow('Not Found')
  })
})
