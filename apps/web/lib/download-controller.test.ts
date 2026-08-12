import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./auth', () => ({
  useAuthStore: {
    getState: () => ({ token: null as string | null }),
  },
}))

vi.mock('./config', () => ({
  PUBLIC_API_URL: 'http://127.0.0.1:9',
}))

vi.mock('./error-store', () => ({
  useErrorStore: { getState: () => ({ addError: vi.fn() }) },
}))

import { setupApiMocks, apiClient } from './__test-helper'
setupApiMocks()

import {
  startDownload,
  getDownloadStatus,
} from './download-controller'

describe('startDownload', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /models/download with model_id', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'started' })
    await startDownload('gpt2')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/models/download', { model_id: 'gpt2', total_bytes_hint: 0 })
  })

  it('includes totalBytesHint when provided', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'started' })
    await startDownload('gpt2', 500000000)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/models/download', { model_id: 'gpt2', total_bytes_hint: 500000000 })
  })

  it('returns started status', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'started' })
    const result = await startDownload('gpt2')
    expect(result.status).toBe('started')
  })
})

describe('getDownloadStatus', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /models/download/{modelId}', async () => {
    apiClient.apiGet.mockResolvedValue({
      model_id: 'gpt2',
      status: 'downloading',
      percentage: 45,
      bytes_downloaded: 100,
      total_bytes: 200,
    })
    const result = await getDownloadStatus('gpt2')
    expect(result.model_id).toBe('gpt2')
    expect(result.status).toBe('downloading')
    expect(result.percentage).toBe(45)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/models/download/gpt2')
  })

  it('URL-encodes special characters in modelId', async () => {
    apiClient.apiGet.mockResolvedValue({ model_id: 'org/model', status: 'complete' })
    await getDownloadStatus('org/model')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/models/download/org%2Fmodel')
  })

  it('returns complete status', async () => {
    apiClient.apiGet.mockResolvedValue({ model_id: 'gpt2', status: 'complete', percentage: 100 })
    const result = await getDownloadStatus('gpt2')
    expect(result.status).toBe('complete')
  })
})
