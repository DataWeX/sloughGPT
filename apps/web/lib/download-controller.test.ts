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
  listDownloads,
  cancelDownload,
  verifyDownload,
  retryDownload,
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
})

describe('listDownloads', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /models/downloads and returns downloads map', async () => {
    apiClient.apiGet.mockResolvedValue({
      downloads: {
        gpt2: { model_id: 'gpt2', status: 'complete', percentage: 100 },
        llama: { model_id: 'llama', status: 'queued', percentage: 0 },
      },
    })
    const result = await listDownloads()
    expect(result.gpt2.status).toBe('complete')
    expect(result.llama.status).toBe('queued')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/models/downloads')
  })
})

describe('cancelDownload', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /models/download/{modelId}/cancel', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'cancelled' })
    await cancelDownload('gpt2')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/models/download/gpt2/cancel')
  })
})

describe('verifyDownload', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /models/download/{modelId}/verify', async () => {
    apiClient.apiPost.mockResolvedValue({
      status: 'verified',
      model_id: 'gpt2',
      verified: true,
      size_on_disk: '500MB',
    })
    const result = await verifyDownload('gpt2')
    expect(result.status).toBe('verified')
    expect(result.verified).toBe(true)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/models/download/gpt2/verify')
  })

  it('returns corrupt status when verification fails', async () => {
    apiClient.apiPost.mockResolvedValue({
      status: 'corrupt',
      model_id: 'gpt2',
      verified: false,
      missing_files: ['model.safetensors'],
      missing_files_count: 1,
    })
    const result = await verifyDownload('gpt2')
    expect(result.verified).toBe(false)
    expect(result.missing_files).toContain('model.safetensors')
  })
})

describe('retryDownload', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /models/download/{modelId}/retry', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'restarted' })
    const result = await retryDownload('gpt2')
    expect(result.status).toBe('restarted')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/models/download/gpt2/retry')
  })
})
