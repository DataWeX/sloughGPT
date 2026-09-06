import { beforeEach, describe, expect, it, vi } from 'vitest'

import { setupApiMocks, apiClient } from './__test-helper'
setupApiMocks()

import { modelController, streamModelEvents } from './model-controller'

describe('modelController', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('list returns mapped models', async () => {
    apiClient.apiGet.mockResolvedValue({ models: ['gpt2', { id: 'bert', name: 'BERT', type: 'huggingface', size_mb: 440 }] })
    const result = await modelController.list()
    expect(result).toHaveLength(2)
    expect(result[0]).toEqual({ id: 'gpt2', name: 'gpt2', type: 'huggingface' })
    expect(result[1]).toEqual({ id: 'bert', name: 'BERT', type: 'huggingface', size_mb: 440 })
  })

  it('list returns empty on error', async () => {
    apiClient.apiGet.mockRejectedValue(new Error('fail'))
    const result = await modelController.list()
    expect(result).toEqual([])
  })

  it('load posts to /models/load', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok' })
    const result = await modelController.load('gpt2')
    expect(result.status).toBe('ok')
  })

  it('load throws on error', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'error', error: 'fail' })
    await expect(modelController.load('x')).rejects.toThrow('fail')
  })

  it('status maps from health', async () => {
    apiClient.apiGet.mockResolvedValue({ model_loaded: true, model_type: 'gpt2', device: 'cpu' })
    const result = await modelController.status()
    expect(result).toEqual({ loaded: true, model_type: 'gpt2', device: 'cpu' })
  })

  it('unloadModel posts to /models/unload', async () => {
    apiClient.apiPost.mockResolvedValue({})
    const result = await modelController.unloadModel()
    expect(apiClient.apiPost).toHaveBeenCalledWith('/models/unload')
  })

  it('isLoaded checks loaded state', async () => {
    apiClient.apiGet.mockResolvedValue({ model_loaded: true, model_type: 'gpt2' })
    expect(await modelController.isLoaded()).toBe(true)
  })

  it('getCurrentModel returns model info', async () => {
    apiClient.apiGet.mockResolvedValue({ model_id: 'gpt2', model_type: 'gpt2', device: 'cpu', loaded_at: 12345 })
    const result = await modelController.getCurrentModel()
    expect(result).toEqual({ model_id: 'gpt2', model_type: 'gpt2', device: 'cpu', loaded_at: 12345 })
    expect(apiClient.apiGet).toHaveBeenCalledWith('/models/current')
  })

  it('getCurrentModel returns null on error', async () => {
    apiClient.apiGet.mockRejectedValue(new Error('fail'))
    const result = await modelController.getCurrentModel()
    expect(result).toBeNull()
  })

  it('getCatalog returns model catalog', async () => {
    apiClient.apiGet.mockResolvedValue({ models: [{ id: 'gpt2', name: 'GPT-2', type: 'local', size_gb: 0.5, cached: true, source: 'huggingface' }], count: 1 })
    const result = await modelController.getCatalog()
    expect(result.models).toHaveLength(1)
    expect(result.count).toBe(1)
  })

  it('getCatalog returns empty on error', async () => {
    apiClient.apiGet.mockRejectedValue(new Error('fail'))
    const result = await modelController.getCatalog()
    expect(result).toEqual({ models: [], count: 0 })
  })

  it('getCatalogStats returns stats', async () => {
    apiClient.apiGet.mockResolvedValue({ total_models: 3, total_size_gb: 1.5, cached_count: 2, sources: { huggingface: 2, local: 1 } })
    const result = await modelController.getCatalogStats()
    expect(result.total_models).toBe(3)
  })

  it('startDownload posts to /models/download', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'started', model_id: 'gpt2' })
    const result = await modelController.startDownload('gpt2')
    expect(result.status).toBe('started')
  })

  it('getDownloadStatus fetches download info', async () => {
    apiClient.apiGet.mockResolvedValue({ model_id: 'gpt2', status: 'downloading', progress: 50, bytes_downloaded: 500, total_bytes: 1000, speed_bps: 100 })
    const result = await modelController.getDownloadStatus('gpt2')
    expect(result.progress).toBe(50)
  })

  it('listDownloads returns list', async () => {
    apiClient.apiGet.mockResolvedValue({ downloads: [], count: 0 })
    const result = await modelController.listDownloads()
    expect(result.count).toBe(0)
  })

  it('listDownloads returns empty on error', async () => {
    apiClient.apiGet.mockRejectedValue(new Error('fail'))
    const result = await modelController.listDownloads()
    expect(result).toEqual({ downloads: [], count: 0 })
  })

  it('cancelDownload posts cancel', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'cancelled' })
    const result = await modelController.cancelDownload('gpt2')
    expect(result.status).toBe('cancelled')
  })

  it('retryDownload posts retry', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'retrying' })
    const result = await modelController.retryDownload('gpt2')
    expect(result.status).toBe('retrying')
  })

  it('verifyDownload posts verify', async () => {
    apiClient.apiPost.mockResolvedValue({ verified: true, model_id: 'gpt2' })
    const result = await modelController.verifyDownload('gpt2')
    expect(result.verified).toBe(true)
  })

  it('getEngineStatus returns engine info', async () => {
    apiClient.apiGet.mockResolvedValue({ engine: 'slo', version: '1.0.0', models_loaded: 1, uptime_s: 3600, memory_usage_mb: 512 })
    const result = await modelController.getEngineStatus()
    expect(result.engine).toBe('slo')
  })

  it('getEngineStatus returns null on error', async () => {
    apiClient.apiGet.mockRejectedValue(new Error('fail'))
    const result = await modelController.getEngineStatus()
    expect(result).toBeNull()
  })

  it('reloadEngine posts reload', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'reloaded' })
    const result = await modelController.reloadEngine()
    expect(result.status).toBe('reloaded')
  })

  it('setPrecision posts mode', async () => {
    apiClient.apiPost.mockResolvedValue({ mode: 'fp16', applied: true })
    const result = await modelController.setPrecision('fp16')
    expect(result.applied).toBe(true)
  })

  it('exportModel posts export params', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', output_path: '/tmp/model' })
    const result = await modelController.exportModel('/tmp/model', 'safetensors')
    expect(result.status).toBe('ok')
  })

  it('getCacheUsage returns cache info', async () => {
    apiClient.apiGet.mockResolvedValue({ total_bytes: 1e9, total_gb: 1.0, model_count: 2, cache_dir: '/tmp' })
    const result = await modelController.getCacheUsage()
    expect(result.total_gb).toBe(1.0)
  })
})
