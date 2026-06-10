import { beforeEach, describe, expect, it, vi } from 'vitest'

// Mock auth store (no token needed for system endpoints)
vi.mock('./auth', () => ({
  useAuthStore: {
    getState: () => ({ token: null as string | null }),
  },
}))

// Mock config to avoid hitting real server
vi.mock('./config', () => ({
  PUBLIC_API_URL: 'http://127.0.0.1:9',
}))

import { setupApiMocks, apiClient } from './__test-helper'
setupApiMocks()

import { systemController } from './system-controller'

describe('systemController.getMetrics', () => {
  beforeEach(() => vi.clearAllMocks())

  it('GETs /system/metrics and returns typed data', async () => {
    const mockData = {
      cpu_percent: 12.3,
      memory_percent: 45.6,
      memory_used_gb: 7.8,
      memory_total_gb: 16.0,
    }
    apiClient.apiGet.mockResolvedValue(mockData)
    const result = await systemController.getMetrics()
    expect(result.cpu_percent).toBe(12.3)
    expect(result.memory_total_gb).toBe(16.0)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/system/metrics', undefined, { silent: true })
  })
})

describe('systemController.getInfo', () => {
  beforeEach(() => vi.clearAllMocks())

  it('GETs /system/info', async () => {
    const mockInfo = {
      platform: 'darwin',
      platform_release: '22.5.0',
      platform_version: '22.5.0',
      architecture: 'arm64',
      processor: 'Apple M2',
      cpu_count: 8,
    }
    apiClient.apiGet.mockResolvedValue(mockInfo)
    const result = await systemController.getInfo()
    expect(result.processor).toBe('Apple M2')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/system/info', undefined, { silent: true })
  })
})

describe('systemController.getDisk', () => {
  beforeEach(() => vi.clearAllMocks())

  it('GETs /system/disk', async () => {
    const mockDisk = { total_gb: 512, used_gb: 123, free_gb: 389, percent: 24 }
    apiClient.apiGet.mockResolvedValue(mockDisk)
    const result = await systemController.getDisk()
    expect(result.percent).toBe(24)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/system/disk', undefined, { silent: true })
  })
})

describe('systemController.getDetailedHealth', () => {
  beforeEach(() => vi.clearAllMocks())

  it('GETs /health/detailed', async () => {
    const mockHealth = {
      status: 'ok',
      uptime_seconds: 12345,
      timestamp: new Date().toISOString(),
      system: { cpu_percent: 10, memory_percent: 20, memory_available_mb: 8000 },
      model_loaded: true,
      model_type: 'gpt2',
      inference: { is_inferencing: false, inference_count: 42 },
    }
    apiClient.apiGet.mockResolvedValue(mockHealth)
    const result = await systemController.getDetailedHealth()
    expect(result.model_loaded).toBe(true)
    expect(result.inference?.inference_count).toBe(42)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/health/detailed', undefined, { silent: true })
  })
})
