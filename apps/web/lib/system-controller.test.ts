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

describe('systemController.getOutput', () => {
  beforeEach(() => vi.clearAllMocks())

  it('GETs /system/output with default n=100', async () => {
    const mockOutput = {
      lines: [{ text: 'test log', level: 'info', source: 'man', ts: 123 }],
      size: 1,
      seq: 1,
    }
    apiClient.apiGet.mockResolvedValue(mockOutput)
    const result = await systemController.getOutput()
    expect(result.lines).toHaveLength(1)
    expect(result.lines[0].text).toBe('test log')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/system/output?n=100', undefined, { silent: true })
  })

  it('GETs /system/output with custom n', async () => {
    apiClient.apiGet.mockResolvedValue({ lines: [], size: 0, seq: 0 })
    await systemController.getOutput(50)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/system/output?n=50', undefined, { silent: true })
  })
})

describe('systemController.getExecutorStatus', () => {
  beforeEach(() => vi.clearAllMocks())

  it('GETs /system/executor and returns pool status', async () => {
    const mockExecutor = {
      initialized: true,
      active_jobs: 1,
      max_workers: 2,
      total_tracked: 5,
      jobs: [
        { job_id: 'j1', status: 'running', tree_id: 'test', submitted_at: 1, elapsed_s: 2.5, cancel_requested: false },
      ],
    }
    apiClient.apiGet.mockResolvedValue(mockExecutor)
    const result = await systemController.getExecutorStatus()
    expect(result.initialized).toBe(true)
    expect(result.active_jobs).toBe(1)
    expect(result.jobs).toHaveLength(1)
    expect(result.jobs[0].job_id).toBe('j1')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/system/executor', undefined, { silent: true })
  })

  it('returns uninitialized state when executor not created', async () => {
    apiClient.apiGet.mockResolvedValue({ initialized: false, active_jobs: 0, max_workers: 0, total_tracked: 0, jobs: [] })
    const result = await systemController.getExecutorStatus()
    expect(result.initialized).toBe(false)
  })
})

describe('systemController.cancelExecutorJob', () => {
  beforeEach(() => vi.clearAllMocks())

  it('POSTs to /system/executor/{id}/cancel', async () => {
    const { apiPost } = await import('./http-client')
    vi.mocked(apiPost).mockResolvedValue({ cancelled: true })
    const result = await systemController.cancelExecutorJob('j1')
    expect(result.cancelled).toBe(true)
    expect(apiPost).toHaveBeenCalledWith('/system/executor/j1/cancel')
  })
})

describe('systemController.purgeExecutorJobs', () => {
  beforeEach(() => vi.clearAllMocks())

  it('POSTs to /system/executor/purge with max_age_s', async () => {
    const { apiPost } = await import('./http-client')
    vi.mocked(apiPost).mockResolvedValue({ purged: 3 })
    const result = await systemController.purgeExecutorJobs(7200)
    expect(result.purged).toBe(3)
    expect(apiPost).toHaveBeenCalledWith('/system/executor/purge?max_age_s=7200')
  })
})

describe('systemController.getInferencePoolStatus', () => {
  beforeEach(() => vi.clearAllMocks())

  it('GETs /system/inference-pool', async () => {
    apiClient.apiGet.mockResolvedValue({ initialized: true, max_workers: 4, queue_timeout: 30 })
    const result = await systemController.getInferencePoolStatus()
    expect(result.initialized).toBe(true)
    expect(result.max_workers).toBe(4)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/system/inference-pool', undefined, { silent: true })
  })
})
