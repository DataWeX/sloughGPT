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

import { dashboardController } from './dashboard-controller'

const mockSummary = {
  health: {
    model_loaded: true,
    model_type: 'qwen',
    uptime_seconds: 3600,
    request_count: 500,
    error_count: 5,
    tokens_per_sec: 12.5,
    avg_latency_ms: 80,
    cpu_percent: 45,
    memory_percent: 62,
    memory_used_mb: 2048,
  },
  active_processes: 2,
  processes: {
    training: { type: 'training', status: 'running', label: 'LoRA Fine-tune', detail: 'epoch 3/10', progress: 30 },
  },
  services: { total: 5, healthy: 4 },
}

describe('dashboardController.getSummary', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /dashboard/summary', async () => {
    apiClient.apiGet.mockResolvedValue(mockSummary)
    const result = await dashboardController.getSummary()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/dashboard/summary')
    expect(result.health.model_loaded).toBe(true)
    expect(result.active_processes).toBe(2)
    expect(result.services.healthy).toBe(4)
  })
})

describe('dashboardController.getEvents', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /dashboard/events with default n', async () => {
    apiClient.apiGet.mockResolvedValue({ events: [{ ts: 1, level: 'info', message: 'test' }], count: 1 })
    const result = await dashboardController.getEvents()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/dashboard/events?n=20')
    expect(result.events).toHaveLength(1)
  })

  it('GETs /dashboard/events with custom n', async () => {
    apiClient.apiGet.mockResolvedValue({ events: [], count: 0 })
    await dashboardController.getEvents(50)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/dashboard/events?n=50')
  })
})
