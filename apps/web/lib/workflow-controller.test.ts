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

import { workflowController } from './workflow-controller'

describe('workflowController.status', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /workflow/status', async () => {
    apiClient.apiGet.mockResolvedValue({
      running: true,
      config: { aggregate_interval_minutes: 30, prune_interval_minutes: 60, export_interval_hours: 24, health_check_interval_seconds: 10 },
      stats: { feedback_records: 100, adapters_count: 5 },
    })

    const result = await workflowController.status()
    expect(result.running).toBe(true)
    expect(result.config?.aggregate_interval_minutes).toBe(30)
    expect(result.stats?.feedback_records).toBe(100)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/workflow/status')
  })

  it('returns { running: false } on empty response', async () => {
    apiClient.apiGet.mockResolvedValue(null)
    const result = await workflowController.status()
    expect(result).toEqual({ running: false })
  })
})

describe('workflowController.start', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /workflow/start', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'started' })

    const result = await workflowController.start()
    expect(result.status).toBe('started')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/workflow/start', {})
  })
})

describe('workflowController.stop', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /workflow/stop', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'stopped' })

    const result = await workflowController.stop()
    expect(result.status).toBe('stopped')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/workflow/stop', {})
  })
})

describe('workflowController.trigger', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /workflow/trigger/{action}', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'triggered' })

    const result = await workflowController.trigger('aggregate')
    expect(result.status).toBe('triggered')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/workflow/trigger/aggregate', {})
  })

  it('works with different actions', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'done' })

    await workflowController.trigger('prune')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/workflow/trigger/prune', {})

    await workflowController.trigger('export')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/workflow/trigger/export', {})
  })
})
