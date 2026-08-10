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
      config: { aggregate_interval_minutes: 30, prune_interval_minutes: 60, export_interval_hours: 24, auto_dpo_interval_minutes: 60, health_check_interval_seconds: 10, background_training_interval_seconds: 300, background_training_enabled: true },
      stats: { workflow_runs: 0, aggregations_performed: 0, prunes_performed: 0, exports_performed: 0, feedback_recorded: 100, auto_train_steps: 0, dpo_train_steps: 0, dpo_train_rejected: 0, user_adapter_trained: 5, user_adapter_rejected: 0, start_time: null },
      pending_thumbs_up: 0,
      auto_train_threshold: 3,
      last_runs: { aggregate: 0, prune: 0, export: 0, dpo: 0, health_check: 0, last_rollback: 0, background_training: 0 },
      systems: {},
    })

    const result = await workflowController.status()
    expect(result.running).toBe(true)
    expect(result.config?.aggregate_interval_minutes).toBe(30)
    expect(result.stats?.feedback_recorded).toBe(100)
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
