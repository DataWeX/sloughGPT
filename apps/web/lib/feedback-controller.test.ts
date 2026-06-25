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

import { feedbackController } from './feedback-controller'

describe('feedbackController', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('recordFeedbackWorkflow POSTs /feedback/workflow-record', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'recorded', feedback_id: 'f1', workflow_active: true })
    const result = await feedbackController.recordFeedbackWorkflow({
      userMessage: 'hi', assistantResponse: 'hello', rating: 'thumbs_up',
    })
    expect(result.status).toBe('recorded')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/feedback/workflow-record', {
      user_message: 'hi', assistant_response: 'hello', rating: 'thumbs_up',
      conversation_id: undefined, quality_score: undefined, user_id: undefined,
    })
  })

  it('getFeedbackStats GETs /meta-weights/stats', async () => {
    apiClient.apiGet.mockResolvedValue({
      db_stats: { conversations: 5, messages: 20, feedback_total: 10, thumbs_up: 7, thumbs_down: 3, ratio: 0.7 },
      current_weights: { temperature: 0.8, repetition_penalty: 1.0 },
      history_length: 10,
    })
    const result = await feedbackController.getFeedbackStats()
    expect(result.db_stats.thumbs_up).toBe(7)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/meta-weights/stats')
  })

  it('getWorkflowStatus GETs /workflow/status', async () => {
    apiClient.apiGet.mockResolvedValue({
      running: true, stats: { workflow_runs: 3, aggregations_performed: 1, prunes_performed: 0, exports_performed: 1, feedback_recorded: 10, start_time: 100 },
      config: { aggregate_interval_minutes: 30, prune_interval_minutes: 60, export_interval_hours: 24, health_check_interval_seconds: 30, auto_aggregate_threshold: 50, auto_prune_threshold: 100, min_feedback_for_aggregation: 5 },
      last_runs: { aggregate: 0, prune: 0, export: 0, health_check: 0 },
      systems: {},
    })
    const result = await feedbackController.getWorkflowStatus()
    expect(result.running).toBe(true)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/workflow/status')
  })

  it('triggerWorkflowAction POSTs /workflow/trigger/{action}', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'triggered', timestamp: 123 })
    const result = await feedbackController.triggerWorkflowAction('aggregate')
    expect(result.status).toBe('triggered')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/workflow/trigger/aggregate')
  })

  it('getTrainingStats GETs /training/status and maps fields', async () => {
    apiClient.apiGet.mockResolvedValue({ pairs_converted: 15, last_training: '2026-01-01', quality_score: 0.85 })
    const result = await feedbackController.getTrainingStats()
    expect(result.feedback_pairs).toBe(15)
    expect(result.last_training).toBe('2026-01-01')
    expect(result.quality_score).toBe(0.85)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/training/status')
  })

  it('getTrainingStats handles null fields', async () => {
    apiClient.apiGet.mockResolvedValue({})
    const result = await feedbackController.getTrainingStats()
    expect(result.feedback_pairs).toBe(0)
    expect(result.last_training).toBeNull()
    expect(result.quality_score).toBeNull()
  })

  it('exportTrainingData POSTs /training/export-text', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'exported', path: '/tmp/export.json', count: 10 })
    const result = await feedbackController.exportTrainingData('json', '/tmp')
    expect(result.count).toBe(10)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/training/export-text', { format: 'json', filepath: '/tmp' })
  })
})
