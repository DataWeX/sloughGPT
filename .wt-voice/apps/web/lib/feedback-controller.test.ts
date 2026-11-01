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
import { userAdaptersController } from './user-adapters-controller'

describe('feedbackController.recordFeedbackWorkflow', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /feedback/workflow-record with feedback data', async () => {
    apiClient.apiPost.mockResolvedValue({
      status: 'recorded',
      feedback_id: 'test-123',
      workflow_active: true,
    })

    const result = await feedbackController.recordFeedbackWorkflow({
      userMessage: 'Hello?',
      assistantResponse: 'Hi there!',
      rating: 'thumbs_up',
      userId: 'user-1',
    })

    expect(result.status).toBe('recorded')
    expect(apiClient.apiPost).toHaveBeenCalledWith(
      '/feedback/workflow-record',
      expect.objectContaining({ rating: 'thumbs_up' }),
    )
  })

  it('handles thumbs_down rating', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'recorded', feedback_id: 'test-456', workflow_active: false })

    await feedbackController.recordFeedbackWorkflow({
      userMessage: 'What is 2+2?',
      assistantResponse: '5',
      rating: 'thumbs_down',
    })

    expect(apiClient.apiPost).toHaveBeenCalledWith(
      '/feedback/workflow-record',
      expect.objectContaining({ rating: 'thumbs_down' }),
    )
  })
})

describe('feedbackController.getFeedbackStats', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('calls /feedback/stats/summary endpoint', async () => {
    apiClient.apiGet.mockResolvedValue({
      thumbs_up: 7,
      thumbs_down: 3,
      total: 10,
      up_ratio: 0.7,
    })

    const result = await feedbackController.getFeedbackStats()

    expect(result.db_stats.thumbs_up).toBe(7)
    expect(result.db_stats.thumbs_down).toBe(3)
    expect(result.db_stats.feedback_total).toBe(10)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/feedback/stats/summary')
  })
})

describe('userAdaptersController.list', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('calls /user-adapters endpoint', async () => {
    apiClient.apiGet.mockResolvedValue({ adapters: [], stats: { total_users: 5, total_size_mb: 0.25 } })

    const result = await userAdaptersController.list()

    expect(result.adapters).toEqual([])
    expect(apiClient.apiGet).toHaveBeenCalledWith('/user-adapters')
  })
})

describe('feedbackController.getWorkflowStatus', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /workflow/status', async () => {
    apiClient.apiGet.mockResolvedValue({
      running: true,
      stats: { workflow_runs: 10 },
    })

    const result = await feedbackController.getWorkflowStatus()

    expect(result.running).toBe(true)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/workflow/status')
  })
})

describe('feedbackController.getTrainingStats', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('calls /training/jobs and handles array response', async () => {
    apiClient.apiGet.mockResolvedValue([
      { id: 'j1', status: 'completed', created_at: '2026-01-01', loss: 0.5 },
      { id: 'j2', status: 'running', progress: 50 },
    ])

    const result = await feedbackController.getTrainingStats()

    expect(result.feedback_pairs).toBe(2)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/training/jobs')
  })

  it('handles empty array', async () => {
    apiClient.apiGet.mockResolvedValue([])
    const result = await feedbackController.getTrainingStats()
    expect(result.feedback_pairs).toBe(0)
  })
})

describe('feedbackController.exportTrainingData', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /training/export-text with format', async () => {
    apiClient.apiPost.mockResolvedValue({
      status: 'exported',
      path: '/data/dpo_123.jsonl',
      count: 10,
    })

    const result = await feedbackController.exportTrainingData('dpo')

    expect(result.count).toBe(10)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/training/export-text', { format: 'dpo', filepath: undefined })
  })
})
