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

  it('calls /meta-weights/stats endpoint', async () => {
    apiClient.apiGet.mockResolvedValue({
      db_stats: { feedback_total: 10, thumbs_up: 7, thumbs_down: 3 },
      current_weights: { temperature: 0.8 },
    })

    await feedbackController.getFeedbackStats()

    expect(apiClient.apiGet).toHaveBeenCalledWith('/meta-weights/stats')
  })
})

describe('userAdaptersController.list', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('calls /user-adapters endpoint', async () => {
    apiClient.apiGet.mockResolvedValue({ stats: { total_users: 5, total_size_mb: 0.25 } })

    await userAdaptersController.list()

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

  it('calls /training/status endpoint', async () => {
    apiClient.apiGet.mockResolvedValue({
      pairs_converted: 3,
      last_training: '2026-01-01',
      quality_score: 0.85,
    })

    const result = await feedbackController.getTrainingStats()

    expect(result.feedback_pairs).toBe(3)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/training/status')
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
