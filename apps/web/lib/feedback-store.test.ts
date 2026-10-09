import { describe, it, expect, vi, beforeEach } from 'vitest'

const mockRecordFeedbackWorkflow = vi.fn()
const mockGetFeedbackStats = vi.fn()
const mockGetWorkflowStatus = vi.fn()
const mockTriggerWorkflowAction = vi.fn()
const mockGetTrainingStats = vi.fn()
const mockExportTrainingData = vi.fn()

vi.mock('./feedback-controller', () => ({
  feedbackController: {
    recordFeedbackWorkflow: (...args: any[]) => mockRecordFeedbackWorkflow(...args),
    getFeedbackStats: (...args: any[]) => mockGetFeedbackStats(...args),
    getWorkflowStatus: (...args: any[]) => mockGetWorkflowStatus(...args),
    triggerWorkflowAction: (...args: any[]) => mockTriggerWorkflowAction(...args),
    getTrainingStats: (...args: any[]) => mockGetTrainingStats(...args),
    exportTrainingData: (...args: any[]) => mockExportTrainingData(...args),
  },
}))

const mockUserAdaptersList = vi.fn()

vi.mock('./user-adapters-controller', () => ({
  userAdaptersController: {
    list: (...args: any[]) => mockUserAdaptersList(...args),
  },
}))

vi.mock('./error-store', () => ({
  useErrorStore: { getState: () => ({ addError: vi.fn() }) },
  addGlobalError: vi.fn(),
}))

const mockMonitorGetState = vi.fn().mockReturnValue({ status: 'connected' as const })

vi.mock('./api-monitor-store', () => ({
  useApiMonitor: { getState: (...args: any[]) => mockMonitorGetState(...args) },
}))

import { useFeedbackStore } from './feedback-store'

describe('useFeedbackStore', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useFeedbackStore.setState({
      stats: null,
      adapterStats: null,
      workflowStatus: null,
      isLoading: false,
      error: null,
    })
  })

  it('has correct initial state', () => {
    const state = useFeedbackStore.getState()
    expect(state.isLoading).toBe(false)
    expect(state.error).toBeNull()
    expect(state.stats).toBeNull()
    expect(state.adapterStats).toBeNull()
    expect(state.workflowStatus).toBeNull()
  })

  describe('recordFeedback', () => {
    it('calls feedbackController.recordFeedbackWorkflow and fetches stats', async () => {
      mockRecordFeedbackWorkflow.mockResolvedValue({ status: 'ok' })
      mockGetFeedbackStats.mockResolvedValue({ db_stats: { thumbs_up: 1 } })
      mockUserAdaptersList.mockResolvedValue({ total_users: 0 })

      const ok = await useFeedbackStore.getState().recordFeedback({
        userMessage: 'hi',
        assistantResponse: 'hello',
        rating: 'thumbs_up',
      })

      expect(ok).toBe(true)
      expect(mockRecordFeedbackWorkflow).toHaveBeenCalledWith({
        userMessage: 'hi',
        assistantResponse: 'hello',
        rating: 'thumbs_up',
        conversationId: undefined,
        qualityScore: undefined,
        userId: undefined,
      })
      expect(mockGetFeedbackStats).toHaveBeenCalled()
      expect(useFeedbackStore.getState().isLoading).toBe(false)
      expect(useFeedbackStore.getState().error).toBeNull()
    })

    it('sets error on failure', async () => {
      mockRecordFeedbackWorkflow.mockRejectedValue(new Error('network fail'))

      const ok = await useFeedbackStore.getState().recordFeedback({
        userMessage: 'hi',
        assistantResponse: 'hello',
        rating: 'thumbs_down',
      })

      expect(ok).toBe(false)
      expect(useFeedbackStore.getState().error).toBe('network fail')
      expect(useFeedbackStore.getState().isLoading).toBe(false)
    })

    it('returns false during reloading status without setting error', async () => {
      mockMonitorGetState.mockReturnValue({ status: 'reloading' })
      mockRecordFeedbackWorkflow.mockRejectedValue(new Error('fail'))

      const ok = await useFeedbackStore.getState().recordFeedback({
        userMessage: 'hi',
        assistantResponse: 'hello',
        rating: 'thumbs_up',
      })

      expect(ok).toBe(false)
      mockMonitorGetState.mockReturnValue({ status: 'connected' })
    })
  })

  describe('fetchStats', () => {
    it('updates stats on success', async () => {
      const stats = { db_stats: { thumbs_up: 5 } }
      mockGetFeedbackStats.mockResolvedValue(stats)

      await useFeedbackStore.getState().fetchStats()
      expect(useFeedbackStore.getState().stats).toEqual(stats)
    })
  })

  describe('fetchAdapterStats', () => {
    it('updates adapterStats on success', async () => {
      const res = { stats: { total_users: 3, total_size_mb: 1.2 } }
      mockUserAdaptersList.mockResolvedValue(res)

      await useFeedbackStore.getState().fetchAdapterStats()
      expect(useFeedbackStore.getState().adapterStats).toEqual(res.stats)
    })
  })

  describe('fetchWorkflowStatus', () => {
    it('updates workflowStatus on success', async () => {
      const status = { running: true, stats: { workflow_runs: 1 } }
      mockGetWorkflowStatus.mockResolvedValue(status)

      await useFeedbackStore.getState().fetchWorkflowStatus()
      expect(useFeedbackStore.getState().workflowStatus).toEqual(status)
    })
  })

  describe('triggerWorkflowAction', () => {
    it('calls controller and refreshes stats', async () => {
      mockTriggerWorkflowAction.mockResolvedValue({ status: 'ok', timestamp: 1 })
      mockUserAdaptersList.mockResolvedValue({ total_users: 1 })
      mockGetWorkflowStatus.mockResolvedValue({ running: true })

      const ok = await useFeedbackStore.getState().triggerWorkflowAction('aggregate')
      expect(ok).toBe(true)
      expect(mockTriggerWorkflowAction).toHaveBeenCalledWith('aggregate')
    })

    it('sets error on failure', async () => {
      mockTriggerWorkflowAction.mockRejectedValue(new Error('bad action'))

      const ok = await useFeedbackStore.getState().triggerWorkflowAction('prune')
      expect(ok).toBe(false)
      expect(useFeedbackStore.getState().error).toBe('bad action')
    })
  })

  describe('reset', () => {
    it('clears all state', async () => {
      mockGetFeedbackStats.mockResolvedValue({ stats: true })
      mockUserAdaptersList.mockResolvedValue({ adapters: true })
      mockGetWorkflowStatus.mockResolvedValue({ status: true })
      await useFeedbackStore.getState().fetchStats()
      await useFeedbackStore.getState().fetchAdapterStats()
      await useFeedbackStore.getState().fetchWorkflowStatus()

      useFeedbackStore.getState().reset()
      const state = useFeedbackStore.getState()
      expect(state.stats).toBeNull()
      expect(state.adapterStats).toBeNull()
      expect(state.workflowStatus).toBeNull()
      expect(state.isLoading).toBe(false)
      expect(state.error).toBeNull()
    })
  })
})
