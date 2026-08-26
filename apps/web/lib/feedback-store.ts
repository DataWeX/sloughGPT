'use client'

import { create } from 'zustand'
import { feedbackController } from './feedback-controller'
import { userAdaptersController } from './user-adapters-controller'
import type { FeedbackStats, WorkflowStatus } from './feedback-controller'
import type { UserAdapterStats } from './user-adapters-controller'
import { useErrorStore, addGlobalError } from './error-store'
import { useApiMonitor } from './api-monitor-store'
import { extractErrorMessage } from '@/lib/error-utils'

interface FeedbackState {
  stats: FeedbackStats | null
  adapterStats: UserAdapterStats | null
  workflowStatus: WorkflowStatus | null
  isLoading: boolean
  error: string | null

  recordFeedback: (params: {
    userMessage: string
    assistantResponse: string
    rating: 'thumbs_up' | 'thumbs_down'
    conversationId?: string
    qualityScore?: number
    userId?: string
  }) => Promise<boolean>

  fetchStats: () => Promise<void>
  fetchAdapterStats: () => Promise<void>
  fetchWorkflowStatus: () => Promise<void>
  triggerWorkflowAction: (action: 'aggregate' | 'prune' | 'export') => Promise<boolean>
  reset: () => void
}

export const useFeedbackStore = create<FeedbackState>()((set, get) => ({
  stats: null,
  adapterStats: null,
  workflowStatus: null,
  isLoading: false,
  error: null,

  recordFeedback: async (params) => {
    set({ isLoading: true, error: null })
    try {
      await feedbackController.recordFeedbackWorkflow(params)
      await get().fetchStats()
      await get().fetchAdapterStats()
      set({ isLoading: false })
      return true
    } catch (err) {
      const error = extractErrorMessage(err, 'Could not record feedback')
      if (useApiMonitor.getState().status === 'reloading') return false
      addGlobalError(err, 'Feedback')
      set({ isLoading: false, error })
      return false
    }
  },

  fetchStats: async () => {
    try {
      const stats = await feedbackController.getFeedbackStats()
      set({ stats })
    } catch (err) {
      if (useApiMonitor.getState().status === 'reloading') return
      const msg = extractErrorMessage(err)
      if (msg.includes('404') || msg.includes('Not Found')) return
      useErrorStore.getState().addError(err, { source: 'Feedback Stats' })
    }
  },

  fetchAdapterStats: async () => {
    try {
      const res = await userAdaptersController.list()
      set({ adapterStats: res?.stats ?? null })
    } catch (err) {
      if (useApiMonitor.getState().status === 'reloading') return
      const msg = extractErrorMessage(err)
      if (msg.includes('404') || msg.includes('Not Found')) return
      useErrorStore.getState().addError(err, { source: 'User Adapters' })
    }
  },

  fetchWorkflowStatus: async () => {
    try {
      const workflowStatus = await feedbackController.getWorkflowStatus()
      set({ workflowStatus })
    } catch (err) {
      if (useApiMonitor.getState().status === 'reloading') return
      const msg = extractErrorMessage(err)
      if (msg.includes('404') || msg.includes('Not Found')) return
      useErrorStore.getState().addError(err, { source: 'Workflow Status' })
    }
  },

  triggerWorkflowAction: async (action) => {
    set({ isLoading: true, error: null })
    try {
      await feedbackController.triggerWorkflowAction(action)
      await get().fetchAdapterStats()
      await get().fetchWorkflowStatus()
      set({ isLoading: false })
      return true
    } catch (err) {
      const error = extractErrorMessage(err, 'Could not trigger action')
      useErrorStore.getState().addError(err, { source: 'Workflow Action' })
      set({ isLoading: false, error })
      return false
    }
  },

  reset: () => {
    set({
      stats: null,
      adapterStats: null,
      workflowStatus: null,
      isLoading: false,
      error: null,
    })
  },
}))
