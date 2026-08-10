/**
 * Feedback Controller — axios-based API for feedback, workflow, and training stats.
 *
 * Usage:
 *   import { feedbackController } from '@/lib/feedback-controller'
 *   await feedbackController.recordFeedbackWorkflow(params)
 */

import { apiGet, apiPost } from './http-client'
import type { WorkflowStatus, FeedbackStats, TrainingStats } from './types'

export type { WorkflowStatus, WorkflowConfig, WorkflowStats, FeedbackStats, TrainingStats } from './types'

export const feedbackController = {
  async recordFeedbackWorkflow(params: {
    userMessage: string
    assistantResponse: string
    rating: 'thumbs_up' | 'thumbs_down'
    conversationId?: string
    qualityScore?: number
    userId?: string
  }): Promise<{ status: string; feedback_id: string; workflow_active: boolean }> {
    return apiPost('/feedback/workflow-record', {
      user_message: params.userMessage,
      assistant_response: params.assistantResponse,
      rating: params.rating,
      conversation_id: params.conversationId,
      quality_score: params.qualityScore,
      user_id: params.userId,
    })
  },

  async getFeedbackStats(): Promise<FeedbackStats & { quality_trend: { thumbs_up_ratio: number } }> {
    return apiGet('/meta-weights/stats')
  },

  async getWorkflowStatus(): Promise<WorkflowStatus> {
    return apiGet('/workflow/status')
  },

  async triggerWorkflowAction(action: 'aggregate' | 'prune' | 'export'): Promise<{ status: string; timestamp: number }> {
    return apiPost(`/workflow/trigger/${action}`)
  },

  async getTrainingStats(): Promise<TrainingStats> {
    const data = await apiGet<{ pairs_converted?: number; last_training?: string; quality_score?: number }>('/training/status')
    return {
      feedback_pairs: data.pairs_converted ?? 0,
      last_training: data.last_training ?? null,
      quality_score: data.quality_score ?? null,
    }
  },

  async exportTrainingData(format: string, filepath?: string): Promise<{ status: string; path?: string; count: number }> {
    return apiPost<{ status: string; path?: string; count: number }>('/training/export-text', { format, filepath })
  },
}
