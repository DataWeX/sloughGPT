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
    const data = await apiGet<{
      db_stats: {
        total_feedback?: number
        thumbs_up?: number
        thumbs_down?: number
        total_regenerations?: number
        active_sessions?: number
      }
      quality_trend?: { thumbs_up_ratio?: number }
      current_weights?: { temperature?: number; repetition_penalty?: number }
      history_length?: number
    }>('/meta-weights/stats')
    const db = data.db_stats ?? {}
    const thumbsUp = db.thumbs_up ?? 0
    const thumbsDown = db.thumbs_down ?? 0
    const total = db.total_feedback ?? 0
    return {
      db_stats: {
        conversations: 0,
        messages: 0,
        feedback_total: total,
        thumbs_up: thumbsUp,
        thumbs_down: thumbsDown,
        ratio: total > 0 ? thumbsUp / total : 0,
      },
      current_weights: {
        temperature: data.current_weights?.temperature ?? 0.7,
        repetition_penalty: data.current_weights?.repetition_penalty ?? 1.1,
      },
      history_length: data.history_length ?? 0,
      quality_trend: { thumbs_up_ratio: data.quality_trend?.thumbs_up_ratio ?? 0 },
    }
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
