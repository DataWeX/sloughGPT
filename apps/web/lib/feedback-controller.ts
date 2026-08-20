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
      thumbs_up?: number
      thumbs_down?: number
      total?: number
      up_ratio?: number
    }>('/feedback/stats/summary')
    const thumbsUp = data.thumbs_up ?? 0
    const thumbsDown = data.thumbs_down ?? 0
    const total = data.total ?? 0
    const ratio = data.up_ratio ?? (total > 0 ? thumbsUp / total : 0)
    return {
      db_stats: {
        conversations: 0,
        messages: 0,
        feedback_total: total,
        thumbs_up: thumbsUp,
        thumbs_down: thumbsDown,
        ratio,
      },
      current_weights: {
        temperature: 0.7,
        repetition_penalty: 1.1,
      },
      history_length: 0,
      quality_trend: { thumbs_up_ratio: ratio },
    }
  },

  async getWorkflowStatus(): Promise<WorkflowStatus> {
    return apiGet('/workflow/status')
  },

  async triggerWorkflowAction(action: 'aggregate' | 'prune' | 'export'): Promise<{ status: string; timestamp: number }> {
    return apiPost(`/workflow/trigger/${action}`)
  },

  async getTrainingStats(): Promise<TrainingStats> {
    const data = await apiGet<{
      jobs?: Array<{ id: string; status: string; progress?: number; loss?: number; created_at?: string }>
      total_tracked_jobs?: number
    }>('/training/jobs')
    const jobs = data.jobs ?? []
    const completed = jobs.filter(j => j.status === 'completed')
    const lastJob = completed.length > 0 ? completed[completed.length - 1] : null
    return {
      feedback_pairs: data.total_tracked_jobs ?? jobs.length,
      last_training: lastJob?.created_at ?? null,
      quality_score: lastJob?.loss ?? null,
    }
  },

  async exportTrainingData(format: string, filepath?: string): Promise<{ status: string; path?: string; count: number }> {
    return apiPost<{ status: string; path?: string; count: number }>('/training/export-text', { format, filepath })
  },
}
