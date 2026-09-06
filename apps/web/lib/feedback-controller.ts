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
    const [summary, metaStats] = await Promise.all([
      apiGet<{
        thumbs_up?: number
        thumbs_down?: number
        total?: number
        up_ratio?: number
      }>('/feedback/stats/summary').catch(() => ({} as Record<string, never>)),
      apiGet<{
        current_weights?: Record<string, unknown>
        history_length?: number
      }>('/meta-weights/stats').catch(() => ({} as Record<string, never>)),
    ])
    const thumbsUp = summary.thumbs_up ?? 0
    const thumbsDown = summary.thumbs_down ?? 0
    const total = summary.total ?? 0
    const ratio = summary.up_ratio ?? (total > 0 ? thumbsUp / total : 0)
    return {
      db_stats: {
        conversations: 0,
        messages: 0,
        feedback_total: total,
        thumbs_up: thumbsUp,
        thumbs_down: thumbsDown,
        ratio,
      },
      current_weights: (metaStats.current_weights as { temperature: number; repetition_penalty: number }) ?? {
        temperature: 0.7,
        repetition_penalty: 1.1,
      },
      history_length: metaStats.history_length ?? 0,
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
    const data = await apiGet<Array<{ id: string; status: string; progress?: number; loss?: number; created_at?: string }> | { jobs?: Array<{ id: string; status: string; progress?: number; loss?: number; created_at?: string }>; total_tracked_jobs?: number }>('/training/jobs')
    const jobs = Array.isArray(data) ? data : (data.jobs ?? [])
    const totalTracked = Array.isArray(data) ? jobs.length : (data.total_tracked_jobs ?? jobs.length)
    const completed = jobs.filter(j => j.status === 'completed')
    const lastJob = completed.length > 0 ? completed[completed.length - 1] : null
    return {
      feedback_pairs: totalTracked,
      last_training: lastJob?.created_at ?? null,
      quality_score: lastJob?.loss ?? null,
    }
  },

  async exportTrainingData(format: string, filepath?: string): Promise<{ status: string; path?: string; count: number }> {
    return apiPost<{ status: string; path?: string; count: number }>('/training/export-text', { format, filepath })
  },
}
