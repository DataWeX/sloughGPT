/**
 * Feedback Controller — axios-based API for feedback, workflow, and training stats.
 *
 * Usage:
 *   import { feedbackController } from '@/lib/feedback-controller'
 *   await feedbackController.recordFeedbackWorkflow(params)
 */

import { apiGet, apiPost } from './http-client'

export interface FeedbackStats {
  db_stats: {
    conversations: number
    messages: number
    feedback_total: number
    thumbs_up: number
    thumbs_down: number
    ratio: number
  }
  current_weights: {
    temperature: number
    repetition_penalty: number
  }
  history_length: number
}

export interface WorkflowConfig {
  aggregate_interval_minutes: number
  prune_interval_minutes: number
  export_interval_hours: number
  health_check_interval_seconds: number
  auto_aggregate_threshold: number
  auto_prune_threshold: number
  min_feedback_for_aggregation: number
}

export interface WorkflowStatus {
  running: boolean
  stats: {
    workflow_runs: number
    aggregations_performed: number
    prunes_performed: number
    exports_performed: number
    feedback_recorded: number
    start_time: number | null
  }
  config: WorkflowConfig
  last_runs: {
    aggregate: number
    prune: number
    export: number
    health_check: number
  }
  systems: Record<string, unknown>
}

export interface TrainingStats {
  feedback_pairs: number
  last_training: string | null
  quality_score: number | null
}

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
