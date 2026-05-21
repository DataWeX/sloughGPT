/**
 * Training Jobs Controller — axios-based API for training job management.
 */

import { apiGet, apiPost, apiDelete } from './http-client'
import type { Checkpoint } from './souls-controller'

export interface TrainingJob {
  id: string
  name: string
  status: string
  progress: number
  created_at: string
}

export interface TrainingStatus {
  status: string
  job_id?: string
  progress?: number
}

export interface RecoverableJob {
  id: string
  name: string
  failed_at: string
}

export interface Webhook {
  id: string
  url: string
  events: string[]
}

export interface WebhookStats {
  total: number
  success_rate: number
}

export const trainingJobsController = {
  async list(): Promise<TrainingJob[]> {
    const data = await apiGet<{ jobs: TrainingJob[] }>('/training/jobs')
    return data.jobs || []
  },

  async listCheckpoints(): Promise<Checkpoint[]> {
    const data = await apiGet<{ checkpoints: Checkpoint[] }>('/auto-train/checkpoints')
    return data.checkpoints || []
  },

  async get(id: string): Promise<TrainingJob | null> {
    try {
      return await apiGet<TrainingJob>(`/training/jobs/${id}`)
    } catch {
      return null
    }
  },

  async create(params: {
    name: string
    model: string
    dataset: string
    epochs?: number
    batch_size?: number
    learning_rate?: number
    device?: string
    use_lora?: boolean
    lora_rank?: number
  }): Promise<TrainingStatus> {
    return apiPost<TrainingStatus>('/training/start', params)
  },

  async stop(id: string): Promise<void> {
    await apiPost(`/training/jobs/${id}/stop`)
  },

  async delete(id: string): Promise<void> {
    await apiDelete(`/training/jobs/${id}`)
  },

  async recoverable(): Promise<RecoverableJob[]> {
    const data = await apiGet<{ jobs: RecoverableJob[] }>('/recovery/recoverable')
    return data.jobs || []
  },

  async recover(id: string): Promise<TrainingStatus> {
    return apiPost<TrainingStatus>(`/recovery/recover/${id}`)
  },

  async listWebhooks(): Promise<Webhook[]> {
    const data = await apiGet<{ webhooks: Webhook[] }>('/training/webhooks')
    return data.webhooks || []
  },

  async createWebhook(url: string, events: string[]): Promise<Webhook> {
    return apiPost<Webhook>('/training/webhooks', { url, events })
  },

  async deleteWebhook(id: string): Promise<void> {
    await apiDelete(`/training/webhooks/${id}`)
  },

  async webhookStats(): Promise<WebhookStats> {
    return apiGet<WebhookStats>('/training/webhooks/stats')
  },

  // Recovery
  async getRecoveryStats(): Promise<Record<string, number>> {
    return apiGet<Record<string, number>>('/recovery/stats')
  },

  async abandon(id: string): Promise<{ message?: string }> {
    return apiDelete<{ message?: string }>(`/recovery/abandon/${id}`)
  },

  // Webhooks
  async testWebhook(url: string): Promise<unknown> {
    return apiPost(`/training/webhooks/test?url=${encodeURIComponent(url)}`)
  },

  // Training status
  async getStatus(): Promise<Record<string, unknown>> {
    return apiGet<Record<string, unknown>>('/training/status')
  },

  async exportFeedbackPairs(minQuality: number, targetCount: number): Promise<Record<string, unknown>> {
    return apiPost<Record<string, unknown>>('/training/export-text', { min_quality: minQuality, target_count: targetCount })
  },

  async loadCheckpoint(name: string): Promise<{ success: boolean }> {
    return apiPost(`/auto-train/checkpoints/${encodeURIComponent(name)}/load`)
  },

  async deleteCheckpoint(name: string): Promise<{ success: boolean }> {
    return apiDelete(`/auto-train/checkpoints/${encodeURIComponent(name)}`)
  },

  async downloadTrainingJob(jobId: string): Promise<Blob> {
    const { apiClient } = await import('./http-client')
    const token = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null
    const baseUrl = apiClient.defaults.baseURL || 'http://localhost:8000'
    const res = await fetch(`${baseUrl}/training/export/${jobId}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    })
    if (!res.ok) throw new Error(`Export failed (${res.status})`)
    return res.blob()
  },
}

export const trainingController = trainingJobsController
