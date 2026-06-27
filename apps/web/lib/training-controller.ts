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
  model?: string
  dataset?: string
  epochs?: number
  current_epoch?: number
  global_step?: number
  loss?: number
  train_loss?: number
  eval_loss?: number
  checkpoint?: string
  data_source?: string
  manifest?: Record<string, unknown>
  message?: string
  loss_history?: Array<{ step: number; value: number; type: 'train' | 'eval' }>
  reward_history?: Array<{ step: number; value: number }>
  result?: Record<string, unknown>
  explanation?: string
  status_message?: string
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

export interface AutoTrainStartRequest {
  algo?: string
  dataset_id?: string
  soul_name?: string
  epochs?: number
  learning_rate?: number
  teacher_model?: string
  temperature?: number
  source_text?: string
  checkpoint_name?: string
}

export interface AutoTrainStartResponse {
  status: string
  teacher?: string
  student?: string
  soul?: string
  epochs?: number
}

export interface WebhookStats {
  total: number
  success_rate: number
}

export interface TrainingBuild {
  name: string
  build_type: 'auto-train' | 'lora' | 'hf-finetune' | 'hf-finetuned-dir' | 'vlm' | 'visual'
  job_id?: string
  model?: string
  dataset?: string
  loss?: number | null
  epochs?: number | null
  model_path?: string
  size_mb?: number
  model_type?: string
  training_dataset?: string
  created_at?: string
  finished_at?: string
  soul?: string
  traits?: Record<string, number>
  verdict?: string
}

export interface TurboTrainStartRequest {
  method?: 'transformer'
  dataset_id?: string
  data_path?: string
  epochs?: number
  batch_size?: number
  learning_rate?: number
  vocab_size?: number
  n_embed?: number
  n_head?: number
  n_encoder_layers?: number
  n_decoder_layers?: number
  dim_feedforward?: number
  dropout?: number
  max_src_len?: number
  max_tgt_len?: number
}

export interface TurboTrainResponse {
  status: string
  model_path?: string
  method?: string
  final_loss?: number
  total_steps?: number
  epochs?: number
  message?: string
}

export const trainingJobsController = {
  async startAutoTrain(params?: AutoTrainStartRequest): Promise<AutoTrainStartResponse> {
    return apiPost<AutoTrainStartResponse>('/auto-train/start', params ?? null)
  },

  async stopAutoTrain(): Promise<void> {
    await apiPost('/auto-train/stop')
  },

  async pauseTraining(): Promise<{ success: boolean }> {
    return apiPost('/training/control/pause')
  },

  async getTrainingLog(): Promise<string[]> {
    try {
      const data = await apiGet<{ lines: string[] }>('/auto-train/log')
      return data.lines ?? []
    } catch { return [] }
  },

  async resumeTraining(): Promise<{ success: boolean }> {
    return apiPost('/training/control/resume')
  },

  async stopUnified(): Promise<void> {
    await apiPost('/training/unified-stop')
  },

  async startTurboTrain(params?: TurboTrainStartRequest): Promise<TurboTrainResponse> {
    return apiPost<TurboTrainResponse>('/auto-train/start-turbo', params ?? null)
  },

  async list(): Promise<TrainingJob[]> {
    const data = await apiGet<TrainingJob[] | { jobs: TrainingJob[] }>('/training/jobs')
    return Array.isArray(data) ? data : data.jobs || []
  },

  async listCheckpoints(): Promise<Checkpoint[]> {
    const data = await apiGet<{ checkpoints: Checkpoint[] }>('/auto-train/checkpoints')
    return data.checkpoints || []
  },

  async listBuilds(): Promise<TrainingBuild[]> {
    const data = await apiGet<{ builds: TrainingBuild[] }>('/training/builds')
    return data.builds || []
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

  async startHFFineTune(params: {
    model: string
    dataset: string
    name?: string
    epochs?: number
    batch_size?: number
    learning_rate?: number
    use_lora?: boolean
    lora_rank?: number
    max_seq_length?: number
  }): Promise<{ job_id: string; status: string; message: string }> {
    return apiPost('/training/hf-start', params)
  },

  async startQuick(params: {
    dataset: string
    name?: string
    model?: string
  }): Promise<{ job_id: string; status: string; config: Record<string, unknown>; explanation: string }> {
    return apiPost('/training/quick', params)
  },

  async getSummary(jobId: string): Promise<{ job_id: string; summary: string; status: string; model: string; dataset: string }> {
    return apiGet(`/training/jobs/${jobId}/summary`)
  },

  async startVisualTrain(params: {
    dataset: string
    vision_encoder?: string
    llm?: string
    connector_hidden_dim?: number
    max_seq_length?: number
    stage1_epochs?: number
    stage2_epochs?: number
    stage1_lr?: number
    stage2_lr?: number
    batch_size?: number
    use_lora?: boolean
    lora_rank?: number
    freeze_vision?: boolean
    name?: string
  }): Promise<{ job_id: string; status: string; message: string }> {
    return apiPost('/training/visual-start', params)
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
    return apiPost('/training/webhooks/test', { url })
  },

  // Training status
  async getStatus(): Promise<Record<string, unknown>> {
    return apiGet<Record<string, unknown>>('/training/status')
  },

  async trainFromFeedback(params?: { epochs?: number; batch_size?: number; learning_rate?: number; use_lora?: boolean }): Promise<{
    status: string; job_id?: string; samples?: number; message?: string
  }> {
    return apiPost('/training/from-feedback', params ?? {})
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
    const { PUBLIC_API_URL } = await import('./config')
    const token = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null
    const res = await fetch(`${PUBLIC_API_URL}/training/export/${jobId}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    })
    if (!res.ok) throw new Error(`Export failed (${res.status})`)
    return res.blob()
  },

  // ---------------------------------------------------------------------------
  // Unified Pipeline
  // ---------------------------------------------------------------------------

  async startUnified(config: UnifiedStartConfig): Promise<{ status: string }> {
    return apiPost('/training/unified-start', config)
  },

  async *streamUnified(): AsyncGenerator<UnifiedStreamEvent> {
    const { PUBLIC_API_URL } = await import('./config')
    const res = await fetch(`${PUBLIC_API_URL}/training/unified-stream`)
    if (!res.ok || !res.body) throw new Error(`Stream error (${res.status})`)
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const event = JSON.parse(line.slice(6))
            yield event as UnifiedStreamEvent
          } catch { /* skip malformed */ }
        }
      }
    }
  },
}

export interface UnifiedStartConfig {
  method?: string
  data_path?: string
  dataset_name?: string
  output_dir?: string
  epochs?: number
  batch_size?: number
  learning_rate?: number
  weight_decay?: number
  warmup_steps?: number
  distill?: boolean
  temperature?: number
  hf_model_name?: string
  vocab_size?: number
  n_embed?: number
  n_layer?: number
  n_head?: number
  block_size?: number
  checkpoint_dir?: string
  skip_generate?: boolean
  skip_distill?: boolean
  skip_train?: boolean
  skip_evaluate?: boolean
  skip_deploy?: boolean
  use_lora?: boolean
  lora_rank?: number
}

export interface UnifiedStreamEvent {
  stream: string
  phase: string
  status: string
  data?: {
    loss?: number
    progress?: number
    epoch?: number
    step?: number
    final_loss?: number
    total_steps?: number
    elapsed?: number
    model_path?: string
    error?: string
  }
  meta?: Record<string, unknown>
  message?: string
}

export const trainingController = trainingJobsController
