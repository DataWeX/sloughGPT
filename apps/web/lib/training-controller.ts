/**
 * Training Jobs Controller — axios-based API for training job management.
 */

import { apiGet, apiPost, apiDelete, apiPatch } from './http-client'
import type { Checkpoint } from './souls-controller'

export interface TrainingJob {
  id: string
  name: string
  status: string
  progress: number
  created_at: string
  finished_at?: string
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
  metrics?: Record<string, unknown>
  epochs_completed?: number
  error?: string
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
  description?: string
  is_active?: boolean
  created_at?: string
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
  method?: 'slonet' | 'transformer'
  dataset_id?: string
  data_path?: string
  epochs?: number
  batch_size?: number
  learning_rate?: number
  vocab_size?: number
  n_embed?: number
  n_head?: number
  n_layer?: number
  block_size?: number
  dropout?: number
  // Legacy fields (ignored)
  n_encoder_layers?: number
  n_decoder_layers?: number
  dim_feedforward?: number
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
    const data = await apiGet<Checkpoint[] | { checkpoints: Checkpoint[] }>('/auto-train/checkpoints')
    return Array.isArray(data) ? data : (data?.checkpoints ?? [])
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

  async startDistill(params: {
    teacher_model: string
    dataset: string
    name?: string
    temperature?: number
    alpha?: number
    beta?: number
    epochs?: number
    embed_dim?: number
    n_layers?: number
    n_heads?: number
    block_size?: number
  }): Promise<{ job_id: string; status: string; message: string }> {
    return apiPost('/training/distill', params)
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
  // Train from server conversation logs (SSE streaming)
  // ---------------------------------------------------------------------------

  async *streamTrainFromSessions(params?: {
    limit?: number
    min_length?: number
    model?: string
    session_ids?: string[]
  }): AsyncGenerator<{
    stream: string
    phase: string
    status: string
    data: Record<string, unknown>
    meta: Record<string, unknown>
    message: string
  }> {
    const { PUBLIC_API_URL } = await import('./config')
    const token = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null
    const headers: Record<string, string> = { 'Content-Type': 'application/json' }
    if (token) headers['Authorization'] = `Bearer ${token}`

    const res = await fetch(`${PUBLIC_API_URL}/mobile/train/from-sessions`, {
      method: 'POST',
      headers,
      body: JSON.stringify(params ?? {}),
    })

    if (!res.ok || !res.body) {
      const text = await res.text().catch(() => '')
      throw new Error(`Training request failed (${res.status}): ${text.slice(0, 200)}`)
    }

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
            yield event
            if (event.status === 'complete' || event.status === 'error') return
          } catch { /* skip malformed */ }
        }
      }
    }

    // Drain remaining buffer
    if (buffer.startsWith('data: ')) {
      try {
        const event = JSON.parse(buffer.slice(6))
        yield event
      } catch { /* skip */ }
    }
  },

  // Blocking fallback for mobile (non-SSE) — consumes SSE stream and returns final result
  async trainFromSessions(params?: {
    limit?: number
    min_length?: number
    model?: string
    session_ids?: string[]
  }): Promise<{
    success: boolean
    checkpoint_name: string
    loss: number
    steps: number
    elapsed_ms: number
    message?: string
  }> {
    let finalResult: Record<string, unknown> = {}
    for await (const event of this.streamTrainFromSessions(params)) {
      if (event.status === 'complete') {
        finalResult = event.data ?? {}
      } else if (event.status === 'error') {
        throw new Error(event.message || 'Training failed')
      }
    }
    return {
      success: !!finalResult.checkpoint_name,
      checkpoint_name: (finalResult.checkpoint_name as string) ?? '',
      loss: (finalResult.loss as number) ?? 0,
      steps: (finalResult.steps as number) ?? 0,
      elapsed_ms: (finalResult.elapsed_ms as number) ?? 0,
      message: undefined,
    }
  },

  async getAutoTrainStatus(): Promise<AutoTrainStatus> {
    return apiGet<AutoTrainStatus>('/mobile/train/auto-status')
  },

  async updateAutoTrainConfig(params: { threshold?: number; interval_s?: number }): Promise<AutoTrainStatus> {
    const query = new URLSearchParams()
    if (params.threshold != null) query.set('threshold', String(params.threshold))
    if (params.interval_s != null) query.set('interval_s', String(params.interval_s))
    const qs = query.toString()
    return apiPatch<AutoTrainStatus>(`/mobile/train/auto-config${qs ? `?${qs}` : ''}`)
  },

  // ---------------------------------------------------------------------------
  // Training data management (MogDB pairs)
  // ---------------------------------------------------------------------------

  async getTrainingStats(): Promise<TrainingDataStats> {
    return apiGet<TrainingDataStats>('/mobile/train/stats')
  },

  async getPendingPairs(limit = 50): Promise<{ pairs: TrainingPair[]; count: number }> {
    const data = await apiGet<{ pairs: TrainingPair[]; count: number }>('/mobile/train/pending', { limit: String(limit) })
    return data
  },

  async getEvalHistory(limit: number = 20): Promise<{ results: EvalHistoryEntry[] }> {
    return apiGet('/lora-eval/history', { limit: String(limit) })
  },

  async runEval(adapterPath: string = 'data/user_adapters/best_aggregated.npz', soul: string = 'assistant'): Promise<{
    status: string
    baseline?: EvalResult
    with_adapter?: EvalResult
    delta?: { perplexity_delta: number; bleu_delta: number; throughput_delta: number; verdict: string }
    report?: string
  }> {
    return apiGet('/lora-eval/run', { adapter_path: adapterPath, soul })
  },

  async listTrainingPairs(params?: {
    limit?: number
    offset?: number
    min_quality?: number
    session_id?: string
    search?: string
  }): Promise<{ pairs: TrainingPair[]; total: number; count: number; offset: number }> {
    const query: Record<string, string> = {}
    if (params?.limit != null) query.limit = String(params.limit)
    if (params?.offset != null) query.offset = String(params.offset)
    if (params?.min_quality != null) query.min_quality = String(params.min_quality)
    if (params?.session_id) query.session_id = params.session_id
    if (params?.search) query.search = params.search
    return apiGet('/mobile/train/pairs', query)
  },

  async getSessionPairs(sessionId: string): Promise<{ pairs: TrainingPair[]; count: number }> {
    const data = await apiGet<{ pairs: TrainingPair[]; count: number }>(`/mobile/train/session/${encodeURIComponent(sessionId)}`)
    return data
  },

  async deletePair(pairId: string): Promise<{ status: string }> {
    return apiDelete<{ status: string }>(`/mobile/train/pair/${encodeURIComponent(pairId)}`)
  },

  async updatePairQuality(pairId: string, quality: number): Promise<{ status: string }> {
    return apiPatch<{ status: string }>(`/mobile/train/pair/${encodeURIComponent(pairId)}`, { quality })
  },

  async deleteSyncedPairs(): Promise<{ status: string; count: number }> {
    return apiDelete<{ status: string; count: number }>('/mobile/train/synced')
  },

  async deletePairsBulk(ids: string[]): Promise<{ status: string; count: number }> {
    const params = ids.map(id => `ids=${encodeURIComponent(id)}`).join('&')
    return apiDelete<{ status: string; count: number }>(`/mobile/train/pairs/bulk?${params}`)
  },

  async listChatSessions(): Promise<ChatSession[]> {
    const data = await apiGet<{ data: ChatSession[] } | ChatSession[]>('/chat/sessions')
    return Array.isArray(data) ? data : (data as any).data || []
  },

  async exportTrainingPairs(params?: {
    min_quality?: number
    session_id?: string
    limit?: number
  }): Promise<Blob> {
    const { PUBLIC_API_URL } = await import('./config')
    const token = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null
    const query = new URLSearchParams()
    if (params?.min_quality != null) query.set('min_quality', String(params.min_quality))
    if (params?.session_id) query.set('session_id', params.session_id)
    if (params?.limit != null) query.set('limit', String(params.limit))
    const qs = query.toString()
    const res = await fetch(`${PUBLIC_API_URL}/mobile/train/export${qs ? `?${qs}` : ''}`, {
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

export interface AutoTrainStatus {
  enabled: boolean
  threshold: number
  interval_s: number
  pending_conversations: number
  total_trains: number
  last_train: string | null
  last_loss: number | null
  last_checkpoint: string | null
  session_count: number
  response_log_count: number
}

export interface TrainingDataStats {
  total: number
  pending: number
  synced: number
  used: number
  by_quality: Record<string, number>
}

export interface TrainingPair {
  id: string
  user_msg: string
  assistant_msg: string
  quality: number
  session_id: string
  timestamp: number
}

export interface ChatSession {
  id: string
  name: string
  updated_at: string
  archived?: boolean
  messages?: Array<{ role: string; content: string }>
}

export interface EvalResult {
  timestamp: string
  perplexity: number
  bleu: number
  avg_response_len: number
  inference_time_sec: number
  tokens_per_sec: number
  personality_score: number
  adapter_path?: string | null
}

export interface EvalHistoryEntry {
  timestamp: string
  baseline: EvalResult
  with_adapter?: EvalResult
  delta?: {
    perplexity_delta: number
    bleu_delta: number
    throughput_delta: number
    verdict: string
  }
  report?: string
}

export const trainingController = trainingJobsController
