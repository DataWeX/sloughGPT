/**
 * Training Jobs Controller — axios-based API for training job management.
 */

import { apiGet, apiPost, apiDelete, apiPatch, authFetch, streamSSE } from './http-client'
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
  method?: string
  epochs?: number
  current_epoch?: number
  global_step?: number
  total_steps?: number
  steps_per_sec?: number
  eta_s?: number | null
  elapsed_s?: number
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

export interface FineTunedModel {
  name: string
  model_path: string
  size_mb: number
  size_bytes?: number
  created_at?: string
  model: string
  dataset: string
  model_name?: string
  final_loss?: number | null
  epochs?: number
}

export interface FineTunedModelLoadResponse {
  status: string
  name: string
  model_path: string
  model_id?: string
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
  // Legacy fields (kept for backward compat)
  n_decoder_layers?: number
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

export interface TurboJobStatus {
  status: 'idle' | 'running' | 'complete' | 'error'
  job_id?: string | null
  global_step?: number
  total_steps?: number
  progress?: number
  loss?: number | null
  learning_rate?: number | null
  steps_per_sec?: number | null
  eta_s?: number | null
  elapsed_s?: number | null
  error?: string | null
  result?: Record<string, unknown> | null
}

export const trainingJobsController = {
  async startAutoTrain(params?: AutoTrainStartRequest): Promise<AutoTrainStartResponse> {
    return apiPost<AutoTrainStartResponse>('/auto-train/start', params ?? null)
  },

  async stopAutoTrain(): Promise<void> {
    await apiPost('/auto-train/stop')
  },

  async pauseTraining(): Promise<{ success: boolean }> {
    return apiPost('/auto-train/pause')
  },

  async getTrainingLog(): Promise<string[]> {
    try {
      const data = await apiGet<{ lines: string[] }>('/auto-train/log')
      return data.lines ?? []
    } catch { return [] }
  },

  async resumeTraining(): Promise<{ success: boolean }> {
    return apiPost('/auto-train/resume')
  },

  async startTurboTrain(params?: TurboTrainStartRequest): Promise<TurboTrainResponse> {
    return apiPost<TurboTrainResponse>('/auto-train/start-turbo', params ?? null)
  },

  async getTurboStatus(): Promise<TurboJobStatus> {
    return apiGet<TurboJobStatus>('/auto-train/turbo/status')
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

  async listFineTuned(): Promise<FineTunedModel[]> {
    try {
      const data = await apiGet<FineTunedModel[] | { models: FineTunedModel[] }>('/training/finetuned-models')
      return Array.isArray(data) ? data : (data?.models ?? [])
    } catch { return [] }
  },

  async loadFineTuned(name: string): Promise<FineTunedModelLoadResponse> {
    return apiPost(`/training/finetuned-models/${encodeURIComponent(name)}/load`)
  },

  async deleteFineTuned(name: string): Promise<{ status: string; name: string }> {
    return apiDelete(`/training/finetuned-models/${encodeURIComponent(name)}`)
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

  async startLoraFinetune(params: {
    model_path: string
    dataset: string
    name?: string
    rank?: number
    alpha?: number
    dropout?: number
    target_modules?: string[]
    epochs?: number
    batch_size?: number
    learning_rate?: number
    max_seq_length?: number
    warmup_steps?: number
    weight_decay?: number
    gradient_clip?: number
    log_interval?: number
    eval_interval?: number
    device?: string
  }): Promise<{ job_id: string; status: string; message: string }> {
    return apiPost('/training/lora-finetune', params)
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

  async deleteCheckpointsBatch(names: string[]): Promise<{ deleted: number }> {
    const results = await Promise.allSettled(names.map(n => this.deleteCheckpoint(n)))
    return { deleted: results.filter(r => r.status === 'fulfilled').length }
  },

  async downloadCheckpoint(name: string): Promise<Blob> {
    const res = await authFetch(`/auto-train/checkpoints/${encodeURIComponent(name)}/download`)
    if (!res.ok) throw new Error(`Download failed (${res.status})`)
    return res.blob()
  },

  async exportMetrics(): Promise<Blob> {
    const res = await authFetch('/auto-train/metrics/export')
    if (!res.ok) throw new Error(`Export failed (${res.status})`)
    return res.blob()
  },

  async getCheckpointInfo(name: string): Promise<Record<string, unknown>> {
    return apiGet<Record<string, unknown>>(`/auto-train/checkpoints/${encodeURIComponent(name)}/info`)
  },

  async downloadTrainingJob(jobId: string): Promise<Blob> {
    const res = await authFetch(`/training/export/${jobId}`)
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
    try {
      for await (const event of streamSSE('/mobile/train/from-sessions', { body: params ?? {} })) {
        yield event as { stream: string; phase: string; status: string; data: Record<string, unknown>; meta: Record<string, unknown>; message: string }
        if (event.status === 'complete' || event.status === 'error') return
      }
    } catch (err) {
      throw new Error(`Training request failed: ${err instanceof Error ? err.message : 'unknown'}`)
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

  // ---------------------------------------------------------------------------
  // On-device SloNet training from sessions (pure NumPy)
  // ---------------------------------------------------------------------------

  async startFromSessionsSloNet(params: {
    epochs?: number
    learning_rate?: number
    n_embed?: number
    n_layer?: number
    n_head?: number
    block_size?: number
    soul_name?: string
    min_pair_quality?: number
    max_pairs?: number
    session_ids?: string[]
  }): Promise<void> {
    await apiPost('/auto-train/from-sessions/start', params)
  },

  async *streamFromSessionsSloNet(): AsyncGenerator<{
    stream: string
    phase: string
    status: string
    data: Record<string, unknown>
    meta: Record<string, unknown>
    message: string
  }> {
    try {
      for await (const event of streamSSE('/auto-train/from-sessions/stream', { method: 'GET' })) {
        yield event as { stream: string; phase: string; status: string; data: Record<string, unknown>; meta: Record<string, unknown>; message: string }
        if (event.status === 'complete' || event.status === 'error') return
      }
    } catch (err) {
      throw new Error(`Stream failed: ${err instanceof Error ? err.message : 'unknown'}`)
    }
  },

  async cancelFromSessionsSloNet(): Promise<void> {
    await apiGet('/auto-train/from-sessions/cancel')
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
    return Array.isArray(data) ? data : (data as { data: ChatSession[] }).data || []
  },

  async exportTrainingPairs(params?: {
    min_quality?: number
    session_id?: string
    limit?: number
  }): Promise<Blob> {
    const query = new URLSearchParams()
    if (params?.min_quality != null) query.set('min_quality', String(params.min_quality))
    if (params?.session_id) query.set('session_id', params.session_id)
    if (params?.limit != null) query.set('limit', String(params.limit))
    const qs = query.toString()
    const res = await authFetch(`/mobile/train/export${qs ? `?${qs}` : ''}`)
    if (!res.ok) throw new Error(`Export failed (${res.status})`)
    return res.blob()
  },

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
  captured_count?: number
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
