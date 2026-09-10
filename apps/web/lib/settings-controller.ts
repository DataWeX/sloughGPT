import { apiGet, apiPatch, apiPost, apiDelete } from './http-client'

export interface GenerationSettings {
  temperature: number
  top_p: number
  top_k: number
  repetition_penalty: number
  max_new_tokens: number
  max_context_length: number
}

export interface TrainingSettings {
  preferred_model: string
  auto_train: boolean
  auto_train_threshold: number
  preferred_method: string
  max_checkpoints: number
  enable_tracking: boolean
}

export interface AdaptiveSettings {
  enabled: boolean
  exploration_rate: number
  learning_enabled: boolean
}

export interface VoiceSettings {
  noise_gate_db: number
  target_level_db: number
  vad_enabled: boolean
  vad_min_speech_ms: number
  agc_enabled: boolean
}

export interface UISettings {
  theme: string
  language: string
  show_confidence: boolean
  compact_mode: boolean
}

export interface AllSettings {
  generation: GenerationSettings
  training: TrainingSettings
  adaptive: AdaptiveSettings
  voice: VoiceSettings
  ui: UISettings
  version: number
}

export interface AdaptiveInsights {
  total_runs: number
  avg_quality: number
  avg_loss: number
  best_quality: number
  best_config: Record<string, number>
  trend: string
  recommendation: string
  message?: string
}

export const settingsController = {
  async getAll(): Promise<AllSettings> {
    return apiGet<AllSettings>('/settings')
  },

  async getGeneration(): Promise<GenerationSettings> {
    return apiGet<GenerationSettings>('/settings/generation')
  },

  async updateGeneration(updates: Partial<GenerationSettings>): Promise<GenerationSettings> {
    return apiPatch<GenerationSettings>('/settings/generation', updates)
  },

  async getTraining(): Promise<TrainingSettings> {
    return apiGet<TrainingSettings>('/settings/training')
  },

  async updateTraining(updates: Partial<TrainingSettings>): Promise<TrainingSettings> {
    return apiPatch<TrainingSettings>('/settings/training', updates)
  },

  async getAdaptive(): Promise<AdaptiveSettings> {
    return apiGet<AdaptiveSettings>('/settings/adaptive')
  },

  async updateAdaptive(updates: Partial<AdaptiveSettings>): Promise<AdaptiveSettings> {
    return apiPatch<AdaptiveSettings>('/settings/adaptive', updates)
  },

  async getVoice(): Promise<VoiceSettings> {
    return apiGet<VoiceSettings>('/settings/voice')
  },

  async updateVoice(updates: Partial<VoiceSettings>): Promise<VoiceSettings> {
    return apiPatch<VoiceSettings>('/settings/voice', updates)
  },

  async getUI(): Promise<UISettings> {
    return apiGet<UISettings>('/settings/ui')
  },

  async updateUI(updates: Partial<UISettings>): Promise<UISettings> {
    return apiPatch<UISettings>('/settings/ui', updates)
  },

  async reset(): Promise<{ status: string; message: string }> {
    return apiPost('/settings/reset')
  },

  async getAdaptiveInsights(): Promise<AdaptiveInsights> {
    return apiGet<AdaptiveInsights>('/settings/adaptive/insights')
  },

  async exportTrainingHistory(format: string = 'json', limit: number = 0): Promise<{ format: string; outcomes?: Record<string, unknown>[]; content?: string; count: number }> {
    return apiGet(`/settings/training/history/export?format=${format}&limit=${limit}`)
  },

  async generateModelCard(name: string, params: Record<string, unknown> = {}): Promise<{ card: Record<string, unknown>; markdown: string }> {
    return apiPost('/settings/model-card', { name, ...params })
  },

  async compareTrainingRuns(runA: string, runB: string): Promise<{ run_a: Record<string, unknown>; run_b: Record<string, unknown>; differences: Record<string, { run_a: unknown; run_b: unknown }>; a_wins: number; b_wins: number }> {
    return apiGet(`/settings/training/compare?run_a=${runA}&run_b=${runB}`)
  },

  async getBatchTrainingStatus(): Promise<{ jobs: Array<Record<string, unknown>>; summary: { total: number; running: number; queued: number; completed: number; failed: number } }> {
    return apiGet('/settings/training/batch-status')
  },

  async listTrainingPresets(): Promise<{ presets: Array<Record<string, unknown>> }> {
    return apiGet('/settings/training/presets')
  },

  async getTrainingPreset(name: string): Promise<Record<string, unknown>> {
    return apiGet(`/settings/training/presets/${name}`)
  },

  async applyTrainingPreset(name: string): Promise<{ preset: string; applied: Record<string, unknown> }> {
    return apiPost(`/settings/training/presets/${name}/apply`)
  },

  async getTrainingRun(runId: string): Promise<Record<string, unknown>> {
    return apiGet(`/settings/training/runs/${runId}`)
  },

  async deleteTrainingRun(runId: string): Promise<{ deleted: boolean; run_id: string }> {
    return apiDelete(`/settings/training/runs/${runId}`)
  },

  async filterTrainingRuns(params: Record<string, string | number> = {}): Promise<{ runs: Array<Record<string, unknown>>; count: number }> {
    const qs = new URLSearchParams()
    for (const [k, v] of Object.entries(params)) {
      if (v !== '' && v !== undefined && v !== null) qs.set(k, String(v))
    }
    return apiGet(`/settings/training/runs?${qs.toString()}`)
  },

  async clearTrainingHistory(): Promise<{ cleared: boolean; removed_count: number }> {
    return apiPost('/settings/training/history/clear')
  },

  async addRunTag(runId: string, tag: string): Promise<Record<string, unknown>> {
    return apiPost(`/settings/training/runs/${runId}/tags?tag=${encodeURIComponent(tag)}`)
  },

  async removeRunTag(runId: string, tag: string): Promise<Record<string, unknown>> {
    return apiDelete(`/settings/training/runs/${runId}/tags/${encodeURIComponent(tag)}`)
  },

  async setRunNotes(runId: string, notes: string): Promise<Record<string, unknown>> {
    return apiPut(`/settings/training/runs/${runId}/notes?notes=${encodeURIComponent(notes)}`)
  },

  async getAllTags(): Promise<{ tags: string[] }> {
    return apiGet('/settings/training/tags')
  },

  async getRunsByTag(tag: string): Promise<{ runs: Array<Record<string, unknown>>; count: number; tag: string }> {
    return apiGet(`/settings/training/tags/${encodeURIComponent(tag)}`)
  },

  async exportTrainingRun(runId: string, format: string = 'json'): Promise<{ run_id: string; format: string; content: string }> {
    return apiGet(`/settings/training/runs/${runId}/export?format=${format}`)
  },

  async toggleBookmark(runId: string): Promise<Record<string, unknown>> {
    return apiPost(`/settings/training/runs/${runId}/bookmark`)
  },

  async getBookmarkedRuns(): Promise<{ runs: Array<Record<string, unknown>>; count: number }> {
    return apiGet('/settings/training/bookmarks')
  },

  async duplicateTrainingRun(runId: string, newRunId: string = ''): Promise<Record<string, unknown>> {
    const params = newRunId ? `?new_run_id=${encodeURIComponent(newRunId)}` : ''
    return apiPost(`/settings/training/runs/${runId}/duplicate${params}`)
  },

  async bulkDeleteRuns(runIds: string[]): Promise<{ deleted_count: number; requested: number }> {
    const ids = runIds.join(',')
    return apiPost(`/settings/training/runs/bulk/delete?run_ids=${encodeURIComponent(ids)}`)
  },

  async bulkAddTag(runIds: string[], tag: string): Promise<{ updated_count: number; tag: string }> {
    const ids = runIds.join(',')
    return apiPost(`/settings/training/runs/bulk/tag?run_ids=${encodeURIComponent(ids)}&tag=${encodeURIComponent(tag)}`)
  },

  async bulkBookmark(runIds: string[], bookmarked: boolean = true): Promise<{ updated_count: number; bookmarked: boolean }> {
    const ids = runIds.join(',')
    return apiPost(`/settings/training/runs/bulk/bookmark?run_ids=${encodeURIComponent(ids)}&bookmarked=${bookmarked}`)
  },
}
