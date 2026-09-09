import { apiGet, apiPatch, apiPost } from './http-client'

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
}
