/**
 * Consciousness controller — API client for the consciousness subsystem.
 */
import { apiGet, apiPost, apiPatch, apiDelete } from '@/lib/http-client'

export interface ConsciousnessStatus {
  status: string
  uptime: number
  episodes: number
  beliefs_count: number
  qualia_count: number
  enabled: boolean
  level: number
  current_qualia: Record<string, number>
  beliefs: Record<string, number>
  narrative: string
  training: {
    is_training: boolean
    total_pairs: number
    current_epoch: number
    loss: number
  }
}

export interface SelfModel {
  traits: Record<string, number>
  beliefs: string[]
  goals: string[]
  memories: string[]
  self_beliefs: Record<string, number>
}

export interface Qualia {
  id: string
  type: string
  intensity: number
  valence: number
  timestamp: string
}

export interface Episode {
  input: string
  response: string
  narrative: string
  qualia: Record<string, number>
  growth_delta: number
  rating: number
  timestamp: string
}

export interface Belief {
  id: string
  statement: string
  confidence: number
  source: string
  timestamp: string
}

export interface Personality {
  traits: Record<string, number>
  description: string
  name: string
}

export interface Persona {
  id: string
  name: string
  personality: Personality
  active: boolean
  created_at: string
}

export interface TrainingStatus {
  status: string
  epoch: number
  loss: number
  episodes_trained: number
}

export interface FeedbackResult {
  accepted: boolean
  belief_updates: string[]
}

export interface BackupResult {
  backup_id: string
  size_bytes: number
  created_at: string
}

export const consciousnessController = {
  async getStatus(): Promise<ConsciousnessStatus> {
    return apiGet<ConsciousnessStatus>('/consciousness/status')
  },

  async getSelfModel(): Promise<SelfModel> {
    return apiGet<SelfModel>('/consciousness/self-model')
  },

  async getQualia(): Promise<Record<string, number>> {
    return apiGet('/consciousness/qualia')
  },

  async reflect(): Promise<{ reflection: string }> {
    return apiPost('/consciousness/reflect')
  },

  async updateConfig(config: Record<string, unknown>): Promise<{ updated: boolean }> {
    return apiPatch('/consciousness/config', config)
  },

  async getTrainingStatus(): Promise<TrainingStatus> {
    return apiGet<TrainingStatus>('/consciousness/train/status')
  },

  async startTraining(config?: Record<string, unknown>): Promise<{ started: boolean }> {
    return apiPost('/consciousness/train/start', config ?? {})
  },

  async evaluate(): Promise<{ overall_score: number; metrics: Record<string, { score: number; weight: number; details: string }>; diagnostics: string[] }> {
    return apiGet('/consciousness/evaluate')
  },

  async getEpisodeHistory(limit?: number): Promise<{ episodes: Episode[]; total: number }> {
    const qs = limit ? `?limit=${limit}` : ''
    return apiGet(`/consciousness/history/episodes${qs}`)
  },

  async getQualiaHistory(limit?: number): Promise<{ history: Qualia[] }> {
    const qs = limit ? `?limit=${limit}` : ''
    return apiGet(`/consciousness/history/qualia${qs}`)
  },

  async getBeliefsHistory(limit?: number): Promise<{ beliefs: Array<{ timestamp: number; step: number; [key: string]: number }> }> {
    const qs = limit ? `?limit=${limit}` : ''
    return apiGet(`/consciousness/history/beliefs${qs}`)
  },

  async submitFeedback(feedback: Record<string, unknown>): Promise<FeedbackResult> {
    return apiPost('/consciousness/feedback', feedback)
  },

  async seedData(data: Record<string, unknown>): Promise<{ seeded: boolean }> {
    const count = (data.count as number) || 30
    return apiPost(`/consciousness/seed?count=${count}`, {})
  },

  async getPersonality(): Promise<Personality> {
    return apiGet<Personality>('/consciousness/personality')
  },

  async updatePersonality(traits: Record<string, number>): Promise<{ updated: boolean }> {
    return apiPatch('/consciousness/personality', { traits })
  },

  async resetPersonality(): Promise<{ reset: boolean }> {
    return apiPost('/consciousness/personality/reset')
  },

  async clearEpisodes(): Promise<{ cleared: boolean; episodes_cleared: number }> {
    return apiPost('/consciousness/clear/episodes')
  },

  async resetBeliefs(): Promise<{ reset: boolean; beliefs: Record<string, number> }> {
    return apiPost('/consciousness/clear/beliefs')
  },

  async getPersonalityHistory(): Promise<{ history: Personality[] }> {
    return apiGet('/consciousness/personality/history')
  },

  async getPersonalityPresets(): Promise<{ presets: Record<string, unknown>[] }> {
    return apiGet('/consciousness/personality/presets')
  },

  async applyPersonalityPreset(presetName: string): Promise<{ applied: boolean }> {
    return apiPost('/consciousness/personality/presets/apply', { preset: presetName })
  },

  async getPersonalityConflicts(): Promise<{ conflicts: string[] }> {
    return apiGet('/consciousness/personality/conflicts')
  },

  async listPersonas(): Promise<{ personas: Persona[] }> {
    return apiGet('/consciousness/personas')
  },

  async savePersona(persona: Omit<Persona, 'id' | 'created_at'>): Promise<{ id: string }> {
    return apiPost('/consciousness/personas/save', persona)
  },

  async getPersona(personaId: string): Promise<Persona> {
    return apiGet<Persona>(`/consciousness/personas/${personaId}`)
  },

  async activatePersona(personaId: string): Promise<{ activated: boolean }> {
    return apiPost(`/consciousness/personas/${personaId}/activate`)
  },

  async deletePersona(personaId: string): Promise<{ deleted: boolean }> {
    return apiDelete(`/consciousness/personas/${personaId}`)
  },

  async healthCheck(): Promise<{ health_score: number; enabled: boolean; level: number; episodes: number; avg_growth: number; positive_ratio: number; qualia: Record<string, number>; last_reflection: string; diagnostics: string[] }> {
    return apiGet('/consciousness/health')
  },

  async backup(): Promise<BackupResult> {
    return apiPost('/consciousness/backup')
  },

  async restore(backupId: string): Promise<{ restored: boolean }> {
    return apiPost('/consciousness/restore', { backup_id: backupId })
  },

  async downloadBackup(): Promise<Blob> {
    return apiGet('/consciousness/backup/download')
  },

  async importBackup(file: File): Promise<{ imported: boolean }> {
    const formData = new FormData()
    formData.append('file', file)
    return apiPost('/consciousness/backup/import', formData)
  },

  async getStats(): Promise<Record<string, unknown>> {
    return apiGet('/consciousness/stats')
  },

  async batch(operations: Array<{ type: string; payload: Record<string, unknown> }>): Promise<unknown> {
    return apiPost('/consciousness/batch', { operations })
  },

  connectStream(onEvent: (event: Record<string, unknown>) => void, onError?: (err: Error) => void): () => void {
    const { createSSEStream } = require('./sse-client') as typeof import('./sse-client')
    const stream = createSSEStream({
      url: '/consciousness/stream',
      onEvent: (envelope: { data: Record<string, unknown>; type?: string }) => {
        if (envelope.type !== 'heartbeat') onEvent(envelope.data)
      },
      onError,
      reconnect: true,
      maxReconnects: Infinity,
      baseReconnectMs: 3000,
      maxReconnectMs: 15_000,
    })
    stream.start()
    return () => { stream.stop() }
  },
}
