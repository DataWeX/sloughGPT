/**
 * Consciousness controller — API client for the consciousness subsystem.
 */
import { apiGet, apiPost, apiPatch, apiDelete } from '@/lib/http-client'
import { createSSEStream } from '@/lib/sse-client'

/** True payload of GET /consciousness/status — engine.get_status() + router-injected training (trainer.get_status()). */
export interface ConsciousnessStatus {
  enabled: boolean
  level: number
  episodes: number
  beliefs: Record<string, number>
  current_qualia: Record<string, number>
  curiosity_topics: Record<string, unknown>
  training: {
    pairs_collected: number
    min_pairs: number
    should_train: boolean
    is_training: boolean
    training_runs: number
    /** Last entry of the trainer's run history, null before the first run. */
    last_result: {
      timestamp: number
      status: string
      pairs: number
      loss: number | null
      adapter: string | null
      elapsed: number
    } | null
  }
}

/** True payload of GET /consciousness/stats — router get_stats(). */
export type ConsciousnessStats = {
  total_episodes: number
  total_feedback: number
  avg_rating: number
  avg_growth: number
  qualia_averages: Record<string, number>
  belief_averages: Record<string, number>
  personality_summary: {
    voice_averages: Record<string, number>
    trait_averages: Record<string, number>
  }
  /** trainer.get_status() — same dict as ConsciousnessStatus['training']. */
  training_status: ConsciousnessStatus['training']
  health_score: number
  active_personas: number
}

export interface SelfModel {
  traits: Record<string, number>
  beliefs: string[]
  goals: string[]
  memories: string[]
  self_beliefs: Record<string, number>
}

export interface QualiaEntry {
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
  input_text: string
  self_insight: string
  qualia: Record<string, number>
  growth_delta: number
  rating: number
  timestamp: number
}

export interface HealthCheckResult {
  health_score: number
  enabled: boolean
  level: number
  episodes: number
  avg_growth: number
  positive_ratio: number
  qualia: Record<string, number>
  last_reflection: string
  diagnostics: string[]
}

export interface EvalReport {
  overall_score: number
  metrics: Record<string, { score: number; weight: number; details: string }>
  diagnostics: string[]
}

export interface BeliefHistoryEntry {
  timestamp: number
  step: number
  [key: string]: number
}

export interface Belief {
  id: string
  statement: string
  confidence: number
  source: string
  timestamp: string
}

/** True payload of GET /consciousness/personality (PersonalityProfile.to_dict()). */
export interface Personality {
  values: string[]
  goals: string[]
  voice: Record<string, number>
  style: Record<string, boolean>
  traits: Record<string, number>
  interests: string[]
  avoid: string[]
}

/** A preset profile with its name-key injected as `name` (payload serves a name-keyed dict). */
export interface PersonalityPreset extends Personality {
  name: string
}

/** Entry of GET /consciousness/personality/history — a voice/traits trajectory over episodes. */
export interface PersonalityHistoryEntry {
  timestamp: number
  step: number
  voice: Record<string, number>
  traits: Record<string, number>
}

/** One conflict entry of GET /consciousness/personality/conflicts (PersonalityManager.get_conflicts). */
export interface PersonalityConflict {
  type: string
  severity: string
  message: string
  fields: string[]
}

/** A saved persona list entry (PersonalityManager.list_personas). */
export interface Persona {
  id: string
  name: string
  values: string[]
  updated_at: number | null
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

  async reflect(): Promise<{
    reflection: string
    narrative?: string
    belief_deltas?: Record<string, number>
    strategy_notes?: string[]
    avg_growth?: number
    trajectory?: string
    episode_count?: number
    updated_beliefs?: Record<string, number>
  }> {
    return apiPost('/consciousness/reflect')
  },

  /** PATCH /consciousness/config — returns the post-update level/enabled. */
  async updateConfig(config: Record<string, unknown>): Promise<{ level: number; enabled: boolean }> {
    return apiPatch('/consciousness/config', config)
  },

  async getTrainingStatus(): Promise<TrainingStatus> {
    return apiGet<TrainingStatus>('/consciousness/train/status')
  },

  async startTraining(config?: Record<string, unknown>): Promise<{ started: boolean }> {
    return apiPost('/consciousness/train/start', config ?? {})
  },

  async evaluate(): Promise<EvalReport> {
    return apiGet('/consciousness/evaluate')
  },

  async getEpisodeHistory(limit?: number): Promise<{ episodes: Episode[]; total: number }> {
    const qs = limit ? `?limit=${limit}` : ''
    return apiGet(`/consciousness/history/episodes${qs}`)
  },

  async getQualiaHistory(limit?: number): Promise<{ history: QualiaEntry[] }> {
    const qs = limit ? `?limit=${limit}` : ''
    return apiGet(`/consciousness/history/qualia${qs}`)
  },

  async getBeliefsHistory(limit?: number): Promise<{ beliefs: BeliefHistoryEntry[] }> {
    const qs = limit ? `?limit=${limit}` : ''
    return apiGet(`/consciousness/history/beliefs${qs}`)
  },

  async submitFeedback(feedback: Record<string, unknown>): Promise<FeedbackResult> {
    return apiPost('/consciousness/feedback', feedback)
  },

  async seedData(data: Record<string, unknown>): Promise<{ seeded: number }> {
    const count = (data.count as number) || 30
    return apiPost(`/consciousness/seed?count=${count}`, {})
  },

  async getPersonality(): Promise<Personality> {
    return apiGet<Personality>('/consciousness/personality')
  },

  /**
   * PATCH /consciousness/personality — body mirrors the backend's
   * PersonalityUpdateRequest (all fields optional; absent keys are dropped
   * by JSON serialization so the backend leaves them untouched).
   */
  async updatePersonality(update: {
    values?: string[]
    goals?: string[]
    voice?: Record<string, number>
    style?: Record<string, boolean>
    traits?: Record<string, number>
    interests?: string[]
    avoid?: string[]
  }): Promise<{ updated: boolean }> {
    return apiPatch('/consciousness/personality', update)
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

  async getPersonalityHistory(): Promise<{ history: PersonalityHistoryEntry[] }> {
    return apiGet('/consciousness/personality/history')
  },

  async getPersonalityPresets(): Promise<{ presets: PersonalityPreset[]; names: string[] }> {
    const resp = await apiGet<{ presets: Record<string, Omit<PersonalityPreset, 'name'>>; names: string[] }>('/consciousness/personality/presets')
    // The endpoint serves presets as a name-keyed dict of profiles (no `name`
    // inside each profile); project to an array with the key injected as `name`.
    return {
      presets: Object.entries(resp.presets ?? {}).map(([name, profile]) => ({ name, ...profile })),
      names: resp.names ?? [],
    }
  },

  async applyPersonalityPreset(presetName: string): Promise<{ applied: boolean }> {
    return apiPost('/consciousness/personality/presets/apply', { preset: presetName })
  },

  async getPersonalityConflicts(): Promise<{ conflicts: PersonalityConflict[] }> {
    return apiGet('/consciousness/personality/conflicts')
  },

  async listPersonas(): Promise<{ personas: Persona[] }> {
    return apiGet('/consciousness/personas')
  },

  /** Body of POST /consciousness/personas/save (SavePersonaRequest); profile = full PersonalityProfile dict. */
  async savePersona(persona: { persona_id: string; name?: string; profile?: Personality }): Promise<{ id: string; name: string; updated_at: number }> {
    return apiPost('/consciousness/personas/save', persona)
  },

  async getPersona(personaId: string): Promise<{ id: string } & Personality> {
    return apiGet(`/consciousness/personas/${personaId}`)
  },

  async activatePersona(personaId: string): Promise<{ activated: boolean }> {
    return apiPost(`/consciousness/personas/${personaId}/activate`)
  },

  async deletePersona(personaId: string): Promise<{ deleted: boolean }> {
    return apiDelete(`/consciousness/personas/${personaId}`)
  },

  async healthCheck(): Promise<HealthCheckResult> {
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

  async getStats(): Promise<ConsciousnessStats> {
    return apiGet<ConsciousnessStats>('/consciousness/stats')
  },

  async batch(
    operations: Array<{ type: string; payload: Record<string, unknown> }>,
  ): Promise<unknown> {
    return apiPost('/consciousness/batch', { operations })
  },

  connectStream(
    onEvent: (event: Record<string, unknown>) => void,
    onError?: (err: Error) => void,
  ): () => void {
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
    return () => {
      stream.stop()
    }
  },
}
