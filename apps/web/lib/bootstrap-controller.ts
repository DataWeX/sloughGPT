'use client'

// ═══════════════════════════════════════════════════════════════════════════
// bootstrap-controller — typed client for GET /bootstrap (card 342d729f)
// ═══════════════════════════════════════════════════════════════════════════
//
// /bootstrap is the one-request boot fan-out read-model: the landing
// journey's first-paint domain reads composed server-side from the member
// endpoints' own handlers. Each member carries the SAME payload its
// individual endpoint serves (envelope unwrapped by the server), so the
// existing controller types stay valid.
//
// Members are nullable twice over: `null` + errors[key] = 'not_ready'
// (member not registered yet — the server answers during model load),
// `null` + a real error string (member failed — degraded, never a 503).
// Consumers treat both as "skeleton", never as "empty".

import { apiGet } from '@/lib/http-client'
import type { HealthStatus } from '@/lib/model-controller'
import type { Session } from '@/lib/session-controller'
import type { TrainingJob } from '@/lib/training-controller'
import type { SoulsResponse } from '@/lib/souls-controller'
import type { Dataset } from '@/lib/dataset-controller'
import type { ModelInfo } from '@/lib/types'

export interface StartupProgress {
  phase: string
  step: number
  total: number
  message: string
}

export interface KnowledgeStats {
  total_items: number
  topic_count?: number
  [k: string]: unknown
}

export interface TurboStatus {
  status: string
  job_id: string | null
  progress?: number
  loss?: number | null
  paused?: boolean
  [k: string]: unknown
}

/** Raw /feedback/stats/summary slice — merged with meta_weights_stats by
 *  buildFeedbackStats() into the FeedbackStats shape components consume. */
export interface FeedbackSummary {
  thumbs_up?: number
  thumbs_down?: number
  total?: number
  up_ratio?: number
}

export interface MetaWeightsStats {
  current_weights?: Record<string, unknown>
  history_length?: number
  db_stats?: Record<string, number>
  [k: string]: unknown
}

export interface DatasetsPayload {
  datasets: Dataset[]
  count: number
}

export interface BootstrapMembers {
  health: HealthStatus | null
  startup_progress: StartupProgress | null
  /** Normalized by bootstrapController.get() via normalizeHfModels. */
  models: ModelInfo[] | null
  consciousness_status: Record<string, unknown> | null
  souls: SoulsResponse | null
  sessions: Session[] | null
  training_jobs: TrainingJob[] | null
  turbo_status: TurboStatus | null
  knowledge_stats: KnowledgeStats | null
  feedback_stats: FeedbackSummary | null
  meta_weights_stats: MetaWeightsStats | null
  datasets: DatasetsPayload | null
}

export interface BootstrapPayload {
  members: BootstrapMembers
  errors: Record<string, string>
}

/**
 * Same normalization modelController.list() applies to /models/hf: the
 * endpoint serves either a bare array or { models: [...] }, with entries
 * that may be bare model-id strings.
 */
export function normalizeHfModels(
  raw: (string | ModelInfo)[] | { models: (string | ModelInfo)[] } | null | undefined,
): ModelInfo[] {
  const list = Array.isArray(raw) ? raw : (raw?.models ?? [])
  return list.map((m) => (typeof m === 'string' ? { id: m, name: m, type: 'huggingface' } : m))
}

class BootstrapController {
  async get(): Promise<BootstrapPayload> {
    const payload = await apiGet<BootstrapPayload>('/bootstrap')
    // Normalize only a present member: null means not_ready/degraded and
    // must stay null so once-only seeders don't latch onto an empty list.
    if (payload?.members?.models) {
      payload.members.models = normalizeHfModels(payload.members.models)
    }
    return payload
  }
}

export const bootstrapController = new BootstrapController()
