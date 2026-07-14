/**
 * User Adapters Controller — axios-based API for per-user LoRA adapters.
 *
 * Usage:
 *   import { userAdaptersController } from '@/lib/user-adapters-controller'
 *   const stats = await userAdaptersController.list()
 */

import { apiGet, apiPost } from './http-client'

export interface UserAdapterStats {
  total_users: number
  total_size_bytes: number
  total_size_mb: number
  adapter_rank: number
  model_dim: number
  avg_size_per_user_kb: number
  auto_management?: {
    aggregate_threshold: number
    prune_threshold: number
    min_feedback_for_aggregation: number
    quality_adapters_count: number
  }
}

export interface UserAdapterInfo {
  user_id: string
  rank: number
  alpha: number
  model_dim: number
  created_at: string
  updated_at: string
  feedback_count: number
}

export const userAdaptersController = {
  async list(): Promise<UserAdapterStats> {
    return apiGet('/user-adapters')
  },

  async get(userId: string): Promise<UserAdapterInfo> {
    return apiGet(`/user-adapters/${encodeURIComponent(userId)}`)
  },

  async getQuality(minFeedbackCount = 3, maxAgeDays?: number): Promise<{ count: number; adapters: UserAdapterInfo[] }> {
    let url = `/user-adapters/quality?min_feedback_count=${minFeedbackCount}`
    if (maxAgeDays) url += `&max_age_days=${maxAgeDays}`
    return apiGet(url)
  },

  async aggregateBest(params: { top_k?: number; min_feedback_count?: number; output_name?: string } = {}): Promise<{
    status: string
    user_count?: number
    output_path?: string
    eval?: {
      baseline?: Record<string, number>
      with_adapter?: Record<string, number>
      perplexity_delta?: number
      bleu_delta?: number
      throughput_delta?: number
      verdict?: string
      report?: string
    }
  }> {
    return apiPost('/user-adapters/aggregate-best', {
      top_k: params.top_k ?? 10,
      min_feedback_count: params.min_feedback_count ?? 5,
      output_name: params.output_name ?? 'best_aggregated',
    })
  },

  async prune(params: { min_feedback_count?: number; max_age_days?: number } = {}): Promise<{
    status: string
    deleted_count: number
    deleted_users: string[]
  }> {
    return apiPost('/user-adapters/prune', {
      min_feedback_count: params.min_feedback_count ?? 1,
      max_age_days: params.max_age_days ?? 30,
    })
  },

  async reset(userId: string): Promise<{ status: string; user_id: string; feedback_count: number }> {
    return apiPost(`/user-adapters/${encodeURIComponent(userId)}/reset`)
  },
}
