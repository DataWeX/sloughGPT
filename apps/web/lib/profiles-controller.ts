/**
 * Profiles controller — API client for serving profiles.
 */
import { apiGet, apiPost } from '@/lib/http-client'

export interface ServingProfile {
  id: string
  name: string
  description: string
  tier: string
  device: string
  quantize: boolean
  quant_bits: number
  inference_pool_size: number
  generate_timeout: number
  enable_guard: boolean
  lazy_autoload: boolean
  memory_limit_mb: number
  workload_mode: string
  compute_threads: number
  io_threads: number
  memory_pressure_warning: number
  memory_pressure_critical: number
  temperature: number
  top_p: number
  top_k: number
  repetition_penalty: number
  max_new_tokens: number
  tags: string[]
  min_ram_gb: number
  recommended_model: string
}

export interface ProfileApplyResult {
  profile: ServingProfile
  live_settings: Record<string, unknown>
  requires_restart: string[]
  active_profile_id: string
}

export interface ProfileRecommendResult {
  recommended_profile_id: string
  detected_ram_gb: number
  has_gpu: boolean
}

export interface ActiveProfileResult {
  active_profile_id: string
  profile: ServingProfile | null
}

export const profilesController = {
  async list(): Promise<ServingProfile[]> {
    const res = await apiGet<{ data: ServingProfile[] }>('/profiles')
    return res.data ?? []
  },

  async get(id: string): Promise<ServingProfile> {
    const res = await apiGet<{ data: ServingProfile }>(`/profiles/${id}`)
    return res.data
  },

  async apply(profileId: string): Promise<ProfileApplyResult> {
    const res = await apiPost<{ data: ProfileApplyResult }>('/profiles/apply', { profile_id: profileId })
    return res.data
  },

  async active(): Promise<ActiveProfileResult> {
    const res = await apiGet<{ data: ActiveProfileResult }>('/profiles/active')
    return res.data
  },

  async recommend(): Promise<ProfileRecommendResult> {
    const res = await apiGet<{ data: ProfileRecommendResult }>('/profiles/recommend')
    return res.data
  },
}
