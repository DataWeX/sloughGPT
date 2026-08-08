/**
 * Registry Controller — API for model registry.
 */

import { apiGet } from './http-client'

export interface RegisteredModel {
  model_id: string
  status: string
  metrics?: Record<string, unknown>
  registered_at?: string
}

export interface RegistryStats {
  total_models: number
  loaded_models: number
  failed_models: number
  circuit_breaker_open: boolean
}

export interface RegistryData {
  models: RegisteredModel[]
  stats: RegistryStats | null
  bestModel: Record<string, unknown> | null
}

class RegistryController {
  async list(): Promise<RegisteredModel[]> {
    const data = await apiGet<{ data?: { models?: RegisteredModel[] }; models?: RegisteredModel[] } | RegisteredModel[]>('/registry/models')
    if (Array.isArray(data)) return data
    return data?.data?.models ?? data?.models ?? []
  }

  async stats(): Promise<RegistryStats | null> {
    const data = await apiGet<{ data?: RegistryStats } | RegistryStats>('/registry/stats')
    if (!data) return null
    return (data as { data?: RegistryStats }).data ?? data as RegistryStats
  }

  async best(): Promise<Record<string, unknown> | null> {
    const data = await apiGet<{ data?: Record<string, unknown> } | Record<string, unknown>>('/registry/best')
    if (!data) return null
    return (data as { data?: Record<string, unknown> }).data ?? data as Record<string, unknown>
  }
}

export const registryController = new RegistryController()
