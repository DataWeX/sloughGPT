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
    const data = await apiGet<{ models?: RegisteredModel[] } | RegisteredModel[]>('/registry/models')
    if (Array.isArray(data)) return data
    return data?.models ?? []
  }

  async stats(): Promise<RegistryStats | null> {
    const data = await apiGet<{
      models_loaded?: number
      models_registered?: number
      healthy?: boolean
      degraded?: boolean
      has_errors?: boolean
      default_model?: string | null
    }>('/registry/stats')
    if (!data) return null
    return {
      total_models: data.models_registered ?? 0,
      loaded_models: data.models_loaded ?? 0,
      failed_models: data.has_errors ? 1 : 0,
      circuit_breaker_open: data.degraded ?? false,
    }
  }

  async best(): Promise<Record<string, unknown> | null> {
    const data = await apiGet<{ default_model?: string | null; models_loaded?: number }>('/registry/best')
    if (!data) return null
    return data as Record<string, unknown>
  }
}

export const registryController = new RegistryController()
