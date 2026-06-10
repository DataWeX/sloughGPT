import { create } from 'zustand'
import { apiGet, apiPost } from '@/lib/api-client'

export interface ModelInfo {
  id: string
  name: string
  type: string
  loaded: boolean
  sizeMb?: number
  sizeGb?: number
  params?: string
  description?: string
  tags?: string[]
  thumbnail?: string
  source?: string
}

export interface SoulInfo {
  name: string
  description: string
  traits: string[]
}

export interface CheckpointInfo {
  name: string
  soul?: string
  loss?: number
  steps?: number
  traits?: Record<string, number>
  created_at?: string
}

export interface HealthStatus {
  status: string
  model_loaded: boolean
  model_type?: string
  is_inferencing: boolean
  inference_count: number
}

interface ModelState {
  models: ModelInfo[]
  currentModel: string | null
  souls: SoulInfo[]
  currentSoul: string | null
  checkpoints: CheckpointInfo[]
  currentCheckpoint: string | null
  health: HealthStatus | null
  loading: boolean
  loadingModelId: string | null
  error: string | null

  refresh: () => Promise<void>
  loadModel: (modelId: string) => Promise<boolean>
  unloadModel: () => Promise<boolean>
  switchSoul: (name: string, checkpointName?: string) => Promise<boolean>
  clearError: () => void
}

export const useModelStore = create<ModelState>()((set, get) => ({
  models: [],
  currentModel: null,
  souls: [],
  currentSoul: null,
  checkpoints: [],
  currentCheckpoint: null,
  health: null,
  loading: false,
  loadingModelId: null,
  error: null,

  refresh: async () => {
    set({ loading: true, error: null })
    try {
      const [models, souls, currentSoul, checkpoints, health] = await Promise.all([
        apiGet<ModelInfo[]>('/models').catch(() => [] as ModelInfo[]),
        apiGet<SoulInfo[]>('/souls').catch(() => [] as SoulInfo[]),
        apiGet<{ name: string }>('/souls/current').catch(() => ({ name: '' })),
        apiGet<CheckpointInfo[]>('/auto-train/checkpoints').catch(() => [] as CheckpointInfo[]),
        apiGet<HealthStatus>('/health').catch(() => null),
      ])

      const loaded = models.find((m) => m.loaded)
      set({
        models,
        souls,
        currentSoul: currentSoul?.name || null,
        checkpoints,
        health,
        currentModel: loaded?.id || health?.model_type || null,
        loading: false,
      })
    } catch (error) {
      set({ error: (error as Error).message, loading: false })
    }
  },

  loadModel: async (modelId: string) => {
    set({ loadingModelId: modelId, error: null })
    try {
      await apiPost('/models/load', { model_id: modelId })
      await get().refresh()
      set({ loadingModelId: null })
      return true
    } catch (error) {
      set({ error: (error as Error).message, loadingModelId: null })
      return false
    }
  },

  unloadModel: async () => {
    try {
      await apiPost('/models/unload')
      await get().refresh()
      return true
    } catch (error) {
      set({ error: (error as Error).message })
      return false
    }
  },

  switchSoul: async (name: string, checkpointName?: string) => {
    try {
      const body: Record<string, string> = { soul: name }
      if (checkpointName) body.checkpoint_name = checkpointName
      await apiPost('/souls/switch', body)
      await get().refresh()
      return true
    } catch (error) {
      set({ error: (error as Error).message })
      return false
    }
  },

  clearError: () => set({ error: null }),
}))
