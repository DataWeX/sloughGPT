'use client'

import React, {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  useCallback,
  type ReactNode,
} from 'react'
import { modelController } from '@/lib/model-controller'
import type { HealthStatus } from '@/lib/model-controller'
import type { ModelInfo } from '@/lib/types'
import { useLiveStatus } from '@/hooks/useLiveStatus'
import { useBootstrap } from '@/contexts/BootstrapContext'
import { logger, trackEvent } from '@/lib/dev-log'
import { extractErrorMessage } from '@/lib/error-utils'

const _log = logger.child('model-context')

export type { ModelInfo } from '@/lib/types'

export interface ModelContextValue {
  models: ModelInfo[]
  currentModel: string | null
  isModelLoaded: boolean
  loading: boolean
  loadingModelId: string | null
  error: string | null

  loadModel: (
    modelId: string,
    opts?: { mode?: string; device?: string },
  ) => Promise<{ success: boolean; error?: string }>
  loadModelPath: (path: string) => Promise<{ success: boolean; error?: string }>
  unloadModel: (modelId: string) => Promise<{ success: boolean; error?: string }>
  refreshModels: () => Promise<void>
  clearError: () => void
}

const ModelContext = createContext<ModelContextValue | null>(null)

export function ModelProvider({ children }: { children: ReactNode }) {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [currentModel, setCurrentModel] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadingModelId, setLoadingModelId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthStatus | null>(null)

  // Subscribe to live health SSE for instant model status updates
  const { health: liveHealth, connectionStatus } = useLiveStatus()

  const refreshHealth = useCallback(async () => {
    try {
      const h = await modelController.getHealth()
      setHealth(h)
    } catch {
      setHealth(null)
    }
  }, [])

  // Sync live health into local health state for immediate UI updates
  useEffect(() => {
    if (liveHealth && connectionStatus === 'connected') {
      setHealth({
        status: liveHealth.health_status || 'healthy',
        model_loaded: liveHealth.model_loaded,
        model_type: liveHealth.model_type || '',
        summary: liveHealth.health_summary,
        inference_count: liveHealth.inference_count,
        is_inferencing: liveHealth.is_inferencing,
      })
    }
  }, [liveHealth, connectionStatus])

  const isModelLoaded = health !== null && health.model_loaded

  const refreshModels = useCallback(async () => {
    try {
      const fetchedModels = await modelController.list()
      setModels(
        fetchedModels.map((m) => ({
          id: m.id,
          name: m.name,
          type: m.type || 'huggingface',
          loaded: m.loaded ?? false,
          size_mb: m.size_mb,
          size_gb: m.size_gb,
          params: m.params,
          description: m.description,
          tags: m.tags,
        })),
      )
    } catch (err) {
      _log.error('Could not fetch models', { exception: String(err) })
    }
  }, [])

  // Initial model list rides in the /bootstrap payload (one request per
  // app load) — no dedicated mount fetch. Actions (load/unload) still
  // refresh through the controller for post-mutation freshness.
  const { data: bootstrap } = useBootstrap()
  const seededRef = useRef(false)
  useEffect(() => {
    if (seededRef.current) return
    const list = bootstrap?.members.models
    if (!list) return
    seededRef.current = true
    setModels(
      list.map((m) => ({
        id: m.id,
        name: m.name,
        type: m.type || 'huggingface',
        loaded: m.loaded ?? false,
        size_mb: m.size_mb,
        size_gb: m.size_gb,
        params: m.params,
        description: m.description,
        tags: m.tags,
      })),
    )
  }, [bootstrap])

  useEffect(() => {
    if (health && health.model_loaded && health.model_type) {
      setCurrentModel(health.model_type)
    }
  }, [health])

  const loadModel = useCallback(
    async (
      modelId: string,
      opts?: { mode?: string; device?: string },
    ): Promise<{ success: boolean; error?: string }> => {
      setLoading(true)
      setLoadingModelId(modelId)
      setError(null)

      try {
        const data = await modelController.load(modelId, opts?.device)
        if (data && (data.status === 'error' || data.error)) {
          const error = String(data.error ?? data.status)
          setError(error)
          return { success: false, error }
        }
        await refreshModels()
        await refreshHealth()
        trackEvent('model_loaded', { model_id: modelId })
        return { success: true }
      } catch (err) {
        const error = extractErrorMessage(err)
        setError(error)
        return { success: false, error }
      } finally {
        setLoading(false)
        setLoadingModelId(null)
      }
    },
    [refreshModels, refreshHealth],
  )

  const loadModelPath = useCallback(
    async (path: string): Promise<{ success: boolean; error?: string }> => {
      setLoading(true)
      setLoadingModelId(path)
      setError(null)

      try {
        const data = await modelController.loadModelPath(path)
        if (data && data.error) {
          const error = String(data.error)
          setError(error)
          return { success: false, error }
        }

        await refreshModels()
        await refreshHealth()
        return { success: true }
      } catch (err) {
        const error = extractErrorMessage(err)
        setError(error)
        return { success: false, error }
      } finally {
        setLoading(false)
        setLoadingModelId(null)
      }
    },
    [refreshModels, refreshHealth],
  )

  const unloadModel = useCallback(
    async (modelId: string): Promise<{ success: boolean; error?: string }> => {
      setLoading(true)
      setLoadingModelId(modelId)
      setError(null)

      try {
        await modelController.unloadModel()

        await refreshModels()
        await refreshHealth()
        trackEvent('model_unloaded', { model_id: modelId })
        return { success: true }
      } catch (err) {
        const error = extractErrorMessage(err)
        setError(error)
        return { success: false, error }
      } finally {
        setLoading(false)
        setLoadingModelId(null)
      }
    },
    [refreshModels, refreshHealth],
  )

  const clearError = useCallback(() => {
    setError(null)
  }, [])

  const value: ModelContextValue = {
    models,
    currentModel,
    isModelLoaded,
    loading,
    loadingModelId,
    error,
    loadModel,
    loadModelPath,
    unloadModel,
    refreshModels,
    clearError,
  }

  return <ModelContext.Provider value={value}>{children}</ModelContext.Provider>
}

export function useModels(): ModelContextValue {
  const context = useContext(ModelContext)
  if (!context) {
    throw new Error('useModels must be used within a ModelProvider')
  }
  return context
}

export function useCurrentModel(): { modelId: string | null; isLoaded: boolean } {
  const { currentModel, isModelLoaded } = useModels()
  return {
    modelId: currentModel,
    isLoaded: isModelLoaded,
  }
}

export function useModelById(modelId: string): ModelInfo | undefined {
  const { models } = useModels()
  return models.find((m) => m.id === modelId)
}

export function useLocalModels(): ModelInfo[] {
  const { models } = useModels()
  return models.filter((m) => m.type === 'local')
}

export function useHuggingFaceModels(): ModelInfo[] {
  const { models } = useModels()
  return models.filter((m) => m.type === 'huggingface')
}
