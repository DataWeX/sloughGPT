'use client'

import React, { createContext, useContext, useEffect, useState, useCallback, type ReactNode } from 'react'
import { modelController } from '@/lib/model-controller'
import type { HealthStatus } from '@/lib/model-controller'
import { useLiveStatus } from '@/hooks/useLiveStatus'

export interface ModelInfo {
  id: string
  name: string
  type: string
  loaded: boolean
  sizeMb?: number
  params?: string
  description?: string
  tags?: string[]
}

export interface ModelContextValue {
  models: ModelInfo[]
  currentModel: string | null
  isModelLoaded: boolean
  loading: boolean
  loadingModelId: string | null
  error: string | null

  loadModel: (modelId: string, opts?: { mode?: string; device?: string }) => Promise<{ success: boolean; error?: string }>
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
    } catch { setHealth(null) }
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
      setModels(fetchedModels.map(m => ({
        id: m.id,
        name: m.name,
        type: m.type || 'huggingface',
        loaded: m.loaded ?? false,
        sizeMb: m.size_mb,
        params: m.params,
        description: m.description,
        tags: m.tags,
      })))
    } catch (err) {
      console.error('Failed to fetch models:', err)
    }
  }, [])

  useEffect(() => {
    void refreshModels()
  }, [refreshModels])

  useEffect(() => {
    if (health && health.model_loaded && health.model_type) {
      setCurrentModel(health.model_type)
    }
  }, [health])

  const loadModel = useCallback(async (
    modelId: string,
    opts?: { mode?: string; device?: string }
  ): Promise<{ success: boolean; error?: string }> => {
    setLoading(true)
    setLoadingModelId(modelId)
    setError(null)

    try {
      await modelController.load(modelId, opts?.device)
      await refreshModels()
      await refreshHealth()
      return { success: true }
    } catch (err) {
      const error = err instanceof Error ? err.message : 'Unknown error'
      setError(error)
      return { success: false, error }
    } finally {
      setLoading(false)
      setLoadingModelId(null)
    }
  }, [refreshModels, refreshHealth])

  const loadModelPath = useCallback(async (path: string): Promise<{ success: boolean; error?: string }> => {
    setLoading(true)
    setLoadingModelId(path)
    setError(null)

    try {
      const data = await modelController.loadModelPath(path)
      if (data && (data as any).error) {
        const error = String((data as any).error)
        setError(error)
        return { success: false, error }
      }

      await refreshModels()
      await refreshHealth()
      return { success: true }
    } catch (err) {
      const error = err instanceof Error ? err.message : 'Unknown error'
      setError(error)
      return { success: false, error }
    } finally {
      setLoading(false)
      setLoadingModelId(null)
    }
  }, [refreshModels, refreshHealth])

  const unloadModel = useCallback(async (modelId: string): Promise<{ success: boolean; error?: string }> => {
    setLoading(true)
    setLoadingModelId(modelId)
    setError(null)

    try {
      await modelController.unloadModel(modelId)

      await refreshModels()
      await refreshHealth()
      return { success: true }
    } catch (err) {
      const error = err instanceof Error ? err.message : 'Unknown error'
      setError(error)
      return { success: false, error }
    } finally {
      setLoading(false)
      setLoadingModelId(null)
    }
  }, [refreshModels, refreshHealth])

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

  return (
    <ModelContext.Provider value={value}>
      {children}
    </ModelContext.Provider>
  )
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
  return models.find(m => m.id === modelId)
}

export function useLocalModels(): ModelInfo[] {
  const { models } = useModels()
  return models.filter(m => m.type === 'local')
}

export function useHuggingFaceModels(): ModelInfo[] {
  const { models } = useModels()
  return models.filter(m => m.type === 'huggingface')
}
