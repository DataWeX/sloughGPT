'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { modelController } from '@/lib/model-controller'
import { generationConfigController } from '@/lib/generation-config-controller'
import { soulsController, type Soul } from '@/lib/souls-controller'
import { trainingJobsController, type FineTunedModel } from '@/lib/training-controller'
import { startDownload, getDownloadStatus } from '@/lib/download-controller'
import { sessionStore } from '@/lib/session-store'
import { useSettings, useUpdateSettings } from '@/lib/store'
import type { DownloadProgressInfo } from '@/lib/chat-utils'
import { logger } from '@/lib/dev-log'
import { formatToastError } from '@/lib/error-utils'
import { DEFAULT_ERROR_MESSAGE } from '@/lib/format-bytes'

const _log = logger.child('chat-model-settings')

interface LearnerInfo {
  total_tokens_ingested: number
  train_steps_completed: number
  current_loss: number | undefined
  loss_history?: Array<{ step: number; loss: number; tokens: number; timestamp: number }>
  n_embed?: number
  n_layer?: number
  n_head?: number
  arch?: string
}

export function useChatModelSettings(
  showToast: (message: string, type?: 'success' | 'error' | 'info') => void,
  refreshHealth: () => Promise<void>,
) {
  const storeSettings = useSettings()
  const updateStoreSettings = useUpdateSettings()

  const [model, setModel] = useState('')
  const [souls, setSouls] = useState<Soul[]>([])
  const [temperature, setTemperature] = useState(storeSettings.defaultTemp)
  const [maxTokens, setMaxTokens] = useState(storeSettings.defaultMaxTokens)
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [modelInfoMap, setModelInfoMap] = useState<Record<string, { cached?: boolean; size_gb?: number }>>({})
  const [downloadProgress, setDownloadProgress] = useState<Record<string, DownloadProgressInfo>>({})
  const [currentSoul, setCurrentSoul] = useState<Soul | null>(null)
  const [currentCheckpoint, setCurrentCheckpoint] = useState<string | undefined>(undefined)
  const [checkpoints, setCheckpoints] = useState<Array<{
    name: string; loss?: number; traits?: string[]; is_loaded?: boolean; eval_verdict?: string
  }>>([])
  const [loadingModel, setLoadingModel] = useState<string | null>(null)
  const [pendingDownload, setPendingDownload] = useState<string | null>(null)
  const [learnerInfo, setLearnerInfo] = useState<LearnerInfo | null>(null)
  const [learnerTraining, setLearnerTraining] = useState(false)
  const [fineTuned, setFineTuned] = useState<FineTunedModel[]>([])
  const [fineTunedLoading, setFineTunedLoading] = useState(false)

  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const startDownloadFlowRef = useRef<(m: string, sizeGb?: number) => Promise<void>>(async () => {})

  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
    }
  }, [])

  const startDownloadFlow = useCallback(async (m: string, sizeGb?: number) => {
    startDownloadFlowRef.current = startDownloadFlow
    setLoadingModel(m)
    const showToastFn = showToast
    showToastFn(`Downloading ${m}...`, 'info')
    try {
      await startDownload(m, sizeGb ? Math.round(sizeGb * 1024 * 1024 * 1024) : 0)
      const pi = setInterval(async () => {
        try {
          const status = await getDownloadStatus(m)
          setDownloadProgress(prev => ({ ...prev, [m]: {
            percentage: status.percentage, status: status.status,
            speed_mb_per_sec: status.speed_mb_per_sec, eta_seconds: status.eta_seconds,
            bytes_downloaded: status.bytes_downloaded, total_bytes: status.total_bytes,
            current_file: status.current_file, files_completed: status.files_completed, files_total: status.files_total,
          } }))
          if (status.status === 'complete' || status.status === 'already_cached') {
            clearInterval(pi)
            pollIntervalRef.current = null
            setDownloadProgress(prev => { const r = { ...prev }; delete r[m]; return r })
            const result = await modelController.load(m)
            if (result && (result.status === 'error' || result.error)) {
              setDownloadProgress(prev => { const r = { ...prev }; delete r[m]; return r })
              showToastFn(`Model load failed: ${result.error || 'unknown reason'}`, 'error')
              setLoadingModel(null)
              return
            }
            await refreshHealth()
            setModel(m)
            showToastFn(`Model ready: ${m}`, 'success')
            setLoadingModel(null)
          } else if (status.status === 'failed' || status.status === 'cancelled') {
            clearInterval(pi)
            pollIntervalRef.current = null
            setDownloadProgress(prev => { const r = { ...prev }; delete r[m]; return r })
            showToastFn(`Download failed: ${status.error || 'unknown reason'}`, 'error')
            setLoadingModel(null)
          }
        } catch { /* poll retry */ }
      }, 2000)
      pollIntervalRef.current = pi
    } catch (err) {
      showToastFn(formatToastError(err, DEFAULT_ERROR_MESSAGE), 'error')
      setLoadingModel(null)
    }
  }, [showToast, refreshHealth])

  const handleSelectModel = useCallback(async (m: string) => {
    if (m === model || loadingModel) return
    const info = modelInfoMap[m]
    if (info?.cached) {
      setLoadingModel(m)
      showToast(`Loading ${m}...`, 'info')
      try {
        const result = await modelController.load(m)
        if (result && (result.status === 'error' || result.error)) {
          showToast(`Could not load ${m}: ${result.error || 'unknown error'}`, 'error')
          setLoadingModel(null)
          return
        }
        await refreshHealth()
        setModel(m)
        showToast(`Model ready: ${m} (${result.device || 'cpu'})`, 'success')
      } catch (err) {
        showToast(formatToastError(err, DEFAULT_ERROR_MESSAGE), 'error')
      } finally {
        setLoadingModel(null)
      }
    } else if (sessionStore.isApproved(m)) {
      setLoadingModel(m)
      await startDownloadFlowRef.current(m, info?.size_gb)
    } else {
      setPendingDownload(m)
    }
  }, [model, loadingModel, modelInfoMap, showToast, refreshHealth])

  const handleSelectSoul = useCallback((s: Soul) => {
    setCurrentSoul(s)
    soulsController.switch(s.name).catch(e => _log.error('Failed to switch soul', { exception: String(e) }))
  }, [])

  const handleUnloadModel = useCallback(async () => {
    if (!model) return
    setLoadingModel(model)
    try {
      await modelController.unloadModel()
      await refreshHealth()
      setModel('')
      showToast('Model stopped', 'info')
    } catch (err) {
      showToast(formatToastError(err, DEFAULT_ERROR_MESSAGE), 'error')
    } finally {
      setLoadingModel(null)
    }
  }, [model, showToast, refreshHealth])

  const fetchFineTuned = useCallback(async () => {
    setFineTunedLoading(true)
    try {
      const models = await trainingJobsController.listFineTuned()
      setFineTuned(models)
    } catch {
      setFineTuned([])
    } finally {
      setFineTunedLoading(false)
    }
  }, [])

  const handleLoadFineTuned = useCallback(async (name: string) => {
    setLoadingModel(name)
    try {
      const result = await trainingJobsController.loadFineTuned(name)
      const loaded = await trainingJobsController.listFineTuned()
      setFineTuned(loaded)
      setModel(result.model_id || name)
      await refreshHealth()
      showToast(`Fine-tuned model loaded: ${name}`, 'success')
    } catch (err) {
      showToast(formatToastError(err, DEFAULT_ERROR_MESSAGE), 'error')
    } finally {
      setLoadingModel(null)
    }
  }, [showToast, refreshHealth])

  const fetchInitialData = useCallback(async (healthModel?: string) => {
    try {
      const [models, genConfig, soulsData, checkpointsData, fineTunedModels] = await Promise.all([
        modelController.list(),
        generationConfigController.get(),
        soulsController.list(),
        soulsController.listCheckpoints(),
        trainingJobsController.listFineTuned(),
      ])

      const fineTunedNames = new Set(fineTunedModels.map(ft => ft.name))
      setFineTuned(fineTunedModels)
      setAvailableModels(models.map(m => m.id).filter(id => !fineTunedNames.has(id)))
      const infoMap: Record<string, { cached?: boolean; size_gb?: number }> = {}
      models.forEach(m => { infoMap[m.id] = { cached: m.cached, size_gb: m.size_gb } })
      setModelInfoMap(infoMap)

      setTemperature(genConfig.temperature)
      setMaxTokens(genConfig.max_new_tokens)
      updateStoreSettings({ defaultTemp: genConfig.temperature, defaultMaxTokens: genConfig.max_new_tokens })

      setSouls(soulsData.souls || [])
      if (soulsData.current_soul) {
        const found = (soulsData.souls || []).find(s => s.name === soulsData.current_soul)
        if (found) setCurrentSoul(found)
      }

      setCheckpoints((checkpointsData.checkpoints || []).map(c => ({
        name: c.name || 'unknown',
        loss: c.loss,
        traits: c.traits ? Object.keys(c.traits) : undefined,
        is_loaded: c.is_loaded || false,
        eval_verdict: c.verdict,
      })))

      if (healthModel) setModel(healthModel)
    } catch (err) {
      _log.error('Failed to fetch initial model data', { exception: String(err) })
    }
  }, [setAvailableModels, setModelInfoMap, setTemperature, setMaxTokens, setSouls, setCurrentSoul, setCheckpoints, setModel, updateStoreSettings])

  const handleSetTemperature = useCallback((v: number) => {
    setTemperature(v)
    updateStoreSettings({ defaultTemp: v })
  }, [updateStoreSettings])

  const handleSetMaxTokens = useCallback((v: number) => {
    setMaxTokens(v)
    updateStoreSettings({ defaultMaxTokens: v })
  }, [updateStoreSettings])

  return {
    model, setModel,
    souls, setSouls,
    temperature, setTemperature: handleSetTemperature,
    maxTokens, setMaxTokens: handleSetMaxTokens,
    availableModels, setAvailableModels,
    modelInfoMap, setModelInfoMap,
    downloadProgress, setDownloadProgress,
    currentSoul, setCurrentSoul,
    currentCheckpoint, setCurrentCheckpoint,
    checkpoints, setCheckpoints,
    loadingModel, setLoadingModel,
    pendingDownload, setPendingDownload,
    learnerInfo, setLearnerInfo,
    learnerTraining, setLearnerTraining,
    fineTuned, fineTunedLoading, fetchFineTuned, handleLoadFineTuned,
    pollIntervalRef,
    startDownloadFlowRef,
    startDownloadFlow,
    handleSelectModel,
    handleSelectSoul,
    handleUnloadModel,
    fetchInitialData,
  }
}
