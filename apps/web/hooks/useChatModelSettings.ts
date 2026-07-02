'use client'

import { useState, useCallback, useRef } from 'react'
import { modelController } from '@/lib/model-controller'
import { generationConfigController } from '@/lib/generation-config-controller'
import { soulsController, type Soul } from '@/lib/souls-controller'
import { startDownload, getDownloadStatus } from '@/lib/download-controller'
import { sessionStore } from '@/lib/session-store'
import type { DownloadProgressInfo } from '@/lib/chat-utils'

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
  const [model, setModel] = useState('')
  const [souls, setSouls] = useState<Soul[]>([])
  const [temperature, setTemperature] = useState(0.8)
  const [maxTokens, setMaxTokens] = useState(200)
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

  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const startDownloadFlowRef = useRef<(m: string, sizeGb?: number) => Promise<void>>(async () => {})

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
            await modelController.load(m)
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
      showToastFn(`Something went wrong: ${err instanceof Error ? err.message : 'unknown error'}`, 'error')
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
        await refreshHealth()
        setModel(m)
        showToast(`Model ready: ${m} (${result.device || 'cpu'})`, 'success')
      } catch (err) {
        showToast(`Something went wrong: ${err instanceof Error ? err.message : 'unknown error'}`, 'error')
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
    soulsController.switch(s.name).catch(e => console.error('Failed to switch soul:', e))
  }, [])

  const handleUnloadModel = useCallback(async () => {
    if (!model) return
    setLoadingModel(model)
    try {
      await modelController.unloadModel(model)
      await refreshHealth()
      setModel('')
      showToast('Model stopped', 'info')
    } catch (err) {
      showToast(`Something went wrong: ${err instanceof Error ? err.message : 'unknown error'}`, 'error')
    } finally {
      setLoadingModel(null)
    }
  }, [model, showToast, refreshHealth])

  const fetchInitialData = useCallback(async (healthModel?: string) => {
    try {
      const [models, genConfig, soulsData, checkpointsData] = await Promise.all([
        modelController.list(),
        generationConfigController.get(),
        soulsController.list(),
        soulsController.listCheckpoints(),
      ])

      setAvailableModels(models.map(m => m.id))
      const infoMap: Record<string, { cached?: boolean; size_gb?: number }> = {}
      models.forEach(m => { infoMap[m.id] = { cached: m.cached, size_gb: m.size_gb } })
      setModelInfoMap(infoMap)

      setTemperature(genConfig.temperature)
      setMaxTokens(genConfig.max_new_tokens)

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
      console.error('Failed to fetch initial model data:', err)
    }
  }, [setAvailableModels, setModelInfoMap, setTemperature, setMaxTokens, setSouls, setCurrentSoul, setCheckpoints, setModel])

  return {
    model, setModel,
    souls, setSouls,
    temperature, setTemperature,
    maxTokens, setMaxTokens,
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
    pollIntervalRef,
    startDownloadFlowRef,
    startDownloadFlow,
    handleSelectModel,
    handleSelectSoul,
    handleUnloadModel,
    fetchInitialData,
  }
}
