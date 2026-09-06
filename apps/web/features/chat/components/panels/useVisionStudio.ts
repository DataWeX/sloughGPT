'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { logger } from '@/lib/dev-log'

interface AnalyzeResult {
  caption: string
  confidence: number
  tags: string[]
  accuracy: number
  supervised: boolean
  images_learned: number
  trained: boolean
  replay_buffer_size: number
  mean_accuracy: number
}

interface TrainingReport {
  images_learned: number
  vocab_size: number
  caption_history: string[]
  accuracy_history: number[]
  mean_accuracy: number
  last_accuracy: number
}

export interface UseVisionStudioReturn {
  tab: string
  setTab: (t: string) => void
  analyzeLoading: boolean
  analyzeResult: AnalyzeResult | null
  analyzeError: string | null
  previewUrl: string | null
  previewFileName: string | null
  trainLabel: string
  setTrainLabel: (v: string) => void
  trainLoading: boolean
  trainResult: { caption: string; accuracy: number } | null
  genPrompt: string
  setGenPrompt: (v: string) => void
  genLoading: boolean
  genResult: string | null
  genError: string | null
  setGenResult: (v: string | null) => void
  trainingReport: TrainingReport | null
  resetLoading: boolean
  dragOver: boolean
  retryLoading: boolean
  fileInputRef: React.RefObject<HTMLInputElement | null>
  dropRef: React.RefObject<HTMLDivElement | null>
  refreshReport: () => Promise<void>
  processFile: (file: File) => Promise<void>
  retryAnalyze: () => Promise<void>
  handleDrop: (e: React.DragEvent) => void
  handleFileSelect: (e: React.ChangeEvent<HTMLInputElement>) => void
  handleTrainWithLabel: () => Promise<void>
  handleGenerateImage: () => Promise<void>
  handleSendGeneratedImage: () => void
  handleReset: () => Promise<void>
  clearPreview: () => void
  setDragOver: (v: boolean) => void
}

export function useVisionStudio(
  sessionId: string | null,
  onGeneratedImage: (dataUrl: string, prompt: string) => void,
  onSendText: (text: string) => void,
  initialCaps?: {
    images_learned?: number
    trained?: boolean
    status?: string
    vocab_size?: number
    mean_accuracy?: number
  }
): UseVisionStudioReturn {
  const [tab, setTab] = useState('analyze')
  const [analyzeLoading, setAnalyzeLoading] = useState(false)
  const [analyzeResult, setAnalyzeResult] = useState<AnalyzeResult | null>(null)
  const [analyzeError, setAnalyzeError] = useState<string | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [previewFileName, setPreviewFileName] = useState<string | null>(null)
  const [trainLabel, setTrainLabel] = useState('')
  const [trainLoading, setTrainLoading] = useState(false)
  const [trainResult, setTrainResult] = useState<{ caption: string; accuracy: number } | null>(null)
  const [genPrompt, setGenPrompt] = useState('')
  const [genLoading, setGenLoading] = useState(false)
  const [genResult, setGenResult] = useState<string | null>(null)
  const [genError, setGenError] = useState<string | null>(null)
  const [trainingReport, setTrainingReport] = useState<TrainingReport | null>(() => {
    if (initialCaps && (initialCaps.images_learned ?? 0) > 0) {
      return {
        images_learned: initialCaps.images_learned ?? 0,
        vocab_size: initialCaps.vocab_size ?? 0,
        caption_history: [],
        accuracy_history: [],
        mean_accuracy: initialCaps.mean_accuracy ?? 0,
        last_accuracy: initialCaps.mean_accuracy ?? 0,
      }
    }
    return null
  })
  const [resetLoading, setResetLoading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [retryLoading, setRetryLoading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const dropRef = useRef<HTMLDivElement>(null)

  const refreshReport = useCallback(async () => {
    try {
      const { multimodalController } = await import('@/lib/multimodal-controller')
      const report = await multimodalController.getTrainingReport()
      setTrainingReport(report)
    } catch {
      // ignore
    }
  }, [])

  useEffect(() => {
    let active = true
    const load = async () => {
      try {
        const { multimodalController } = await import('@/lib/multimodal-controller')
        const report = await multimodalController.getTrainingReport()
        if (active) setTrainingReport(report)
      } catch {
        // ignore
      }
    }
    void load()
    return () => { active = false }
  }, [])

  const processFile = useCallback(async (file: File) => {
    setAnalyzeLoading(true)
    setAnalyzeError(null)
    setAnalyzeResult(null)
    const reader = new FileReader()
    reader.onload = async () => {
      setPreviewUrl(reader.result as string)
      setPreviewFileName(file.name)
      try {
        const { multimodalController } = await import('@/lib/multimodal-controller')
        const result = await multimodalController.analyzeImage(file)
        setAnalyzeResult(result)
      } catch (err) {
        setAnalyzeError(err instanceof Error ? err.message : 'Analysis failed')
      } finally {
        setAnalyzeLoading(false)
      }
    }
    reader.readAsDataURL(file)
  }, [])

  const retryAnalyze = useCallback(async () => {
    if (!previewUrl) return
    setRetryLoading(true)
    try {
      const blobRes = await fetch(previewUrl)
      const blob = await blobRes.blob()
      const file = new File([blob], previewFileName || 'retry.png', { type: blob.type || 'image/png' })
      const { multimodalController } = await import('@/lib/multimodal-controller')
      const result = await multimodalController.analyzeImage(file)
      setAnalyzeResult(result)
    } catch (err) {
      setAnalyzeError(err instanceof Error ? err.message : 'Retry failed')
    } finally {
      setRetryLoading(false)
    }
  }, [previewUrl, previewFileName])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith('image/')) processFile(file)
  }, [processFile])

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) processFile(file)
    e.target.value = ''
  }, [processFile])

  const handleTrainWithLabel = useCallback(async () => {
    if (!previewUrl || !trainLabel.trim()) return
    setTrainLoading(true)
    try {
      const blobRes = await fetch(previewUrl)
      const blob = await blobRes.blob()
      const file = new File([blob], previewFileName || 'train.png', { type: blob.type || 'image/png' })
      const { multimodalController } = await import('@/lib/multimodal-controller')
      const dataUrl = await new Promise<string>((resolve) => {
        const reader = new FileReader()
        reader.onload = () => resolve(reader.result as string)
        reader.readAsDataURL(file)
      })
      const result = await multimodalController.trainImage(dataUrl, previewFileName ?? undefined, trainLabel.trim())
      setTrainResult({ caption: result.caption, accuracy: result.accuracy })
      setTrainLabel('')
      refreshReport()
    } catch {
      // ignore
    } finally {
      setTrainLoading(false)
    }
  }, [previewUrl, trainLabel, previewFileName, refreshReport])

  const handleGenerateImage = useCallback(async () => {
    if (!genPrompt.trim()) return
    setGenLoading(true)
    setGenError(null)
    setGenResult(null)
    try {
      const { multimodalController } = await import('@/lib/multimodal-controller')
      const result = await multimodalController.generateImage(genPrompt.trim())
      setGenResult(result.image)
    } catch (err) {
      setGenError(err instanceof Error ? err.message : 'Generation failed')
    } finally {
      setGenLoading(false)
    }
  }, [genPrompt])

  const handleSendGeneratedImage = useCallback(() => {
    if (genResult) {
      onGeneratedImage(genResult, genPrompt)
      setGenResult(null)
      setGenPrompt('')
    }
  }, [genResult, genPrompt, onGeneratedImage])

  const handleReset = useCallback(async () => {
    setResetLoading(true)
    try {
      const { multimodalController } = await import('@/lib/multimodal-controller')
      await multimodalController.resetModel()
      setTrainingReport(null)
      refreshReport()
    } catch {
      // ignore
    } finally {
      setResetLoading(false)
    }
  }, [refreshReport])

  const clearPreview = useCallback(() => {
    setPreviewUrl(null)
    setPreviewFileName(null)
    setAnalyzeResult(null)
    setAnalyzeError(null)
    setTrainResult(null)
  }, [])

  return {
    tab, setTab,
    analyzeLoading, analyzeResult, analyzeError,
    previewUrl, previewFileName,
    trainLabel, setTrainLabel, trainLoading, trainResult,
    genPrompt, setGenPrompt, genLoading, genResult, genError, setGenResult,
    trainingReport, resetLoading, dragOver, retryLoading,
    fileInputRef, dropRef,
    refreshReport, processFile, retryAnalyze,
    handleDrop, handleFileSelect, handleTrainWithLabel,
    handleGenerateImage, handleSendGeneratedImage,
    handleReset, clearPreview, setDragOver,
  }
}
