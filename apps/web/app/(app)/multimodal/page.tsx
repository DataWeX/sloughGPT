'use client'
export const dynamic = 'force-dynamic'

import { useCallback, useEffect, useRef, useState } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Skeleton } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { multimodalController } from '@/lib/controllers'
import type { MultimodalCapabilities, TrainingReport, TrainingStatus } from '@/lib/multimodal-controller'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { apiPost } from '@/lib/http-client'
import { logger } from '@/lib/dev-log'
import dynamicNext from 'next/dynamic'
import CapabilitiesCard from '@/components/multimodal/CapabilitiesCard'
import ImageTrainingCard from '@/components/multimodal/ImageTrainingCard'
import BatchTrainingCard from '@/components/multimodal/BatchTrainingCard'
import VisualDatasetCard from '@/components/multimodal/VisualDatasetCard'
import PreferenceOptimizationCard from '@/components/multimodal/PreferenceOptimizationCard'
import ImageGenerationCard from '@/components/multimodal/ImageGenerationCard'
import AudioCard from '@/components/multimodal/AudioCard'
import { VoiceSection } from '@/components/multimodal/VoiceSection'
import { ImageSection } from '@/components/multimodal/ImageSection'


const TrainingCard = dynamicNext(() => import('@/components/multimodal/TrainingCard'), { ssr: false })

export default function MultimodalPage() {
  const addToast = useToastStore(s => s.addToast)
  const [caps, setCaps] = useState<MultimodalCapabilities | null>(null)
  const [report, setReport] = useState<TrainingReport | null>(null)
  const [trainStatus, setTrainStatus] = useState<TrainingStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [batchUploading, setBatchUploading] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [generatedImage, setGeneratedImage] = useState<string | null>(null)
  const [transcribing, setTranscribing] = useState(false)
  const [transcript, setTranscript] = useState<string | null>(null)
  const [synthesizing, setSynthesizing] = useState(false)
  const [synthAudio, setSynthAudio] = useState<{ audio: string; duration_sec: number } | null>(null)
  const [creatingDataset, setCreatingDataset] = useState(false)
  const [dpoRunning, setDpoRunning] = useState(false)
  const [dpoStatus, setDpoStatus] = useState<string>('idle')
  const [dpoResult, setDpoResult] = useState<Record<string, unknown> | null>(null)
  const [dpoError, setDpoError] = useState<string | null>(null)
  const [dpoAccepted, setDpoAccepted] = useState(0)
  const [dpoRejected, setDpoRejected] = useState(0)
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null)

  const fetchAll = useCallback(async () => {
    try {
      // Graceful degradation: training report/status may be unavailable, UI handles null
      const [c, r, s] = await Promise.all([
        multimodalController.getCapabilities(),
        multimodalController.getTrainingReport().catch(() => null),
        multimodalController.getTrainingStatus().catch(() => null),
      ])
      setCaps(c)
      setReport(r)
      setTrainStatus(s)
    } catch {
      addToast('Could not load data', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  const startPolling = useCallback(() => {
    if (pollIntervalRef.current) return
    pollIntervalRef.current = setInterval(async () => {
      try {
        const status = await multimodalController.getTrainingStatus()
        setTrainStatus(status)
        if (!status.running) {
          if (pollIntervalRef.current) { clearInterval(pollIntervalRef.current); pollIntervalRef.current = null }
          fetchAll()
          if (status.completed > 0) addToast(`Training complete: ${status.completed} images, ${status.errors} errors`, status.errors > 0 ? 'error' : 'success')
        }
      } catch (err) {
        logger.error('Could not poll training status', { exception: String(err) })
      }
    }, 2000)
  }, [fetchAll, addToast])

  const pollDPOStatus = useCallback(async () => {
    try {
      const s = await multimodalController.getDPOStatus()
      setDpoStatus(s.status)
      setDpoAccepted(s.accepted_count)
      setDpoRejected(s.rejected_count)
      if (s.status === 'completed' && s.result) {
        setDpoResult(s.result); setDpoRunning(false)
        addToast('DPO training complete', 'success')
      } else if (s.status === 'error') {
        setDpoError(s.result?.error as string || 'Could not dpo'); setDpoRunning(false)
        addToast(s.result?.error as string || 'Could not dpo', 'error')
      } else if (s.status === 'idle') { setDpoRunning(false) }
    } catch (err) {
      logger.error('Could not dpo status poll', { exception: String(err) })
    }
  }, [addToast])

  useEffect(() => { fetchAll() }, [fetchAll])
  useEffect(() => () => { if (pollIntervalRef.current) clearInterval(pollIntervalRef.current) }, [])
  useEffect(() => { if (trainStatus?.running && !pollIntervalRef.current) startPolling() }, [trainStatus?.running, startPolling])
  useEffect(() => { if (dpoStatus === 'running') { const interval = setInterval(pollDPOStatus, 3000); return () => clearInterval(interval) } }, [dpoStatus, pollDPOStatus])

  const handleUploadImage = async (file: File) => {
    setUploading(true)
    try {
      const reader = new FileReader()
      reader.onload = async () => {
        const dataUrl = reader.result as string
        const result = await multimodalController.trainImage(dataUrl, file.name)
        addToast(`Trained on "${file.name}" — caption: ${result.caption}`, 'success')
        fetchAll()
      }
      reader.readAsDataURL(file)
    } catch { addToast('Could not upload', 'error')
    } finally { setUploading(false) }
  }

  const handleBatchUpload = async (files: File[]) => {
    if (files.length === 0) return
    setBatchUploading(true)
    try {
      const result = await multimodalController.trainBatch(files)
      addToast(`Training started: ${result.total_images} images`, 'success')
      startPolling()
    } catch { addToast('Could not upload', 'error')
    } finally { setBatchUploading(false) }
  }

  const handleBatchDir = async (dirPath: string) => {
    setBatchUploading(true)
    try {
      const result = await multimodalController.trainBatchFromDir(dirPath)
      addToast(`Training started: ${result.total_images} images from ${dirPath}`, 'success')
      startPolling()
    } catch (err: unknown) { addToast(extractErrorMessage(err, 'Could not start training'), 'error')
    } finally { setBatchUploading(false) }
  }

  const handleCreateVisualDataset = async (name: string, imageDir: string) => {
    setCreatingDataset(true)
    try {
      const result = await apiPost<{ dataset: string; entries: number }>('/multimodal/visual-dataset', {
        name,
        image_dir: imageDir,
        caption_prompt: 'Describe this image in detail.',
        auto_caption: true,
      })
      addToast(`Dataset "${result.dataset}" created: ${result.entries} entries`, 'success')
    } catch (err: unknown) { addToast(extractErrorMessage(err, 'Could not dataset creation'), 'error')
    } finally { setCreatingDataset(false) }
  }

  const handleTriggerDPO = async () => {
    if (dpoRunning || dpoStatus === 'running') return
    setDpoRunning(true); setDpoError(null); setDpoResult(null); setDpoStatus('running')
    try {
      const result = await apiPost<{ status: string }>('/multimodal/dpo')
      addToast(`DPO training started: ${result.status || ''}`, 'success')
    } catch (err: unknown) {
      const msg = extractErrorMessage(err, 'Could not dpo trigger')
      setDpoError(msg); setDpoStatus('error'); setDpoRunning(false); addToast(msg, 'error')
    }
  }

  const handleGenerateImage = async (prompt: string) => {
    setGenerating(true); setGeneratedImage(null)
    try {
      const result = await multimodalController.generateImage(prompt)
      setGeneratedImage(result.image)
      addToast(`Generated: "${result.prompt}"`, 'success')
    } catch { addToast('Could not image generation', 'error')
    } finally { setGenerating(false) }
  }

  const handleTranscribe = async (file: File) => {
    setTranscribing(true); setTranscript(null)
    try {
      const result = await multimodalController.transcribeAudio(file)
      setTranscript(result.text)
    } catch { addToast('Could not speech-to-text', 'error')
    } finally { setTranscribing(false) }
  }

  const handleSynthesize = async (text: string) => {
    setSynthesizing(true); setSynthAudio(null)
    try {
      const result = await multimodalController.synthesizeSpeech(text)
      setSynthAudio({ audio: result.audio, duration_sec: result.duration_sec })
      addToast(`Voice generated (${result.duration_sec.toFixed(1)}s)`, 'success')
    } catch { addToast('Could not speech generation', 'error')
    } finally { setSynthesizing(false) }
  }

  const headerRight = (
    <Button variant="outline" size="sm" onClick={fetchAll} disabled={loading}>
      <IconRefresh className="h-4 w-4 mr-1" /> Refresh
    </Button>
  )

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'r' && !e.metaKey && !e.ctrlKey) { e.preventDefault(); void fetchAll() }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [fetchAll])

  return (
    <PageContainer
      title="Multimodal"
      subtitle="Vision, speech, and image generation"
      headerRight={headerRight}
      loading={loading}
    >
      <CapabilitiesCard caps={caps} />
      {report && <TrainingCard report={report} trainStatus={trainStatus} />}
      <ImageTrainingCard uploading={uploading} onUpload={handleUploadImage} />
      <BatchTrainingCard batchUploading={batchUploading} trainStatus={trainStatus} onFileUpload={handleBatchUpload} onDirUpload={handleBatchDir} />
      <VisualDatasetCard creatingDataset={creatingDataset} onCreate={handleCreateVisualDataset} />
      <PreferenceOptimizationCard dpoRunning={dpoRunning} dpoStatus={dpoStatus} dpoResult={dpoResult} dpoError={dpoError} dpoAccepted={dpoAccepted} dpoRejected={dpoRejected} onTrigger={handleTriggerDPO} />
      <ImageGenerationCard generating={generating} onGenerate={handleGenerateImage} generatedImage={generatedImage} />
      <AudioCard transcribing={transcribing} transcript={transcript} synthesizing={synthesizing} synthAudio={synthAudio} onTranscribe={handleTranscribe} onSynthesize={handleSynthesize} />
      <VoiceSection />
      <ImageSection />
    </PageContainer>
  )
}
