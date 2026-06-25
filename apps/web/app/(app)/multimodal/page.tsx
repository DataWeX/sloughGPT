'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/tags'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { StatCard, KpiGrid, Skeleton, ProgressBar } from '@/components/ui'
import { IconRefresh, IconUpload, IconTrash, IconCheck } from '@/components/ui'
import { cn } from '@/lib/cn'
import { multimodalController, visualController } from '@/lib/controllers'
import type { MultimodalCapabilities, TrainingReport, TrainingStatus } from '@/lib/multimodal-controller'
import { useToastStore } from '@/lib/toast-store'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'

export default function MultimodalPage() {
  const addToast = useToastStore(s => s.addToast)
  const [caps, setCaps] = useState<MultimodalCapabilities | null>(null)
  const [report, setReport] = useState<TrainingReport | null>(null)
  const [trainStatus, setTrainStatus] = useState<TrainingStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [batchUploading, setBatchUploading] = useState(false)
  const [batchDirPath, setBatchDirPath] = useState('')
  const [visualDatasetName, setVisualDatasetName] = useState('')
  const [visualImageDir, setVisualImageDir] = useState('')
  const [creatingDataset, setCreatingDataset] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [generatedImage, setGeneratedImage] = useState<string | null>(null)
  const [genPrompt, setGenPrompt] = useState('')
  const [transcribing, setTranscribing] = useState(false)
  const [transcript, setTranscript] = useState<string | null>(null)
  const [synthesizing, setSynthesizing] = useState(false)
  const [synthText, setSynthText] = useState('')
  const [visualInferring, setVisualInferring] = useState(false)
  const [visualOutput, setVisualOutput] = useState<string | null>(null)
  const [visualPrompt, setVisualPrompt] = useState('Describe this image in detail.')
  const [visualLoaded, setVisualLoaded] = useState(false)
  const [visualTrainDataPath, setVisualTrainDataPath] = useState('')
  const [visualTraining, setVisualTraining] = useState(false)
  const [visualTrainStatus, setVisualTrainStatus] = useState<string>('idle')
  const [visualTrainProgress, setVisualTrainProgress] = useState<number | null>(null)
  const [visualTrainCurrentStage, setVisualTrainCurrentStage] = useState<string | null>(null)
  const [visualTrainLoss, setVisualTrainLoss] = useState<number | null>(null)
  const [visualTrainResult, setVisualTrainResult] = useState<any>(null)
  const [visualTrainError, setVisualTrainError] = useState<string | null>(null)
  const [visualModelDir, setVisualModelDir] = useState('models/visual-finetuned')
  const [visualLoadLoading, setVisualLoadLoading] = useState(false)
  const [dpoRunning, setDpoRunning] = useState(false)
  const [dpoStatus, setDpoStatus] = useState<string>('idle')
  const [dpoResult, setDpoResult] = useState<any>(null)
  const [dpoError, setDpoError] = useState<string | null>(null)
  const [dpoLastRun, setDpoLastRun] = useState<string | null>(null)
  const [dpoAccepted, setDpoAccepted] = useState(0)
  const [dpoRejected, setDpoRejected] = useState(0)
  const [visualCheckpoints, setVisualCheckpoints] = useState<any[]>([])
  const [ckptLoading, setCkptLoading] = useState(false)
  const [deletingCkpt, setDeletingCkpt] = useState<string | null>(null)
  const [loadingCkpt, setLoadingCkpt] = useState<string | null>(null)
  const visualImageInputRef = useRef<HTMLInputElement>(null)
  const [visualImageBase64, setVisualImageBase64] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const batchFileInputRef = useRef<HTMLInputElement>(null)
  const audioInputRef = useRef<HTMLInputElement>(null)
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null)

  const fetchCheckpoints = useCallback(async () => {
    try {
      const res = await visualController.listCheckpoints()
      setVisualCheckpoints(res.checkpoints || [])
    } catch { /* ignore */ }
  }, [])

  const fetchAll = useCallback(async () => {
    try {
      const [c, r, s, visualStatus] = await Promise.all([
        multimodalController.getCapabilities(),
        multimodalController.getTrainingReport().catch(() => null),
        multimodalController.getTrainingStatus().catch(() => null),
        visualController.getVisualStatus().catch(() => ({ loaded: false })),
      ])
      setCaps(c)
      setReport(r)
      setTrainStatus(s)
      setVisualLoaded(visualStatus.loaded)
      fetchCheckpoints()
    } catch {
      addToast('Failed to load data', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast, fetchCheckpoints])

  const startPolling = useCallback(() => {
    if (pollIntervalRef.current) return
    pollIntervalRef.current = setInterval(async () => {
      try {
        const status = await multimodalController.getTrainingStatus()
        setTrainStatus(status)
        if (!status.running) {
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current)
            pollIntervalRef.current = null
          }
          fetchAll()
          if (status.completed > 0) {
            addToast(`Training complete: ${status.completed} images, ${status.errors} errors`, status.errors > 0 ? 'error' : 'success')
          }
        }
      } catch {
        // ignore polling errors
      }
    }, 2000)
  }, [fetchAll, addToast])

  useEffect(() => { fetchAll() }, [fetchAll])

  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
      }
    }
  }, [])

  useEffect(() => {
    if (trainStatus?.running && !pollIntervalRef.current) {
      startPolling()
    }
  }, [trainStatus?.running, startPolling])

  const handleUploadImage = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
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
    } catch {
      addToast('Upload failed', 'error')
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleBatchUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length === 0) return
    setBatchUploading(true)
    try {
      const result = await multimodalController.trainBatch(files)
      addToast(`Training started: ${result.total_images} images`, 'success')
      startPolling()
    } catch {
      addToast('Upload failed', 'error')
    } finally {
      setBatchUploading(false)
      if (batchFileInputRef.current) batchFileInputRef.current.value = ''
    }
  }

  const handleBatchDir = async () => {
    if (!batchDirPath.trim()) return
    setBatchUploading(true)
    try {
      const result = await multimodalController.trainBatchFromDir(batchDirPath.trim())
      addToast(`Training started: ${result.total_images} images from ${batchDirPath}`, 'success')
      startPolling()
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Training failed'
      addToast(msg, 'error')
    } finally {
      setBatchUploading(false)
    }
  }

  const handleCreateVisualDataset = async () => {
    if (!visualDatasetName.trim() || !visualImageDir.trim()) return
    setCreatingDataset(true)
    try {
      const result = await visualController.createVisualDataset(
        visualDatasetName.trim(),
        visualImageDir.trim(),
      )
      addToast(`Dataset "${result.dataset}" created: ${result.entries} entries`, 'success')
      setVisualDatasetName('')
      setVisualImageDir('')
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Dataset creation failed'
      addToast(msg, 'error')
    } finally {
      setCreatingDataset(false)
    }
  }

  const pollVisualTrain = useCallback(async () => {
    try {
      const status = await visualController.getVisualTrainStatus()
      setVisualTrainStatus(status.status)
      setVisualTrainProgress(status.progress)
      setVisualTrainCurrentStage(status.current_stage)
      setVisualTrainLoss(status.current_loss)
      if (status.status === 'completed') {
        setVisualTrainResult(status.result)
        setVisualTraining(false)
        addToast('Visual training complete', 'success')
      } else if (status.status === 'error') {
        setVisualTrainError(status.error || 'Training failed')
        setVisualTraining(false)
        addToast(status.error || 'Training failed', 'error')
      }
    } catch {
      // silent
    }
  }, [addToast])

  useEffect(() => {
    if (visualTrainStatus === 'running') {
      const interval = setInterval(pollVisualTrain, 3000)
      return () => clearInterval(interval)
    }
  }, [visualTrainStatus, pollVisualTrain])

  const handleVisualTrain = async () => {
    if (!visualTrainDataPath.trim() || visualTraining) return
    setVisualTraining(true)
    setVisualTrainResult(null)
    setVisualTrainError(null)
    setVisualTrainStatus('running')
    try {
      const dataPath = visualTrainDataPath.trim().startsWith('/')
        ? visualTrainDataPath.trim()
        : `datasets/${visualTrainDataPath.trim().replace(/^datasets\//, '')}/corpus.jsonl`
      const result = await visualController.startVisualTrain({
        data_path: dataPath,
      })
      addToast(`Visual training started: ${result.job_id}`, 'success')
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Training failed'
      setVisualTrainError(msg)
      setVisualTrainStatus('error')
      setVisualTraining(false)
      addToast(msg, 'error')
    }
  }

  const handleVisualLoad = async () => {
    setVisualLoadLoading(true)
    try {
      const result = await visualController.loadVisualModel(visualModelDir.trim() || undefined)
      setVisualLoaded(true)
      addToast(result.message || 'Model loaded', 'success')
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Loading failed'
      addToast(msg, 'error')
    } finally {
      setVisualLoadLoading(false)
    }
  }

  const pollDPOStatus = useCallback(async () => {
    try {
      const s = await visualController.getDPOStatus()
      setDpoStatus(s.status)
      setDpoLastRun(s.last_run)
      setDpoAccepted(s.accepted_count)
      setDpoRejected(s.rejected_count)
      if (s.status === 'completed' && s.result) {
        setDpoResult(s.result)
        setDpoRunning(false)
        addToast('DPO training complete', 'success')
      } else if (s.status === 'error') {
        setDpoError(s.result?.error as string || 'DPO failed')
        setDpoRunning(false)
        addToast(s.result?.error as string || 'DPO failed', 'error')
      } else if (s.status === 'idle') {
        setDpoRunning(false)
      }
    } catch {
      // silent
    }
  }, [addToast])

  useEffect(() => {
    if (dpoStatus === 'running') {
      const interval = setInterval(pollDPOStatus, 3000)
      return () => clearInterval(interval)
    }
  }, [dpoStatus, pollDPOStatus])

  const handleTriggerDPO = async () => {
    if (dpoRunning || dpoStatus === 'running') return
    setDpoRunning(true)
    setDpoError(null)
    setDpoResult(null)
    setDpoStatus('running')
    try {
      const result = await visualController.triggerDPO()
      addToast(`DPO training started: ${result.job_id || result.status}`, 'success')
      setDpoStatus('running')
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'DPO trigger failed'
      setDpoError(msg)
      setDpoStatus('error')
      setDpoRunning(false)
      addToast(msg, 'error')
    }
  }

  const handleGenerateImage = async () => {
    if (!genPrompt.trim()) return
    setGenerating(true)
    setGeneratedImage(null)
    try {
      const result = await multimodalController.generateImage(genPrompt)
      setGeneratedImage(result.image)
      addToast(`Generated: "${result.prompt}"`, 'success')
    } catch {
      addToast('Image generation failed', 'error')
    } finally {
      setGenerating(false)
    }
  }

  const handleTranscribe = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setTranscribing(true)
    setTranscript(null)
    try {
      const result = await multimodalController.transcribeAudio(file)
      setTranscript(result.text)
    } catch {
      addToast('Speech-to-text failed', 'error')
    } finally {
      setTranscribing(false)
      if (audioInputRef.current) audioInputRef.current.value = ''
    }
  }

  const handleSynthesize = async () => {
    if (!synthText.trim()) return
    setSynthesizing(true)
    try {
      const result = await multimodalController.synthesizeSpeech(synthText)
      addToast(`Voice generated (${result.duration_sec.toFixed(1)}s)`, 'success')
    } catch {
      addToast('Speech generation failed', 'error')
    } finally {
      setSynthesizing(false)
    }
  }

  const handleVisualImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      const dataUrl = reader.result as string
      setVisualImageBase64(dataUrl)
      addToast(`Selected "${file.name}"`, 'info')
    }
    reader.readAsDataURL(file)
  }

  const handleVisualInfer = async () => {
    if (!visualImageBase64) { addToast('Select an image first', 'error'); return }
    setVisualInferring(true)
    setVisualOutput(null)
    try {
      const base64 = visualImageBase64.split(',')[1] || visualImageBase64
      const result = await visualController.visualInference(base64, visualPrompt)
      setVisualOutput(result.text)
      addToast(`Generated ${result.tokens_generated} tokens in ${(result.elapsed_ms / 1000).toFixed(1)}s`, 'success')
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Something went wrong'
      addToast(msg, 'error')
    } finally {
      setVisualInferring(false)
    }
  }

  const capList: { label: string; ok: boolean }[] = caps ? [
    { label: 'Speech-to-text', ok: caps.speech_to_text },
    { label: 'Image captioning', ok: caps.image_caption },
    { label: 'Vision model', ok: !!caps.vision_model },
    { label: 'Speech model', ok: !!caps.speech_model },
    { label: 'Trained', ok: caps.trained },
    { label: 'Vision model loaded', ok: visualLoaded },
  ] : []

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Multimodal" subtitle="Vision, speech, and image generation" />}
        right={
          <Button variant="outline" size="sm" onClick={fetchAll} disabled={loading}>
            <IconRefresh className="h-3.5 w-3.5 mr-1" /> Refresh
          </Button>
        }
      />

      <div className="space-y-4">
        {loading ? (
          <div className="space-y-3">
            <Skeleton className="h-24 rounded-lg" />
            <Skeleton className="h-48 rounded-lg" />
          </div>
        ) : (
          <>
            {/* Overview */}
            <Card>
              <CardHeader><CardTitle className="text-base">Capabilities</CardTitle></CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2 mb-4">
                  {capList.map(c => (
                    <Badge
                      key={c.label}
                      label={c.label}
                      variant={c.ok ? 'success' : 'warning'}
                      size="sm"
                    />
                  ))}
                </div>
                <KpiGrid columns={4}>
                  <StatCard label="Images learned" value={caps?.images_learned ?? 0} />
                  <StatCard label="Memory" value={`${caps?.replay_buffer_size ?? 0} items`} />
                  <StatCard label="Learning method" value={caps?.learning_method || '—'} />
                  <StatCard label="Status" value={caps?.status || '—'} />
                </KpiGrid>
              </CardContent>
            </Card>

            {/* Training progress */}
            {report && (
              <Card>
                <CardHeader><CardTitle className="text-base">Training</CardTitle></CardHeader>
                <CardContent>
                  <KpiGrid columns={4}>
                    <StatCard label="Images learned" value={report.images_learned} />
                    <StatCard label="Vocabulary" value={`${report.vocab_size} words`} />
                    <StatCard label="Unique captions" value={report.unique_captions} />
                    <StatCard label="Diversity" value={`${(report.diversity_ratio * 100).toFixed(0)}%`} />
                  </KpiGrid>
                  {trainStatus && trainStatus.total > 0 && (
                    <div className="mt-3 space-y-1">
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>{trainStatus.completed}/{trainStatus.total} images</span>
                        <span>{trainStatus.progress_pct}%</span>
                      </div>
                      <ProgressBar value={trainStatus.progress_pct} max={100} variant="default" />
                      {trainStatus.current_caption && (
                        <p className="text-[10px] text-muted-foreground/60 truncate">
                          Current: {trainStatus.current_caption}
                        </p>
                      )}
                    </div>
                  )}
                  {report.accuracy_history.length > 1 && (
                    <div className="mt-4">
                      <p className="text-xs font-medium text-muted-foreground mb-2">Accuracy over time</p>
                      <div className="h-32">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={report.accuracy_history.map((acc, i) => ({ step: i + 1, accuracy: acc }))}>
                            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                            <XAxis dataKey="step" stroke="var(--muted-foreground)" fontSize={10} />
                            <YAxis stroke="var(--muted-foreground)" fontSize={10} domain={[0, 1]} />
                            <Tooltip
                              contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: '6px', fontSize: '11px' }}
                              labelStyle={{ color: 'var(--foreground)' }}
                            />
                            <Line type="monotone" dataKey="accuracy" stroke="var(--primary)" strokeWidth={2} dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  )}
                  {report.caption_history.length > 0 && (
                    <div className="mt-3">
                      <p className="text-xs font-medium text-muted-foreground mb-1">Recent captions</p>
                      <div className="max-h-24 overflow-y-auto space-y-0.5">
                        {report.caption_history.slice(-6).reverse().map((c, i) => (
                          <p key={i} className="text-[10px] text-muted-foreground/70 leading-relaxed border-l-2 border-primary/20 pl-2">
                            {c}
                          </p>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Image upload & train */}
            <Card>
              <CardHeader><CardTitle className="text-base">Image Training</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                <p className="text-xs text-muted-foreground">Upload an image to train the vision model. It learns features and generates a caption automatically.</p>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 text-xs"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploading}
                  >
                    <IconUpload className="h-3.5 w-3.5 mr-1" />
                    {uploading ? 'Training…' : 'Upload image'}
                  </Button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleUploadImage}
                  />
                </div>
              </CardContent>
            </Card>

            {/* Batch training */}
            <Card>
              <CardHeader><CardTitle className="text-base">Train with multiple images</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                <p className="text-xs text-muted-foreground">Train on multiple images at once. Upload files or specify a server directory path.</p>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 text-xs"
                    onClick={() => batchFileInputRef.current?.click()}
                    disabled={batchUploading || trainStatus?.running}
                  >
                    <IconUpload className="h-3.5 w-3.5 mr-1" />
                    {batchUploading ? 'Starting…' : 'Upload images'}
                  </Button>
                  <input
                    ref={batchFileInputRef}
                    type="file"
                    accept="image/*"
                    multiple
                    className="hidden"
                    onChange={handleBatchUpload}
                  />
                </div>
                <div className="flex items-center gap-2">
                  <Input
                    value={batchDirPath}
                    onChange={e => setBatchDirPath(e.target.value)}
                    placeholder="/path/to/images on server"
                    className="h-8 text-xs flex-1"
                    aria-label="Server directory path for batch training"
                  />
                  <Button
                    size="sm"
                    className="h-8 text-xs shrink-0"
                    onClick={handleBatchDir}
                    disabled={!batchDirPath.trim() || trainStatus?.running}
                  >
                    {trainStatus?.running ? 'Training…' : 'Train from directory'}
                  </Button>
                </div>
                {trainStatus && trainStatus.running && (
                  <div className="space-y-1 pt-2">
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>{trainStatus.completed}/{trainStatus.total} images</span>
                      <span>{trainStatus.progress_pct}%</span>
                    </div>
                    <ProgressBar value={trainStatus.progress_pct} max={100} variant="default" />
                    {trainStatus.current_image && (
                      <p className="text-[10px] text-muted-foreground/60 truncate">
                        Processing: {trainStatus.current_image}
                      </p>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Visual Dataset Creation */}
            <Card>
              <CardHeader><CardTitle className="text-base">Image description dataset</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                <p className="text-xs text-muted-foreground">Create a training dataset from a folder of images. The AI writes descriptions for each image automatically.</p>
                <div className="flex items-center gap-2">
                  <Input
                    value={visualDatasetName}
                    onChange={e => setVisualDatasetName(e.target.value)}
                    placeholder="Dataset name"
                    className="h-8 text-xs flex-1"
                    aria-label="Visual dataset name"
                  />
                  <Input
                    value={visualImageDir}
                    onChange={e => setVisualImageDir(e.target.value)}
                    placeholder="/path/to/images"
                    className="h-8 text-xs flex-1"
                    aria-label="Image directory for visual dataset"
                  />
                </div>
                <Button
                  size="sm"
                  className="h-8 text-xs"
                  onClick={handleCreateVisualDataset}
                  disabled={!visualDatasetName.trim() || !visualImageDir.trim() || creatingDataset}
                >
                  {creatingDataset ? 'Creating…' : 'Create dataset'}
                </Button>
              </CardContent>
            </Card>

            {/* Visual Training */}
            <Card>
              <CardHeader><CardTitle className="text-base">Train vision model</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                <p className="text-xs text-muted-foreground">Train a visual model on an image description dataset created above.</p>
                <div className="flex items-center gap-2">
                  <Input
                    value={visualTrainDataPath}
                    onChange={e => setVisualTrainDataPath(e.target.value)}
                    placeholder="datasets/my-dataset/corpus.jsonl"
                    className="h-8 text-xs flex-1"
                    aria-label="Training data path"
                  />
                  <Button
                    size="sm"
                    className="h-8 text-xs shrink-0"
                    onClick={handleVisualTrain}
                    disabled={!visualTrainDataPath.trim() || visualTraining || visualTrainStatus === 'running'}
                  >
                    {visualTraining || visualTrainStatus === 'running' ? 'Training…' : 'Start training'}
                  </Button>
                </div>
                {visualTrainStatus === 'running' && (
                  <div className="space-y-1 pt-1">
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>{visualTrainCurrentStage || 'Training…'}</span>
                      {visualTrainLoss !== null && <span>Loss: {visualTrainLoss.toFixed(4)}</span>}
                    </div>
                    <ProgressBar
                      value={visualTrainProgress || 0}
                      max={100}
                      variant="default"
                    />
                  </div>
                )}
                {visualTrainResult && (
                  <div className="p-2 rounded bg-success/10 border border-success/20 text-xs text-success">
                    Training complete
                  </div>
                )}
                {visualTrainError && (
                  <div className="p-2 rounded bg-destructive/10 border border-destructive/20 text-xs text-destructive">
                    {visualTrainError}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Visual Model Load */}
            <Card>
              <CardHeader><CardTitle className="text-base">Load trained model</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                <p className="text-xs text-muted-foreground">Load a trained visual model for inference testing.</p>
                <div className="flex items-center gap-2">
                  <Input
                    value={visualModelDir}
                    onChange={e => setVisualModelDir(e.target.value)}
                    placeholder="models/visual-finetuned"
                    className="h-8 text-xs flex-1"
                    aria-label="Model directory"
                  />
                  <Button
                    size="sm"
                    className="h-8 text-xs shrink-0"
                    onClick={handleVisualLoad}
                    disabled={visualLoadLoading}
                  >
                    {visualLoadLoading ? 'Loading…' : 'Load model'}
                  </Button>
                </div>
                {visualLoaded && (
                  <p className="text-xs text-success flex items-center gap-1">
                    <IconCheck className="h-3.5 w-3.5" />
                    Vision model loaded and ready for inference
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Visual Checkpoints */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">Visual Checkpoints</CardTitle>
                  <Button size="sm" variant="ghost" className="h-7 text-xs gap-1" onClick={() => { setCkptLoading(true); fetchCheckpoints().finally(() => setCkptLoading(false)) }}>
                    <IconRefresh className="h-3.5 w-3.5" />
                    Refresh
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {visualCheckpoints.length === 0 ? (
                  <p className="text-xs text-muted-foreground">No checkpoints saved yet.</p>
                ) : (
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {visualCheckpoints.map((cp: any) => (
                      <div key={cp.name} className="flex items-center justify-between gap-2 rounded-md border px-3 py-2 text-xs">
                        <div className="min-w-0 flex-1">
                          <p className="font-medium truncate">{cp.name}</p>
                          <p className="text-muted-foreground">
                            {cp.llm || '—'} · {cp.final_loss != null ? `loss ${cp.final_loss.toFixed(4)}` : ''}{cp.total_steps ? ` · ${cp.total_steps} steps` : ''}{cp.mean_accuracy != null ? ` · acc ${(cp.mean_accuracy * 100).toFixed(0)}%` : ''}{cp.size_mb ? ` · ${cp.size_mb.toFixed(1)} MB` : ''}
                          </p>
                          {cp.description && <p className="text-muted-foreground truncate mt-0.5">{cp.description}</p>}
                        </div>
                        <div className="flex items-center gap-1 shrink-0">
                          <Button
                            size="sm"
                            variant="ghost"
                            className="h-7 text-xs"
                            onClick={async () => {
                              setLoadingCkpt(cp.name)
                              try {
                                const res = await visualController.loadCheckpoint(cp.name)
                                addToast(`Checkpoint "${res.name}" loaded`, 'success')
                                fetchCheckpoints()
                              } catch { addToast('Failed to load checkpoint', 'error') }
                              finally { setLoadingCkpt(null) }
                            }}
                            disabled={loadingCkpt === cp.name}
                          >
                            {loadingCkpt === cp.name ? '…' : 'Load'}
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="h-7 text-xs text-error"
                            onClick={async () => {
                              if (!window.confirm(`Delete checkpoint "${cp.name}"?`)) return
                              setDeletingCkpt(cp.name)
                              try {
                                await visualController.deleteCheckpoint(cp.name)
                                addToast('Checkpoint deleted', 'success')
                                fetchCheckpoints()
                              } catch { addToast('Failed to delete checkpoint', 'error') }
                              finally { setDeletingCkpt(null) }
                            }}
                            disabled={deletingCkpt === cp.name}
                          >
                            {deletingCkpt === cp.name ? '…' : <IconTrash className="h-3.5 w-3.5" />}
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Visual DPO */}
            <Card>
              <CardHeader><CardTitle className="text-base">DPO fine-tune</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                <p className="text-xs text-muted-foreground">
                  Run Direct Preference Optimization on feedback pairs (thumbs up/down) to align the model.
                </p>
                <div className="flex items-center gap-2">
                  <Button
                    size="sm"
                    className="h-8 text-xs shrink-0"
                    onClick={handleTriggerDPO}
                    disabled={dpoRunning || dpoStatus === 'running'}
                  >
                    {dpoRunning || dpoStatus === 'running' ? 'Running…' : 'Run DPO'}
                  </Button>
                  {(dpoAccepted > 0 || dpoRejected > 0) && (
                    <span className="text-xs text-muted-foreground">
                      {dpoAccepted} accepted / {dpoRejected} rejected
                    </span>
                  )}
                </div>
                {(dpoStatus === 'running') && (
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span className="animate-pulse h-2 w-2 rounded-full bg-primary" />
                    DPO training in progress…
                  </div>
                )}
                {dpoResult && dpoResult.status === 'accepted' && (
                  <div className="space-y-1 p-2 rounded bg-success/10 border border-success/20 text-xs">
                    <p className="text-success font-medium">✓ DPO accepted — model updated</p>
                    {dpoResult.steps > 0 && <p className="text-muted-foreground">{dpoResult.steps} steps · avg loss {dpoResult.avg_loss?.toFixed(4)}</p>}
                    {dpoResult.ppl_before != null && (
                      <p className="text-muted-foreground">PPL: {dpoResult.ppl_before?.toFixed(2)} → {dpoResult.ppl_after?.toFixed(2)} ({dpoResult.ppl_delta_pct > 0 ? '+' : ''}{dpoResult.ppl_delta_pct?.toFixed(1)}%)</p>
                    )}
                    {dpoResult.pairs_trained > 0 && <p className="text-muted-foreground">{dpoResult.pairs_trained} pairs trained</p>}
                    {dpoResult.elapsed_seconds > 0 && <p className="text-muted-foreground">Took {dpoResult.elapsed_seconds}s</p>}
                  </div>
                )}
                {dpoResult && dpoResult.status === 'rejected' && (
                  <div className="p-2 rounded bg-destructive/10 border border-destructive/20 text-xs text-destructive">
                    DPO rejected — PPL degradation above threshold
                  </div>
                )}
                {dpoError && (
                  <div className="p-2 rounded bg-destructive/10 border border-destructive/20 text-xs text-destructive">
                    {dpoError}
                  </div>
                )}
                {dpoLastRun && dpoResult && (
                  <p className="text-[10px] text-muted-foreground">Last run: {dpoLastRun}</p>
                )}
              </CardContent>
            </Card>

            {/* Visual Inference */}
            <Card>
              <CardHeader><CardTitle className="text-base">Test the vision model</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                <p className="text-xs text-muted-foreground">
                  Upload an image and ask a question about it.
                  {visualLoaded ? (
                    <span className="text-success ml-1">Vision model is loaded.</span>
                  ) : (
                    <span className="text-muted-foreground ml-1">Train a vision model on the Training page first.</span>
                  )}
                </p>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 text-xs"
                    onClick={() => visualImageInputRef.current?.click()}
                    disabled={!visualLoaded || visualInferring}
                  >
                    <IconUpload className="h-3.5 w-3.5 mr-1" />
                    {visualImageBase64 ? 'Change image' : 'Select image'}
                  </Button>
                  <input
                    ref={visualImageInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleVisualImageSelect}
                  />
                </div>
                {visualImageBase64 && (
                  <div className="flex items-center gap-2">
                    <img src={visualImageBase64} alt="Selected" className="h-16 w-16 rounded object-cover border border-border/50" />
                    <span className="text-xs text-muted-foreground">Image selected for inference</span>
                  </div>
                )}
                <div className="flex items-center gap-2">
                  <Input
                    value={visualPrompt}
                    onChange={e => setVisualPrompt(e.target.value)}
                    placeholder="Describe this image in detail..."
                    className="h-8 text-xs flex-1"
                  />
                  <Button
                    size="sm"
                    className="h-8 text-xs shrink-0"
                    onClick={handleVisualInfer}
                    disabled={!visualLoaded || !visualImageBase64 || visualInferring}
                  >
                    {visualInferring ? 'Generating…' : 'Generate'}
                  </Button>
                </div>
                {visualOutput && (
                  <div className="rounded-md border border-border/40 bg-muted/30 p-3 text-xs leading-relaxed max-h-48 overflow-y-auto">
                    {visualOutput}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Image generation */}
            <Card>
              <CardHeader><CardTitle className="text-base">Image Generation</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center gap-2">
                  <Input
                    value={genPrompt}
                    onChange={e => setGenPrompt(e.target.value)}
                    placeholder="A cat in a spacesuit…"
                    className="h-8 text-xs flex-1"
                    onKeyDown={e => { if (e.key === 'Enter') handleGenerateImage() }}
                    aria-label="Image generation prompt"
                  />
                  <Button
                    size="sm"
                    className="h-8 text-xs shrink-0"
                    onClick={handleGenerateImage}
                    disabled={generating || !genPrompt.trim()}
                  >
                    {generating ? 'Generating…' : 'Generate'}
                  </Button>
                </div>
                {generatedImage && (
                  <div className="rounded-lg border border-border/50 overflow-hidden">
                    <img src={generatedImage} alt="Generated" className="w-full h-auto max-h-64 object-contain bg-muted/20" />
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Audio */}
            <Card>
              <CardHeader><CardTitle className="text-base">Audio</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-2">Speech-to-text</p>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-8 text-xs"
                      onClick={() => audioInputRef.current?.click()}
                      disabled={transcribing}
                    >
                      <IconUpload className="h-3.5 w-3.5 mr-1" />
                      {transcribing ? 'Transcribing…' : 'Upload audio'}
                    </Button>
                    <input
                      ref={audioInputRef}
                      type="file"
                      accept="audio/*"
                      className="hidden"
                      onChange={handleTranscribe}
                    />
                  </div>
                  {transcript && (
                    <div className="mt-2 p-2 rounded bg-muted/30 border border-border/40 text-xs text-muted-foreground">
                      {transcript}
                    </div>
                  )}
                </div>
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-2">Text-to-speech</p>
                  <div className="flex items-center gap-2">
                    <Input
                      value={synthText}
                      onChange={e => setSynthText(e.target.value)}
                      placeholder="Text to speak…"
                      className="h-8 text-xs flex-1"
                      onKeyDown={e => { if (e.key === 'Enter') handleSynthesize() }}
                      aria-label="Text to synthesize"
                    />
                    <Button
                      size="sm"
                      className="h-8 text-xs shrink-0"
                      onClick={handleSynthesize}
                      disabled={synthesizing || !synthText.trim()}
                    >
                      {synthesizing ? 'Synthesizing…' : 'Speak'}
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  )
}
