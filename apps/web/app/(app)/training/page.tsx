'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui'
import { StatCard, KpiGrid } from '@/components/strui'
import { useToastStore } from '@/lib/toast-store'
import { modelController, trainingController } from '@/lib/controllers'
import { feedbackController } from '@/lib/feedback-controller'
import { datasetController } from '@/lib/controllers'
import type { Dataset } from '@/lib/dataset-controller'
import { DatasetImportModal } from '@/components/DatasetImportModal'
import { cn } from '@/lib/cn'
import { formatSize } from '@/lib/chat-utils'
import { TestModelDialog } from '@/components/training/TestModelDialog'
import { LossChart } from '@/components/training/LossChart'
import { useTrainingSession } from '@/hooks/useTrainingSession'
import { useTrainingDatasets } from '@/hooks/useTrainingDatasets'
import { useTrainingCheckpoints } from '@/hooks/useTrainingCheckpoints'
import { useTestDialog } from '@/hooks/useTestDialog'

function datasetLabel(ds: Dataset): string {
  const size = formatSize(ds.size)
  if (ds.type === 'vlm' && ds.vlm_metadata) {
    return `${ds.name} (VLM: ${ds.vlm_metadata.image_count} images, ${size})`
  }
  const suffix = ds.samples && ds.samples > 0 ? ` (${ds.samples.toLocaleString()} samples, ${size})` : ` (${size})`
  return `${ds.name}${suffix}`
}

type InputMode = 'dataset' | 'text'
type Method = 'distill' | 'finetune' | 'vlm' | 'unified'

function EstimatedTime({ method, datasetId, datasets, epochs, batchSize, sampleCount }: {
  method: string; datasetId: string | null; datasets: Dataset[]; epochs: number; batchSize: number; sampleCount?: number
}) {
  const ds = datasets.find(d => d.id === datasetId)
  const samples = sampleCount ?? ds?.samples ?? 0
  if (!datasetId || samples === 0) return null
  const steps = Math.ceil(samples / batchSize) * epochs
  const secsPerStep = method === 'finetune' ? 90 : 2
  const total = steps * secsPerStep
  if (total < 60) return <p className="text-[11px] text-muted-foreground/60">~{total}s on CPU</p>
  if (total < 3600) return <p className="text-[11px] text-muted-foreground/60">~{Math.ceil(total / 60)}m on CPU</p>
  return <p className="text-[11px] text-muted-foreground/60">~{(total / 3600).toFixed(1)}h on CPU</p>
}

export default function TrainingPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const addToast = useToastStore(s => s.addToast)
  const initialLoadDone = useRef(false)
  const qtTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const session = useTrainingSession()
  const datasets = useTrainingDatasets(addToast)
  const checkpoints = useTrainingCheckpoints()
  const test = useTestDialog()

  // ===== Quick Train =====
  const [quickDataset, setQuickDataset] = useState('')
  const [quickStats, setQuickStats] = useState<import('@/lib/dataset-controller').DatasetStats | null>(null)
  const [quickExplanation, setQuickExplanation] = useState('')
  const [quickJobId, setQuickJobId] = useState('')
  const [quickStatusMessage, setQuickStatusMessage] = useState('')
  const [quickTraining, setQuickTraining] = useState(false)
  const [quickComplete, setQuickComplete] = useState(false)

  // ===== Advanced config =====
  const TRAINING_CONFIG_KEY = 'sloughgpt-training-config'
  const loadSavedConfig = useCallback(() => {
    try {
      const saved = localStorage.getItem(TRAINING_CONFIG_KEY)
      return saved ? JSON.parse(saved) : null
    } catch { return null }
  }, [])
  const saved = useRef(loadSavedConfig())

  const [method, setMethod] = useState<Method>(saved.current?.method ?? 'distill')
  const [inputMode, setInputMode] = useState<InputMode>(saved.current?.inputMode ?? 'dataset')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [algo, setAlgo] = useState(saved.current?.algo ?? 'bpe')
  const [unifiedDistill, setUnifiedDistill] = useState(saved.current?.unifiedDistill ?? false)
  const [trainingEpochs, setTrainingEpochs] = useState(saved.current?.trainingEpochs ?? 5)
  const [trainingLR, setTrainingLR] = useState(saved.current?.trainingLR ?? 1e-3)
  const [trainingBatchSize, setTrainingBatchSize] = useState(saved.current?.trainingBatchSize ?? 64)
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [selectedModel, setSelectedModel] = useState(saved.current?.selectedModel ?? '')
  const [useLoRA, setUseLoRA] = useState(saved.current?.useLoRA ?? true)

  const [textInput, setTextInput] = useState('')

  useEffect(() => {
    try {
      localStorage.setItem(TRAINING_CONFIG_KEY, JSON.stringify({
        method, inputMode, algo, unifiedDistill, trainingEpochs, trainingLR,
        trainingBatchSize, selectedModel, useLoRA,
      }))
    } catch {}
  }, [method, inputMode, algo, unifiedDistill, trainingEpochs, trainingLR, trainingBatchSize, selectedModel, useLoRA])

  // ===== VLM config =====
  const [visualVisionEncoder, setVlmVisionEncoder] = useState('google/siglip-base-patch16-224')
  const [visualLLM, setVlmLLM] = useState('Qwen/Qwen2.5-0.5B-Instruct')
  const [visualStage1Epochs, setVlmStage1Epochs] = useState(1)
  const [visualStage2Epochs, setVlmStage2Epochs] = useState(2)

  // ===== Turbo config =====
  const [showTurboAdvanced, setShowTurboAdvanced] = useState(false)
  const [turboEpochs, setTurboEpochs] = useState(3)
  const [turboLR, setTurboLR] = useState(3e-4)
  const [turboEmbed, setTurboEmbed] = useState(128)
  const [turboHeads, setTurboHeads] = useState(4)
  const [turboLayers, setTurboLayers] = useState(3)

  // ===== Feedback Training =====
  const [feedbackTraining, setFeedbackTraining] = useState(false)
  const [feedbackJobId, setFeedbackJobId] = useState('')
  const [feedbackComplete, setFeedbackComplete] = useState(false)
  const [feedbackResult, setFeedbackResult] = useState<{ model_path?: string; final_loss?: number; samples?: number } | null>(null)
  const [feedbackError, setFeedbackError] = useState('')
  const [feedbackStats, setFeedbackStats] = useState<{ total: number; thumbs_up: number; thumbs_down: number } | null>(null)
  const fbIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    feedbackController.getFeedbackStats().then(s => {
      const db = s?.db_stats
      if (db) setFeedbackStats({ total: db.feedback_total, thumbs_up: db.thumbs_up, thumbs_down: db.thumbs_down })
    }).catch(() => {})
  }, [])

  // ===== Fine-tuned model loading =====
  const [loadingFinetunedModel, setLoadingFinetunedModel] = useState(false)

  // ===== Optimistic job entries (show immediately while backend processes) =====
  const [optimisticJobs, setOptimisticJobs] = useState<any[]>([])
  const allJobs = [...optimisticJobs, ...checkpoints.jobs]

  // ===== Visibility-based polling pause =====
  const visibilityRef = useRef<boolean>(true)

  // ===== Loading timeout: show retry suggestion if data doesn't load in 15s =====
  const [loadingTimedOut, setLoadingTimedOut] = useState(false)
  useEffect(() => {
    if (checkpoints.loadingCheckpoints || checkpoints.loadingJobs) {
      setLoadingTimedOut(false)
      const t = setTimeout(() => setLoadingTimedOut(true), 15000)
      return () => clearTimeout(t)
    }
  }, [checkpoints.loadingCheckpoints, checkpoints.loadingJobs])

  const startTraining = useCallback(async (checkpointName?: string) => {
    const hasDataset = inputMode === 'dataset' && datasets.selectedDataset
    const hasText = inputMode === 'text' && textInput.trim()

    if (!hasDataset && !hasText && !checkpointName) {
      addToast('Select a dataset or paste text to train on', 'error'); return
    }

    if ((method === 'finetune' || method === 'unified') && !hasDataset) {
      addToast(`${method === 'unified' ? 'Unified training' : 'Continue training'} requires a dataset.`, 'error'); return
    }

    if (method === 'vlm' && !hasDataset) {
      addToast('Vision model training requires a dataset with image-text pairs', 'error'); return
    }

    const body: Record<string, unknown> = { algo, epochs: trainingEpochs, learning_rate: trainingLR }
    if (trainingBatchSize) body.batch_size = trainingBatchSize
    if (checkpointName) body.checkpoint_name = checkpointName
    if (hasDataset) body.dataset_id = datasets.selectedDataset
    if (hasText) body.source_text = textInput.trim()

    const tempId = `pending-${Date.now()}`
    const now = new Date().toISOString()
    setOptimisticJobs(prev => [...prev, {
      id: tempId, name: `${method} started`, status: 'running',
      created_at: now, status_message: 'Starting...',
    }])
    const clearOptimistic = () => setOptimisticJobs([])

    if (method === 'finetune') {
      session.startFineTune({
        model: selectedModel || 'gpt2',
        dataset: datasets.selectedDataset || 'custom',
        epochs: trainingEpochs,
        batchSize: trainingBatchSize,
        lr: trainingLR,
        useLoRA,
      }, addToast, () => { clearOptimistic(); checkpoints.fetchJobs() })
    } else if (method === 'vlm') {
      session.startVisualTraining({
        dataset: datasets.selectedDataset,
        visionEncoder: visualVisionEncoder,
        llm: visualLLM,
        stage1Epochs: visualStage1Epochs,
        stage2Epochs: visualStage2Epochs,
        useLoRA,
      }, addToast, () => { clearOptimistic(); checkpoints.fetchJobs() })
    } else if (method === 'unified') {
      session.startUnifiedTraining({
        method: 'auto',
        dataset: datasets.selectedDataset,
        epochs: trainingEpochs,
        batchSize: trainingBatchSize,
        lr: trainingLR,
        distill: unifiedDistill,
        useLoRA,
        hfModel: selectedModel || undefined,
      }, addToast)
    } else {
      session.startSSETraining(body, addToast, () => {
        clearOptimistic(); checkpoints.fetchCheckpoints()
      })
    }
  }, [method, inputMode, textInput, algo, trainingEpochs, trainingLR, trainingBatchSize,
      selectedModel, useLoRA, unifiedDistill, datasets.selectedDataset, visualVisionEncoder, visualLLM,
      visualStage1Epochs, visualStage2Epochs, addToast, session, checkpoints])

  const startTurboTrain = useCallback(async () => {
    if (!datasets.selectedDataset) { addToast('Select a dataset first', 'error'); return }
    session.startTurboTrain(datasets.selectedDataset, {
      epochs: turboEpochs, lr: turboLR, embed: turboEmbed, heads: turboHeads, layers: turboLayers,
    }, addToast)
  }, [datasets.selectedDataset, turboEpochs, turboLR, turboEmbed, turboHeads, turboLayers, addToast, session])

  // ===== Quick Train logic =====
  const startQuickTrain = useCallback(async () => {
    if (!quickDataset) { addToast('Pick a dataset first', 'error'); return }
    setQuickTraining(true)
    setQuickComplete(false)
    setQuickStatusMessage('Setting up training...')
    try {
      const res = await trainingController.startQuick({ dataset: quickDataset })
      setQuickJobId(res.job_id)
      setQuickExplanation(res.explanation)
      setQuickStatusMessage('Training started — this may take a few minutes...')

      // Poll job status
      let retries = 0
      const poll = async () => {
        try {
          const jobs = await trainingController.list()
          const job = jobs.find((j: import('@/lib/training-controller').TrainingJob) => j.id === res.job_id)
          if (!job) return
          if (job.message) setQuickStatusMessage(job.message)
          if (job.status === 'completed') {
            setQuickTraining(false)
            setQuickComplete(true)
            setQuickExplanation(job.explanation || res.explanation)
            checkpoints.fetchCheckpoints()
            checkpoints.fetchJobs()
            addToast('Training complete!', 'success')
          } else if (job.status === 'failed') {
            setQuickTraining(false)
            addToast(job.message || 'Training failed', 'error')
          } else {
            retries = 0
            qtTimeoutRef.current = setTimeout(poll, 3000)
          }
        } catch {
          retries += 1
          if (retries >= 5) {
            setQuickTraining(false)
            addToast('Lost connection to training server', 'error')
            return
          }
          qtTimeoutRef.current = setTimeout(poll, 5000)
        }
      }
      qtTimeoutRef.current = setTimeout(poll, 2000)
    } catch (err: unknown) {
      setQuickTraining(false)
      addToast('Something went wrong starting training', 'error')
    }
  }, [quickDataset, addToast, checkpoints])

  useEffect(() => {
    if (quickDataset) {
      datasetController.getStats(quickDataset).then(setQuickStats).catch(() => setQuickStats(null))
    } else {
      setQuickStats(null)
    }
  }, [quickDataset])

  // ===== Feedback Training handlers =====
  const startFeedbackTrain = useCallback(async () => {
    setFeedbackTraining(true)
    setFeedbackComplete(false)
    setFeedbackError('')
    setFeedbackResult(null)
    try {
      const res = await trainingController.trainFromFeedback({ epochs: 3, batch_size: 16, use_lora: true })
      if (res.status === 'no_data') {
        setFeedbackTraining(false)
        addToast('No feedback data available for training', 'error')
        return
      }
      const jid = res.job_id
      if (!jid) {
        setFeedbackTraining(false)
        addToast('Failed to start feedback training', 'error')
        return
      }
      setFeedbackJobId(jid)
      setOptimisticJobs(prev => [{
        id: jid, name: 'Feedback Training', status: 'running', progress: 0,
        data_source: 'feedback', created_at: new Date().toISOString(),
      }, ...prev])

      // Poll job status
      let retries = 0
      const poll = async () => {
        try {
          const jobs = await trainingController.list()
          const job = jobs.find((j: import('@/lib/training-controller').TrainingJob) => j.id === jid)
          if (!job) return
          if (job.status === 'completed') {
            setFeedbackTraining(false)
            setFeedbackComplete(true)
            setFeedbackResult({
              model_path: job.checkpoint,
              final_loss: job.loss ?? job.train_loss,
              samples: (job as any).samples_used,
            })
            checkpoints.fetchCheckpoints()
            checkpoints.fetchJobs()
            addToast('Feedback training complete!', 'success')
            if (fbIntervalRef.current) { clearInterval(fbIntervalRef.current); fbIntervalRef.current = null }
          } else if (job.status === 'failed') {
            setFeedbackTraining(false)
            setFeedbackError(job.message || 'Training failed')
            addToast(job.message || 'Feedback training failed', 'error')
            if (fbIntervalRef.current) { clearInterval(fbIntervalRef.current); fbIntervalRef.current = null }
          }
        } catch { retries++ }
      }
      fbIntervalRef.current = setInterval(poll, 3000)
    } catch {
      setFeedbackTraining(false)
      addToast('Failed to start feedback training', 'error')
    }
  }, [addToast, checkpoints])

  // Clear optimistic jobs when training ends (success or error)
  useEffect(() => {
    if (session.phase === 'complete' || session.phase === 'error') {
      setOptimisticJobs([])
    }
  }, [session.phase])

  // Warn before leaving during active training
  useEffect(() => {
    if (!session.trainingRunning) return
    const onBeforeUnload = (e: BeforeUnloadEvent) => { e.preventDefault() }
    window.addEventListener('beforeunload', onBeforeUnload)
    return () => window.removeEventListener('beforeunload', onBeforeUnload)
  }, [session.trainingRunning])

  // Effects: cleanup on unmount
  useEffect(() => {
    return () => {
      if (qtTimeoutRef.current) { clearTimeout(qtTimeoutRef.current); qtTimeoutRef.current = null }
      if (fbIntervalRef.current) { clearInterval(fbIntervalRef.current); fbIntervalRef.current = null }
    }
  }, [])

  useEffect(() => {
    void datasets.fetchDatasets()
    void checkpoints.fetchCheckpoints()
    void checkpoints.fetchBuilds()
    void checkpoints.fetchJobs()
    const urlDataset = searchParams.get('dataset')
    if (urlDataset) {
      datasets.setSelectedDataset(urlDataset)
    } else if (!initialLoadDone.current && datasets.datasets.length > 0) {
      initialLoadDone.current = true
      datasets.setSelectedDataset(datasets.datasets[0].id)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Pause checkpoint polling when page is hidden
  useEffect(() => {
    const onVisibility = () => { visibilityRef.current = !document.hidden }
    document.addEventListener('visibilitychange', onVisibility)
    let id: ReturnType<typeof setInterval>
    const tick = () => {
      if (visibilityRef.current) void checkpoints.fetchCheckpoints()
    }
    id = setInterval(tick, 10000)
    return () => {
      clearInterval(id)
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [checkpoints])

  useEffect(() => {
    modelController.list().then(models => {
      const ids = models.map(m => m.id)
      setAvailableModels(ids)
      setSelectedModel((prev: string) => prev || ids[0] || '')
    }).catch(() => addToast('Could not load model list — training may be limited', 'info'))
  }, [addToast])

  useEffect(() => {
    if (datasets.selectedDataset && inputMode === 'dataset') {
      datasetController.preview(datasets.selectedDataset, 3).then(datasets.setDatasetPreview).catch(() => datasets.setDatasetPreview(null))
    } else {
      datasets.setDatasetPreview(null)
    }
  }, [datasets, inputMode])

  const runningJob = allJobs.find(j => j.status === 'running')
  const completedCount = allJobs.filter(j => j.status === 'completed').length

  const canStart = session.trainingRunning ||
    (inputMode === 'dataset' && !datasets.selectedDataset) ||
    (inputMode === 'text' && !textInput.trim()) ||
    (method === 'finetune' && !selectedModel)

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        className="items-start"
        left={<AppRouteHeaderLead title="Teach me" subtitle="Teach your agent from your data" />}
        right={
          <div className="flex items-center gap-2">
            <Button size="sm" variant="ghost" onClick={() => { void checkpoints.fetchJobs(); void checkpoints.fetchCheckpoints() }}>Refresh</Button>
          </div>
        }
      />

      <div className="space-y-4">
        {/* Stats */}
        <KpiGrid columns={4}>
          <StatCard label="Training runs" value={allJobs.length} />
          <StatCard label="Running" value={runningJob ? 1 : 0} />
          <StatCard label="Completed" value={completedCount} />
          <StatCard label="Saved versions" value={checkpoints.checkpoints.length} />
        </KpiGrid>

        {/* Quick Train — one-click training */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Quick Train</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-xs text-muted-foreground">
              Pick a dataset, hit train — we figure out the rest.
            </p>
            <div className="flex items-center gap-2">
              {datasets.datasets.length === 0 ? (
                <>
                  <span className="text-xs text-muted-foreground">No datasets yet — import one to get started.</span>
                  <Button size="sm" variant="outline" onClick={() => datasets.setImportModalOpen(true)}>
                    + Import
                  </Button>
                </>
              ) : (
              <>
                <select
                  value={quickDataset}
                  onChange={e => setQuickDataset(e.target.value)}
                  className="h-8 rounded-md border border-border/60 bg-background px-2 text-xs font-mono text-foreground flex-1 max-w-sm"
                  disabled={quickTraining}
                  aria-label="Dataset for quick training"
                >
                  <option value="">Select a dataset...</option>
                  {datasets.datasets.map(ds => (
                    <option key={ds.id} value={ds.id}>{datasetLabel(ds)}</option>
                  ))}
                </select>
                <Button size="sm" disabled={!quickDataset || quickTraining} onClick={startQuickTrain}>
                  {quickTraining ? 'Training...' : 'Start training'}
                </Button>
              </>
              )}
            </div>

            {/* Auto-config explanation */}
            {quickExplanation && !quickTraining && !quickComplete && (
              <p className="text-xs text-muted-foreground/80 italic">{quickExplanation}</p>
            )}
            {quickStats && !quickTraining && !quickComplete && (
              <>
                <p className="text-xs text-muted-foreground/60">
                  Detected: {quickStats.format} ({quickStats.samples} samples, {Math.round(quickStats.avg_length || 0)} avg chars)
                </p>
                {(quickStats.samples < 5 || (quickStats.avg_length || 0) < 50 || quickStats.format === 'text') && (
                  <div className="rounded border border-amber-500/20 bg-amber-500/5 p-2 space-y-1">
                    {quickStats.samples < 5 && (
                      <p className="text-[11px] text-amber-600 dark:text-amber-400">⚠ Very few samples — training may overfit. Add more data for better results.</p>
                    )}
                    {(quickStats.avg_length || 0) < 50 && quickStats.samples >= 5 && (
                      <p className="text-[11px] text-amber-600 dark:text-amber-400">⚠ Very short text — longer samples help the model learn patterns.</p>
                    )}
                    {quickStats.format === 'text' && (
                      <p className="text-[11px] text-muted-foreground">Tip: Conversation format (JSONL with messages) trains better than plain text.</p>
                    )}
                  </div>
                )}
              </>
            )}

            {/* Training progress */}
            {quickTraining && (
              <div className="space-y-2" role="status" aria-live="polite" aria-label="Training progress">
                <p className="text-sm text-muted-foreground">{quickStatusMessage}</p>
                <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-primary animate-pulse rounded-full" style={{ width: '40%' }} />
                </div>
              </div>
            )}

            {/* Completion */}
            {quickComplete && (
              <div className="rounded-lg border border-success/20 bg-success/5 p-3 space-y-2">
                <p className="text-sm font-medium text-success">Training complete!</p>
                {quickExplanation && <p className="text-xs text-muted-foreground">{quickExplanation}</p>}
                <div className="flex flex-wrap gap-2">
                  <Button size="sm" onClick={() => router.push('/chat')}>Try in chat</Button>
                  <Button size="sm" variant="ghost" onClick={() => { setQuickComplete(false); setQuickExplanation(''); setQuickDataset(''); setQuickStats(null) }}>
                    Train another
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Advanced: Start training (method, model, params) */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Train another way</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">

            {/* Progress display (during training) */}
            {session.trainingRunning && (() => {
              const elapsed = session.startTime ? Math.floor((Date.now() - session.startTime) / 1000) : 0
              const eta = session.progress > 0 ? Math.floor((elapsed / session.progress) * (100 - session.progress)) : 0
              const fmtTime = (s: number) => {
                if (s < 60) return `${s}s`
                if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`
                return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`
              }
              return (
                <div className="space-y-3 mb-4">
                  <div className="flex items-center gap-2 text-sm">
                    <span className="relative flex h-2 w-2 shrink-0">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60" />
                      <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
                    </span>
                    <span className="font-medium">Training in progress</span>
                    {session.totalEpochs > 0 && (
                      <span className="text-xs text-muted-foreground">
                        Epoch {session.epoch} of {session.totalEpochs}
                      </span>
                    )}
                    {session.loss != null && (
                      <span className="text-xs text-muted-foreground">
                        Loss: {session.loss.toFixed(4)}
                      </span>
                    )}
                    {elapsed > 5 && session.progress > 5 && (
                      <span className="text-xs text-muted-foreground">
                        {fmtTime(elapsed)} elapsed · ~{fmtTime(eta)} left
                      </span>
                    )}
                  </div>
                {session.message && (
                  <p className="text-xs text-muted-foreground/80">{session.message}</p>
                )}
                <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-primary transition-all duration-500 rounded-full" style={{ width: `${Math.min(session.progress, 100)}%` }} />
                </div>
                {session.lossHistory.length > 1 && (
                  <div className="h-12 w-full">
                    <LossChart
                      data={session.lossHistory.map(p => ({ step: p.step, value: p.loss, type: 'train' as const }))}
                      height={48}
                      showLegend={false}
                      live={session.trainingRunning}
                    />
                  </div>
                )}
                <Button size="sm" variant="outline" onClick={session.stopTraining}>Stop training</Button>
              </div>
              )
            })()}

            {/* Complete state */}
            {session.phase === 'complete' && !session.trainingRunning && (
              <div className="rounded-lg border border-success/20 bg-success/5 p-4 space-y-3 mb-4">
                <p className="text-sm font-medium text-success">Training complete!</p>
                {session.finetunedModelPath && (
                  <div className="text-xs text-muted-foreground space-y-1">
                    <p>Model: <span className="font-mono">{session.finetunedModelPath}</span></p>
                    {session.finetunedModelLoss != null && <p>Final loss: {session.finetunedModelLoss.toFixed(4)}</p>}
                  </div>
                )}
                {session.distillCheckpoint && (
                  <div className="text-xs text-muted-foreground space-y-1">
                    <p>Checkpoint: <span className="font-mono">{session.distillCheckpoint}</span></p>
                    {session.distillFinalLoss != null && <p>Final loss: {session.distillFinalLoss.toFixed(4)}</p>}
                    {session.distillEpochs != null && <p>Epochs: {session.distillEpochs}</p>}
                  </div>
                )}
                {session.unifiedModelPath && (
                  <div className="text-xs text-muted-foreground space-y-1">
                    <p>Model: <span className="font-mono">{session.unifiedModelPath}</span></p>
                    {session.unifiedFinalLoss != null && <p>Final loss: {session.unifiedFinalLoss.toFixed(4)}</p>}
                    {session.unifiedTotalSteps != null && <p>Steps: {session.unifiedTotalSteps}</p>}
                    {session.unifiedElapsed != null && <p>Time: {session.unifiedElapsed.toFixed(1)}s</p>}
                  </div>
                )}
                {session.visualOutputDir && (
                  <div className="text-xs text-muted-foreground space-y-1">
                    <p>Visual: <span className="font-mono">{session.visualOutputDir}</span></p>
                    {session.finetunedModelLoss != null && <p>Final loss: {session.finetunedModelLoss.toFixed(4)}</p>}
                  </div>
                )}
                {session.evalResult && (
                  <pre className="text-xs text-muted-foreground bg-muted/50 rounded p-2 max-h-40 overflow-y-auto">{session.evalResult}</pre>
                )}
                <div className="flex flex-wrap gap-2 pt-1">
                  <Button size="sm" onClick={() => test.setTestDialogOpen(true)}>
                    Test the model
                  </Button>
                  {session.finetunedModelPath && (
                    <Button size="sm" variant="outline" onClick={async () => {
                      setLoadingFinetunedModel(true)
                      try {
                        await modelController.loadModelPath(session.finetunedModelPath!)
                        addToast('Model loaded', 'success')
                      } catch {
                        addToast('Failed to load model', 'error')
                      } finally {
                        setLoadingFinetunedModel(false)
                      }
                    }} disabled={loadingFinetunedModel}>
                      {loadingFinetunedModel ? 'Loading...' : 'Load model for chat'}
                    </Button>
                  )}
                  {session.distillCheckpoint && (
                    <Button size="sm" variant="outline" onClick={async () => {
                      try {
                        await checkpoints.handleLoadCheckpoint(session.distillCheckpoint!, addToast)
                      } catch {
                        addToast('Failed to load trained version', 'error')
                      }
                    }}>
                      Load checkpoint
                    </Button>
                  )}
                  {session.unifiedModelPath && (
                    <Button size="sm" variant="outline" onClick={async () => {
                      setLoadingFinetunedModel(true)
                      try {
                        await modelController.loadModelPath(session.unifiedModelPath!)
                        addToast('Model loaded', 'success')
                      } catch {
                        addToast('Failed to load model', 'error')
                      } finally {
                        setLoadingFinetunedModel(false)
                      }
                    }} disabled={loadingFinetunedModel}>
                      {loadingFinetunedModel ? 'Loading...' : 'Load model for chat'}
                    </Button>
                  )}
                  {session.visualOutputDir && (
                    <Button size="sm" variant="outline" onClick={async () => {
                      setLoadingFinetunedModel(true)
                      try {
                        await modelController.loadVisualModel(session.visualOutputDir!)
                        addToast('Vision model loaded', 'success')
                      } catch {
                        addToast('Failed to load vision model', 'error')
                      } finally {
                        setLoadingFinetunedModel(false)
                      }
                    }} disabled={loadingFinetunedModel}>
                      {loadingFinetunedModel ? 'Loading...' : 'Load VLM for chat'}
                    </Button>
                  )}
                  <Button size="sm" variant="outline" onClick={() => router.push('/chat')}>
                    Try in chat
                  </Button>
                  <Button size="sm" variant="ghost" onClick={session.resetTraining}>
                    Train another
                  </Button>
                </div>
              </div>
            )}

            {/* Error state */}
            {session.phase === 'error' && !session.trainingRunning && (
              <div className="rounded-lg border border-destructive/20 bg-destructive/5 p-4 space-y-2 mb-4">
                <p className="text-sm font-medium text-destructive">Training failed</p>
                {session.message && <p className="text-xs text-muted-foreground">{session.message}</p>}
                <Button size="sm" variant="ghost" onClick={session.resetTraining}>
                  Try again
                </Button>
              </div>
            )}

            {/* Idle state — simple dataset picker + start button */}
            {!session.trainingRunning && !['complete', 'error'].includes(session.phase) && (
              <>
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <select
                      value={datasets.selectedDataset}
                      onChange={e => datasets.setSelectedDataset(e.target.value)}
                      className="h-8 rounded-md border border-border/60 bg-background px-2 text-xs font-mono text-foreground flex-1 max-w-sm"
                      aria-label="Dataset for fine-tuning"
                    >
                      {!datasets.selectedDataset && <option value="">Select a dataset...</option>}
                      {datasets.datasets.map(ds => (
                        <option key={ds.id} value={ds.id}>{datasetLabel(ds)}</option>
                      ))}
                    </select>
                    <Button size="sm" variant="outline" onClick={() => datasets.setImportModalOpen(true)}>
                      + Import
                    </Button>
                    <DatasetImportModal
                      open={datasets.importModalOpen}
                      onOpenChange={datasets.setImportModalOpen}
                      onImportComplete={(datasetId: string) => {
                        void datasets.fetchDatasets().then(() => datasets.setSelectedDataset(datasetId))
                      }}
                    />
                  </div>
                  {datasets.datasetPreview && datasets.datasetPreview.samples.length > 0 && (
                    <div className="rounded-md border border-border/40 bg-muted/30 p-3 text-xs">
                      <div className="font-medium text-muted-foreground mb-2">
                        Preview ({datasets.datasetPreview.total_samples} samples total)
                      </div>
                      <div className="space-y-1 font-mono text-muted-foreground">
                        {datasets.datasetPreview.samples.slice(0, 3).map((sample, i) => (
                          <div key={i} className="truncate">{sample.content}</div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-3">
                  <Button size="sm" disabled={canStart} onClick={() => startTraining()}>
                    {method === 'unified' ? 'Start unified training' : method === 'distill' ? (inputMode === 'text' && textInput.trim() ? 'Train on pasted text' : 'Start') : 'Start'}
                  </Button>
                  <EstimatedTime
                    method={method}
                    datasetId={datasets.selectedDataset}
                    datasets={datasets.datasets}
                    epochs={trainingEpochs}
                    batchSize={trainingBatchSize}
                    sampleCount={datasets.datasetPreview?.total_samples}
                  />
                </div>
              </>
            )}

            {/* Advanced settings */}
            {!session.trainingRunning && (
              <div className="border-t border-border/40 pt-3">
                <button
                  className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                >
                  {showAdvanced ? 'Hide' : 'Show'} advanced settings
                </button>
                {showAdvanced && (
                  <div className="space-y-3 mt-3">
                    {/* Method toggle */}
                    <div className="flex items-center gap-4" role="radiogroup" aria-label="Training method">
                      <div className="flex items-center gap-1 text-sm">
                        <button role="radio" aria-checked={method === 'distill'} className={cn('px-3 py-1.5 rounded-md transition-colors', method === 'distill' ? 'bg-primary text-primary-foreground font-medium' : 'text-muted-foreground hover:text-foreground')} onClick={() => setMethod('distill')}>Distill</button>
                        <button role="radio" aria-checked={method === 'finetune'} className={cn('px-3 py-1.5 rounded-md transition-colors', method === 'finetune' ? 'bg-primary text-primary-foreground font-medium' : 'text-muted-foreground hover:text-foreground')} onClick={() => setMethod('finetune')}>Continue training</button>
                        <button role="radio" aria-checked={method === 'vlm'} className={cn('px-3 py-1.5 rounded-md transition-colors', method === 'vlm' ? 'bg-primary text-primary-foreground font-medium' : 'text-muted-foreground hover:text-foreground')} onClick={() => setMethod('vlm')}>Vision model</button>
                        <button role="radio" aria-checked={method === 'unified'} className={cn('px-3 py-1.5 rounded-md transition-colors', method === 'unified' ? 'bg-primary text-primary-foreground font-medium' : 'text-muted-foreground hover:text-foreground')} onClick={() => setMethod('unified')}>Auto</button>
                      </div>
                      {method === 'distill' && <span className="text-xs text-muted-foreground/70">A large model teaches a smaller one to copy its style</span>}
                      {method === 'finetune' && <span className="text-xs text-muted-foreground/70">Continue training an existing model on new data</span>}
                      {method === 'vlm' && <span className="text-xs text-muted-foreground/70">Teach the AI to understand images and text</span>}
                      {method === 'unified' && <span className="text-xs text-muted-foreground/70">Automatically pick the best method</span>}
                    </div>

                    {/* Data source toggle */}
                    {method !== 'vlm' && method !== 'unified' && (
                      <div className="flex items-center gap-1 text-sm" role="radiogroup" aria-label="Data source">
                        <button role="radio" aria-checked={inputMode === 'dataset'} className={cn('px-3 py-1.5 rounded-md transition-colors', inputMode === 'dataset' ? 'bg-primary text-primary-foreground font-medium' : 'text-muted-foreground hover:text-foreground')} onClick={() => setInputMode('dataset')}>Use a dataset</button>
                        <button role="radio" aria-checked={inputMode === 'text'} className={cn('px-3 py-1.5 rounded-md transition-colors', inputMode === 'text' ? 'bg-primary text-primary-foreground font-medium' : 'text-muted-foreground hover:text-foreground')} onClick={() => setInputMode('text')}>Paste text</button>
                      </div>
                    )}

                    {/* Text input (advanced) */}
                    {inputMode === 'text' && method !== 'vlm' && method !== 'unified' && (
                      <div className="relative">
                        <textarea
                          value={textInput}
                          onChange={e => setTextInput(e.target.value)}
                          placeholder="Paste any text to train on — stories, docs, conversations, code..."
                          rows={4}
                          className="w-full rounded-md border border-border/60 bg-background p-3 pb-7 text-xs font-mono text-foreground resize-y min-h-[80px]"
                          aria-label="Training text input"
                        />
                        <span className="absolute bottom-1.5 right-2 text-[10px] text-muted-foreground/50 tabular-nums" aria-live="polite">
                          {textInput.length > 0 ? `${textInput.length.toLocaleString()} chars · ~${Math.ceil(textInput.length / 4)} tokens` : ''}
                        </span>
                      </div>
                    )}

                    {/* Model selector (for fine-tune / unified) */}
                    {(method === 'finetune' || method === 'unified') && availableModels.length > 0 && (
                      <div className="flex flex-col gap-1.5">
                        <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Base model</label>
                        <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} className="h-8 rounded-md border border-border/60 bg-background px-2 text-xs font-mono text-foreground max-w-sm" aria-label="Base model for fine-tuning">
                          {availableModels.map(id => <option key={id} value={id}>{id}</option>)}
                        </select>
                      </div>
                    )}

                    {/* VLM config */}
                    {method === 'vlm' && (
                      <div className="grid grid-cols-2 gap-3 p-3 rounded-lg border border-border/40 bg-muted/20">
                        <div className="flex flex-col gap-1">
                          <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Vision Encoder</label>
                          <select value={visualVisionEncoder} onChange={e => setVlmVisionEncoder(e.target.value)} className="h-7 rounded-md border border-border/60 bg-background px-2 text-[11px] font-mono" aria-label="Vision encoder model">
                            <option value="google/siglip-base-patch16-224">SigLIP Base (patch16)</option>
                            <option value="google/siglip-large-patch16-384">SigLIP Large (patch16)</option>
                            <option value="openai/clip-vit-base-patch32">CLIP ViT-B/32</option>
                          </select>
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Language Model</label>
                          <input value={visualLLM} onChange={e => setVlmLLM(e.target.value)} className="h-7 rounded-md border border-border/60 bg-background px-2 text-[11px] font-mono" placeholder="Qwen/Qwen2.5-0.5B-Instruct" />
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Stage 1 Epochs (connector)</label>
                          <input type="number" min={1} max={10} value={visualStage1Epochs} onChange={e => setVlmStage1Epochs(Number(e.target.value))} className="h-7 rounded-md border border-border/60 bg-background px-2 text-[11px] w-20" />
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Stage 2 Epochs (full)</label>
                          <input type="number" min={1} max={50} value={visualStage2Epochs} onChange={e => setVlmStage2Epochs(Number(e.target.value))} className="h-7 rounded-md border border-border/60 bg-background px-2 text-[11px] w-20" />
                        </div>
                      </div>
                    )}

                    {/* Numeric parameters */}
                    <div className="flex flex-wrap items-end gap-3">
                      {(method === 'distill' || method === 'unified') && availableModels.length > 0 && (
                        <div className="flex flex-col gap-1">
                          <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Base model</label>
                          <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} className="h-7 rounded-md border border-border/60 bg-background px-2 text-xs font-mono text-foreground" aria-label="Base model for distillation">
                            {availableModels.map(id => <option key={id} value={id}>{id}</option>)}
                          </select>
                        </div>
                      )}
                      <div className="flex flex-col gap-1">
                        <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Epochs</label>
                        <input type="number" min={1} max={200} value={trainingEpochs} onChange={e => setTrainingEpochs(Math.max(1, parseInt(e.target.value) || 1))} className="h-7 w-16 rounded-md border border-border/60 bg-background px-2 text-xs font-mono text-foreground" />
                      </div>
                      <div className="flex flex-col gap-1">
                        <label className="text-[10px] text-muted-foreground uppercase tracking-wider">LR</label>
                        <input type="number" min={1e-6} max={1} step={1e-4} value={trainingLR} onChange={e => setTrainingLR(parseFloat(e.target.value) || 1e-3)} className="h-7 w-20 rounded-md border border-border/60 bg-background px-2 text-xs font-mono text-foreground" />
                      </div>
                      <div className="flex flex-col gap-1">
                        <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Batch</label>
                        <input type="number" min={1} max={1024} value={trainingBatchSize} onChange={e => setTrainingBatchSize(Math.max(1, parseInt(e.target.value) || 64))} className="h-7 w-16 rounded-md border border-border/60 bg-background px-2 text-xs font-mono text-foreground" />
                      </div>
                      {(method === 'finetune' || method === 'unified') && (
                        <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground h-7">
                          <input type="checkbox" checked={useLoRA} onChange={e => setUseLoRA(e.target.checked)} className="rounded border-border/60" />
                          Advanced
                        </label>
                      )}
                      {method === 'unified' && (
                        <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground h-7">
                          <input type="checkbox" checked={unifiedDistill} onChange={e => setUnifiedDistill(e.target.checked)} className="rounded border-border/60" />
                          Distillation
                        </label>
                      )}
                      {method === 'distill' && (
                        <div className="flex flex-col gap-1">
                          <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Tokenizer</label>
                          <div className="flex items-center gap-1 text-[11px]" role="radiogroup" aria-label="Tokenizer algorithm">
                            <button role="radio" aria-checked={algo === 'bpe'} className={`px-1.5 py-0.5 rounded-sm ${algo === 'bpe' ? 'bg-muted font-medium text-foreground' : 'text-muted-foreground'}`} onClick={() => setAlgo('bpe')}>BPE</button>
                            <button role="radio" aria-checked={algo === 'unigram'} className={`px-1.5 py-0.5 rounded-sm ${algo === 'unigram' ? 'bg-muted font-medium text-foreground' : 'text-muted-foreground'}`} onClick={() => setAlgo('unigram')}>Unigram</button>
                          </div>
                        </div>
                      )}
                      {method === 'vlm' && datasets.datasets.filter(ds => ds.type === 'vlm').length === 0 && (
                        <p className="text-xs text-muted-foreground/70">
                          No VLM datasets found. Create one from the{' '}
                          <a href="/multimodal" className="text-primary underline underline-offset-2">Multimodal</a> page.
                        </p>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Turbo Train — encoder-decoder Transformer via torch shim */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Turbo Train</CardTitle>
          </CardHeader>
          <CardContent>
            {session.turboPhase === 'training' ? (
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-sm">
                  <span className="relative flex h-2 w-2 shrink-0">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60" />
                    <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
                  </span>
                  <span className="font-medium">Training encoder-decoder Transformer...</span>
                </div>
                <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-primary animate-pulse rounded-full" style={{ width: '60%' }} />
                </div>
                <p className="text-xs text-muted-foreground">Using our own torch shim — no PyTorch, no downloads</p>
              </div>
            ) : session.turboPhase === 'complete' && session.turboResult ? (
              <div className="rounded-lg border border-success/20 bg-success/5 p-4 space-y-3">
                <p className="text-sm font-medium text-success">Turbo training complete!</p>
                {session.turboResult!.final_loss != null && <p className="text-xs text-muted-foreground">Final loss: {session.turboResult!.final_loss.toFixed(4)}</p>}
                {session.turboResult!.total_steps != null && <p className="text-xs text-muted-foreground">Steps: {session.turboResult!.total_steps}</p>}
                {session.turboResult!.model_path && <p className="text-xs text-muted-foreground">Model: {session.turboResult!.model_path}</p>}
                <div className="flex gap-2">
                  <Button size="sm" variant="outline" onClick={() => session.setTurboPhase('idle')}>Train another</Button>
                  <Button size="sm" variant="ghost" onClick={async () => {
                    if (session.turboResult!.model_path) {
                      try {
                        await modelController.loadModelPath(session.turboResult!.model_path)
                        addToast('Model loaded for chat', 'success')
                      } catch { addToast('Failed to load model', 'error') }
                    }
                  }}>Load for chat</Button>
                </div>
              </div>
            ) : session.turboPhase === 'error' ? (
              <div className="rounded-lg border border-destructive/20 bg-destructive/5 p-4 space-y-3">
                <p className="text-sm font-medium text-destructive">Training failed</p>
                <p className="text-xs text-muted-foreground">{session.turboError || 'Unknown error'}</p>
                <Button size="sm" variant="outline" onClick={() => session.setTurboPhase('idle')}>Dismiss</Button>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  {datasets.datasets.length === 0 ? (
                    <>
                      <span className="text-xs text-muted-foreground">No datasets yet — import one to use Turbo Train.</span>
                      <Button size="sm" variant="outline" onClick={() => datasets.setImportModalOpen(true)}>+ Import</Button>
                    </>
                  ) : (
                  <>
                    <select value={datasets.selectedDataset} onChange={e => datasets.setSelectedDataset(e.target.value)}
                      className="h-8 rounded-md border border-border/60 bg-background px-2 text-xs font-mono text-foreground flex-1 max-w-xs"
                      aria-label="Dataset for turbo training">
                      {datasets.datasets.map(ds => <option key={ds.id} value={ds.id}>{ds.name}</option>)}
                    </select>
                    <Button size="sm" onClick={startTurboTrain} disabled={!datasets.selectedDataset || datasets.datasets.length === 0}>
                      Train with Turbo
                    </Button>
                  </>
                  )}
                </div>
                <p className="text-[11px] text-muted-foreground">
                  Encoder-decoder Transformer via our own torch shim. No PyTorch, no downloads, runs on CPU.
                  Saves model to <code className="text-xs">models/turbo-trained/</code>
                </p>
                <div className="border-t border-border/40 pt-2">
                  <button className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                    onClick={() => setShowTurboAdvanced(!showTurboAdvanced)}>
                    {showTurboAdvanced ? 'Hide' : 'Show'} advanced settings
                  </button>
                  {showTurboAdvanced && (
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-3">
                      <div className="space-y-1.5">
                        <label className="text-xs text-muted-foreground">Epochs</label>
                        <input type="number" value={turboEpochs} onChange={e => setTurboEpochs(Number(e.target.value))}
                          className="w-full h-8 rounded border border-border/50 bg-background px-2 text-xs" min={1} max={100} />
                      </div>
                      <div className="space-y-1.5">
                        <label className="text-xs text-muted-foreground">Learning rate</label>
                        <input type="number" value={turboLR} onChange={e => setTurboLR(Number(e.target.value))}
                          className="w-full h-8 rounded border border-border/50 bg-background px-2 text-xs" step={1e-5} min={1e-5} max={1} />
                      </div>
                      <div className="space-y-1.5">
                        <label className="text-xs text-muted-foreground">Embed dim</label>
                        <input type="number" value={turboEmbed} onChange={e => setTurboEmbed(Number(e.target.value))}
                          className="w-full h-8 rounded border border-border/50 bg-background px-2 text-xs" min={16} max={1024} />
                      </div>
                      <div className="space-y-1.5">
                        <label className="text-xs text-muted-foreground">Heads</label>
                        <input type="number" value={turboHeads} onChange={e => setTurboHeads(Number(e.target.value))}
                          className="w-full h-8 rounded border border-border/50 bg-background px-2 text-xs" min={1} max={64} />
                      </div>
                      <div className="space-y-1.5">
                        <label className="text-xs text-muted-foreground">Encoder/Decoder layers</label>
                        <input type="number" value={turboLayers} onChange={e => setTurboLayers(Number(e.target.value))}
                          className="w-full h-8 rounded border border-border/50 bg-background px-2 text-xs" min={1} max={24} />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Train from Feedback */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Train from Feedback</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {feedbackStats && (
              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                <span>{feedbackStats.total} feedback entries</span>
                <span className="text-success">{feedbackStats.thumbs_up} thumbs up</span>
                <span className="text-destructive">{feedbackStats.thumbs_down} thumbs down</span>
              </div>
            )}
            {!feedbackStats && !feedbackTraining && !feedbackComplete && !feedbackError && (
              <p className="text-xs text-muted-foreground">Loading feedback stats...</p>
            )}

            {feedbackError && (
              <div className="rounded-lg border border-destructive/20 bg-destructive/5 p-3 space-y-2">
                <p className="text-sm font-medium text-destructive">Training failed</p>
                <p className="text-xs text-muted-foreground">{feedbackError}</p>
                <Button size="sm" variant="outline" onClick={() => setFeedbackError('')}>Dismiss</Button>
              </div>
            )}

            {feedbackComplete && feedbackResult && (
              <div className="rounded-lg border border-success/20 bg-success/5 p-3 space-y-2">
                <p className="text-sm font-medium text-success">Feedback training complete!</p>
                {feedbackResult.final_loss != null && (
                  <p className="text-xs text-muted-foreground">Final loss: {feedbackResult.final_loss.toFixed(4)}</p>
                )}
                {feedbackResult.samples != null && (
                  <p className="text-xs text-muted-foreground">Samples used: {feedbackResult.samples}</p>
                )}
                <div className="flex gap-2">
                  <Button size="sm" variant="outline" onClick={async () => {
                    if (feedbackResult.model_path) {
                      try {
                        await modelController.loadModelPath(feedbackResult.model_path)
                        addToast('Model loaded for chat', 'success')
                      } catch { addToast('Failed to load model', 'error') }
                    }
                  }}>Load for chat</Button>
                  <Button size="sm" variant="ghost" onClick={() => {
                    setFeedbackComplete(false); setFeedbackResult(null)
                  }}>Dismiss</Button>
                </div>
              </div>
            )}

            {feedbackTraining && (
              <div className="space-y-2" role="status" aria-live="polite">
                <p className="text-sm text-muted-foreground">Training from feedback data...</p>
                <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-primary animate-pulse rounded-full" style={{ width: '50%' }} />
                </div>
              </div>
            )}

            {!feedbackTraining && !feedbackComplete && !feedbackError && (
              <Button size="sm" disabled={!feedbackStats || feedbackStats.total === 0} onClick={startFeedbackTrain}>
                {!feedbackStats ? 'Loading...' : feedbackStats.total === 0 ? 'No feedback data' : `Train on ${feedbackStats.total} entries`}
              </Button>
            )}
          </CardContent>
        </Card>

        {/* Checkpoints */}
        {(checkpoints.checkpoints.length > 0 || checkpoints.loadingCheckpoints) && (
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-base">Trained models</CardTitle>
              <Button size="sm" variant="ghost" onClick={() => test.setTestDialogOpen(true)}>
                Test model
              </Button>
            </CardHeader>
            <CardContent>
              {checkpoints.loadingCheckpoints && checkpoints.checkpoints.length === 0 ? (
                loadingTimedOut ? (
                  <div className="py-6 text-center space-y-2">
                    <p className="text-sm text-muted-foreground">Taking longer than expected</p>
                    <Button size="sm" variant="ghost" onClick={() => { setLoadingTimedOut(false); void checkpoints.fetchCheckpoints() }}>
                      Retry
                    </Button>
                  </div>
                ) : (
                <div className="grid gap-2 sm:grid-cols-2">
                  {[1,2].map(i => (
                    <div key={i} className="flex items-center justify-between rounded-lg border border-border/50 p-3">
                      <div className="space-y-1.5 flex-1">
                        <Skeleton className="h-4 w-40" />
                        <Skeleton className="h-3 w-28" />
                      </div>
                      <Skeleton className="h-5 w-12 rounded" />
                    </div>
                  ))}
                </div>
                )
              ) : (
              <div className="grid gap-2 sm:grid-cols-2">
                {checkpoints.checkpoints.slice().reverse().map((cp: any) => (
                  <div key={cp.name} className={cn("flex items-center justify-between rounded-lg border p-3 text-sm", checkpoints.activeCheckpoint === cp.name ? "border-primary/30 bg-primary/5" : "border-border/50")}>
                    <div className="min-w-0 flex-1">
                      <p className="truncate font-medium text-xs">{cp.name}</p>
                      {cp.description ? (
                        <p className="text-[11px] text-muted-foreground mt-0.5">{cp.description}</p>
                      ) : (
                        <div className="flex flex-wrap gap-x-2 gap-y-0.5 text-[11px] text-muted-foreground mt-0.5">
                          {cp.loss != null && <span>Loss: {cp.loss.toFixed(4)}</span>}
                          {cp.epochs_trained != null && <span>{cp.epochs_trained} epochs</span>}
                          {cp.training_dataset && cp.training_dataset !== 'gpt2-generated' && <span>Dataset: {cp.training_dataset}</span>}
                        </div>
                      )}
                    </div>
                    <div className="flex items-center gap-1 shrink-0 ml-2">
                      {checkpoints.activeCheckpoint === cp.name ? (
                        <span className="text-xs px-2 py-0.5 rounded-full bg-primary/10 text-primary font-medium">Active</span>
                      ) : (
                        <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={() => checkpoints.handleLoadCheckpoint(cp.name, addToast)}>Load</Button>
                      )}
                      <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={() => startTraining(cp.name)}>Continue</Button>
                      <Button size="sm" variant="ghost" className="h-6 text-xs text-destructive hover:text-destructive" onClick={() => checkpoints.handleDeleteCheckpoint(cp.name, addToast)}>Del</Button>
                    </div>
                  </div>
                ))}
              </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Builds */}
        {(checkpoints.builds.length > 0 || checkpoints.loadingBuilds) && (
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-base">Builds</CardTitle>
              <Button size="sm" variant="ghost" onClick={() => checkpoints.fetchBuilds()}>
                Refresh
              </Button>
            </CardHeader>
            <CardContent className="p-0">
              {checkpoints.loadingBuilds ? (
                loadingTimedOut ? (
                  <div className="px-4 py-6 text-center space-y-2">
                    <p className="text-sm text-muted-foreground">Taking longer than expected</p>
                    <Button size="sm" variant="ghost" onClick={() => { setLoadingTimedOut(false); void checkpoints.fetchBuilds() }}>
                      Retry
                    </Button>
                  </div>
                ) : (
                <div className="divide-y divide-border/50">
                  {[1,2].map(i => (
                    <div key={i} className="flex items-center justify-between px-4 py-3">
                      <div className="space-y-1.5 flex-1">
                        <Skeleton className="h-4 w-56" />
                        <Skeleton className="h-3 w-40" />
                      </div>
                      <Skeleton className="h-5 w-14 rounded" />
                    </div>
                  ))}
                </div>
                )
              ) : (
              <div className="divide-y divide-border/50">
                {checkpoints.builds.slice().reverse().map((b, i) => (
                  <div key={`${b.build_type}-${b.name}-${i}`} className="flex items-center justify-between px-4 py-3 text-sm">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="truncate font-medium">{b.name}</span>
                        <span className={cn(
                          'text-[10px] uppercase px-1.5 py-0.5 rounded font-medium',
                          b.build_type === 'auto-train' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300' : '',
                          b.build_type === 'hf-finetune' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300' : '',
                          b.build_type === 'hf-finetuned-dir' ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300' : '',
                          b.build_type === 'lora' ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300' : '',
                          b.build_type === 'vlm' ? 'bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300' : '',
                        )}>
                          {b.build_type === 'auto-train' ? 'Auto' : b.build_type === 'hf-finetune' ? 'HF' : b.build_type === 'hf-finetuned-dir' ? 'Dir' : b.build_type === 'vlm' ? 'Visual' : 'Adv'}
                        </span>
                      </div>
                      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-muted-foreground mt-0.5">
                        {b.model && <span>Model: {b.model}</span>}
                        {b.dataset && <span>Dataset: {b.dataset}</span>}
                        {b.loss != null && <span>Loss: {Number(b.loss).toFixed(4)}</span>}
                        {b.epochs != null && <span>{b.epochs} epochs</span>}
                        {b.size_mb != null && <span>{(b.size_mb).toFixed(1)} MB</span>}
                        {b.training_dataset && b.training_dataset !== 'gpt2-generated' && <span>Data: {b.training_dataset}</span>}
                      </div>
                    </div>
                    <div className="flex items-center gap-1 shrink-0 ml-2">
                      {b.build_type === 'auto-train' && checkpoints.checkpoints.find(cp => cp.name === b.name) && (
                        checkpoints.checkpoints.find(cp => cp.name === b.name)!.name === checkpoints.activeCheckpoint
                          ? <span className="text-xs px-2 py-0.5 rounded-full bg-primary/10 text-primary">Active</span>
                          : <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={() => checkpoints.handleLoadCheckpoint(b.name, addToast)}>Load</Button>
                      )}
                      {b.build_type === 'vlm' && (
                        <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={async () => {
                          try { await modelController.loadVisualModel(b.model_path!); addToast(`Vision model loaded: ${b.name}`, 'success') }
                          catch { addToast('Failed to load vision model', 'error') }
                        }}>
                          Visual Chat
                        </Button>
                      )}
                      {b.model_path && b.build_type !== 'vlm' && (
                        <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={async () => {
                          try {
                            await modelController.loadModelPath(b.model_path!)
                            addToast(`Loaded ${b.name}`, 'success')
                          } catch { addToast('Failed to load model', 'error') }
                        }}>
                          Use
                        </Button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Job history */}
        {(allJobs.length > 0 || checkpoints.loadingJobs) && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Job history</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              {checkpoints.loadingJobs ? (
                loadingTimedOut ? (
                  <div className="px-4 py-6 text-center space-y-2">
                    <p className="text-sm text-muted-foreground">Taking longer than expected</p>
                    <Button size="sm" variant="ghost" onClick={() => { setLoadingTimedOut(false); void checkpoints.fetchJobs() }}>
                      Retry
                    </Button>
                  </div>
                ) : (
                <div className="divide-y divide-border/50">
                  {[1,2,3].map(i => (
                    <div key={i} className="flex items-center justify-between px-4 py-3">
                      <div className="space-y-1.5 flex-1">
                        <Skeleton className="h-4 w-48" />
                        <Skeleton className="h-3 w-32" />
                      </div>
                      <Skeleton className="h-5 w-12 rounded-full" />
                    </div>
                  ))}
                </div>
                )
              ) : (
              <div className="divide-y divide-border/50">
                {allJobs.slice().reverse().map((job) => {
                  const relativeTime = (() => {
                    if (!job.created_at) return ''
                    const diff = Date.now() - new Date(job.created_at).getTime()
                    const mins = Math.floor(diff / 60000)
                    if (mins < 1) return 'just now'
                    if (mins < 60) return `${mins}m ago`
                    const hrs = Math.floor(mins / 60)
                    if (hrs < 24) return `${hrs}h ago`
                    return `${Math.floor(hrs / 24)}d ago`
                  })()
                  return (
                  <div key={job.id} role="button" tabIndex={0} className="flex items-center justify-between px-4 py-3 text-sm cursor-pointer hover:bg-muted/20 transition-colors" onClick={() => router.push(`/training/job/${job.id}`)} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); router.push(`/training/job/${job.id}`) } }} aria-label={`View job ${job.name || job.id}`}>
                    <div className="min-w-0 flex-1">
                      <p className="truncate font-medium">{job.name || job.id}</p>
                      {job.status_message ? (
                        <p className="text-xs text-muted-foreground mt-0.5">{job.status_message}</p>
                      ) : (
                        <p className="text-xs text-muted-foreground mt-0.5">{job.status} &middot; {relativeTime}</p>
                      )}
                    </div>
                    <div className="flex items-center gap-2 shrink-0 ml-3" onClick={e => e.stopPropagation()}>
                      {job.status === 'running' && (
                        <>
                          <span className="relative flex h-2 w-2"><span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-success/60" /><span className="relative inline-flex h-2 w-2 rounded-full bg-success" /></span>
                          <Button size="sm" variant="ghost" className="h-6 text-xs text-destructive hover:text-destructive" onClick={async () => {
                            try { await trainingController.stop(job.id); addToast('Stopped', 'info'); void checkpoints.fetchJobs() }
                            catch { addToast('Failed to stop job', 'error') }
                          }}>
                            Stop
                          </Button>
                        </>
                      )}
                      {job.status === 'completed' && (
                        <>
                          {job.checkpoint && (
                            <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={async () => {
                              try { await checkpoints.handleLoadCheckpoint(job.checkpoint!, addToast) }
                              catch { addToast('Failed to load trained version', 'error') }
                            }}>
                              Use
                            </Button>
                          )}
                          <span className="text-xs text-success shrink-0">Done</span>
                        </>
                      )}
                      {job.status === 'failed' && <span className="text-xs text-destructive shrink-0">Failed</span>}
                      <Button size="sm" variant="ghost" className="h-6 text-xs text-muted-foreground hover:text-destructive" onClick={async () => {
                        if (!confirm(`Delete job "${job.name || job.id}"?`)) return
                        try {
                          await trainingController.delete(job.id)
                          addToast('Job deleted', 'info')
                          void checkpoints.fetchJobs()
                        } catch { addToast('Failed to delete job', 'error') }
                      }}>
                        Delete
                      </Button>
                    </div>
                  </div>
                  )
                })}
              </div>
              )}
            </CardContent>
          </Card>
        )}

        {!session.trainingRunning && !checkpoints.loadingCheckpoints && checkpoints.checkpoints.length === 0 && allJobs.length === 0 && (
          <Card className="border-dashed py-8">
            <CardContent className="text-center space-y-3">
              <p className="text-sm text-muted-foreground">No training activity yet.</p>
              <div className="text-xs text-muted-foreground/70 space-y-1.5 max-w-sm mx-auto text-left">
                <p>• Pick a dataset above and click <span className="font-medium text-foreground">Start training</span></p>
                <p>• Use <span className="font-medium text-foreground">Paste text</span> to train on your own content</p>
                <p>• Conversation-format data (JSONL) trains better than plain text</p>
                <p>• Small datasets (5-10 samples) work for personality fine-tuning</p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      <TestModelDialog
        open={test.testDialogOpen}
        prompt={test.testPrompt}
        output={test.testOutput}
        loading={test.testLoading}
        onClose={() => test.setTestDialogOpen(false)}
        onPromptChange={test.setTestPrompt}
        onGenerate={test.handleTestModel}
        onClear={test.clearTest}
      />
    </div>
  )
}
