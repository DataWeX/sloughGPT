'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { StatCard, KpiGrid } from '@/components/strui'
import { useToastStore } from '@/lib/toast-store'
import { modelController } from '@/lib/controllers'
import { datasetController } from '@/lib/controllers'
import type { Dataset } from '@/lib/dataset-controller'
import { DatasetImportModal } from '@/components/DatasetImportModal'
import { cn } from '@/lib/cn'
import { formatSize } from '@/lib/chat-utils'
import { TestModelDialog } from '@/components/training/TestModelDialog'
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

export default function TrainingPage() {
  const addToast = useToastStore(s => s.addToast)
  const initialLoadDone = useRef(false)

  const session = useTrainingSession()
  const datasets = useTrainingDatasets(addToast)
  const checkpoints = useTrainingCheckpoints()
  const test = useTestDialog()

  // ===== Config =====
  const [method, setMethod] = useState<Method>('distill')
  const [inputMode, setInputMode] = useState<InputMode>('dataset')
  const [textInput, setTextInput] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [algo, setAlgo] = useState('bpe')
  const [unifiedDistill, setUnifiedDistill] = useState(false)
  const [trainingEpochs, setTrainingEpochs] = useState(5)
  const [trainingLR, setTrainingLR] = useState(1e-3)
  const [trainingBatchSize, setTrainingBatchSize] = useState(64)
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [useLoRA, setUseLoRA] = useState(true)

  // ===== VLM config =====
  const [vlmVisionEncoder, setVlmVisionEncoder] = useState('google/siglip-base-patch16-224')
  const [vlmLLM, setVlmLLM] = useState('Qwen/Qwen2.5-0.5B-Instruct')
  const [vlmStage1Epochs, setVlmStage1Epochs] = useState(1)
  const [vlmStage2Epochs, setVlmStage2Epochs] = useState(2)

  // ===== Turbo config =====
  const [turboEpochs, setTurboEpochs] = useState(3)
  const [turboLR, setTurboLR] = useState(3e-4)
  const [turboEmbed, setTurboEmbed] = useState(128)
  const [turboHeads, setTurboHeads] = useState(4)
  const [turboLayers, setTurboLayers] = useState(3)

  // ===== Fine-tuned model loading =====
  const [loadingFinetunedModel, setLoadingFinetunedModel] = useState(false)

  const startTraining = useCallback(async (checkpointName?: string) => {
    const hasDataset = inputMode === 'dataset' && datasets.selectedDataset
    const hasText = inputMode === 'text' && textInput.trim()

    if (!hasDataset && !hasText && !checkpointName) {
      addToast('Select a dataset or paste text to train on', 'error'); return
    }

    if ((method === 'finetune' || method === 'unified') && !hasDataset) {
      addToast(`${method === 'unified' ? 'Unified training' : 'Fine-tune'} requires a dataset.`, 'error'); return
    }

    if (method === 'vlm' && !hasDataset) {
      addToast('VLM training requires a dataset with image-text pairs', 'error'); return
    }

    const body: Record<string, unknown> = { algo, epochs: trainingEpochs, learning_rate: trainingLR }
    if (trainingBatchSize) body.batch_size = trainingBatchSize
    if (checkpointName) body.checkpoint_name = checkpointName
    if (hasDataset) body.dataset_id = datasets.selectedDataset
    if (hasText) body.source_text = textInput.trim()

    if (method === 'finetune') {
      session.startFineTune({
        model: selectedModel || 'gpt2',
        dataset: datasets.selectedDataset || 'custom',
        epochs: trainingEpochs,
        batchSize: trainingBatchSize,
        lr: trainingLR,
        useLoRA,
      }, addToast, () => { checkpoints.fetchJobs() })
    } else if (method === 'vlm') {
      session.startVLMTraining({
        dataset: datasets.selectedDataset,
        visionEncoder: vlmVisionEncoder,
        llm: vlmLLM,
        stage1Epochs: vlmStage1Epochs,
        stage2Epochs: vlmStage2Epochs,
        useLoRA,
      }, addToast, () => { checkpoints.fetchJobs() })
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
      session.startSSETraining(body, addToast, () => { checkpoints.fetchCheckpoints() })
    }
  }, [method, inputMode, textInput, algo, trainingEpochs, trainingLR, trainingBatchSize,
      selectedModel, useLoRA, unifiedDistill, datasets.selectedDataset, vlmVisionEncoder, vlmLLM,
      vlmStage1Epochs, vlmStage2Epochs, addToast, session, checkpoints])

  const startTurboTrain = useCallback(async () => {
    if (!datasets.selectedDataset) { addToast('Select a dataset first', 'error'); return }
    session.startTurboTrain(datasets.selectedDataset, {
      epochs: turboEpochs, lr: turboLR, embed: turboEmbed, heads: turboHeads, layers: turboLayers,
    }, addToast)
  }, [datasets.selectedDataset, turboEpochs, turboLR, turboEmbed, turboHeads, turboLayers, addToast, session])

  // Effects
  useEffect(() => { return () => { /* esRef cleanup handled in hook */ } }, [])

  useEffect(() => {
    void datasets.fetchDatasets()
    void checkpoints.fetchCheckpoints()
    void checkpoints.fetchBuilds()
    void checkpoints.fetchJobs()
    if (!initialLoadDone.current && datasets.datasets.length > 0) {
      initialLoadDone.current = true
      datasets.setSelectedDataset(datasets.datasets[0].id)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const id = setInterval(() => void checkpoints.fetchCheckpoints(), 10000)
    return () => clearInterval(id)
  }, [checkpoints.fetchCheckpoints])

  useEffect(() => {
    modelController.list().then(models => {
      const ids = models.map(m => m.id)
      setAvailableModels(ids)
      setSelectedModel(prev => prev || ids[0] || '')
    }).catch(() => {})
  }, [])

  useEffect(() => {
    if (datasets.selectedDataset && inputMode === 'dataset') {
      datasetController.preview(datasets.selectedDataset, 3).then(datasets.setDatasetPreview).catch(() => datasets.setDatasetPreview(null))
    } else {
      datasets.setDatasetPreview(null)
    }
  }, [datasets.selectedDataset, inputMode, datasets.setDatasetPreview])

  const runningJob = checkpoints.jobs.find(j => j.status === 'running')
  const completedCount = checkpoints.jobs.filter(j => j.status === 'completed').length

  const canStart = session.trainingRunning ||
    (inputMode === 'dataset' && !datasets.selectedDataset) ||
    (inputMode === 'text' && !textInput.trim()) ||
    (method === 'finetune' && !selectedModel)

  return (
    <div className="sl-page mx-auto max-w-6xl">
      <AppRouteHeader
        className="items-start"
        left={<AppRouteHeaderLead title="Training" subtitle="Teach the model from your data — just a click away" />}
        right={
          <div className="flex items-center gap-2">
            <Button size="sm" variant="ghost" onClick={() => { void checkpoints.fetchJobs(); void checkpoints.fetchCheckpoints() }}>Refresh</Button>
          </div>
        }
      />

      <div className="space-y-4">
        {/* Stats */}
        <KpiGrid columns={4}>
          <StatCard label="Total Jobs" value={checkpoints.jobs.length} />
          <StatCard label="Running" value={runningJob ? 1 : 0} />
          <StatCard label="Completed" value={completedCount} />
          <StatCard label="Checkpoints" value={checkpoints.checkpoints.length} />
        </KpiGrid>

        {/* Start training */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Start training</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">

            {/* Progress display (during training) */}
            {session.trainingRunning && (
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
                </div>
                {session.message && (
                  <p className="text-xs text-muted-foreground/80">{session.message}</p>
                )}
                <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-primary transition-all duration-500 rounded-full" style={{ width: `${Math.min(session.progress, 100)}%` }} />
                </div>
                {session.lossHistory.length > 1 && (() => {
                  const maxLoss = Math.max(...session.lossHistory.map(l => l.loss)) || 1
                  const h = 28, w = 200
                  const path = session.lossHistory.map((p, i) => {
                    const x = (i / (session.lossHistory.length - 1)) * w
                    const y = h - ((p.loss / maxLoss) * (h - 2)) - 1
                    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
                  }).join(' ')
                  return (
                    <div className="h-12 w-full">
                      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-full" preserveAspectRatio="none">
                        <path d={path} fill="none" stroke="hsl(var(--primary))" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </div>
                  )
                })()}
                <Button size="sm" variant="outline" onClick={session.stopTraining}>Stop training</Button>
              </div>
            )}

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
                {session.vlmOutputDir && (
                  <div className="text-xs text-muted-foreground space-y-1">
                    <p>VLM: <span className="font-mono">{session.vlmOutputDir}</span></p>
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
                        addToast('Fine-tuned model loaded', 'success')
                      } catch {
                        addToast('Failed to load fine-tuned model', 'error')
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
                        addToast('Failed to load checkpoint', 'error')
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
                        addToast('Unified model loaded', 'success')
                      } catch {
                        addToast('Failed to load unified model', 'error')
                      } finally {
                        setLoadingFinetunedModel(false)
                      }
                    }} disabled={loadingFinetunedModel}>
                      {loadingFinetunedModel ? 'Loading...' : 'Load model for chat'}
                    </Button>
                  )}
                  {session.vlmOutputDir && (
                    <Button size="sm" variant="outline" onClick={async () => {
                      setLoadingFinetunedModel(true)
                      try {
                        await modelController.loadVLM(session.vlmOutputDir!)
                        addToast('VLM loaded for chat', 'success')
                      } catch {
                        addToast('Failed to load VLM', 'error')
                      } finally {
                        setLoadingFinetunedModel(false)
                      }
                    }} disabled={loadingFinetunedModel}>
                      {loadingFinetunedModel ? 'Loading...' : 'Load VLM for chat'}
                    </Button>
                  )}
                  <Button size="sm" variant="outline" onClick={() => window.location.href = '/chat'}>
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

            {/* Idle state */}
            {!session.trainingRunning && !['complete', 'error'].includes(session.phase) && (
              <>
                {/* Method toggle */}
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-1 text-sm">
                    <button
                      className={cn('px-3 py-1.5 rounded-md transition-colors', method === 'distill' ? 'bg-primary text-primary-foreground font-medium' : 'text-muted-foreground hover:text-foreground')}
                      onClick={() => setMethod('distill')}
                    >
                      Distill
                    </button>
                    <button
                      className={cn('px-3 py-1.5 rounded-md transition-colors', method === 'finetune' ? 'bg-primary text-primary-foreground font-medium' : 'text-muted-foreground hover:text-foreground')}
                      onClick={() => setMethod('finetune')}
                    >
                      Fine-tune
                    </button>
                    <button
                      className={cn('px-3 py-1.5 rounded-md transition-colors', method === 'vlm' ? 'bg-primary text-primary-foreground font-medium' : 'text-muted-foreground hover:text-foreground')}
                      onClick={() => setMethod('vlm')}
                    >
                      VLM
                    </button>
                    <button
                      className={cn('px-3 py-1.5 rounded-md transition-colors', method === 'unified' ? 'bg-primary text-primary-foreground font-medium' : 'text-muted-foreground hover:text-foreground')}
                      onClick={() => setMethod('unified')}
                    >
                      Unified
                    </button>
                  </div>
                  {method === 'distill' && (
                    <span className="text-xs text-muted-foreground/70">Teacher model distills into a compact student</span>
                  )}
                  {method === 'finetune' && (
                    <span className="text-xs text-muted-foreground/70">Continue training an existing model on new data</span>
                  )}
                  {method === 'vlm' && (
                    <span className="text-xs text-muted-foreground/70">Vision + Language model (SigLIP encoder + LLM)</span>
                  )}
                  {method === 'unified' && (
                    <span className="text-xs text-muted-foreground/70">Composite pipeline: auto-detect method, optional distillation</span>
                  )}
                </div>

                {/* Model selector (for fine-tune / unified) */}
                {(method === 'finetune' || method === 'unified') && availableModels.length > 0 && (
                  <div className="flex flex-col gap-1.5">
                    <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Base model</label>
                    <select
                      value={selectedModel}
                      onChange={e => setSelectedModel(e.target.value)}
                      className="h-8 rounded-md border border-border/60 bg-background px-2 text-xs font-mono text-foreground max-w-sm"
                    >
                      {availableModels.map(id => (
                        <option key={id} value={id}>{id}</option>
                      ))}
                    </select>
                  </div>
                )}

                {/* VLM config */}
                {method === 'vlm' && (
                  <div className="grid grid-cols-2 gap-3 p-3 rounded-lg border border-border/40 bg-muted/20">
                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Vision Encoder</label>
                      <select
                        value={vlmVisionEncoder}
                        onChange={e => setVlmVisionEncoder(e.target.value)}
                        className="h-7 rounded-md border border-border/60 bg-background px-2 text-[11px] font-mono"
                      >
                        <option value="google/siglip-base-patch16-224">SigLIP Base (patch16)</option>
                        <option value="google/siglip-large-patch16-384">SigLIP Large (patch16)</option>
                        <option value="openai/clip-vit-base-patch32">CLIP ViT-B/32</option>
                      </select>
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Language Model</label>
                      <input
                        value={vlmLLM}
                        onChange={e => setVlmLLM(e.target.value)}
                        className="h-7 rounded-md border border-border/60 bg-background px-2 text-[11px] font-mono"
                        placeholder="Qwen/Qwen2.5-0.5B-Instruct"
                      />
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Stage 1 Epochs (connector)</label>
                      <input type="number" min={1} max={10}
                        value={vlmStage1Epochs}
                        onChange={e => setVlmStage1Epochs(Number(e.target.value))}
                        className="h-7 rounded-md border border-border/60 bg-background px-2 text-[11px] w-20"
                      />
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Stage 2 Epochs (full)</label>
                      <input type="number" min={1} max={50}
                        value={vlmStage2Epochs}
                        onChange={e => setVlmStage2Epochs(Number(e.target.value))}
                        className="h-7 rounded-md border border-border/60 bg-background px-2 text-[11px] w-20"
                      />
                    </div>
                  </div>
                )}

                {/* Data source toggle */}
                {method !== 'vlm' && method !== 'unified' && (
                  <div className="flex items-center gap-1 text-sm">
                    <button
                      className={cn('px-3 py-1.5 rounded-md transition-colors', inputMode === 'dataset' ? 'bg-primary text-primary-foreground font-medium' : 'text-muted-foreground hover:text-foreground')}
                      onClick={() => setInputMode('dataset')}
                    >
                      Use a dataset
                    </button>
                    <button
                      className={cn('px-3 py-1.5 rounded-md transition-colors', inputMode === 'text' ? 'bg-primary text-primary-foreground font-medium' : 'text-muted-foreground hover:text-foreground')}
                      onClick={() => setInputMode('text')}
                    >
                      Paste text
                    </button>
                  </div>
                )}

                {/* Dataset picker */}
                {inputMode === 'dataset' && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <select
                        value={datasets.selectedDataset}
                        onChange={e => datasets.setSelectedDataset(e.target.value)}
                        className="h-8 rounded-md border border-border/60 bg-background px-2 text-xs font-mono text-foreground flex-1 max-w-sm"
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
                )}

                {/* Text input */}
                {inputMode === 'text' && (
                  <div className="space-y-2">
                    <textarea
                      value={textInput}
                      onChange={e => setTextInput(e.target.value)}
                      placeholder="Paste any text to train on — stories, docs, conversations, code..."
                      rows={6}
                      className="w-full rounded-md border border-border/60 bg-background p-3 text-xs font-mono text-foreground resize-y min-h-[100px]"
                    />
                    <label className="inline-flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer hover:text-foreground transition-colors">
                      <input
                        type="file"
                        accept=".txt,.md,.json,.jsonl"
                        className="hidden"
                        onChange={e => {
                          const file = e.target.files?.[0]
                          if (!file) return
                          const reader = new FileReader()
                          reader.onload = () => {
                            const text = reader.result as string
                            setTextInput(prev => prev + (prev ? '\n\n' : '') + text)
                            addToast(`Loaded ${file.name} (${(text.length / 1024).toFixed(1)} KB)`, 'success')
                          }
                          reader.readAsText(file)
                          e.target.value = ''
                        }}
                      />
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5 5-5M12 15V3" /></svg>
                      Upload a file
                    </label>
                  </div>
                )}

                {/* Start button */}
                <div className="flex items-center gap-3">
                  <Button size="sm" disabled={canStart} onClick={() => startTraining()}>
                    {method === 'unified' ? 'Start unified training' : method === 'distill' ? (inputMode === 'text' && textInput.trim() ? 'Train on pasted text' : 'Start distill') : 'Start fine-tune'}
                  </Button>
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
                  <div className="flex flex-wrap items-end gap-3 mt-3">
                    {(method === 'distill' || method === 'unified') && availableModels.length > 0 && (
                      <div className="flex flex-col gap-1">
                        <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Base model</label>
                        <select
                          value={selectedModel}
                          onChange={e => setSelectedModel(e.target.value)}
                          className="h-7 rounded-md border border-border/60 bg-background px-2 text-xs font-mono text-foreground"
                        >
                          {availableModels.map(id => (
                            <option key={id} value={id}>{id}</option>
                          ))}
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
                        LoRA
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
                        <div className="flex items-center gap-1 text-[11px]">
                          <button className={`px-1.5 py-0.5 rounded-sm ${algo === 'bpe' ? 'bg-muted font-medium text-foreground' : 'text-muted-foreground'}`} onClick={() => setAlgo('bpe')}>BPE</button>
                          <button className={`px-1.5 py-0.5 rounded-sm ${algo === 'unigram' ? 'bg-muted font-medium text-foreground' : 'text-muted-foreground'}`} onClick={() => setAlgo('unigram')}>Unigram</button>
                        </div>
                      </div>
                    )}
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
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
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
                  <div className="space-y-1.5">
                    <label className="text-xs text-muted-foreground">Dataset</label>
                    <select value={datasets.selectedDataset} onChange={e => datasets.setSelectedDataset(e.target.value)}
                      className="w-full h-8 rounded border border-border/50 bg-background px-2 text-xs">
                      {datasets.datasets.length === 0 && <option value="">No datasets</option>}
                      {datasets.datasets.map(ds => <option key={ds.id} value={ds.id}>{ds.name}</option>)}
                    </select>
                  </div>
                </div>
                <Button size="sm" onClick={startTurboTrain} disabled={!datasets.selectedDataset || datasets.datasets.length === 0}>
                  Train with Turbo
                </Button>
                <p className="text-[11px] text-muted-foreground">
                  Encoder-decoder Transformer via our own torch shim. No PyTorch, no downloads, runs on CPU.
                  Saves model to <code className="text-xs">models/turbo-trained/</code>
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Checkpoints */}
        {checkpoints.checkpoints.length > 0 && (
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-base">Checkpoints</CardTitle>
              <Button size="sm" variant="ghost" onClick={() => test.setTestDialogOpen(true)}>
                Test model
              </Button>
            </CardHeader>
            <CardContent>
              <div className="grid gap-2 sm:grid-cols-2">
                {checkpoints.checkpoints.slice().reverse().map((cp: any) => (
                  <div key={cp.name} className={cn("flex items-center justify-between rounded-lg border p-3 text-sm", checkpoints.activeCheckpoint === cp.name ? "border-primary/30 bg-primary/5" : "border-border/50")}>
                    <div className="min-w-0 flex-1">
                      <p className="truncate font-mono text-xs font-medium">{cp.name}</p>
                      <div className="flex flex-wrap gap-x-2 gap-y-0.5 text-[11px] text-muted-foreground mt-0.5">
                        {cp.loss != null && <span>Loss: {cp.loss.toFixed(4)}</span>}
                        {cp.epochs_trained != null && <span>{cp.epochs_trained} epochs</span>}
                        {cp.training_dataset && cp.training_dataset !== 'gpt2-generated' && <span>Dataset: {cp.training_dataset}</span>}
                        {cp.model_type && <span>{cp.model_type}</span>}
                        {cp.vocab_size != null && <span>Vocab: {cp.vocab_size}</span>}
                      </div>
                      {cp.traits && Object.keys(cp.traits).length > 0 && (
                        <p className="text-xs text-muted-foreground mt-0.5">Traits: {Object.entries(cp.traits).map(([k, v]) => `${k}: ${v}`).join(', ')}</p>
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
            </CardContent>
          </Card>
        )}

        {/* Builds */}
        {checkpoints.builds.length > 0 && (
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-base">Builds</CardTitle>
              <Button size="sm" variant="ghost" onClick={() => checkpoints.fetchBuilds()}>
                Refresh
              </Button>
            </CardHeader>
            <CardContent className="p-0">
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
                          {b.build_type === 'auto-train' ? 'Auto' : b.build_type === 'hf-finetune' ? 'HF' : b.build_type === 'hf-finetuned-dir' ? 'Dir' : b.build_type === 'vlm' ? 'VLM' : 'LoRA'}
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
                          try { await modelController.loadVLM(b.model_path!); addToast(`VLM loaded: ${b.name}`, 'success') }
                          catch { addToast('Failed to load VLM', 'error') }
                        }}>
                          VLM Chat
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
            </CardContent>
          </Card>
        )}

        {/* Job history */}
        {checkpoints.jobs.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Job history</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <div className="divide-y divide-border/50">
                {checkpoints.jobs.slice().reverse().map((job) => (
                  <div key={job.id} className="flex items-center justify-between px-4 py-3 text-sm">
                    <div className="min-w-0 flex-1">
                      <p className="truncate font-medium">{job.name}</p>
                      <p className="text-xs text-muted-foreground mt-0.5">{job.status} &middot; {job.created_at ? new Date(job.created_at).toLocaleDateString() : ''}</p>
                    </div>
                    {job.status === 'running' && <span className="relative flex h-2 w-2 shrink-0"><span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-success/60" /><span className="relative inline-flex h-2 w-2 rounded-full bg-success" /></span>}
                    {job.status === 'completed' && (
                      <div className="flex items-center gap-2 shrink-0">
                        {job.type === 'vlm' && job.output_dir && (
                          <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={async () => {
                            try { await modelController.loadVLM(job.model_path || job.output_dir); addToast('VLM loaded for chat', 'success') }
                            catch { addToast('Failed to load VLM', 'error') }
                          }}>
                            VLM Chat
                          </Button>
                        )}
                        <span className="text-xs text-success">Done</span>
                      </div>
                    )}
                    {job.status === 'failed' && <span className="text-xs text-destructive shrink-0">Failed</span>}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {!session.trainingRunning && checkpoints.checkpoints.length === 0 && checkpoints.jobs.length === 0 && (
          <Card className="border-dashed py-8">
            <CardContent className="text-center text-sm text-muted-foreground">
              No training activity yet. Select a dataset or paste text above to get started.
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
