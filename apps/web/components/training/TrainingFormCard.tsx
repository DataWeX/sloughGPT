'use client'

import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { useState, useRef } from 'react'
import { extractErrorMessage } from '@/lib/error-utils'
import { getJsonItem } from '@/lib/format-bytes'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button, Input } from '@sloughgpt/strui'
import { DatasetSelector } from '@/components/training/DatasetSelector'
import { TrainingErrorBanner } from '@/components/training/TrainingStatus'
import dynamic from 'next/dynamic'
import { downloadJson } from '@/lib/download-utils'

const LossChart = dynamic(() => import('@/components/training/LossChart').then(m => m.LossChart), { ssr: false })
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { Switch } from '@sloughgpt/strui'
import { ToggleGroup, ToggleGroupItem } from '@sloughgpt/strui'
import { modelController, trainingJobsController } from '@/lib/controllers'
import { useToastStore } from '@/lib/toast-store'
import type { Dataset } from '@/lib/dataset-controller'
import type { TrainingFormState, NativePreset } from '@/hooks/useTrainingForm'
import { NATIVE_PRESETS } from '@/hooks/useTrainingForm'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'
import type { UseTrainingSessionReturn } from '@/hooks/useTrainingSession'
import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'

const PRESETS_KEY = 'sloughgpt-training-presets'

interface SavedPreset {
  id: string
  name: string
  method: string
  epochs: number
  batchSize: number
  lr: number
  useLoRA: boolean
}

function loadPresets(): SavedPreset[] {
  return getJsonItem<SavedPreset[]>(PRESETS_KEY, [])
}

function savePresets(presets: SavedPreset[]) {
  localStorage.setItem(PRESETS_KEY, JSON.stringify(presets))
}

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

export function TrainingFormCard({
  form,
  datasets,
  session,
  checkpoints,
  onTest,
}: {
  form: TrainingFormState
  datasets: UseTrainingDatasetsReturn
  session: UseTrainingSessionReturn
  checkpoints: UseTrainingCheckpointsReturn
  onTest: () => void
}) {
  const router = useRouter()
  const addToast = useToastStore(s => s.addToast)
  const [customPresets, setCustomPresets] = useState<SavedPreset[]>([])
  const [presetName, setPresetName] = useState('')

  const applyPreset = (p: SavedPreset) => {
    form.setMethod(p.method as 'distill' | 'finetune' | 'vlm')
    form.setTrainingEpochs(p.epochs)
    form.setTrainingBatchSize(p.batchSize)
    form.setTrainingLR(p.lr)
    form.setUseLoRA(p.useLoRA)
    form.setShowAdvanced(p.useLoRA || p.batchSize < 32)
    addToast(`Applied "${p.name}"`, 'info')
  }

  const saveCurrentAsPreset = () => {
    const name = presetName.trim()
    if (!name) return addToast('Enter a preset name', 'error')
    const preset: SavedPreset = {
      id: Date.now().toString(36),
      name,
      method: form.method,
      epochs: form.trainingEpochs,
      batchSize: form.trainingBatchSize,
      lr: form.trainingLR,
      useLoRA: form.useLoRA,
    }
    const updated = [...customPresets, preset]
    setCustomPresets(updated)
    savePresets(updated)
    setPresetName('')
    addToast(`Saved preset "${name}"`, 'success')
  }

  const deletePreset = (id: string) => {
    const updated = customPresets.filter(p => p.id !== id)
    setCustomPresets(updated)
    savePresets(updated)
  }

  const exportPresets = () => {
    if (customPresets.length === 0) return addToast('No custom presets to export', 'error')
    downloadJson(customPresets, 'training-presets.json')
    addToast(`Exported ${customPresets.length} preset(s)`, 'success')
  }

  const importRef = useRef<HTMLInputElement>(null)

  const importPresets = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      try {
        const data = JSON.parse(reader.result as string)
        if (!Array.isArray(data)) throw new Error('Invalid format')
        const valid = data.filter((p: SavedPreset): p is SavedPreset =>
          typeof p === 'object' && p !== null && typeof p.id === 'string' && typeof p.name === 'string' && typeof p.method === 'string'
        )
        if (valid.length === 0) throw new Error('No valid presets found')
        const merged = [...customPresets]
        for (const preset of valid) {
          if (!merged.some(p => p.id === preset.id)) merged.push(preset)
        }
        setCustomPresets(merged)
        savePresets(merged)
        addToast(`Imported ${valid.length} preset(s)`, 'success')
      } catch (err) {
        addToast(extractErrorMessage(err, 'Import failed'), 'error')
      }
    }
    reader.readAsText(file)
    e.target.value = ''
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Train</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">

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
                  <span className="text-xs text-muted-foreground">Epoch {session.epoch} of {session.totalEpochs}</span>
                )}
                {session.loss != null && (
                  <span className="text-xs text-muted-foreground">Loss: {session.loss.toFixed(4)}</span>
                )}
                {elapsed > 5 && session.progress > 5 && (
                  <span className="text-xs text-muted-foreground">{fmtTime(elapsed)} elapsed · ~{fmtTime(eta)} left</span>
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
              {session.paused && (
                <div className="text-xs text-warning flex items-center gap-1.5 mt-1">
                  <span className="inline-block h-1.5 w-1.5 rounded-full bg-warning" />
                  Training paused
                </div>
              )}
              <div className="flex gap-2 pt-1">
                {session.paused ? (
                  <Button size="sm" variant="outline" onClick={session.resumeTraining}>Resume</Button>
                ) : (
                  <Button size="sm" variant="outline" onClick={session.pauseTraining}>Pause</Button>
                )}
                <Button size="sm" variant="outline" onClick={session.stopTraining}>Stop</Button>
              </div>
            </div>
          )
        })()}

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
              <Button size="sm" onClick={onTest}>Test the model</Button>
              {session.finetunedModelPath && (
                <Button size="sm" variant="outline" onClick={async () => {
                  form.setLoadingFinetunedModel(true)
                  try {
                    const path = session.finetunedModelPath
                    const name = path?.split('/').pop() || path || ''
                    await trainingJobsController.loadFineTuned(name)
                    addToast('Model loaded', 'success')
                  } catch {
                    addToast('Failed to load model', 'error')
                  } finally {
                    form.setLoadingFinetunedModel(false)
                  }
                }} disabled={form.loadingFinetunedModel}>
                  {form.loadingFinetunedModel ? 'Loading...' : 'Load model for chat'}
                </Button>
              )}
              {session.distillCheckpoint && (
                <Button size="sm" variant="outline" onClick={async () => {
                  try {
                    const cp = session.distillCheckpoint
                    if (cp) await checkpoints.handleLoadCheckpoint(cp, addToast)
                  } catch {
                    addToast('Failed to load trained version', 'error')
                  }
                }}>
                  Load checkpoint
                </Button>
              )}
              {session.visualOutputDir && (
                <Button size="sm" variant="outline" onClick={async () => {
                  form.setLoadingFinetunedModel(true)
                  try {
                    await modelController.loadVisualModel(session.visualOutputDir!)
                    addToast('Vision model loaded', 'success')
                  } catch {
                    addToast('Failed to load vision model', 'error')
                  } finally {
                    form.setLoadingFinetunedModel(false)
                  }
                }} disabled={form.loadingFinetunedModel}>
                  {form.loadingFinetunedModel ? 'Loading...' : 'Load VLM for chat'}
                </Button>
              )}
              <Button size="sm" variant="outline" onClick={() => router.push('/chat')}>Try in chat</Button>
              <Button size="sm" variant="ghost" onClick={session.resetTraining}>Train another</Button>
            </div>
          </div>
        )}

        {session.phase === 'error' && !session.trainingRunning && (
          <TrainingErrorBanner
            error={session.message || 'Training failed'}
            onRetry={session.resetTraining}
            onDismiss={session.resetTraining}
          />
        )}

        {!session.trainingRunning && !['complete', 'error'].includes(session.phase) && (
          <>
            <DatasetSelector
              datasets={datasets}
              value={datasets.selectedDataset}
              onChange={datasets.setSelectedDataset}
              showImport
            />
            {datasets.datasetPreview && datasets.datasetPreview.samples.length > 0 && (
              <div className="rounded-md border border-border/40 bg-muted/30 p-3 text-xs">
                <div className="font-medium text-muted-foreground mb-2">
                  Dataset preview
                </div>
                <div className="grid grid-cols-3 gap-2 mb-2">
                  <div>
                    <span className="text-muted-foreground/60">Samples:</span>{' '}
                    <span className="font-mono">{(datasets.datasetPreview.total_samples ?? 0).toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground/60">Characters:</span>{' '}
                    <span className="font-mono">{(datasets.datasetPreview.total_chars ?? 0).toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground/60">Avg length:</span>{' '}
                    <span className="font-mono">{(datasets.datasetPreview.total_samples ?? 0) > 0 ? Math.round((datasets.datasetPreview.total_chars ?? 0) / (datasets.datasetPreview.total_samples ?? 1)).toLocaleString() : 0}</span>
                  </div>
                </div>
                {datasets.datasetPreview.languages && Object.keys(datasets.datasetPreview.languages).length > 0 && (
                  <div className="mb-2">
                    <span className="text-muted-foreground/60">Languages:</span>{' '}
                    {Object.entries(datasets.datasetPreview.languages).map(([lang, count], i, arr) => (
                      <span key={lang} className="font-mono">{lang} ({count}){i < arr.length - 1 ? ', ' : ''}</span>
                    ))}
                  </div>
                )}
                {datasets.datasetPreview.total_samples < 5 && (
                  <div className="text-warning text-[11px] mb-1">Very small dataset — consider adding more samples for better results.</div>
                )}
                <div className="space-y-1 font-mono text-muted-foreground mt-2 border-t border-border/30 pt-2">
                  {datasets.datasetPreview.samples.slice(0, 3).map((sample, i) => (
                    <div key={i} className="truncate">{sample.content}</div>
                  ))}
                </div>
              </div>
            )}
            <div className="flex items-center gap-3">
              <Button size="sm" disabled={!form.canStart} onClick={() => form.startTraining()}>
                  {form.method === 'distill' ? (form.inputMode === 'text' && form.textInput.trim() ? 'Train on pasted text' : 'Start training') : form.method === 'vlm' ? 'Start vision training' : 'Start training'}
              </Button>
              <EstimatedTime
                method={form.method}
                datasetId={datasets.selectedDataset}
                datasets={datasets.datasets}
                epochs={form.trainingEpochs}
                batchSize={form.trainingBatchSize}
                sampleCount={datasets.datasetPreview?.total_samples}
              />
            </div>
          </>
        )}

        {!session.trainingRunning && (
          <div className="border-t border-border/40 pt-3">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Quick presets</span>
            </div>
            <div className="flex gap-2 mb-2">
              <button
                onClick={() => { form.setMethod('distill'); form.setTrainingEpochs(3); form.setTrainingBatchSize(64); form.setTrainingLR(1e-3); form.setShowAdvanced(false) }}
                className="flex-1 rounded-md border border-border/60 bg-muted/30 px-3 py-2 text-left hover:bg-muted/60 transition-colors"
              >
                <p className="text-xs font-medium">Fast test</p>
                <p className="text-[10px] text-muted-foreground">3 epochs, ~2 min</p>
              </button>
              <button
                onClick={() => { form.setMethod('distill'); form.setTrainingEpochs(10); form.setTrainingBatchSize(32); form.setTrainingLR(5e-4); form.setShowAdvanced(false) }}
                className="flex-1 rounded-md border border-border/60 bg-muted/30 px-3 py-2 text-left hover:bg-muted/60 transition-colors"
              >
                <p className="text-xs font-medium">Balanced</p>
                <p className="text-[10px] text-muted-foreground">10 epochs, ~8 min</p>
              </button>
              <button
                onClick={() => { form.setMethod('finetune'); form.setTrainingEpochs(50); form.setTrainingBatchSize(16); form.setTrainingLR(2e-5); form.setUseLoRA(true); form.setShowAdvanced(true) }}
                className="flex-1 rounded-md border border-border/60 bg-muted/30 px-3 py-2 text-left hover:bg-muted/60 transition-colors"
              >
                <p className="text-xs font-medium">Thorough</p>
                <p className="text-[10px] text-muted-foreground">50 epochs + LoRA</p>
              </button>
              <button
                onClick={() => { form.setMethod('native'); form.setTrainingEpochs(5); form.setTrainingBatchSize(8); form.setTrainingLR(1e-3); form.setNativeEmbed(128); form.setNativeLayers(4); form.setNativeHeads(4); form.setNativeBlockSize(128); form.setShowAdvanced(true) }}
                className="flex-1 rounded-md border border-border/60 bg-muted/30 px-3 py-2 text-left hover:bg-muted/60 transition-colors"
              >
                <p className="text-xs font-medium">Native</p>
                <p className="text-[10px] text-muted-foreground">128d×4L transformer</p>
              </button>
            </div>
            {customPresets.length > 0 && (
              <div className="flex flex-wrap gap-1.5 mb-2">
                {customPresets.map(p => (
                  <div key={p.id} className="group flex items-center gap-1 rounded-md border border-primary/20 bg-primary/5 px-2 py-1">
                    <button type="button" onClick={() => applyPreset(p)} className="text-[11px] font-medium text-primary hover:underline">
                      {p.name}
                    </button>
                    <button type="button" onClick={() => deletePreset(p.id)} className="text-[10px] text-muted-foreground hover:text-destructive opacity-0 group-hover:opacity-100 transition-opacity">×</button>
                  </div>
                ))}
              </div>
            )}
            <div className="flex items-center gap-2">
              <Input
                value={presetName}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setPresetName(e.target.value)}
                placeholder="Save current settings as..."
                className="h-7 text-xs flex-1"
                onKeyDown={(e: React.KeyboardEvent) => { if (e.key === 'Enter') saveCurrentAsPreset() }}
              />
              <Button variant="outline" size="sm" className="h-7 text-xs" onClick={saveCurrentAsPreset} disabled={!presetName.trim()}>
                Save
              </Button>
              <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={exportPresets} disabled={customPresets.length === 0}>
                Export
              </Button>
              <input ref={importRef} type="file" accept=".json" className="hidden" onChange={importPresets} />
              <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => importRef.current?.click()}>
                Import
              </Button>
            </div>
            <button
              className="text-xs text-muted-foreground hover:text-foreground transition-colors mt-2"
              onClick={() => form.setShowAdvanced(!form.showAdvanced)}
            >
              {form.showAdvanced ? 'Hide' : 'Show'} advanced settings
            </button>
            {form.showAdvanced && (
              <div className="space-y-3 mt-3">
                <div className="flex items-center gap-4" role="radiogroup" aria-label="Training method">
                  <ToggleGroup type="single" value={form.method} onValueChange={(v) => { if (v) form.setMethod(v as 'distill' | 'finetune' | 'vlm' | 'native') }}>
                    <ToggleGroupItem value="distill" className="px-3 py-1.5 rounded-md text-sm data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium">Train from scratch</ToggleGroupItem>
                    <ToggleGroupItem value="finetune" className="px-3 py-1.5 rounded-md text-sm data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium">Continue training</ToggleGroupItem>
                    <ToggleGroupItem value="native" className="px-3 py-1.5 rounded-md text-sm data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium">Native SloNet</ToggleGroupItem>
                    <ToggleGroupItem value="vlm" className="px-3 py-1.5 rounded-md text-sm data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium">Vision model</ToggleGroupItem>
                  </ToggleGroup>
                  {form.method === 'distill' && <span className="text-xs text-muted-foreground/70">Train a small model from text data — no teacher needed</span>}
                  {form.method === 'finetune' && <span className="text-xs text-muted-foreground/70">Continue training an existing model on new data</span>}
                  {form.method === 'native' && <span className="text-xs text-muted-foreground/70">Train a pure transformer from scratch — SloNet architecture, own tokenizer</span>}
                  {form.method === 'vlm' && <span className="text-xs text-muted-foreground/70">Teach the AI to understand images and text</span>}
                </div>

                {form.method !== 'vlm' && (
                  <div className="flex items-center gap-1 text-sm" role="radiogroup" aria-label="Data source">
                    <ToggleGroup type="single" value={form.inputMode} onValueChange={(v) => { if (v) form.setInputMode(v as 'dataset' | 'text') }}>
                      <ToggleGroupItem value="dataset" className="px-3 py-1.5 rounded-md text-sm data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium">Use a dataset</ToggleGroupItem>
                      <ToggleGroupItem value="text" className="px-3 py-1.5 rounded-md text-sm data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium">Paste text</ToggleGroupItem>
                    </ToggleGroup>
                  </div>
                )}

                {form.inputMode === 'text' && form.method !== 'vlm' && (
                  <div className="relative">
                    <textarea
                      value={form.textInput}
                      onChange={e => form.setTextInput(e.target.value)}
                      placeholder="Paste any text to train on — stories, docs, conversations, code..."
                      rows={4}
                      className="w-full rounded-md border border-border/60 bg-background p-3 pb-7 text-xs font-mono text-foreground resize-y min-h-[80px]"
                      aria-label="Training text input"
                    />
                    <span className="absolute bottom-1.5 right-2 text-[10px] text-muted-foreground/50 tabular-nums" aria-live="polite">
                      {form.textInput.length > 0 ? `${form.textInput.length.toLocaleString()} chars · ~${Math.ceil(form.textInput.length / 4)} tokens` : ''}
                    </span>
                  </div>
                )}

                {(form.method === 'finetune') && form.availableModels.length > 0 && (
                  <div className="flex flex-col gap-1.5">
                    <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Base model</label>
                    <Select value={form.selectedModel} onValueChange={form.setSelectedModel}>
                      <SelectTrigger className="h-8 text-xs font-mono max-w-sm" aria-label="Base model for fine-tuning">
                        <SelectValue placeholder="Select model..." />
                      </SelectTrigger>
                      <SelectContent>
                        {form.availableModels.map(id => <SelectItem key={id} value={id}>{id}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </div>
                )}

                {form.method === 'vlm' && (
                  <div className="grid grid-cols-2 gap-3 p-3 rounded-lg border border-border/40 bg-muted/20">
                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Vision Encoder</label>
                      <Select value={form.visualVisionEncoder} onValueChange={form.setVlmVisionEncoder}>
                        <SelectTrigger className="h-7 text-[11px] font-mono" aria-label="Vision encoder model">
                          <SelectValue placeholder="Select encoder..." />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="google/siglip-base-patch16-224">SigLIP Base (patch16)</SelectItem>
                          <SelectItem value="google/siglip-large-patch16-384">SigLIP Large (patch16)</SelectItem>
                          <SelectItem value="openai/clip-vit-base-patch32">CLIP ViT-B/32</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Language Model</label>
                      <input value={form.visualLLM} onChange={e => form.setVlmLLM(e.target.value)} className="h-7 rounded-md border border-border/60 bg-background px-2 text-[11px] font-mono" placeholder="Qwen/Qwen2.5-0.5B-Instruct" />
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Stage 1 Epochs (connector)</label>
                      <input type="number" min={1} max={10} value={form.visualStage1Epochs} onChange={e => form.setVlmStage1Epochs(Number(e.target.value))} className="h-7 rounded-md border border-border/60 bg-background px-2 text-[11px] w-20" />
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Stage 2 Epochs (full)</label>
                      <input type="number" min={1} max={50} value={form.visualStage2Epochs} onChange={e => form.setVlmStage2Epochs(Number(e.target.value))} className="h-7 rounded-md border border-border/60 bg-background px-2 text-[11px] w-20" />
                    </div>
                  </div>
                )}

                {form.method === 'native' && (
                  <div className="grid grid-cols-2 gap-3 p-3 rounded-lg border border-border/40 bg-muted/20">
                    <div className="col-span-2 flex flex-wrap gap-1.5">
                      {NATIVE_PRESETS.map(p => (
                        <button
                          key={p.name}
                          type="button"
                          onClick={() => form.applyPreset(p)}
                          className="px-2 py-0.5 rounded text-[10px] font-medium border border-border/60 bg-background hover:bg-primary/10 hover:text-primary transition-colors"
                          title={`${p.description} (${p.params})`}
                        >
                          {p.name} {p.params}
                        </button>
                      ))}
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Embed dim</label>
                      <input type="number" min={16} max={1024} step={16} value={form.nativeEmbed} onChange={e => form.setNativeEmbed(Number(e.target.value))} className="h-7 rounded-md border border-border/60 bg-background px-2 text-[11px] font-mono" aria-label="Embedding dimension" />
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Layers</label>
                      <input type="number" min={1} max={24} value={form.nativeLayers} onChange={e => form.setNativeLayers(Number(e.target.value))} className="h-7 rounded-md border border-border/60 bg-background px-2 text-[11px] font-mono" aria-label="Number of layers" />
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Heads</label>
                      <input type="number" min={1} max={64} value={form.nativeHeads} onChange={e => form.setNativeHeads(Number(e.target.value))} className="h-7 rounded-md border border-border/60 bg-background px-2 text-[11px] font-mono" aria-label="Number of attention heads" />
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Block size</label>
                      <input type="number" min={8} max={2048} step={8} value={form.nativeBlockSize} onChange={e => form.setNativeBlockSize(Number(e.target.value))} className="h-7 rounded-md border border-border/60 bg-background px-2 text-[11px] font-mono" aria-label="Context block size" />
                    </div>
                    <div className="col-span-2 text-[10px] text-muted-foreground/60">
                      Params: ~{Math.round(form.nativeEmbed * form.nativeLayers * 12 * (1 + 4 * form.nativeEmbed / 1024) / 1000)}K — saves to models/slonet-native/
                    </div>
                  </div>
                )}

                <div className="flex flex-wrap items-end gap-3">
                  {form.method === 'distill' && form.availableModels.length > 0 && (
                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Base model</label>
                      <Select value={form.selectedModel} onValueChange={form.setSelectedModel}>
                        <SelectTrigger className="h-7 text-xs font-mono" aria-label="Base model for distillation">
                          <SelectValue placeholder="Select model..." />
                        </SelectTrigger>
                        <SelectContent>
                          {form.availableModels.map(id => <SelectItem key={id} value={id}>{id}</SelectItem>)}
                        </SelectContent>
                      </Select>
                    </div>
                  )}
                  <div className="flex flex-col gap-1">
                    <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Epochs</label>
                    <input type="number" min={1} max={200} value={form.trainingEpochs} onChange={e => form.setTrainingEpochs(Math.max(1, parseInt(e.target.value) || 1))} className="h-7 w-16 rounded-md border border-border/60 bg-background px-2 text-xs font-mono text-foreground" aria-label="Training epochs" />
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-[10px] text-muted-foreground uppercase tracking-wider">LR</label>
                    <input type="number" min={1e-6} max={1} step={1e-4} value={form.trainingLR} onChange={e => form.setTrainingLR(parseFloat(e.target.value) || 1e-3)} className="h-7 w-20 rounded-md border border-border/60 bg-background px-2 text-xs font-mono text-foreground" aria-label="Learning rate" />
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Batch</label>
                    <input type="number" min={1} max={1024} value={form.trainingBatchSize} onChange={e => form.setTrainingBatchSize(Math.max(1, parseInt(e.target.value) || 64))} className="h-7 w-16 rounded-md border border-border/60 bg-background px-2 text-xs font-mono text-foreground" aria-label="Batch size" />
                  </div>
                  {(form.method === 'finetune') && (
                    <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground h-7">
                      <Switch checked={form.useLoRA} onCheckedChange={form.setUseLoRA} />
                      Advanced
                    </label>
                  )}
                  {form.method === 'distill' && (
                    <div className="flex flex-col gap-1">
                      <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Tokenizer</label>
                      <div className="flex items-center gap-1 text-[11px]" role="radiogroup" aria-label="Tokenizer algorithm">
                        <ToggleGroup type="single" value={form.algo} onValueChange={(v) => { if (v) form.setAlgo(v as 'bpe' | 'unigram') }}>
                          <ToggleGroupItem value="bpe" className="px-1.5 py-0.5 rounded-sm text-[11px] data-[state=on]:bg-muted data-[state=on]:font-medium data-[state=on]:text-foreground">BPE</ToggleGroupItem>
                          <ToggleGroupItem value="unigram" className="px-1.5 py-0.5 rounded-sm text-[11px] data-[state=on]:bg-muted data-[state=on]:font-medium data-[state=on]:text-foreground">Unigram</ToggleGroupItem>
                        </ToggleGroup>
                      </div>
                    </div>
                  )}
                  {form.method === 'vlm' && datasets.datasets.filter(ds => ds.type === 'vlm').length === 0 && (
                    <p className="text-xs text-muted-foreground/70">
                      No VLM datasets found. Create one from the{' '}
                      <Link href="/multimodal" className="text-primary underline underline-offset-2">Multimodal</Link> page.
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
