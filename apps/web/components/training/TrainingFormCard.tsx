'use client'

import { useRouter } from 'next/navigation'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { DatasetImportModal } from '@/components/DatasetImportModal'
import dynamic from 'next/dynamic'

const LossChart = dynamic(() => import('@/components/training/LossChart').then(m => m.LossChart), { ssr: false })
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { modelController } from '@/lib/controllers'
import { useToastStore } from '@/lib/toast-store'
import { cn } from '@/lib/cn'
import type { Dataset } from '@/lib/dataset-controller'
import type { TrainingFormState, Method } from '@/hooks/useTrainingForm'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'
import type { UseTrainingSessionReturn } from '@/hooks/useTrainingSession'
import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'

function datasetLabel(ds: Dataset): string {
  const size = ds.size != null ? `${(ds.size / 1024).toFixed(1)} KB` : ''
  if (ds.type === 'vlm' && ds.vlm_metadata) {
    return `${ds.name} (VLM: ${ds.vlm_metadata.image_count} images, ${size})`
  }
  const suffix = ds.samples && ds.samples > 0 ? ` (${ds.samples.toLocaleString()} samples, ${size})` : ` (${size})`
  return `${ds.name}${suffix}`
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

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Train another way</CardTitle>
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
                <div className="text-xs text-amber-600 dark:text-amber-400 flex items-center gap-1.5 mt-1">
                  <span className="inline-block h-1.5 w-1.5 rounded-full bg-amber-500" />
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
              <Button size="sm" onClick={onTest}>Test the model</Button>
              {session.finetunedModelPath && (
                <Button size="sm" variant="outline" onClick={async () => {
                  form.setLoadingFinetunedModel(true)
                  try {
                    await modelController.loadModelPath(session.finetunedModelPath!)
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
                  form.setLoadingFinetunedModel(true)
                  try {
                    await modelController.loadModelPath(session.unifiedModelPath!)
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
          <div className="rounded-lg border border-destructive/20 bg-destructive/5 p-4 space-y-2 mb-4">
            <p className="text-sm font-medium text-destructive">Training failed</p>
            {session.message && <p className="text-xs text-muted-foreground">{session.message}</p>}
            <Button size="sm" variant="ghost" onClick={session.resetTraining}>Try again</Button>
          </div>
        )}

        {!session.trainingRunning && !['complete', 'error'].includes(session.phase) && (
          <>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Select value={datasets.selectedDataset} onValueChange={datasets.setSelectedDataset}>
                  <SelectTrigger className="h-8 text-xs font-mono flex-1 max-w-sm" aria-label="Dataset for fine-tuning">
                    <SelectValue placeholder="Select a dataset..." />
                  </SelectTrigger>
                  <SelectContent>
                    {datasets.datasets.map(ds => (
                      <SelectItem key={ds.id} value={ds.id}>{datasetLabel(ds)}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Button size="sm" variant="outline" onClick={() => datasets.setImportModalOpen(true)}>+ Import</Button>
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
              <Button size="sm" disabled={form.canStart} onClick={() => form.startTraining()}>
                {form.method === 'unified' ? 'Start unified training' : form.method === 'distill' ? (form.inputMode === 'text' && form.textInput.trim() ? 'Train on pasted text' : 'Start') : 'Start'}
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
            <button
              className="text-xs text-muted-foreground hover:text-foreground transition-colors"
              onClick={() => form.setShowAdvanced(!form.showAdvanced)}
            >
              {form.showAdvanced ? 'Hide' : 'Show'} advanced settings
            </button>
            {form.showAdvanced && (
              <div className="space-y-3 mt-3">
                <div className="flex items-center gap-4" role="radiogroup" aria-label="Training method">
                  <ToggleGroup type="single" value={form.method} onValueChange={(v) => { if (v) form.setMethod(v as 'distill' | 'finetune' | 'vlm' | 'unified') }}>
                    <ToggleGroupItem value="distill" className="px-3 py-1.5 rounded-md text-sm data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium">Distill</ToggleGroupItem>
                    <ToggleGroupItem value="finetune" className="px-3 py-1.5 rounded-md text-sm data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium">Continue training</ToggleGroupItem>
                    <ToggleGroupItem value="vlm" className="px-3 py-1.5 rounded-md text-sm data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium">Vision model</ToggleGroupItem>
                    <ToggleGroupItem value="unified" className="px-3 py-1.5 rounded-md text-sm data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium">Auto</ToggleGroupItem>
                  </ToggleGroup>
                  {form.method === 'distill' && <span className="text-xs text-muted-foreground/70">A large model teaches a smaller one to copy its style</span>}
                  {form.method === 'finetune' && <span className="text-xs text-muted-foreground/70">Continue training an existing model on new data</span>}
                  {form.method === 'vlm' && <span className="text-xs text-muted-foreground/70">Teach the AI to understand images and text</span>}
                  {form.method === 'unified' && <span className="text-xs text-muted-foreground/70">Automatically pick the best method</span>}
                </div>

                {form.method !== 'vlm' && form.method !== 'unified' && (
                  <div className="flex items-center gap-1 text-sm" role="radiogroup" aria-label="Data source">
                    <ToggleGroup type="single" value={form.inputMode} onValueChange={(v) => { if (v) form.setInputMode(v as 'dataset' | 'text') }}>
                      <ToggleGroupItem value="dataset" className="px-3 py-1.5 rounded-md text-sm data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium">Use a dataset</ToggleGroupItem>
                      <ToggleGroupItem value="text" className="px-3 py-1.5 rounded-md text-sm data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium">Paste text</ToggleGroupItem>
                    </ToggleGroup>
                  </div>
                )}

                {form.inputMode === 'text' && form.method !== 'vlm' && form.method !== 'unified' && (
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

                {(form.method === 'finetune' || form.method === 'unified') && form.availableModels.length > 0 && (
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

                <div className="flex flex-wrap items-end gap-3">
                  {(form.method === 'distill' || form.method === 'unified') && form.availableModels.length > 0 && (
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
                    <input type="number" min={1} max={200} value={form.trainingEpochs} onChange={e => form.setTrainingEpochs(Math.max(1, parseInt(e.target.value) || 1))} className="h-7 w-16 rounded-md border border-border/60 bg-background px-2 text-xs font-mono text-foreground" />
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-[10px] text-muted-foreground uppercase tracking-wider">LR</label>
                    <input type="number" min={1e-6} max={1} step={1e-4} value={form.trainingLR} onChange={e => form.setTrainingLR(parseFloat(e.target.value) || 1e-3)} className="h-7 w-20 rounded-md border border-border/60 bg-background px-2 text-xs font-mono text-foreground" />
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Batch</label>
                    <input type="number" min={1} max={1024} value={form.trainingBatchSize} onChange={e => form.setTrainingBatchSize(Math.max(1, parseInt(e.target.value) || 64))} className="h-7 w-16 rounded-md border border-border/60 bg-background px-2 text-xs font-mono text-foreground" />
                  </div>
                  {(form.method === 'finetune' || form.method === 'unified') && (
                    <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground h-7">
                      <Switch checked={form.useLoRA} onCheckedChange={form.setUseLoRA} />
                      Advanced
                    </label>
                  )}
                  {form.method === 'unified' && (
                    <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground h-7">
                      <Switch checked={form.unifiedDistill} onCheckedChange={form.setUnifiedDistill} />
                      Distillation
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
  )
}
