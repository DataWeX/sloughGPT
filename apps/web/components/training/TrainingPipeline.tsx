'use client'

import { useState, useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { DatasetSelector } from '@/components/training/DatasetSelector'
import { TrainingErrorBanner } from '@/components/training/TrainingStatus'
import dynamic from 'next/dynamic'
import type { TrainingFormState } from '@/hooks/useTrainingForm'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'
import type { UseTrainingSessionReturn } from '@/hooks/useTrainingSession'
import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'
import { ToggleGroup, ToggleGroupItem } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { TrainingPresets } from '@/components/training/TrainingPresets'

const LossChart = dynamic(() => import('@/components/training/LossChart').then(m => m.LossChart), { ssr: false })

const STEPS = [
  { id: 'data', label: 'Data', description: 'Pick your training data' },
  { id: 'configure', label: 'Configure', description: 'Set training parameters' },
  { id: 'train', label: 'Train', description: 'Run training' },
  { id: 'results', label: 'Results', description: 'View checkpoints & eval' },
] as const

type StepId = typeof STEPS[number]['id']

function StepIndicator({ current, completed }: { current: StepId; completed: Set<StepId> }) {
  const idx = STEPS.findIndex(s => s.id === current)
  return (
    <div className="flex items-center gap-1" role="navigation" aria-label="Training steps">
      {STEPS.map((step, i) => {
        const isDone = completed.has(step.id)
        const isCurrent = step.id === current
        return (
          <div key={step.id} className="flex items-center gap-1">
            {i > 0 && <div className={`w-6 h-px ${isDone || isCurrent ? 'bg-primary' : 'bg-border'}`} />}
            <div className="flex items-center gap-1.5">
              <div className={`flex h-6 w-6 items-center justify-center rounded-full text-[10px] font-medium transition-colors ${
                isCurrent ? 'bg-primary text-primary-foreground' :
                isDone ? 'bg-primary/15 text-primary' :
                'bg-muted text-muted-foreground'
              }`}>
                {isDone ? '✓' : i + 1}
              </div>
              <span className={`text-xs ${isCurrent ? 'font-medium text-foreground' : 'text-muted-foreground'}`}>
                {step.label}
              </span>
            </div>
          </div>
        )
      })}
    </div>
  )
}

export function TrainingPipeline({
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
  const [step, setStep] = useState<StepId>('data')
  const [completedSteps, setCompletedSteps] = useState<Set<StepId>>(new Set())

  const completeStep = (id: StepId) => {
    setCompletedSteps(prev => new Set(prev).add(id))
  }

  const runningJob = form.allJobs.find(j => j.status === 'running')
  const isTraining = session.trainingRunning || !!runningJob

  const canAdvanceData = !!datasets.selectedDataset || (form.inputMode === 'text' && form.textInput.trim().length > 0)

  const hpErrors = useMemo(() => {
    const errors: string[] = []
    if (form.trainingEpochs < 1 || form.trainingEpochs > 500) errors.push('Epochs must be 1–500')
    if (form.trainingBatchSize < 1 || form.trainingBatchSize > 256) errors.push('Batch size must be 1–256')
    if (form.trainingLR <= 0 || form.trainingLR > 1) errors.push('Learning rate must be 0–1')
    return errors
  }, [form.trainingEpochs, form.trainingBatchSize, form.trainingLR])

  const canAdvanceConfigure = form.canStart && hpErrors.length === 0

  const stepIdx = STEPS.findIndex(s => s.id === step)

  const advance = () => {
    completeStep(step)
    const next = STEPS[stepIdx + 1]
    if (next) setStep(next.id)
  }

  const goBack = () => {
    const prev = STEPS[stepIdx - 1]
    if (prev) setStep(prev.id)
  }

  if (isTraining) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base">Training in progress</CardTitle>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span className="relative flex h-2 w-2 shrink-0">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
              </span>
              {session.epoch > 0 && session.totalEpochs > 0 && (
                <span>Epoch {session.epoch}/{session.totalEpochs}</span>
              )}
              {session.loss != null && (
                <span>Loss: {session.loss.toFixed(4)}</span>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {session.lossHistory.length > 0 && (
            <LossChart data={session.lossHistory.map(p => ({ step: p.step, value: p.loss, type: 'train' as const }))} height={200} />
          )}

          {session.phase === 'complete' && (
            <div className="space-y-3">
              <div className="rounded-md bg-success/10 border border-success/20 p-3 text-sm text-success">
                Training complete
                {session.distillCheckpoint && <span className="text-muted-foreground ml-1">— {session.distillCheckpoint}</span>}
              </div>
              <div className="flex flex-wrap gap-2">
                <Button size="sm" onClick={onTest}>Test model</Button>
                {session.distillCheckpoint && (
                  <Button size="sm" variant="outline" onClick={() => {
                    checkpoints.handleLoadCheckpoint(session.distillCheckpoint!, () => {})
                  }}>Load checkpoint</Button>
                )}
                <Button size="sm" variant="ghost" onClick={() => {
                  session.resetTraining()
                  setStep('results')
                  completeStep('train')
                }}>View results</Button>
              </div>
            </div>
          )}

          {session.phase === 'error' && (
            <TrainingErrorBanner
              error={session.message || 'Training failed'}
              onRetry={session.resetTraining}
              onDismiss={session.resetTraining}
            />
          )}
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      <StepIndicator current={step} completed={completedSteps} />

      {step === 'data' && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">1. Pick your data</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-xs text-muted-foreground">
              Choose a dataset or paste text to train on. Conversation-format data (JSONL) trains better than plain text.
            </p>

            <DatasetSelector
              datasets={datasets}
              value={datasets.selectedDataset}
              onChange={datasets.setSelectedDataset}
              showImport
            />

            {datasets.datasetPreview && datasets.datasetPreview.samples.length > 0 && (
              <div className="rounded-md border border-border/40 bg-muted/30 p-3 text-xs">
                <div className="font-medium text-muted-foreground mb-2">Preview</div>
                <div className="grid grid-cols-3 gap-2 mb-2">
                  <div>
                    <span className="text-muted-foreground/60">Samples: </span>
                    <span className="font-mono">{(datasets.datasetPreview.total_samples ?? 0).toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground/60">Characters: </span>
                    <span className="font-mono">{(datasets.datasetPreview.total_chars ?? 0).toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground/60">Avg: </span>
                    <span className="font-mono">
                      {(datasets.datasetPreview.total_samples ?? 0) > 0
                        ? Math.round((datasets.datasetPreview.total_chars ?? 0) / (datasets.datasetPreview.total_samples ?? 1)).toLocaleString()
                        : 0} chars
                    </span>
                  </div>
                </div>
                <div className="space-y-1 font-mono text-muted-foreground border-t border-border/30 pt-2">
                  {datasets.datasetPreview.samples.slice(0, 3).map((sample, i) => (
                    <div key={i} className="truncate">{sample.content}</div>
                  ))}
                </div>
              </div>
            )}

            <div className="flex items-center gap-2 pt-2">
              <Button size="sm" disabled={!canAdvanceData} onClick={advance}>
                Next: Configure
              </Button>
              {!datasets.selectedDataset && form.inputMode !== 'text' && (
                <span className="text-[11px] text-muted-foreground">Select a dataset or switch to paste text in the next step</span>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {step === 'configure' && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">2. Configure training</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Top-level mode: Text or Vision */}
            <div className="flex items-center gap-1 text-sm" role="radiogroup" aria-label="Training mode">
              <ToggleGroup type="single" value={form.method === 'vlm' ? 'vlm' : 'text'} onValueChange={(v) => { if (v) form.setMethod(v === 'vlm' ? 'vlm' : 'distill') }}>
                <ToggleGroupItem value="text" className="px-3 py-1.5 rounded-md text-sm data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium">Text</ToggleGroupItem>
                <ToggleGroupItem value="vlm" className="px-3 py-1.5 rounded-md text-sm data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium">Vision</ToggleGroupItem>
              </ToggleGroup>
            </div>

            <div className="text-xs text-muted-foreground/70">
              {form.method === 'vlm' ? 'Teach the AI to understand images and text' : 'Train a model on text data'}
            </div>

            <TrainingPresets
              onApply={form.applyPreset}
              customPresets={form.customPresets}
              onSave={form.saveCustomPreset}
              onDelete={form.deleteCustomPreset}
            />

            {form.method !== 'vlm' ? (
              <>
                {/* Text training sub-method */}
                <div className="flex items-center gap-1 text-sm" role="radiogroup" aria-label="Training method">
                  <ToggleGroup type="single" value={form.method} onValueChange={(v) => { if (v) form.setMethod(v as 'distill' | 'finetune' | 'native') }}>
                    <ToggleGroupItem value="distill" className="px-3 py-1.5 rounded-md text-sm data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium">Train from scratch</ToggleGroupItem>
                    <ToggleGroupItem value="finetune" className="px-3 py-1.5 rounded-md text-sm data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium">Continue training</ToggleGroupItem>
                    <ToggleGroupItem value="native" className="px-3 py-1.5 rounded-md text-sm data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium">Native SloNet</ToggleGroupItem>
                  </ToggleGroup>
                </div>

                <div className="text-xs text-muted-foreground/70">
                  {form.method === 'distill' && 'Train a small model from text data — no teacher needed'}
                  {form.method === 'finetune' && 'Continue training an existing model on new data'}
                  {form.method === 'native' && 'Train a pure transformer from scratch — SloNet architecture'}
                </div>

                {/* Data source */}
                <div className="flex items-center gap-1 text-sm" role="radiogroup" aria-label="Data source">
                  <ToggleGroup type="single" value={form.inputMode} onValueChange={(v) => { if (v) form.setInputMode(v as 'dataset' | 'text') }}>
                    <ToggleGroupItem value="dataset" className="px-3 py-1.5 rounded-md text-sm data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium">Use a dataset</ToggleGroupItem>
                    <ToggleGroupItem value="text" className="px-3 py-1.5 rounded-md text-sm data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium">Paste text</ToggleGroupItem>
                  </ToggleGroup>
                </div>

                {form.inputMode === 'text' && (
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

                {form.method === 'finetune' && form.availableModels.length > 0 && (
                  <div className="flex flex-col gap-1.5">
                    <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Base model</label>
                    <Select value={form.selectedModel} onValueChange={form.setSelectedModel}>
                      <SelectTrigger className="h-8 text-xs font-mono max-w-sm" aria-label="Base model">
                        <SelectValue placeholder="Select model..." />
                      </SelectTrigger>
                      <SelectContent>
                        {form.availableModels.map(id => <SelectItem key={id} value={id}>{id}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </div>
                )}

                <div className="grid grid-cols-3 gap-3">
                  <div className="flex flex-col gap-1">
                    <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Epochs</label>
                    <input type="number" min={1} max={500} value={form.trainingEpochs}
                      onChange={e => form.setTrainingEpochs(Number(e.target.value))}
                      className={`h-8 rounded-md bg-background px-2 text-xs font-mono ${
                        form.trainingEpochs < 1 || form.trainingEpochs > 500
                          ? 'border border-destructive/60 text-destructive'
                          : 'border border-border/60'
                      }`} />
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Batch size</label>
                    <input type="number" min={1} max={256} value={form.trainingBatchSize}
                      onChange={e => form.setTrainingBatchSize(Number(e.target.value))}
                      className={`h-8 rounded-md bg-background px-2 text-xs font-mono ${
                        form.trainingBatchSize < 1 || form.trainingBatchSize > 256
                          ? 'border border-destructive/60 text-destructive'
                          : 'border border-border/60'
                      }`} />
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Learning rate</label>
                    <input type="text" value={form.trainingLR}
                      onChange={e => form.setTrainingLR(Number(e.target.value) || 1e-3)}
                      className={`h-8 rounded-md bg-background px-2 text-xs font-mono ${
                        form.trainingLR <= 0 || form.trainingLR > 1
                          ? 'border border-destructive/60 text-destructive'
                          : 'border border-border/60'
                      }`} />
                  </div>
                </div>

                {hpErrors.length > 0 && (
                  <div className="text-[11px] text-destructive space-y-0.5">
                    {hpErrors.map(e => <div key={e}>{e}</div>)}
                  </div>
                )}

                {form.method === 'finetune' && (
                  <label className="flex items-center gap-2 text-xs">
                    <input type="checkbox" checked={form.useLoRA} onChange={e => form.setUseLoRA(e.target.checked)}
                      className="rounded border-border" />
                    Use LoRA (parameter-efficient fine-tuning)
                  </label>
                )}
              </>
            ) : (
              /* Vision training — no data source or hyperparams needed */
              <div className="text-sm text-muted-foreground py-2">
                Vision training uses the selected dataset to teach image understanding.
              </div>
            )}

            <div className="flex items-center gap-2 pt-2">
              <Button size="sm" disabled={!canAdvanceConfigure} onClick={advance}>
                Next: Train
              </Button>
              <Button size="sm" variant="ghost" onClick={goBack}>Back</Button>
            </div>
          </CardContent>
        </Card>
      )}

      {step === 'train' && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">3. Train</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="rounded-md bg-muted/30 border border-border/40 p-3 text-xs text-muted-foreground space-y-1">
              <div>Method: <span className="font-medium text-foreground">{form.method}</span></div>
              {datasets.selectedDataset && <div>Dataset: <span className="font-medium text-foreground">{datasets.selectedDataset}</span></div>}
              <div>Epochs: <span className="font-medium text-foreground">{form.trainingEpochs}</span></div>
              <div>Batch size: <span className="font-medium text-foreground">{form.trainingBatchSize}</span></div>
              <div>Learning rate: <span className="font-medium text-foreground">{form.trainingLR}</span></div>
              {form.useLoRA && <div>LoRA: <span className="font-medium text-foreground">enabled</span></div>}
            </div>

            <div className="flex items-center gap-2">
              <Button size="sm" disabled={!form.canStart} onClick={() => form.startTraining()}>
                {form.method === 'distill' && form.inputMode === 'text' && form.textInput.trim()
                  ? 'Train on pasted text'
                  : form.method === 'vlm' ? 'Start vision training' : 'Start training'}
              </Button>
              <Button size="sm" variant="ghost" onClick={goBack}>Back</Button>
            </div>
          </CardContent>
        </Card>
      )}

      {step === 'results' && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">4. Results</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {checkpoints.checkpoints.length === 0 ? (
              <div className="text-sm text-muted-foreground py-4 text-center">
                No checkpoints yet. Run a training job to see results here.
              </div>
            ) : (
              <div className="space-y-3">
                <div className="text-xs text-muted-foreground">
                  {checkpoints.checkpoints.length} checkpoint(s) saved
                </div>
                <div className="space-y-2">
                  {checkpoints.checkpoints.slice(0, 10).map(cp => (
                    <div key={cp.name} className="flex items-center justify-between rounded-md border border-border/40 bg-muted/20 px-3 py-2">
                      <div className="min-w-0">
                        <div className="text-xs font-medium truncate">{cp.name}</div>
                        <div className="text-[10px] text-muted-foreground">
                          {cp.loss != null && <span>Loss: {cp.loss.toFixed(4)}</span>}
                          {cp.tags && cp.tags.length > 0 && <span className="ml-2">Tags: {cp.tags.join(', ')}</span>}
                        </div>
                      </div>
                      <Button size="sm" variant="ghost" className="shrink-0" onClick={() => {
                        checkpoints.handleLoadCheckpoint(cp.name, () => {})
                      }}>
                        Load
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="flex items-center gap-2 pt-2">
              <Button size="sm" variant="ghost" onClick={() => setStep('train')}>Train more</Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
