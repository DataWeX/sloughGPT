'use client'

import { useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, ToggleGroup, ToggleGroupItem, Select, SelectTrigger, SelectValue, SelectContent, SelectItem, Input, Label } from '@sloughgpt/strui'
import { TrainingPresets } from '@/components/training/TrainingPresets'
import type { StepProps } from './DataStep'

export function ConfigureStep({ form, datasets, checkpoints, onNext, onBack }: StepProps) {
  const hpErrors = useMemo(() => {
    const errors: string[] = []
    if (form.trainingEpochs < 1 || form.trainingEpochs > 500) errors.push('Epochs must be 1–500')
    if (form.trainingBatchSize < 1 || form.trainingBatchSize > 256) errors.push('Batch size must be 1–256')
    if (form.trainingLR <= 0 || form.trainingLR > 1) errors.push('Learning rate must be 0–1')
    return errors
  }, [form.trainingEpochs, form.trainingBatchSize, form.trainingLR])

  const canAdvance = form.canStart && hpErrors.length === 0

  return (
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
          getCurrentState={() => ({
            name: '',
            description: '',
            method: form.method as 'distill' | 'finetune' | 'native' | 'vlm',
            epochs: form.trainingEpochs,
            lr: form.trainingLR,
            batchSize: form.trainingBatchSize,
            useLoRA: form.useLoRA,
            nativeEmbed: form.nativeEmbed,
            nativeLayers: form.nativeLayers,
            nativeHeads: form.nativeHeads,
            nativeBlockSize: form.nativeBlockSize,
          })}
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
                <span className="absolute bottom-1.5 right-2 text-[10px] text-muted-foreground/50 tabular-nums" aria-live="polite" aria-atomic="true">
                  {form.textInput.length > 0 ? `${form.textInput.length.toLocaleString()} chars · ~${Math.ceil(form.textInput.length / 4)} tokens` : ''}
                </span>
              </div>
            )}

            {form.method === 'finetune' && form.availableModels.length > 0 && (
              <div className="flex flex-col gap-1.5">
                <label htmlFor="train-base-model" className="text-[10px] text-muted-foreground uppercase tracking-wider">Base model</label>
                <Select value={form.selectedModel} onValueChange={form.setSelectedModel}>
                  <SelectTrigger id="train-base-model" className="h-8 text-xs font-mono max-w-sm" aria-label="Base model">
                    <SelectValue placeholder="Select model..." />
                  </SelectTrigger>
                  <SelectContent>
                    {form.availableModels.map(id => <SelectItem key={id} value={id}>{id}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="flex flex-col gap-1">
                <Label htmlFor="train-epochs" variant="uppercase">Epochs</Label>
                <Input id="train-epochs" type="number" min={1} max={500} value={form.trainingEpochs}
                  onChange={e => form.setTrainingEpochs(Number(e.target.value))}
                  error={form.trainingEpochs < 1 || form.trainingEpochs > 500}
                  aria-invalid={form.trainingEpochs < 1 || form.trainingEpochs > 500}
                  aria-describedby={hpErrors.length > 0 ? 'configure-hp-errors' : undefined}
                  className="h-8 text-xs font-mono" />
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="train-batch" variant="uppercase">Batch size</Label>
                <Input id="train-batch" type="number" min={1} max={256} value={form.trainingBatchSize}
                  onChange={e => form.setTrainingBatchSize(Number(e.target.value))}
                  error={form.trainingBatchSize < 1 || form.trainingBatchSize > 256}
                  aria-invalid={form.trainingBatchSize < 1 || form.trainingBatchSize > 256}
                  aria-describedby={hpErrors.length > 0 ? 'configure-hp-errors' : undefined}
                  className="h-8 text-xs font-mono" />
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="train-lr" variant="uppercase">Learning rate</Label>
                <Input id="train-lr" type="text" inputMode="decimal" value={form.trainingLR}
                  onChange={e => form.setTrainingLR(Number(e.target.value) || 1e-3)}
                  error={form.trainingLR <= 0 || form.trainingLR > 1}
                  aria-invalid={form.trainingLR <= 0 || form.trainingLR > 1}
                  aria-describedby={hpErrors.length > 0 ? 'configure-hp-errors' : undefined}
                  className="h-8 text-xs font-mono" />
              </div>
            </div>

            {hpErrors.length > 0 && (
              <div id="configure-hp-errors" className="text-[11px] text-destructive space-y-0.5">
                {hpErrors.map(e => <div key={e}>{e}</div>)}
              </div>
            )}

            {form.method === 'finetune' && (
              <div className="space-y-2">
                <label className="flex items-center gap-2 text-xs">
                  <input type="checkbox" checked={form.useLoRA} onChange={e => form.setUseLoRA(e.target.checked)}
                    className="rounded border-border" aria-label="Enable LoRA" />
                  Use LoRA (parameter-efficient fine-tuning)
                </label>
                {form.useLoRA && (
                  <div className="grid grid-cols-2 gap-3 pl-5">
                    <div className="flex flex-col gap-1">
                      <Label htmlFor="lora-rank" variant="uppercase">Rank</Label>
                      <Input id="lora-rank" type="number" min={1} max={128} value={form.loraRank}
                        onChange={e => form.setLoraRank(Number(e.target.value) || 8)}
                        className="h-8 text-xs font-mono" />
                    </div>
                    <div className="flex flex-col gap-1">
                      <Label htmlFor="lora-alpha" variant="uppercase">Alpha</Label>
                      <Input id="lora-alpha" type="number" min={1} max={256} value={form.loraAlpha}
                        onChange={e => form.setLoraAlpha(Number(e.target.value) || 16)}
                        className="h-8 text-xs font-mono" />
                    </div>
                  </div>
                )}
              </div>
            )}

            {checkpoints && checkpoints.checkpoints.length > 0 && (
              <div className="flex flex-col gap-1.5">
                <label htmlFor="resume-checkpoint" className="text-[10px] text-muted-foreground uppercase tracking-wider">
                  Resume from checkpoint (optional)
                </label>
                <select
                  id="resume-checkpoint"
                  value={form.resumeCheckpoint}
                  onChange={e => form.setResumeCheckpoint(e.target.value)}
                  aria-label="Resume from checkpoint"
                  className="h-8 rounded-md border border-border bg-background px-2 text-xs font-mono"
                >
                  <option value="">Start fresh</option>
                  {checkpoints.checkpoints.map(c => (
                    <option key={c.name} value={c.name}>
                      {c.name} {c.loss != null ? `(${c.loss.toFixed(4)})` : ''}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </>
        ) : (
          /* Vision training — no data source or hyperparams needed */
          <div className="text-sm text-muted-foreground py-2">
            Vision training uses the selected dataset to teach image understanding.
          </div>
        )}

        <div className="flex items-center gap-2 pt-2">
          <Button size="sm" disabled={!canAdvance} onClick={onNext}>
            Next: Train
          </Button>
          <Button size="sm" variant="ghost" onClick={onBack}>Back</Button>
        </div>
      </CardContent>
    </Card>
  )
}
