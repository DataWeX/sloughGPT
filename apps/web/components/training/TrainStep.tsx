'use client'

import { Card, CardContent, CardHeader, CardTitle, Button } from '@sloughgpt/strui'
import type { StepProps } from './DataStep'
import type { TrainingFormState } from '@/hooks/useTrainingForm'

const METHOD_LABELS: Record<TrainingFormState['method'], string> = {
  distill: 'Train from scratch',
  finetune: 'Continue training',
  native: 'Native SloNet',
  vlm: 'Vision (image + text)',
}

interface SummaryRow {
  label: string
  value: string
}

export function TrainStep({ form, datasets, onBack }: StepProps) {
  const datasetName = (() => {
    if (!datasets.selectedDataset) return undefined
    const ds = datasets.datasets.find(d => d.id === datasets.selectedDataset)
    return ds?.name || datasets.selectedDataset
  })()

  const rows: SummaryRow[] = []
  if (form.method === 'vlm') {
    if (form.visualVisionEncoder) rows.push({ label: 'Vision encoder', value: form.visualVisionEncoder })
    if (form.visualLLM) rows.push({ label: 'Language model', value: form.visualLLM })
    rows.push({ label: 'Stage 1 epochs', value: String(form.visualStage1Epochs) })
    rows.push({ label: 'Stage 2 epochs', value: String(form.visualStage2Epochs) })
  } else {
    rows.push({ label: 'Epochs', value: String(form.trainingEpochs) })
    rows.push({ label: 'Batch size', value: String(form.trainingBatchSize) })
    rows.push({ label: 'Learning rate', value: String(form.trainingLR) })
    if (form.method === 'native') {
      rows.push({ label: 'Embed dim', value: String(form.nativeEmbed) })
      rows.push({ label: 'Layers', value: String(form.nativeLayers) })
      rows.push({ label: 'Heads', value: String(form.nativeHeads) })
      rows.push({ label: 'Block size', value: String(form.nativeBlockSize) })
    }
  }
  if (form.method === 'finetune') {
    if (form.selectedModel) rows.push({ label: 'Base model', value: form.selectedModel })
    rows.push({ label: 'LoRA', value: form.useLoRA ? 'enabled' : 'disabled' })
  }
  if (form.method === 'vlm') {
    rows.push({ label: 'LoRA', value: form.useLoRA ? 'enabled' : 'disabled' })
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">3. Train</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="rounded-md bg-muted/30 border border-border/40 p-3 text-xs text-muted-foreground space-y-1">
          <div>Method: <span className="font-medium text-foreground">{METHOD_LABELS[form.method]}</span></div>
          {datasetName && <div>Dataset: <span className="font-medium text-foreground">{datasetName}</span></div>}
          {form.inputMode === 'text' && form.textInput.trim() && (
            <div>Source: <span className="font-medium text-foreground">Pasted text ({form.textInput.length.toLocaleString()} chars)</span></div>
          )}
          {rows.map(row => (
            <div key={row.label}>{row.label}: <span className="font-medium text-foreground">{row.value}</span></div>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <Button size="sm" disabled={!form.canStart} onClick={() => form.startTraining()}>
            {form.method === 'distill' && form.inputMode === 'text' && form.textInput.trim()
              ? 'Train on pasted text'
              : form.method === 'vlm' ? 'Start vision training' : 'Start training'}
          </Button>
          <Button size="sm" variant="ghost" onClick={onBack}>Back</Button>
        </div>
      </CardContent>
    </Card>
  )
}
