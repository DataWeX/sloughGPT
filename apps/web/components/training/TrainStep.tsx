'use client'

import { Card, CardContent, CardHeader, CardTitle, Button } from '@sloughgpt/strui'
import type { StepProps } from './DataStep'

export function TrainStep({ form, datasets, onBack }: StepProps) {
  return (
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
          <Button size="sm" variant="ghost" onClick={onBack}>Back</Button>
        </div>
      </CardContent>
    </Card>
  )
}
