'use client'

import { Card, CardContent, CardHeader, CardTitle, Button } from '@sloughgpt/strui'
import { DatasetSelector } from '@/components/training/DatasetSelector'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'
import type { TrainingFormState } from '@/hooks/useTrainingForm'

export interface StepProps {
  form: TrainingFormState
  datasets: UseTrainingDatasetsReturn
  onNext: () => void
  onBack: () => void
}

export function DataStep({ form, datasets, onNext }: StepProps) {
  const canAdvance = !!datasets.selectedDataset || (form.inputMode === 'text' && form.textInput.trim().length > 0)

  return (
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
                <span className="font-numeric">{(datasets.datasetPreview.total_samples ?? 0).toLocaleString()}</span>
              </div>
              <div>
                <span className="text-muted-foreground/60">Characters: </span>
                <span className="font-numeric">{(datasets.datasetPreview.total_chars ?? 0).toLocaleString()}</span>
              </div>
              <div>
                <span className="text-muted-foreground/60">Avg: </span>
                <span className="font-numeric">
                  {(datasets.datasetPreview.total_samples ?? 0) > 0
                    ? Math.round((datasets.datasetPreview.total_chars ?? 0) / (datasets.datasetPreview.total_samples ?? 1)).toLocaleString()
                    : 0} chars
                </span>
              </div>
            </div>
            <div className="space-y-1 font-numeric text-muted-foreground border-t border-border/30 pt-2">
              {datasets.datasetPreview.samples.slice(0, 3).map((sample, i) => (
                <div key={i} className="truncate">{sample.content}</div>
              ))}
            </div>
          </div>
        )}

        <div className="flex items-center gap-2 pt-2">
          <Button size="sm" onClick={onNext}>
            Next: Configure
          </Button>
          {!canAdvance && (
            <span className="text-[11px] text-muted-foreground">Select a dataset or switch to paste text in the next step</span>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
