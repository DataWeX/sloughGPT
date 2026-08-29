'use client'

import { useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button } from '@sloughgpt/strui'
import { DatasetSelector } from '@/components/training/DatasetSelector'
import { datasetController } from '@/lib/controllers'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'
import type { TrainingFormState } from '@/hooks/useTrainingForm'

import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'

export interface StepProps {
  form: TrainingFormState
  datasets: UseTrainingDatasetsReturn
  checkpoints?: UseTrainingCheckpointsReturn
  onNext: () => void
  onBack: () => void
  addToast?: (msg: string, type?: 'success' | 'error' | 'info') => void
}

const POPULAR_KAGGLE = [
  { id: 'nlp-datasets/tinyshakespeare', label: 'Tiny Shakespeare' },
  { id: 'HuggingFaceH4/ultrachat_200k', label: 'UltraChat 200K' },
  { id: 'datasets/opensubtitles', label: 'OpenSubtitles' },
  { id: 'heliosbrahma/mental-health-chatbot-dataset', label: 'Mental Health Chat' },
]

const POPULAR_HF = [
  { id: 'HuggingFaceH4/tinyshakespeare', label: 'Tiny Shakespeare' },
  { id: 'HuggingFaceH4/ultrachat_200k', label: 'UltraChat 200K' },
  { id: 'HuggingFaceH4/cosmopedia', label: 'Cosmopedia' },
  { id: 'HuggingFaceH4/smollm-corpus', label: 'SmolLM Corpus' },
]

export function DataStep({ form, datasets, onNext, addToast }: StepProps) {
  const canAdvance = !!datasets.selectedDataset || (form.inputMode === 'text' && form.textInput.trim().length > 0)
  const [importingKaggle, setImportingKaggle] = useState<string | null>(null)
  const [importingHF, setImportingHF] = useState<string | null>(null)

  const handleQuickKaggle = useCallback(async (datasetId: string) => {
    setImportingKaggle(datasetId)
    try {
      const result = await datasetController.importFromKaggle({
        dataset: datasetId,
        name: datasetId.split('/').pop() || datasetId,
      })
      addToast?.(`Imported ${datasetId}`, 'success')
      await datasets.fetchDatasets()
      datasets.setSelectedDataset(result.dataset_id)
    } catch {
      addToast?.(`Could not import ${datasetId}`, 'error')
    } finally {
      setImportingKaggle(null)
    }
  }, [addToast, datasets])

  const handleQuickHF = useCallback(async (datasetId: string) => {
    setImportingHF(datasetId)
    try {
      const result = await datasetController.importFromHuggingFace({
        dataset_id: datasetId,
        name: datasetId.split('/').pop() || datasetId,
      })
      addToast?.(`Imported ${datasetId}`, 'success')
      await datasets.fetchDatasets()
      datasets.setSelectedDataset(result.dataset_id)
    } catch {
      addToast?.(`Could not import ${datasetId}`, 'error')
    } finally {
      setImportingHF(null)
    }
  }, [addToast, datasets])

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

        <div className="space-y-2">
          <div className="text-[10px] text-muted-foreground uppercase tracking-wider">Quick import from Kaggle</div>
          <div className="flex flex-wrap gap-1.5">
            {POPULAR_KAGGLE.map(ds => (
              <Button
                key={ds.id}
                size="sm"
                variant="secondary"
                className="h-7 text-[11px]"
                disabled={importingKaggle === ds.id}
                onClick={() => handleQuickKaggle(ds.id)}
              >
                {importingKaggle === ds.id ? 'Importing...' : ds.label}
              </Button>
            ))}
          </div>
          <p className="text-[10px] text-muted-foreground/70">
            One-click import popular NLP datasets. Requires Kaggle API credentials configured on the server.
          </p>
        </div>

        <div className="space-y-2">
          <div className="text-[10px] text-muted-foreground uppercase tracking-wider">Quick import from HuggingFace</div>
          <div className="flex flex-wrap gap-1.5">
            {POPULAR_HF.map(ds => (
              <Button
                key={ds.id}
                size="sm"
                variant="secondary"
                className="h-7 text-[11px]"
                disabled={importingHF === ds.id}
                onClick={() => handleQuickHF(ds.id)}
              >
                {importingHF === ds.id ? 'Importing...' : ds.label}
              </Button>
            ))}
          </div>
          <p className="text-[10px] text-muted-foreground/70">
            One-click import from HuggingFace Hub. No extra server setup required.
          </p>
        </div>

        {datasets.datasetPreview && datasets.datasetPreview.samples.length > 0 && (
          <div className="rounded-md border border-border/40 bg-muted/30 p-3 text-xs">
            <div className="font-medium text-muted-foreground mb-2">Preview</div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mb-2">
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
