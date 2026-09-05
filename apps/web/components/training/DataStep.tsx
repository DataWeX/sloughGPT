'use client'

import { useState, useCallback, useMemo } from 'react'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Button,
  Badge,
  SearchInput,
  EmptyCard,
  Tooltip,
  TooltipTrigger,
  TooltipContent,
  Spinner,
  IconFile,
  IconUpload,
} from '@sloughgpt/strui'
import { SectionHeader } from '@sloughgpt/strui'
import { FoldSection } from '@sloughgpt/strui'
import { DatasetSelector } from '@/components/training/DatasetSelector'
import { datasetController } from '@/lib/controllers'
import type { Dataset } from '@/lib/dataset-controller'
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
  { id: 'heptapod/titanic', label: 'Titanic' },
  { id: 'uciml/iris', label: 'Iris' },
  { id: 'rounakbanik/pokemon', label: 'Pokemon' },
  { id: 'unsdsn/world-happiness', label: 'World Happiness' },
]

const POPULAR_HF = [
  { id: 'HuggingFaceH4/tinyshakespeare', label: 'Tiny Shakespeare' },
  { id: 'HuggingFaceH4/ultrachat_200k', label: 'UltraChat 200K' },
  { id: 'HuggingFaceH4/cosmopedia', label: 'Cosmopedia' },
  { id: 'HuggingFaceH4/smollm-corpus', label: 'SmolLM Corpus' },
]

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function DatasetTooltipContent({ dataset }: { dataset: Dataset }) {
  return (
    <div className="space-y-1.5 max-w-[220px]">
      <div className="font-medium text-foreground">{dataset.name}</div>
      <div className="flex flex-wrap gap-1.5">
        {dataset.source && (
          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
            {dataset.source}
          </Badge>
        )}
        {dataset.type && (
          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
            {dataset.type.toUpperCase()}
          </Badge>
        )}
      </div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px] text-muted-foreground">
        {dataset.samples != null && dataset.samples > 0 && (
          <>
            <span>Samples</span>
            <span className="text-right font-numeric">{dataset.samples.toLocaleString()}</span>
          </>
        )}
        <span>Size</span>
        <span className="text-right font-numeric">{formatSize(dataset.size)}</span>
        {dataset.tags && dataset.tags.length > 0 && (
          <>
            <span>Tags</span>
            <span className="text-right">{dataset.tags.slice(0, 2).join(', ')}</span>
          </>
        )}
      </div>
      {dataset.vlm_metadata && (
        <div className="text-[10px] text-muted-foreground border-t border-border/30 pt-1">
          {dataset.vlm_metadata.image_count} images · {dataset.vlm_metadata.type}
        </div>
      )}
    </div>
  )
}

function DatasetChip({
  dataset,
  selected,
  onSelect,
}: {
  dataset: Dataset
  selected: boolean
  onSelect: (id: string) => void
}) {
  return (
    <Tooltip delayDuration={300}>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={() => onSelect(dataset.id)}
          className={`
            inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px]
            transition-all duration-150
            ${selected
              ? 'bg-primary/15 text-primary border border-primary/30'
              : 'bg-muted/40 text-muted-foreground border border-border/60 hover:bg-muted/70 hover:text-foreground hover:border-border'
            }
          `}
        >
          <span className="truncate max-w-[120px]">{dataset.name}</span>
          {dataset.samples != null && dataset.samples > 0 && (
            <span className="text-[9px] opacity-60 font-numeric">
              {dataset.samples.toLocaleString()}
            </span>
          )}
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" variant="muted" sideOffset={4}>
        <DatasetTooltipContent dataset={dataset} />
      </TooltipContent>
    </Tooltip>
  )
}

export function DataStep({ form, datasets, onNext, addToast }: StepProps) {
  const canAdvance = !!datasets.selectedDataset || (form.inputMode === 'text' && form.textInput.trim().length > 0)
  const [importingKaggle, setImportingKaggle] = useState<string | null>(null)
  const [importingHF, setImportingHF] = useState<string | null>(null)
  const [search, setSearch] = useState('')

  const filteredDatasets = useMemo(() => {
    if (!search.trim()) return datasets.datasets
    const q = search.toLowerCase()
    return datasets.datasets.filter(
      ds =>
        ds.name.toLowerCase().includes(q) ||
        ds.source?.toLowerCase().includes(q) ||
        ds.tags?.some(t => t.toLowerCase().includes(q)),
    )
  }, [datasets.datasets, search])

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

        {datasets.datasets.length > 0 && (
          <div className="space-y-3">
            <SectionHeader
              title="Your Datasets"
              description={`${datasets.datasets.length} dataset${datasets.datasets.length !== 1 ? 's' : ''} available`}
              action={
                <SearchInput
                  value={search}
                  onChange={setSearch}
                  placeholder="Filter datasets..."
                  className="h-7 w-48 text-xs"
                  aria-label="Filter datasets"
                />
              }
            />

            {search.trim() && filteredDatasets.length === 0 ? (
              <div className="text-xs text-muted-foreground py-2">
                No datasets match &ldquo;{search}&rdquo;
              </div>
            ) : (
              <div className="flex flex-wrap gap-1.5 max-h-[180px] overflow-y-auto overscroll-contain pr-1">
                {filteredDatasets.map(ds => (
                  <DatasetChip
                    key={ds.id}
                    dataset={ds}
                    selected={datasets.selectedDataset === ds.id}
                    onSelect={datasets.setSelectedDataset}
                  />
                ))}
              </div>
            )}
          </div>
        )}

        {datasets.datasets.length === 0 && !search.trim() && (
          <EmptyCard
            message="No datasets yet"
            description="Import a dataset or paste text in the next step."
            icon={<IconUpload className="h-5 w-5" />}
            action={
              <Button size="sm" variant="outline" onClick={() => datasets.setImportModalOpen(true)}>
                + Import Dataset
              </Button>
            }
          />
        )}

        <div className="space-y-2">
          <FoldSection
            heading={
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground uppercase tracking-wider">Quick Import</span>
                <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                  Kaggle &middot; HuggingFace
                </Badge>
              </div>
            }
          >
            <div className="space-y-4">
              <div className="space-y-2">
                <div className="text-[10px] text-muted-foreground uppercase tracking-wider">Kaggle</div>
                <div className="flex flex-wrap gap-1.5">
                  {POPULAR_KAGGLE.map(ds => (
                    <Tooltip key={ds.id} delayDuration={200}>
                      <TooltipTrigger asChild>
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-6 text-[10px] px-2"
                          disabled={importingKaggle === ds.id}
                          onClick={() => handleQuickKaggle(ds.id)}
                        >
                          {importingKaggle === ds.id ? (
                            <Spinner className="h-2.5 w-2.5 mr-1" />
                          ) : null}
                          {importingKaggle === ds.id ? 'Importing...' : ds.label}
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent side="bottom" variant="muted" sideOffset={4}>
                        <div className="text-[10px] max-w-[180px]">
                          Import from Kaggle. Requires API credentials on server.
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <div className="text-[10px] text-muted-foreground uppercase tracking-wider">HuggingFace</div>
                <div className="flex flex-wrap gap-1.5">
                  {POPULAR_HF.map(ds => (
                    <Tooltip key={ds.id} delayDuration={200}>
                      <TooltipTrigger asChild>
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-6 text-[10px] px-2"
                          disabled={importingHF === ds.id}
                          onClick={() => handleQuickHF(ds.id)}
                        >
                          {importingHF === ds.id ? (
                            <Spinner className="h-2.5 w-2.5 mr-1" />
                          ) : null}
                          {importingHF === ds.id ? 'Importing...' : ds.label}
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent side="bottom" variant="muted" sideOffset={4}>
                        <div className="text-[10px] max-w-[180px]">
                          One-click import from HuggingFace Hub. No server setup required.
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  ))}
                </div>
              </div>
            </div>
          </FoldSection>
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
