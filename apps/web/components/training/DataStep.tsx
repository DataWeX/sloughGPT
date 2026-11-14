'use client'

import { useCallback, useMemo, useState } from 'react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Button,
  EmptyCard,
  SearchInput,
  SectionHeader,
  IconGrid,
  IconSparkle,
  IconUpload,
} from '@sloughgpt/strui'
import { DatasetSelector } from '@/components/training/DatasetSelector'
import { DatasetChip } from '@/components/training/dataset-chip'
import { DatasetPreviewPanel } from '@/components/training/dataset-preview'
import {
  QuickImportSection,
  type QuickImportSource,
} from '@/components/training/quick-import-section'
import { datasetController } from '@/lib/controllers'
import { isTrainingCorpus } from '@/lib/dataset-controller'
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

const KAGGLE_PRESETS = [
  { id: 'heptapod/titanic', label: 'Titanic' },
  { id: 'uciml/iris', label: 'Iris' },
  { id: 'rounakbanik/pokemon', label: 'Pokemon' },
  { id: 'unsdsn/world-happiness', label: 'World Happiness' },
]

const HF_PRESETS = [
  { id: 'HuggingFaceH4/tinyshakespeare', label: 'Tiny Shakespeare' },
  { id: 'HuggingFaceH4/ultrachat_200k', label: 'UltraChat 200K' },
  { id: 'HuggingFaceH4/cosmopedia', label: 'Cosmopedia' },
  { id: 'HuggingFaceH4/smollm-corpus', label: 'SmolLM Corpus' },
]

type QuickSourceKey = 'kaggle' | 'huggingface'

export function DataStep({ datasets, onNext, addToast }: StepProps) {
  const canAdvance = true
  const [importing, setImporting] = useState<{ source: QuickSourceKey; id: string } | null>(null)
  const [search, setSearch] = useState('')

  const corpusDatasets = useMemo(
    () => datasets.datasets.filter(isTrainingCorpus),
    [datasets.datasets],
  )

  const filteredDatasets = useMemo(() => {
    if (!search.trim()) return corpusDatasets
    const q = search.toLowerCase()
    return corpusDatasets.filter(
      (ds) =>
        ds.name.toLowerCase().includes(q) ||
        ds.id.toLowerCase().includes(q) ||
        ds.source?.toLowerCase().includes(q) ||
        ds.tags?.some((t) => t.toLowerCase().includes(q)),
    )
  }, [corpusDatasets, search])

  const handleQuickImport = useCallback(
    async (source: QuickSourceKey, datasetId: string) => {
      setImporting({ source, id: datasetId })
      const name = datasetId.split('/').pop() || datasetId
      try {
        const result =
          source === 'kaggle'
            ? await datasetController.importFromKaggle({ dataset: datasetId, name })
            : await datasetController.importFromHuggingFace({ dataset_id: datasetId, name })
        addToast?.(`Imported ${datasetId}`, 'success')
        await datasets.fetchDatasets()
        datasets.setSelectedDataset(result.dataset_id)
      } catch {
        addToast?.(`Could not import ${datasetId}`, 'error')
      } finally {
        setImporting(null)
      }
    },
    [addToast, datasets],
  )

  const quickSources: QuickImportSource[] = useMemo(
    () => [
      {
        key: 'kaggle',
        label: 'Kaggle',
        icon: <IconGrid className="h-3 w-3" />,
        hint: 'Requires Kaggle API credentials configured on the server.',
        presets: KAGGLE_PRESETS,
        importingId: importing?.source === 'kaggle' ? importing.id : null,
        onImport: (id) => void handleQuickImport('kaggle', id),
      },
      {
        key: 'huggingface',
        label: 'HuggingFace',
        icon: <IconSparkle className="h-3 w-3" />,
        hint: 'No server setup — pulls straight from the HuggingFace Hub.',
        presets: HF_PRESETS,
        importingId: importing?.source === 'huggingface' ? importing.id : null,
        onImport: (id) => void handleQuickImport('huggingface', id),
      },
    ],
    [importing, handleQuickImport],
  )

  const hasDatasets = datasets.datasets.length > 0

  return (
    <Card>
      <CardHeader className="pb-4">
        <CardTitle className="text-base">1. Pick your data</CardTitle>
        <CardDescription className="text-xs">
          Choose a dataset or paste text to train on. Conversation-format data (JSONL) trains better
          than plain text.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Layer 1 — the one decision this step exists for. */}
        <section
          aria-label="Active dataset"
          className="space-y-2 rounded-lg border border-border bg-muted p-3"
        >
          <h3 className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Active dataset
          </h3>
          <DatasetSelector
            datasets={datasets}
            value={datasets.selectedDataset}
            onChange={datasets.setSelectedDataset}
            showImport
          />
        </section>

        {/* Layer 2 — the library the user already owns. */}
        {hasDatasets && (
          <section aria-label="Your datasets" className="space-y-3">
            <SectionHeader
              title="Your Datasets"
              description={
                corpusDatasets.length > 0
                  ? `${corpusDatasets.length} dataset${corpusDatasets.length !== 1 ? 's' : ''} available`
                  : 'System stores hidden — no training corpora yet'
              }
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
            ) : filteredDatasets.length === 0 ? (
              <div className="text-xs text-muted-foreground py-2">
                Only system stores found — import a corpus to train.
              </div>
            ) : (
              <div className="flex flex-wrap gap-1.5 max-h-[180px] overflow-y-auto overscroll-contain pr-1">
                {filteredDatasets.map((ds) => (
                  <DatasetChip
                    key={ds.id}
                    dataset={ds}
                    selected={datasets.selectedDataset === ds.id}
                    onSelect={datasets.setSelectedDataset}
                  />
                ))}
              </div>
            )}
          </section>
        )}

        {!hasDatasets && !search.trim() && (
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

        {/* Layer 3 — acquisition: folded away once the user has data. */}
        <QuickImportSection sources={quickSources} defaultOpen={!hasDatasets} />

        {/* Layer 4 — evidence for the choice just made. */}
        {datasets.datasetPreview && datasets.datasetPreview.samples.length > 0 && (
          <DatasetPreviewPanel preview={datasets.datasetPreview} />
        )}

        <div className="flex flex-wrap items-center gap-2 border-t border-border/50 pt-3">
          <Button size="sm" className="h-8 px-3 text-xs" onClick={onNext} disabled={!canAdvance}>
            Next: Configure
          </Button>
          {!datasets.selectedDataset && (
            <span className="text-[11px] text-muted-foreground/70">
              Select a dataset or switch to paste text in the next step
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
