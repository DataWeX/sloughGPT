'use client'

import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { DatasetImportDialog } from '@/components/DatasetImportDialog'
import type { Dataset } from '@/lib/dataset-controller'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'

export function datasetLabel(ds: Dataset): string {
  const hasSize = ds.size != null && Number.isFinite(ds.size) && ds.size > 0
  const size = hasSize ? `${(ds.size / 1024).toFixed(1)} KB` : ''
  const parts: string[] = [ds.name]

  if (ds.type === 'vlm' && ds.vlm_metadata) {
    parts.push('VLM', `${ds.vlm_metadata.image_count} images`)
  } else if (ds.samples && ds.samples > 0) {
    parts.push(`${ds.samples.toLocaleString()} samples`)
  }

  if (ds.source) parts.push(ds.source)
  if (size) parts.push(size)

  return parts.join(' · ')
}

export function DatasetSelector({
  datasets,
  value,
  onChange,
  disabled,
  showImport,
  showAllKinds,
}: {
  datasets: UseTrainingDatasetsReturn
  value: string
  onChange: (id: string) => void
  disabled?: boolean
  showImport?: boolean
  /** Show adapter/system/media entries too (default hides them — training needs corpora). */
  showAllKinds?: boolean
}) {
  // Training selects corpora: hide adapter/system/media entries tagged by the
  // just-cache classifier. Untagged entries (older backends/mocks) still show.
  const visible = showAllKinds
    ? datasets.datasets
    : datasets.datasets.filter((ds) => ds.kind == null || ds.kind === 'dataset')
  return (
    <div className="flex items-center gap-1.5">
      {visible.length === 0 ? (
        <>
          <span className="text-[10px] text-muted-foreground/60">
            No datasets — import one to get started.
          </span>
          <Button
            size="sm"
            variant="outline"
            className="h-7 text-[10px]"
            onClick={() => datasets.setImportModalOpen(true)}
          >
            + Import
          </Button>
        </>
      ) : (
        <>
          <Select value={value} onValueChange={onChange} disabled={disabled}>
            <SelectTrigger
              className="h-7 text-[11px] font-mono flex-1 max-w-sm"
              aria-label="Dataset selector"
            >
              <SelectValue placeholder="Select a dataset..." />
            </SelectTrigger>
            <SelectContent>
              {visible.map((ds) => (
                <SelectItem key={ds.id} value={ds.id}>
                  {datasetLabel(ds)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {showImport && (
            <Button
              size="sm"
              variant="outline"
              className="h-7 text-[10px]"
              onClick={() => datasets.setImportModalOpen(true)}
            >
              + Import
            </Button>
          )}
        </>
      )}
      <DatasetImportDialog
        open={datasets.importModalOpen}
        onOpenChange={datasets.setImportModalOpen}
        onImportComplete={(datasetId: string) => {
          void datasets
            .fetchDatasets()
            .then(() => datasets.setSelectedDataset(datasetId))
            .catch(() => {})
        }}
      />
    </div>
  )
}
