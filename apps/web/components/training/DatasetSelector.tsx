'use client'

import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { DatasetImportModal } from '@/components/DatasetImportModal'
import type { Dataset } from '@/lib/dataset-controller'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'

export function datasetLabel(ds: Dataset): string {
  const size = ds.size != null ? `${(ds.size / 1024).toFixed(1)} KB` : ''
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
}: {
  datasets: UseTrainingDatasetsReturn
  value: string
  onChange: (id: string) => void
  disabled?: boolean
  showImport?: boolean
}) {
  return (
    <div className="flex items-center gap-2">
      {datasets.datasets.length === 0 ? (
        <>
          <span className="text-xs text-muted-foreground">No datasets — import one to get started.</span>
          <Button size="sm" variant="outline" onClick={() => datasets.setImportModalOpen(true)}>
            + Import
          </Button>
        </>
      ) : (
        <>
          <Select value={value} onValueChange={onChange} disabled={disabled}>
            <SelectTrigger className="h-8 text-xs font-mono flex-1 max-w-sm" aria-label="Dataset selector">
              <SelectValue placeholder="Select a dataset..." />
            </SelectTrigger>
            <SelectContent>
              {datasets.datasets.map(ds => (
                <SelectItem key={ds.id} value={ds.id}>{datasetLabel(ds)}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          {showImport && (
            <Button size="sm" variant="outline" onClick={() => datasets.setImportModalOpen(true)}>
              + Import
            </Button>
          )}
        </>
      )}
      <DatasetImportModal
        open={datasets.importModalOpen}
        onOpenChange={datasets.setImportModalOpen}
        onImportComplete={(datasetId: string) => {
          void datasets.fetchDatasets()
            .then(() => datasets.setSelectedDataset(datasetId))
            .catch(() => {})
        }}
      />
    </div>
  )
}
