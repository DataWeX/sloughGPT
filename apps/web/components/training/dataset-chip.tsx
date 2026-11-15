'use client'

import { Badge, Tooltip, TooltipContent, TooltipTrigger } from '@sloughgpt/strui'
import { humanizeDatasetName } from '@/lib/dataset-controller'
import type { Dataset } from '@/lib/dataset-controller'

export function formatSize(bytes: number | undefined | null): string {
  if (bytes == null || !Number.isFinite(bytes) || bytes <= 0) return 'Unknown size'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function DatasetTooltipContent({ dataset }: { dataset: Dataset }) {
  const hasSize = dataset.size != null && Number.isFinite(dataset.size) && dataset.size > 0
  const hasSamples = dataset.samples != null && dataset.samples > 0
  const hasTags = dataset.tags && dataset.tags.length > 0
  const hasVlm = dataset.vlm_metadata != null
  const createdDate = dataset.created_at ? new Date(dataset.created_at) : null
  const isValidDate = createdDate && !Number.isNaN(createdDate.getTime())

  return (
    <div className="space-y-1.5 max-w-[240px]">
      <div className="font-medium text-foreground text-[11px]">
        {humanizeDatasetName(dataset.name)}
      </div>
      {dataset.name !== humanizeDatasetName(dataset.name) && (
        <div className="text-[10px] text-muted-foreground/60 font-mono">
          {dataset.id || dataset.name}
        </div>
      )}
      <div className="flex flex-wrap gap-1">
        {dataset.source && (
          <Badge variant="outline" size="sm">
            {dataset.source}
          </Badge>
        )}
        {dataset.type && (
          <Badge variant="outline" size="sm">
            {dataset.type.toUpperCase()}
          </Badge>
        )}
      </div>
      <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[10px] text-muted-foreground">
        {hasSamples && (
          <>
            <span>Samples</span>
            <span className="text-right font-numeric">{dataset.samples!.toLocaleString()}</span>
          </>
        )}
        <span>Size</span>
        <span className="text-right font-numeric">{formatSize(dataset.size)}</span>
        {isValidDate && (
          <>
            <span>Created</span>
            <span className="text-right">{createdDate!.toLocaleDateString()}</span>
          </>
        )}
        {hasTags && (
          <>
            <span>Tags</span>
            <span className="text-right truncate max-w-[120px]">
              {dataset.tags!.slice(0, 3).join(', ')}
            </span>
          </>
        )}
      </div>
      {hasVlm && (
        <div className="text-[10px] text-muted-foreground border-t border-border/30 pt-1">
          {dataset.vlm_metadata!.image_count.toLocaleString()} images · {dataset.vlm_metadata!.type}
          {dataset.vlm_metadata!.auto_captioned && ' · auto-captioned'}
        </div>
      )}
      {!hasSize && !hasSamples && (
        <div className="text-[10px] text-muted-foreground/60 italic">
          No size data available — preview for details
        </div>
      )}
    </div>
  )
}

export function DatasetChip({
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
          aria-label={dataset.id || dataset.name}
          aria-pressed={selected}
          title={dataset.id || dataset.name}
          className={`
            inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px]
            transition-all duration-150
            focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring
            focus-visible:ring-offset-1 focus-visible:ring-offset-background
            ${
              selected
                ? 'bg-primary/15 text-primary border border-primary/30 shadow-sm'
                : 'bg-muted/40 text-muted-foreground border border-border/60 hover:bg-muted/70 hover:text-foreground hover:border-border'
            }
          `}
        >
          <span className="truncate max-w-[120px]">{humanizeDatasetName(dataset.name)}</span>
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
