'use client'

import { Badge, Tooltip, TooltipContent, TooltipTrigger } from '@sloughgpt/strui'
import { humanizeDatasetName, isDatasetEmpty } from '@/lib/dataset-controller'
import type { Dataset } from '@/lib/dataset-controller'

export function formatSize(bytes: number | undefined | null): string {
  if (bytes == null || !Number.isFinite(bytes) || bytes <= 0) return 'Unknown size'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

/** Uppercase micro-label + tabular value — the tooltip's stat tile. */
function TooltipStat({ label, value, muted }: { label: string; value: string; muted?: boolean }) {
  return (
    <div className="rounded-md bg-muted/40 px-2 py-1.5">
      <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground/60">
        {label}
      </p>
      <p
        className={`font-numeric text-xs font-semibold ${
          muted ? 'text-muted-foreground/50' : 'text-foreground'
        }`}
      >
        {value}
      </p>
    </div>
  )
}

/** Label/value meta row — secondary facts sit quieter than the stats. */
function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="shrink-0 text-muted-foreground/60">{label}</span>
      <span className="truncate text-right text-foreground/80">{value}</span>
    </div>
  )
}

export function DatasetTooltipContent({ dataset }: { dataset: Dataset }) {
  const hasSize = dataset.size != null && Number.isFinite(dataset.size) && dataset.size > 0
  const hasSamples = dataset.samples != null && dataset.samples > 0
  const isEmpty = isDatasetEmpty(dataset)
  const hasTags = dataset.tags && dataset.tags.length > 0
  const hasVlm = dataset.vlm_metadata != null
  const createdDate = dataset.created_at ? new Date(dataset.created_at) : null
  const isValidDate = createdDate && !Number.isNaN(createdDate.getTime())
  const label = humanizeDatasetName(dataset.name)

  return (
    <div className="w-[240px] space-y-2">
      {/* Header — name + status at a glance */}
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate text-[11px] font-semibold leading-tight text-foreground">
            {label}
          </div>
          {dataset.name !== label && (
            <div className="truncate font-mono text-[10px] leading-tight text-muted-foreground/50">
              {dataset.id || dataset.name}
            </div>
          )}
        </div>
        <Badge variant={isEmpty ? 'warning' : 'success'} size="sm" className="shrink-0">
          {isEmpty ? 'Empty' : 'Ready'}
        </Badge>
      </div>

      {/* Origin chips */}
      {(dataset.source || dataset.type) && (
        <div className="flex flex-wrap gap-1">
          {dataset.source && (
            <Badge variant="outline" size="sm">
              {dataset.source}
            </Badge>
          )}
          {dataset.type && (
            <Badge variant="primary" size="sm">
              {dataset.type.toUpperCase()}
            </Badge>
          )}
        </div>
      )}

      {/* Stats — the numbers the eye should land on */}
      <div className="grid grid-cols-2 gap-1.5">
        <TooltipStat
          label="Samples"
          value={hasSamples ? dataset.samples!.toLocaleString() : '—'}
          muted={!hasSamples}
        />
        <TooltipStat
          label="Size"
          value={isEmpty ? '—' : formatSize(dataset.size)}
          muted={isEmpty || !hasSize}
        />
      </div>

      {/* Meta rows — quieter second tier */}
      {(isValidDate || hasTags) && (
        <div className="space-y-0.5 border-t border-border/30 pt-1.5 text-[11px]">
          {isValidDate && <MetaRow label="Created" value={createdDate!.toLocaleDateString()} />}
          {hasTags && <MetaRow label="Tags" value={dataset.tags!.slice(0, 3).join(', ')} />}
        </div>
      )}

      {hasVlm && (
        <div className="border-t border-border/30 pt-1.5 text-[11px] text-muted-foreground">
          {dataset.vlm_metadata!.image_count.toLocaleString()} images · {dataset.vlm_metadata!.type}
          {dataset.vlm_metadata!.auto_captioned && ' · auto-captioned'}
        </div>
      )}

      {/* Footer — honesty notes */}
      {isEmpty ? (
        <div className="rounded-md bg-warning/10 px-2 py-1 text-[11px] font-medium text-warning">
          Empty — no training text retrieved
        </div>
      ) : !hasSize && !hasSamples ? (
        <div className="text-[11px] italic text-muted-foreground/60">
          No size data available — preview for details
        </div>
      ) : null}
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
        {/* No native title attr here — it fires alongside the custom tooltip
            on the same hover and shows the raw id in a browser box (double
            tooltip). aria-label carries the a11y name; the tooltip carries
            the detail. */}
        <button
          type="button"
          onClick={() => onSelect(dataset.id)}
          aria-label={dataset.id || dataset.name}
          aria-pressed={selected}
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
          {isDatasetEmpty(dataset) ? (
            <span
              aria-hidden="true"
              data-testid="dataset-empty-dot"
              className="h-1.5 w-1.5 shrink-0 rounded-full bg-warning"
            />
          ) : (
            dataset.samples != null &&
            dataset.samples > 0 && (
              <span className="text-[11px] opacity-60 font-numeric">
                {dataset.samples.toLocaleString()}
              </span>
            )
          )}
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" variant="muted" sideOffset={6}>
        <DatasetTooltipContent dataset={dataset} />
      </TooltipContent>
    </Tooltip>
  )
}
