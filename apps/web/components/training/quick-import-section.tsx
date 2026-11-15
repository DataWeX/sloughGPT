'use client'

import type { ReactNode } from 'react'
import { Badge, Button, FoldSection } from '@sloughgpt/strui'

export interface QuickImportPreset {
  id: string
  label: string
}

export interface QuickImportSource {
  /** Stable key — also scopes the in-flight import state. */
  key: string
  label: string
  icon: ReactNode
  /** Static note shown under the chips (requirements, effort). */
  hint: string
  presets: QuickImportPreset[]
  /** Preset id currently importing, or null. */
  importingId: string | null
  onImport: (datasetId: string) => void
}

/**
 * One elevated source block: icon + label + count, the preset chips, and the
 * requirement note. Kaggle and HuggingFace are the same shape, so the shape
 * lives here once instead of being pasted twice.
 */
function SourceGroup({ source }: { source: QuickImportSource }) {
  return (
    <section
      aria-label={`${source.label} presets`}
      className="rounded-lg border border-border bg-muted p-3"
    >
      <div className="mb-2.5 flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <span
            aria-hidden="true"
            className="flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-muted/70 text-muted-foreground"
          >
            {source.icon}
          </span>
          <h4 className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            {source.label}
          </h4>
        </div>
        <span className="text-[10px] tabular-nums text-muted-foreground/60">
          {source.presets.length} preset{source.presets.length !== 1 ? 's' : ''}
        </span>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {source.presets.map((preset) => {
          const isImporting = source.importingId === preset.id
          return (
            <Button
              key={preset.id}
              size="sm"
              variant="outline"
              className="h-7 px-2.5 text-[11px]"
              title={`${source.label}: ${preset.id}`}
              loading={isImporting}
              loadingText="Importing..."
              disabled={source.importingId !== null && !isImporting}
              onClick={() => source.onImport(preset.id)}
            >
              {preset.label}
            </Button>
          )
        })}
      </div>

      <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground/70">{source.hint}</p>
    </section>
  )
}

/**
 * Collapsible "Quick Import" band holding every preset source.
 * Folded by default; pass `defaultOpen` to surface it for first-run users.
 */
export function QuickImportSection({
  sources,
  defaultOpen = false,
}: {
  sources: QuickImportSource[]
  defaultOpen?: boolean
}) {
  return (
    <FoldSection
      open={defaultOpen}
      heading={
        <span className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-medium text-foreground">Quick Import</span>
          <Badge size="sm" variant="outline">
            {sources.map((s) => s.label).join(' · ')}
          </Badge>
        </span>
      }
    >
      <div className="space-y-3">
        {sources.map((source) => (
          <SourceGroup key={source.key} source={source} />
        ))}
      </div>
    </FoldSection>
  )
}
