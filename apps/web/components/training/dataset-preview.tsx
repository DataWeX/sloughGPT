'use client'

import type { DatasetPreview } from '@/lib/dataset-controller'

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border/50 bg-card px-2.5 py-2">
      <p className="text-[10px] uppercase tracking-wider text-muted-foreground/70">{label}</p>
      <p className="font-numeric text-xs tabular-nums text-foreground">{value}</p>
    </div>
  )
}

/** Tertiary layer: what the selected dataset actually contains. */
export function DatasetPreviewPanel({ preview }: { preview: DatasetPreview }) {
  const avg =
    (preview.total_samples ?? 0) > 0
      ? Math.round((preview.total_chars ?? 0) / (preview.total_samples ?? 1)).toLocaleString()
      : '0'

  return (
    <section
      aria-label="Dataset preview"
      className="space-y-2 rounded-lg border border-border bg-muted p-3"
    >
      <div className="flex items-center justify-between gap-2">
        <h4 className="text-xs font-medium uppercase tracking-wider text-muted-foreground/80">
          Preview
        </h4>
        <span className="text-[10px] text-muted-foreground/60">
          first {Math.min(preview.samples.length, 3)} of{' '}
          {(preview.total_samples ?? 0).toLocaleString()} samples
        </span>
      </div>

      <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-3">
        <Stat label="Samples" value={(preview.total_samples ?? 0).toLocaleString()} />
        <Stat label="Characters" value={(preview.total_chars ?? 0).toLocaleString()} />
        <Stat label="Avg / sample" value={`${avg} chars`} />
      </div>

      <div className="space-y-1 border-t border-border/40 pt-2 font-numeric text-[11px] text-muted-foreground/70">
        {preview.samples.slice(0, 3).map((sample, i) => (
          <div key={i} className="truncate">
            {sample.content}
          </div>
        ))}
      </div>
    </section>
  )
}
