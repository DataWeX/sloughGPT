'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import type { Checkpoint } from '@/lib/souls-controller'

interface TrainingRunCardProps {
  checkpoint: Checkpoint
  index: number
  isBest?: boolean
  onLoad?: (checkpoint: Checkpoint) => void
}

function formatDuration(seconds?: number | null): string {
  if (seconds == null || seconds <= 0) return '—'
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return m > 0 ? `${m}m ${s}s` : `${s}s`
}

function lossColor(loss?: number | null): string {
  if (loss == null) return 'text-muted-foreground'
  if (loss < 1) return 'text-emerald-500'
  if (loss < 2) return 'text-green-500'
  if (loss < 3) return 'text-yellow-500'
  return 'text-orange-500'
}

function verdictBadge(verdict?: string) {
  if (!verdict) return null
  const map: Record<string, { label: string; className: string }> = {
    overfit: { label: 'Overfit', className: 'bg-red-500/10 text-red-500 border-red-500/20' },
    underfit: { label: 'Underfit', className: 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20' },
    good: { label: 'Good', className: 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20' },
    stale: { label: 'Stale', className: 'bg-muted text-muted-foreground border-muted-foreground/20' },
  }
  const v = map[verdict] ?? { label: verdict, className: 'bg-muted text-muted-foreground border-muted-foreground/20' }
  return (
    <Badge variant="outline" className={`text-[10px] px-1 py-0 h-4 ${v.className}`}>
      {v.label}
    </Badge>
  )
}

export function TrainingRunCard({ checkpoint: c, index, isBest, onLoad }: TrainingRunCardProps) {
  const [expanded, setExpanded] = useState(false)

  return (
    <Card
      className={`transition-all ${isBest ? 'border-emerald-500/30 bg-emerald-500/5' : ''}`}
      data-testid="training-run-card"
    >
      <CardHeader
        className="flex flex-row items-center justify-between cursor-pointer py-3"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-[10px] text-muted-foreground/50 font-mono w-4 text-right shrink-0">
            {index + 1}
          </span>
          <span className="text-sm font-medium truncate">{c.name}</span>
          {isBest && (
            <Badge variant="outline" className="text-[10px] px-1 py-0 h-4 bg-emerald-500/10 text-emerald-500 border-emerald-500/20 shrink-0">
              best
            </Badge>
          )}
          {verdictBadge(c.verdict)}
        </div>

        <div className="flex items-center gap-3 text-[11px] text-muted-foreground shrink-0">
          {c.loss != null && (
            <span className={`font-mono ${lossColor(c.loss)}`}>
              {c.loss.toFixed(3)}
            </span>
          )}
          {c.training_duration_s != null && (
            <span className="font-mono">{formatDuration(c.training_duration_s)}</span>
          )}
          {c.model_type && (
            <Badge variant="secondary" className="text-[10px] px-1 py-0 h-4">
              {c.model_type}
            </Badge>
          )}
          <span className="text-[10px] text-muted-foreground/40">
            {expanded ? '\u25B2' : '\u25BC'}
          </span>
        </div>
      </CardHeader>

      {expanded && (
        <CardContent className="pt-0 pb-3 space-y-3">
          <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-[11px]">
            {c.soul && <Row label="Soul" value={c.soul} />}
            {c.loss != null && <Row label="Loss" value={c.loss.toFixed(4)} className={lossColor(c.loss)} />}
            {c.final_val_loss != null && <Row label="Val Loss" value={c.final_val_loss.toFixed(4)} />}
            {c.training_duration_s != null && <Row label="Duration" value={formatDuration(c.training_duration_s)} />}
            {c.epochs_trained != null && <Row label="Epochs" value={`${c.epochs_trained}${c.epochs != null ? `/${c.epochs}` : ''}`} />}
            {c.steps != null && <Row label="Steps" value={c.steps.toLocaleString()} />}
            {c.vocab_size != null && <Row label="Vocab" value={c.vocab_size.toLocaleString()} />}
            {c.model_type && <Row label="Type" value={c.model_type} />}
            {c.lineage && <Row label="Lineage" value={c.lineage} />}
            {c.training_dataset && <Row label="Dataset" value={c.training_dataset} />}
            {c.size_mb != null && <Row label="Size" value={`${c.size_mb} MB`} />}
            {c.is_loaded && (
              <Row label="Status" value="Loaded" className="text-emerald-500" />
            )}
          </div>

          {c.traits && Object.keys(c.traits).length > 0 && (
            <div className="space-y-1">
              <span className="text-[10px] text-muted-foreground/50 uppercase tracking-wider">Traits</span>
              <div className="flex flex-wrap gap-1">
                {Object.entries(c.traits).map(([k, v]) => (
                  <Badge key={k} variant="secondary" className="text-[10px] px-1 py-0 h-4">
                    {k}: {typeof v === 'number' ? v.toFixed(2) : String(v)}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {c.personality && Object.keys(c.personality).length > 0 && (
            <div className="space-y-1">
              <span className="text-[10px] text-muted-foreground/50 uppercase tracking-wider">Personality</span>
              <div className="flex flex-wrap gap-1">
                {Object.entries(c.personality).map(([k, v]) => (
                  <Badge key={k} variant="secondary" className="text-[10px] px-1 py-0 h-4">
                    {k}: {typeof v === 'number' ? v.toFixed(2) : String(v)}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {(c.tagline || c.description) && (
            <div className="space-y-1">
              {c.tagline && <p className="text-[11px] text-muted-foreground italic">{c.tagline}</p>}
              {c.description && <p className="text-[11px] text-muted-foreground/70">{c.description}</p>}
            </div>
          )}

          {onLoad && (
            <div className="flex justify-end pt-1">
              <Button
                variant="outline"
                size="sm"
                className="h-7 text-[11px]"
                onClick={() => onLoad(c)}
                disabled={c.is_loaded}
              >
                {c.is_loaded ? 'Loaded' : 'Load'}
              </Button>
            </div>
          )}
        </CardContent>
      )}
    </Card>
  )
}

function Row({ label, value, className }: { label: string; value: string; className?: string }) {
  return (
    <>
      <span className="text-muted-foreground/50">{label}</span>
      <span className={`truncate ${className ?? ''}`}>{value}</span>
    </>
  )
}
