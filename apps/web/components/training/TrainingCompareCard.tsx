'use client'

import { useState, useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import type { Checkpoint } from '@/lib/souls-controller'

interface TrainingCompareCardProps {
  checkpoints: Checkpoint[]
  onLoad?: (name: string) => void
}

function diffColor(a?: number | null, b?: number | null, lowerBetter = true): string {
  if (a == null || b == null) return ''
  const diff = a - b
  if (Math.abs(diff) < 0.001) return ''
  const improved = lowerBetter ? diff < 0 : diff > 0
  return improved ? 'text-emerald-500' : 'text-red-500'
}

function diffLabel(a?: number | null, b?: number | null): string {
  if (a == null || b == null) return ''
  const diff = a - b
  if (Math.abs(diff) < 0.001) return '='
  const sign = diff > 0 ? '+' : ''
  return `${sign}${diff.toFixed(4)}`
}

function Row({ label, a, b, format, lowerBetter }: { label: string; a?: number | null; b?: number | null; format?: (v: number) => string; lowerBetter?: boolean }) {
  const fmt = format ?? ((v: number) => v.toFixed(4))
  const diff = diffLabel(a, b)
  const color = diffColor(a, b, lowerBetter ?? true)

  return (
    <div className="grid grid-cols-[1fr_auto_1fr] gap-2 items-center text-[11px] py-1 border-b border-border/30 last:border-0">
      <span className="text-right font-mono">{a != null ? fmt(a) : '—'}</span>
      <div className="flex flex-col items-center min-w-[80px]">
        <span className="text-muted-foreground/50">{label}</span>
        {diff && <span className={`font-mono ${color}`}>{diff}</span>}
      </div>
      <span className="font-mono">{b != null ? fmt(b) : '—'}</span>
    </div>
  )
}

function TraitDiff({ label, a, b }: { label: string; a?: number; b?: number }) {
  if (a == null && b == null) return null
  const diff = (a ?? 0) - (b ?? 0)
  const improved = diff > 0
  const color = Math.abs(diff) < 0.01 ? '' : improved ? 'text-emerald-500' : 'text-red-500'

  return (
    <div className="flex items-center justify-between text-[10px] py-0.5">
      <span className="text-muted-foreground/50 capitalize">{label}</span>
      <div className="flex items-center gap-2">
        <span className="font-mono w-8 text-right">{a != null ? a.toFixed(2) : '—'}</span>
        <span className="text-muted-foreground/30">→</span>
        <span className="font-mono w-8">{b != null ? b.toFixed(2) : '—'}</span>
        {Math.abs(diff) >= 0.01 && <span className={`font-mono ${color}`}>{diff > 0 ? '+' : ''}{diff.toFixed(2)}</span>}
      </div>
    </div>
  )
}

export function TrainingCompareCard({ checkpoints, onLoad }: TrainingCompareCardProps) {
  const [idxA, setIdxA] = useState(0)
  const [idxB, setIdxB] = useState(1)
  const [showTraits, setShowTraits] = useState(false)

  const sorted = useMemo(() =>
    [...checkpoints].sort((a, b) => {
      if (!a.born_at && !b.born_at) return 0
      if (!a.born_at) return 1
      if (!b.born_at) return -1
      return new Date(b.born_at).getTime() - new Date(a.born_at).getTime()
    }),
    [checkpoints]
  )

  if (sorted.length < 2) return null

  const a = sorted[Math.min(idxA, sorted.length - 1)]
  const b = sorted[Math.min(idxB, sorted.length - 1)]

  const allTraitKeys = useMemo(() => {
    const keys = new Set<string>()
    if (a.traits) Object.keys(a.traits).forEach(k => keys.add(k))
    if (b.traits) Object.keys(b.traits).forEach(k => keys.add(k))
    return [...keys]
  }, [a, b])

  return (
    <Card data-testid="training-compare-card">
      <CardHeader className="flex flex-row items-center justify-between py-3">
        <CardTitle className="text-base">Compare Checkpoints</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1">
            <label className="text-[10px] text-muted-foreground/50 uppercase tracking-wider">Checkpoint A</label>
            <select
              className="w-full rounded border border-border bg-background px-2 py-1.5 text-[11px]"
              value={idxA}
              onChange={e => setIdxA(Number(e.target.value))}
              data-testid="select-a"
            >
              {sorted.map((c, i) => (
                <option key={c.name} value={i}>{c.name} {c.loss != null ? `(${c.loss.toFixed(3)})` : ''}</option>
              ))}
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-[10px] text-muted-foreground/50 uppercase tracking-wider">Checkpoint B</label>
            <select
              className="w-full rounded border border-border bg-background px-2 py-1.5 text-[11px]"
              value={idxB}
              onChange={e => setIdxB(Number(e.target.value))}
              data-testid="select-b"
            >
              {sorted.map((c, i) => (
                <option key={c.name} value={i}>{c.name} {c.loss != null ? `(${c.loss.toFixed(3)})` : ''}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 text-[11px]">
          <div className="rounded border border-border/50 p-2 space-y-1">
            <span className="text-muted-foreground/50">A</span>
            <p className="font-medium truncate">{a.name}</p>
            {a.model_type && <Badge variant="secondary" className="text-[9px] px-1 py-0 h-3">{a.model_type}</Badge>}
          </div>
          <div className="rounded border border-border/50 p-2 space-y-1">
            <span className="text-muted-foreground/50">B</span>
            <p className="font-medium truncate">{b.name}</p>
            {b.model_type && <Badge variant="secondary" className="text-[9px] px-1 py-0 h-3">{b.model_type}</Badge>}
          </div>
        </div>

        <div className="space-y-0">
          <Row label="Loss" a={a.loss} b={b.loss} lowerBetter />
          <Row label="Val Loss" a={a.final_val_loss} b={b.final_val_loss} lowerBetter />
          <Row label="Duration" a={a.training_duration_s} b={b.training_duration_s} format={v => `${Math.round(v)}s`} lowerBetter={false} />
          <Row label="Epochs" a={a.epochs_trained} b={b.epochs_trained} format={v => String(Math.round(v))} lowerBetter={false} />
          <Row label="Steps" a={a.steps} b={b.steps} format={v => v.toLocaleString()} lowerBetter={false} />
          <Row label="Vocab" a={a.vocab_size} b={b.vocab_size} format={v => v.toLocaleString()} lowerBetter={false} />
          <Row label="Size" a={a.size_mb} b={b.size_mb} format={v => `${v} MB`} lowerBetter />
        </div>

        {allTraitKeys.length > 0 && (
          <div className="space-y-1">
            <Button
              variant="ghost"
              size="sm"
              className="h-6 text-[10px] text-muted-foreground"
              onClick={() => setShowTraits(!showTraits)}
            >
              {showTraits ? 'Hide traits' : `Show traits (${allTraitKeys.length})`}
            </Button>
            {showTraits && (
              <div className="rounded border border-border/30 p-2 space-y-0.5">
                {allTraitKeys.map(k => (
                  <TraitDiff key={k} label={k} a={a.traits?.[k]} b={b.traits?.[k]} />
                ))}
              </div>
            )}
          </div>
        )}

        {onLoad && (
          <div className="flex justify-end gap-2 pt-1">
            <Button variant="outline" size="sm" className="h-7 text-[11px]" onClick={() => onLoad(a.name)} disabled={a.is_loaded}>
              Load A
            </Button>
            <Button variant="outline" size="sm" className="h-7 text-[11px]" onClick={() => onLoad(b.name)} disabled={b.is_loaded}>
              Load B
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
