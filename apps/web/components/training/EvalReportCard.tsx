'use client'

import { useCallback, useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import { trainingController } from '@/lib/controllers'
import { logger } from '@/lib/dev-log'
import type { EvalHistoryEntry } from '@/lib/training-controller'

function VerdictBadge({ verdict }: { verdict: string }) {
  const color = verdict === 'improved' ? 'bg-success/15 text-success'
    : verdict === 'degraded' ? 'bg-destructive/15 text-destructive'
    : 'bg-muted text-muted-foreground'
  return (
    <span className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${color}`}>
      {verdict === 'improved' ? '↑ Improved' : verdict === 'degraded' ? '↓ Degraded' : verdict || '—'}
    </span>
  )
}

function MetricDelta({ label, value, invert }: { label: string; value?: number; invert?: boolean }) {
  if (value == null) return null
  const sign = value > 0 ? '+' : ''
  const isGood = invert ? value < 0 : value > 0
  const color = Math.abs(value) < 0.01 ? 'text-muted-foreground' : isGood ? 'text-success' : 'text-destructive'
  return (
    <div className="text-center">
      <p className="text-[10px] text-muted-foreground">{label}</p>
      <p className={`text-xs font-mono font-medium ${color}`}>{sign}{(value * 100).toFixed(1)}%</p>
    </div>
  )
}

function EvalRow({ entry }: { entry: EvalHistoryEntry }) {
  const d = entry.delta
  return (
    <div className="flex items-center justify-between rounded-lg border border-border/50 p-3 text-sm">
      <div className="min-w-0 flex-1 space-y-1">
        <div className="flex items-center gap-2">
          <p className="text-[10px] text-muted-foreground">{new Date(entry.timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</p>
          {d?.verdict && <VerdictBadge verdict={d.verdict} />}
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-0.5 text-[11px] text-muted-foreground">
          {entry.baseline && (
            <span>PPL: {entry.baseline.perplexity.toFixed(2)}</span>
          )}
          {entry.baseline && (
            <span>BLEU: {entry.baseline.bleu.toFixed(2)}</span>
          )}
          {entry.baseline && (
            <span>{entry.baseline.tokens_per_sec.toFixed(1)} tok/s</span>
          )}
        </div>
      </div>
      {d && (
        <div className="flex items-center gap-3 shrink-0 ml-3">
          <MetricDelta label="PPL" value={d.perplexity_delta} invert />
          <MetricDelta label="BLEU" value={d.bleu_delta} />
          <MetricDelta label="Speed" value={d.throughput_delta} />
        </div>
      )}
    </div>
  )
}

export function EvalReportCard() {
  const [entries, setEntries] = useState<EvalHistoryEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [expanded, setExpanded] = useState(false)

  const fetchHistory = useCallback(async () => {
    setLoading(true)
    try {
      const { results } = await trainingController.getEvalHistory(20)
      setEntries(results)
    } catch {
      logger.warning('EvalReportCard: eval endpoint not available', {})
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void fetchHistory() }, [fetchHistory])

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Evaluation History</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {[1, 2].map(i => <Skeleton key={i} className="h-14 w-full" />)}
        </CardContent>
      </Card>
    )
  }

  if (entries.length === 0) return (
    <Card>
      <CardHeader><CardTitle className="text-base">Evaluation History</CardTitle></CardHeader>
      <CardContent>
        <p className="text-xs text-muted-foreground text-center py-4">No evaluation reports yet. Train a model to see eval results.</p>
      </CardContent>
    </Card>
  )

  const display = expanded ? entries : entries.slice(0, 3)

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Evaluation History</CardTitle>
        <Button size="sm" variant="ghost" onClick={() => void fetchHistory()}>Refresh</Button>
      </CardHeader>
      <CardContent className="space-y-2">
        {display.map((entry, i) => (
          <EvalRow key={entry.timestamp + i} entry={entry} />
        ))}
        {entries.length > 3 && (
          <Button size="sm" variant="ghost" className="w-full" onClick={() => setExpanded(!expanded)}>
            {expanded ? 'Show less' : `Show all ${entries.length}`}
          </Button>
        )}
      </CardContent>
    </Card>
  )
}
