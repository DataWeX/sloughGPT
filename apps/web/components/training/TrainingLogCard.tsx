'use client'

import { useState, useCallback, useEffect, useRef, memo } from 'react'
import { ActionCard, Button, Skeleton } from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
import { trainingFacade } from '@/lib/training-facade'

const logsApi = trainingFacade.jobs

/** Concise rollup parsed from raw log lines (newest wins per metric). */
export interface LogSummary {
  step: string | null
  loss: string | null
  epoch: string | null
  rate: string | null
  errors: number
  warnings: number
}

const METRIC_RES = {
  step: /(?:step|global_step)[=: ](\d+)/i,
  loss: /(?:train_loss|loss)[=: ]([0-9]+\.[0-9]+)/i,
  epoch: /epoch[=: ](\d+)/i,
  rate: /(?:lr|learning_rate)[=: ]([0-9.e+-]+)/i,
} as const

export function summarizeLogLines(lines: string[]): LogSummary {
  const summary: LogSummary = {
    step: null,
    loss: null,
    epoch: null,
    rate: null,
    errors: 0,
    warnings: 0,
  }
  for (const line of lines) {
    const lower = line.toLowerCase()
    if (lower.includes('error') || lower.includes('failed') || lower.includes('traceback')) {
      summary.errors += 1
    } else if (lower.includes('warn')) {
      summary.warnings += 1
    }
    for (const [key, re] of Object.entries(METRIC_RES)) {
      const m = re.exec(line)
      if (m) summary[key as 'step' | 'loss' | 'epoch' | 'rate'] = m[1]
    }
  }
  return summary
}

const POLL_INTERVAL_MS = 5000
const MAX_VISIBLE_LINES = 500

interface TrainingLogCardProps {
  trainingRunning: boolean
  className?: string
}

function LogSummaryStrip({ summary }: { summary: LogSummary }) {
  const cells: { label: string; value: string | null; tone?: string }[] = [
    { label: 'step', value: summary.step },
    { label: 'loss', value: summary.loss, tone: 'text-success' },
    { label: 'epoch', value: summary.epoch },
    { label: 'lr', value: summary.rate },
  ]
  return (
    <div className="mb-2 flex flex-wrap items-center gap-1.5" aria-live="polite">
      {cells.map(
        (c) =>
          c.value != null && (
            <code
              key={c.label}
              className="rounded bg-muted/40 px-1.5 py-0.5 font-mono text-[10px] tabular-nums"
            >
              <span className="text-muted-foreground/60">{c.label} </span>
              <span className={c.tone ?? 'text-foreground'}>{c.value}</span>
            </code>
          ),
      )}
      {summary.errors > 0 && (
        <code className="rounded bg-destructive/10 px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-destructive">
          {summary.errors} error{summary.errors === 1 ? '' : 's'}
        </code>
      )}
      {summary.warnings > 0 && summary.errors === 0 && (
        <code className="rounded bg-warning/10 px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-warning">
          {summary.warnings} warning{summary.warnings === 1 ? '' : 's'}
        </code>
      )}
    </div>
  )
}

export const TrainingLogCard = memo(function TrainingLogCard({
  trainingRunning,
  className,
}: TrainingLogCardProps) {
  const [lines, setLines] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const prevLineCountRef = useRef(0)

  const fetchLogs = useCallback(async () => {
    try {
      const logs = await logsApi.trainingLog()
      setLines(logs)
      setError(null)
    } catch {
      if (lines.length === 0) setError('Could not load logs')
    }
  }, [lines.length])

  const fetchLogsLoading = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const logs = await logsApi.trainingLog()
      setLines(logs)
    } catch {
      if (lines.length === 0) setError('Could not load logs')
    } finally {
      setLoading(false)
    }
  }, [lines.length])

  // Auto-refresh during training
  useEffect(() => {
    if (!trainingRunning || !expanded) return
    void fetchLogs()
    const id = setInterval(() => {
      if (!document.hidden) void fetchLogs()
    }, POLL_INTERVAL_MS)
    return () => clearInterval(id)
  }, [trainingRunning, expanded, fetchLogs])

  // Auto-scroll when new lines arrive
  useEffect(() => {
    if (lines.length > prevLineCountRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
    prevLineCountRef.current = lines.length
  }, [lines])

  return (
    <ActionCard
      title="Training logs"
      className={className}
      actions={
        <>
          {expanded && trainingRunning && (
            <span className="text-[9px] text-muted-foreground/60 animate-pulse">live</span>
          )}
          <Button
            size="sm"
            variant="ghost"
            className="h-6 text-[10px]"
            onClick={() => {
              if (!expanded) {
                void fetchLogsLoading()
              }
              setExpanded(!expanded)
            }}
          >
            {expanded ? 'Hide' : 'Show'}
          </Button>
        </>
      }
      testId="training-log"
    >
      {expanded && (
        <>
          {lines.length > 0 && <LogSummaryStrip summary={summarizeLogLines(lines)} />}
          {loading ? (
            <div className="space-y-1">
              <Skeleton className="h-3 w-full" />
              <Skeleton className="h-3 w-3/4" />
              <Skeleton className="h-3 w-1/2" />
            </div>
          ) : error ? (
            <StatusBanner
              variant="error"
              message={error}
              dismissible={false}
              onRetry={() => void fetchLogsLoading()}
            />
          ) : lines.length === 0 ? (
            <p className="text-[10px] text-muted-foreground/60">No logs yet.</p>
          ) : (
            <div
              ref={scrollRef}
              className="max-h-64 overflow-y-auto rounded-lg bg-muted/20 p-2.5 font-mono text-[10px] leading-relaxed"
            >
              {lines.length > MAX_VISIBLE_LINES && (
                <div className="mb-1 text-[9px] text-muted-foreground/60">
                  Showing last {MAX_VISIBLE_LINES} of {lines.length} lines
                </div>
              )}
              {lines.slice(-MAX_VISIBLE_LINES).map((line, i) => (
                <div key={i} className="whitespace-pre-wrap break-all">
                  {line}
                </div>
              ))}
            </div>
          )}
          <Button
            size="sm"
            variant="ghost"
            className="h-6 text-[10px] mt-1"
            onClick={fetchLogsLoading}
            disabled={loading}
          >
            Refresh
          </Button>
        </>
      )}
    </ActionCard>
  )
})
