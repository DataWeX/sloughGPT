'use client'

import { useState, useCallback, useEffect, useRef, memo } from 'react'
import { ActionCard, Button, Skeleton, Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
import { trainingJobsController } from '@/lib/training-controller'

const POLL_INTERVAL_MS = 5000
const MAX_VISIBLE_LINES = 500

interface TrainingLogCardProps {
  trainingRunning: boolean
  className?: string
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
      const logs = await trainingJobsController.getTrainingLog()
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
      const logs = await trainingJobsController.getTrainingLog()
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
          {loading ? (
            <div className="space-y-1">
              <Skeleton className="h-3 w-full" />
              <Skeleton className="h-3 w-3/4" />
              <Skeleton className="h-3 w-1/2" />
            </div>
          ) : error ? (
            <StatusBanner variant="error" message={error} dismissible={false} onRetry={() => void fetchLogsLoading()} />
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
                <div key={i} className="whitespace-pre-wrap break-all">{line}</div>
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
