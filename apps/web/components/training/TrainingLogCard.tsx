'use client'

import { useState, useCallback, useEffect, useRef, memo } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, Skeleton } from '@sloughgpt/strui'
import { trainingJobsController } from '@/lib/training-controller'

const POLL_INTERVAL_MS = 5000

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
  const [expanded, setExpanded] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const prevLineCountRef = useRef(0)

  const fetchLogs = useCallback(async () => {
    try {
      const logs = await trainingJobsController.getTrainingLog()
      setLines(logs)
    } catch {
      // Silent — will retry on next poll
    }
  }, [])

  const fetchLogsLoading = useCallback(async () => {
    setLoading(true)
    try {
      const logs = await trainingJobsController.getTrainingLog()
      setLines(logs)
    } catch {
      // Silent
    } finally {
      setLoading(false)
    }
  }, [])

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
    <Card className={className}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Training logs</CardTitle>
          <div className="flex items-center gap-2">
            {expanded && trainingRunning && (
              <span className="text-[10px] text-muted-foreground animate-pulse">live</span>
            )}
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                if (!expanded) {
                  void fetchLogsLoading()
                }
                setExpanded(!expanded)
              }}
            >
              {expanded ? 'Hide' : 'Show'}
            </Button>
          </div>
        </div>
      </CardHeader>
      {expanded && (
        <CardContent>
          {loading ? (
            <div className="space-y-1">
              <Skeleton className="h-3 w-full" />
              <Skeleton className="h-3 w-3/4" />
              <Skeleton className="h-3 w-1/2" />
            </div>
          ) : lines.length === 0 ? (
            <p className="text-xs text-muted-foreground">No logs yet.</p>
          ) : (
            <div
              ref={scrollRef}
              className="max-h-72 overflow-y-auto rounded bg-muted/30 p-3 font-mono text-xs leading-relaxed"
            >
              {lines.map((line, i) => (
                <div key={i} className="whitespace-pre-wrap break-all">{line}</div>
              ))}
            </div>
          )}
          <Button
            size="sm"
            variant="ghost"
            className="mt-2"
            onClick={fetchLogsLoading}
            disabled={loading}
          >
            Refresh
          </Button>
        </CardContent>
      )}
    </Card>
  )
})
