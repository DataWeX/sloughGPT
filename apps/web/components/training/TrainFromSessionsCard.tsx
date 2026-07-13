'use client'

import { useCallback, useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import { trainingController } from '@/lib/controllers'
import { useToastStore } from '@/lib/toast-store'
import type { AutoTrainStatus } from '@/lib/training-controller'

export function TrainFromSessionsCard() {
  const addToast = useToastStore(s => s.addToast)
  const [status, setStatus] = useState<AutoTrainStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [training, setTraining] = useState(false)

  const fetchStatus = useCallback(async () => {
    try {
      const s = await trainingController.getAutoTrainStatus()
      setStatus(s)
    } catch {
      // endpoint might not be available
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void fetchStatus() }, [fetchStatus])

  const handleTrainFromSessions = useCallback(async () => {
    setTraining(true)
    try {
      const result = await trainingController.trainFromSessions({ limit: 50, min_length: 5 })
      addToast(
        `Trained from conversations — loss ${result.loss.toFixed(4)}, ${result.steps} steps`,
        'success'
      )
      void fetchStatus()
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Training failed'
      addToast(msg, 'error')
    } finally {
      setTraining(false)
    }
  }, [addToast, fetchStatus])

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">From conversations</CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-16 w-full" />
        </CardContent>
      </Card>
    )
  }

  if (!status) return null

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">From conversations</CardTitle>
        {status.enabled && (
          <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <span className="relative flex h-1.5 w-1.5 shrink-0">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-500/60" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-green-500" />
            </span>
            Auto-training on
          </span>
        )}
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">
          Train from your server&apos;s own conversation logs — no data upload needed.
        </p>

        <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground/70">
          {status.enabled && (
            <>
              <span>Threshold: {status.threshold} conversations</span>
              <span>Interval: {status.interval_s}s</span>
              <span>Pending: {status.pending_conversations}</span>
            </>
          )}
          {status.total_trains > 0 && (
            <>
              <span>Trains: {status.total_trains}</span>
              {status.last_loss != null && <span>Last loss: {status.last_loss.toFixed(4)}</span>}
              {status.last_checkpoint && <span>Latest: {status.last_checkpoint}</span>}
            </>
          )}
        </div>

        <Button
          size="sm"
          onClick={() => void handleTrainFromSessions()}
          disabled={training}
        >
          {training ? 'Training...' : 'Train from conversations'}
        </Button>
      </CardContent>
    </Card>
  )
}
