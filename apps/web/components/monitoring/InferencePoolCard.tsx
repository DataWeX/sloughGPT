'use client'
import { notify } from '@/lib/notify'

import { memo, useState, useEffect, useCallback } from 'react'
import { Card, CardContent, Button, StatCard, KpiGrid, Spinner } from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
import { systemController, type InferencePoolStatus } from '@/lib/system-controller'
// Module scope: `addToast` must keep a stable identity — it sits in
// useCallback/useEffect dependency arrays, and a per-render arrow would
// re-fire those effects on every render.
const addToast = (message: string, type?: string) =>
  notify.toast(message, { type: type as 'success' | 'error' | 'info' })

interface InferencePoolCardProps {
  onRefresh?: () => void
}

export const InferencePoolCard = memo(function InferencePoolCard({
  onRefresh,
}: InferencePoolCardProps) {
  const [status, setStatus] = useState<InferencePoolStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchStatus = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const s = await systemController.getInferencePoolStatus()
      setStatus(s)
    } catch {
      notify.toast('Could not load inference pool status', { type: 'error' })
      setError('Failed to load')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    let active = true
    const load = async () => {
      setLoading(true)
      setError(null)
      try {
        const s = await systemController.getInferencePoolStatus()
        if (active) setStatus(s)
      } catch {
        if (active) {
          notify.toast('Could not load inference pool status', { type: 'error' })
          setError('Failed to load')
        }
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    return () => {
      active = false
    }
  }, [])

  return (
    <Card data-testid="inference-pool">
      <div className="flex items-center justify-between border-b border-border/30 pb-2 pt-3 px-4">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          Inference Pool
        </span>
        <Button
          size="sm"
          variant="ghost"
          className="h-7 text-xs"
          onClick={onRefresh}
          disabled={loading}
          aria-label="Refresh inference pool"
        >
          <Spinner className="h-3 w-3" />
        </Button>
      </div>
      <CardContent className="pt-3">
        {loading && !status ? (
          <KpiGrid>
            <StatCard label="Initialized" value="" loading />
            <StatCard label="Max Workers" value="" loading />
            <StatCard label="Queue Timeout" value="" loading />
          </KpiGrid>
        ) : error ? (
          <StatusBanner variant="error" message={error} dismissible={false} />
        ) : status?.error ? (
          <StatusBanner variant="error" message={status.error} dismissible={false} />
        ) : status ? (
          <KpiGrid>
            <StatCard label="Initialized" value={status.initialized ? 'Yes' : 'No'} />
            <StatCard
              label="Max Workers"
              value={status.max_workers != null ? String(status.max_workers) : '—'}
            />
            <StatCard
              label="Queue Timeout"
              value={status.queue_timeout != null ? `${status.queue_timeout}ms` : '—'}
            />
          </KpiGrid>
        ) : null}
      </CardContent>
    </Card>
  )
})
