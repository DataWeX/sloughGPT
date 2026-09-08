'use client'

import { memo, useState, useEffect, useCallback } from 'react'
import { Card, CardContent, Button, StatCard, KpiGrid, Spinner } from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
import { systemController, type InferencePoolStatus } from '@/lib/system-controller'
import { useToastStore } from '@/lib/toast-store'

interface InferencePoolCardProps {
  onRefresh?: () => void
}

export const InferencePoolCard = memo(function InferencePoolCard({ onRefresh }: InferencePoolCardProps) {
  const [status, setStatus] = useState<InferencePoolStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const addToast = useToastStore(s => s.addToast)

  const fetchStatus = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const s = await systemController.getInferencePoolStatus()
      setStatus(s)
    } catch {
      addToast('Could not load inference pool status', 'error')
      setError('Failed to load')
    } finally {
      setLoading(false)
    }
  }, [addToast])

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
          addToast('Could not load inference pool status', 'error')
          setError('Failed to load')
        }
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    return () => { active = false }
  }, [addToast])

  return (
    <Card data-testid="inference-pool">
      <div className="flex items-center justify-between border-b border-border/30 pb-2 pt-3 px-4">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Inference Pool</span>
        <Button size="sm" variant="ghost" className="h-7 text-xs" onClick={onRefresh} disabled={loading} aria-label="Refresh inference pool">
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
            <StatCard label="Max Workers" value={status.max_workers != null ? String(status.max_workers) : '—'} />
            <StatCard label="Queue Timeout" value={status.queue_timeout != null ? `${status.queue_timeout}ms` : '—'} />
          </KpiGrid>
        ) : null}
      </CardContent>
    </Card>
  )
})
