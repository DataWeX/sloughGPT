'use client'

import { memo, useState, useEffect, useCallback } from 'react'
import { cn, Card, CardContent } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
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

  if (loading) return null

  if (error || !status) {
    return (
      <Card className="p-3">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Inference Pool</span>
          {onRefresh && (
            <Button variant="outline" size="sm" className="text-[10px] h-7" onClick={() => { void fetchStatus(); onRefresh() }}>
              Retry
            </Button>
          )}
        </div>
        <CardContent className="p-0">
          <p className="text-xs text-muted-foreground py-2">{error || 'Not available'}</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Inference Pool</span>
        {onRefresh && (
          <Button variant="outline" size="sm" className="text-[10px] h-7" onClick={() => { void fetchStatus(); onRefresh() }}>
            Refresh
          </Button>
        )}
      </div>
      <CardContent className="p-0">
        <KpiGrid columns={3}>
          <StatCard
            label="Status"
            value={status.initialized ? 'Active' : 'Inactive'}
            icon={<span className={cn('inline-block w-2 h-2 rounded-full', status.initialized ? 'bg-success' : 'bg-muted-foreground')} />}
          />
          {status.max_workers != null && (
            <StatCard label="Max Workers" value={status.max_workers.toString()} numeric />
          )}
          {status.queue_timeout != null && (
            <StatCard label="Queue Timeout" value={`${status.queue_timeout}s`} />
          )}
        </KpiGrid>
        {status.error && (
          <div className="mt-2 rounded bg-destructive/10 px-2 py-1.5 text-[11px] text-destructive">
            {status.error}
          </div>
        )}
      </CardContent>
    </Card>
  )
})
