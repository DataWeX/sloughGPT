'use client'
export const dynamic = 'force-dynamic'

import { useState, useCallback, useEffect } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, CardHeader, CardTitle, Button, KpiGrid, StatCard } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { rateLimitController, type RateLimitStatus, type RateLimitCheck } from '@/lib/rate-limit-controller'

export default function RateLimitPage() {
  const addToast = useToastStore(s => s.addToast)
  const [status, setStatus] = useState<RateLimitStatus | null>(null)
  const [checkResult, setCheckResult] = useState<RateLimitCheck | null>(null)
  const [loading, setLoading] = useState(true)
  const [checking, setChecking] = useState(false)

  const fetchStatus = useCallback(async () => {
    setLoading(true)
    try {
      const s = await rateLimitController.getStatus()
      setStatus(s)
    } catch {
      addToast('Could not load rate limit status', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => { void fetchStatus() }, [fetchStatus])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'r' && !e.metaKey && !e.ctrlKey) { e.preventDefault(); void fetchStatus() }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [fetchStatus])

  const handleCheck = useCallback(async () => {
    setChecking(true)
    try {
      const result = await rateLimitController.check()
      setCheckResult(result)
    } catch {
      addToast('Could not check rate limit', 'error')
    } finally {
      setChecking(false)
    }
  }, [addToast])

  return (
    <PageContainer
      title="Rate Limiting"
      subtitle="Rate limit configuration and status"
      headerRight={
        <Button size="sm" variant="ghost" onClick={() => void fetchStatus()}>Refresh</Button>
      }
    >
      <KpiGrid columns={3}>
        <StatCard
          label="Status"
          value={status?.enabled ? 'Active' : 'Inactive'}
          icon={<span className={`inline-block w-2 h-2 rounded-full ${status?.enabled ? 'bg-success' : 'bg-muted-foreground'}`} />}
        />
        <StatCard label="Requests/min" value={status?.requests_per_minute?.toString() ?? '—'} />
        <StatCard label="Burst Size" value={status?.burst_size?.toString() ?? '—'} />
      </KpiGrid>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Check Rate Limit</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-xs text-muted-foreground">
            Simulates a request check to see if the current IP would be rate limited.
          </p>
          <Button size="sm" onClick={() => void handleCheck()} disabled={checking}>
            {checking ? 'Checking...' : 'Check Now'}
          </Button>
          {checkResult && (
            <div className="rounded bg-muted/30 px-3 py-2 text-xs space-y-1">
              <div className="flex justify-between">
                <span className="font-medium">Allowed: {checkResult.allowed ? 'Yes' : 'No'}</span>
                {!checkResult.allowed && <span className="text-warning">Wait: {checkResult.wait_time.toFixed(1)}s</span>}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="rounded bg-muted p-3 text-xs overflow-auto whitespace-pre-wrap">
            {status ? JSON.stringify(status, null, 2) : 'Loading...'}
          </pre>
        </CardContent>
      </Card>
    </PageContainer>
  )
}
