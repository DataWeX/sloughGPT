'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, KpiGrid, StatCard, Skeleton } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { modelController } from '@/lib/model-controller'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'

interface EngineStatus {
  engine: string
  version: string
  models_loaded: number
  uptime_s: number
  memory_usage_mb: number
}

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${seconds}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  return `${h}h ${m}m`
}

export default function EngineStatusCard() {
  const [status, setStatus] = useState<EngineStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [reloading, setReloading] = useState(false)
  const addToast = useToastStore(s => s.addToast)

  const fetchStatus = useCallback(async () => {
    try {
      const result = await modelController.getEngineStatus()
      setStatus(result)
    } catch (err) {
      addToast(extractErrorMessage(err, 'Could not load engine status'), 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => { fetchStatus() }, [fetchStatus])

  const handleReload = async () => {
    setReloading(true)
    try {
      await modelController.reloadEngine()
      addToast('Engine reloaded', 'success')
      await fetchStatus()
    } catch (err) {
      addToast(extractErrorMessage(err, 'Could not reload engine'), 'error')
    } finally {
      setReloading(false)
    }
  }

  if (loading) {
    return (
      <Card>
        <CardHeader><CardTitle className="text-base">Engine</CardTitle></CardHeader>
        <CardContent>
          <KpiGrid columns={3}>
            <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
            <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
            <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
          </KpiGrid>
        </CardContent>
      </Card>
    )
  }

  if (!status) return null

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Engine</CardTitle>
        <div className="flex items-center gap-1">
          <Button size="sm" variant="ghost" onClick={fetchStatus} aria-label="Refresh engine status">
            <IconRefresh className="h-3.5 w-3.5" />
          </Button>
          <Button size="sm" variant="outline" onClick={handleReload} disabled={reloading}>
            {reloading ? 'Reloading...' : 'Reload'}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <KpiGrid columns={3}>
          <StatCard label="Engine" value={status.engine} />
          <StatCard label="Version" value={status.version} />
          <StatCard label="Models Loaded" value={status.models_loaded} />
          <StatCard label="Uptime" value={formatUptime(status.uptime_s)} />
          <StatCard label="Memory" value={`${status.memory_usage_mb} MB`} />
        </KpiGrid>
      </CardContent>
    </Card>
  )
}
