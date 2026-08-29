'use client'

import { useEffect, useState, useCallback } from 'react'
import { cn, Card, CardContent, CardHeader, CardTitle, Button } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { workflowController, type WorkflowStatus } from '@/lib/workflow-controller'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'

interface WorkflowCardProps {
  onRefresh?: () => void
}

export function WorkflowCard({ onRefresh }: WorkflowCardProps) {
  const [status, setStatus] = useState<WorkflowStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const addToast = useToastStore((s) => s.addToast)

  const refetch = useCallback(async () => {
    setLoading(true)
    try {
      const s = await workflowController.status()
      setStatus(s)
    } catch (e: unknown) {
      logger.warning('Could not workflow status fetch', { exception: String(e) })
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    let failures = 0
    let timer: ReturnType<typeof setTimeout> | null = null
    let generation = 0
    let fetching = false

    const run = async (gen: number) => {
      if (fetching) return
      fetching = true
      try {
        const s = await workflowController.status()
        if (gen === generation) { setStatus(s); failures = 0 }
      } catch (e: unknown) {
        if (gen === generation) {
          failures++
          logger.warning('Could not workflow poll', { exception: String(e) })
        }
      } finally {
        if (gen === generation) setLoading(false)
        fetching = false
      }
      if (gen !== generation) return
      const delay = Math.min(10_000 * Math.pow(2, failures), 60_000)
      timer = setTimeout(() => tick(), delay)
    }

    const tick = () => {
      const gen = ++generation
      if (timer !== null) { clearTimeout(timer); timer = null }
      run(gen)
    }

    run(0)
    return () => { generation++; if (timer !== null) clearTimeout(timer) }
  }, [])

  const handleStart = async () => {
    try {
      await workflowController.start()
      await refetch()
      onRefresh?.()
    } catch (e: unknown) {
      logger.warning('Could not workflow start', { exception: String(e) })
      addToast('Could not start workflow', 'error')
    }
  }

  const handleStop = async () => {
    try {
      await workflowController.stop()
      await refetch()
      onRefresh?.()
    } catch (e: unknown) {
      logger.warning('Could not workflow stop', { exception: String(e) })
      addToast('Could not stop workflow', 'error')
    }
  }

  if (loading && !status) {
    return (
      <Card className="p-4" aria-busy="true">
        <CardContent className="p-0">
          <div className="animate-pulse h-20 rounded bg-muted/50" />
        </CardContent>
      </Card>
    )
  }

  const stats = (status?.stats || {}) as Partial<WorkflowStatus['stats']>
  const config = (status?.config || {}) as Partial<WorkflowStatus['config']>
  const lastRuns = (status?.last_runs || {}) as Partial<WorkflowStatus['last_runs']>

  const formatTime = (ts: number) => {
    if (!ts || ts === 0) return 'Never'
    const diff = Date.now() / 1000 - ts
    if (diff < 60) return `${Math.floor(diff)}s ago`
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
    return `${Math.floor(diff / 3600)}h ago`
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">Background Training</CardTitle>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            className="h-7 w-7 p-0"
            onClick={refetch}
            aria-label="Refresh"
          >
            <IconRefresh className="h-3.5 w-3.5" aria-hidden="true" />
          </Button>
          {status?.running ? (
            <Button variant="outline" size="sm" className="h-7 text-xs" onClick={handleStop} aria-label="Stop background training">
              Stop
            </Button>
          ) : (
            <Button size="sm" className="h-7 text-xs" onClick={handleStart} aria-label="Start background training">
              Start
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <div className={cn('h-2 w-2 rounded-full', status?.running ? 'bg-success' : 'bg-muted-foreground/40')} />
            <span className="text-xs text-muted-foreground">
              {status?.running ? 'Running' : 'Stopped'}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-muted-foreground">Auto-trains:</span>
              <span className="ml-1 font-medium">{stats?.auto_train_steps || 0}</span>
            </div>
            <div>
              <span className="text-muted-foreground">User adapter trained:</span>
              <span className="ml-1 font-medium">{stats?.user_adapter_trained || 0}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Rejected:</span>
              <span className="ml-1 font-medium">{stats?.user_adapter_rejected || 0}</span>
            </div>
            <div>
              <span className="text-muted-foreground">DPO trains:</span>
              <span className="ml-1 font-medium">{stats?.dpo_train_steps || 0}</span>
            </div>
          </div>

          <div className="text-[10px] text-muted-foreground space-y-1">
            <div className="flex justify-between">
              <span>Background interval:</span>
              <span>{config?.background_training_interval_seconds || 300}s</span>
            </div>
            <div className="flex justify-between">
              <span>Last background training:</span>
              <span>{formatTime(lastRuns?.background_training || 0)}</span>
            </div>
            <div className="flex justify-between">
              <span>Last aggregate:</span>
              <span>{formatTime(lastRuns?.aggregate || 0)}</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
