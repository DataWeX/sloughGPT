'use client'

import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, StatCard, KpiGrid } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { workflowController } from '@/lib/workflow-controller'
import { WorkflowPipeline } from '@/components/workflow/WorkflowPipeline'
import { WorkflowHealthCard } from '@/components/workflow/WorkflowHealthCard'
import { useToastStore } from '@/lib/toast-store'

export function WorkflowSection() {
  const [status, setStatus] = useState<Awaited<ReturnType<typeof workflowController.status>> | null>(null)
  const [loading, setLoading] = useState(true)
  const [toggling, setToggling] = useState(false)
  const [triggerMsg, setTriggerMsg] = useState<string | null>(null)
  const [triggering, setTriggering] = useState(false)
  const addToast = useToastStore(s => s.addToast)

  const fetchStatus = async () => {
    setLoading(true)
    try {
      setStatus(await workflowController.status())
    } catch {
      addToast('Failed to load workflow status', 'error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchStatus() }, [])

  const handleToggle = async () => {
    setToggling(true)
    try {
      if (status?.running) {
        await workflowController.stop()
      } else {
        await workflowController.start()
      }
      await fetchStatus()
    } catch {
      addToast('Failed to toggle workflow', 'error')
    } finally {
      setToggling(false)
    }
  }

  const handleTrigger = async (action: string) => {
    setTriggering(true)
    setTriggerMsg(null)
    try {
      const res = await workflowController.trigger(action)
      setTriggerMsg(`${action}: ${res.status ?? 'done'}`)
      await fetchStatus()
    } catch {
      setTriggerMsg(`${action} failed`)
    } finally {
      setTriggering(false)
    }
  }

  return (
    <>
      <div className="flex items-center justify-between border-b border-border/30 pb-2 pt-1">
        <h2 className="text-base font-medium">Feedback Pipeline</h2>
        <Button size="sm" variant="ghost" onClick={fetchStatus}>
          <IconRefresh className="h-4 w-4" />
        </Button>
      </div>

      {loading ? (
        <KpiGrid>
          <StatCard label="Status" value="Loading..." />
          <StatCard label="Feedback" value="..." />
          <StatCard label="Adapters" value="..." />
          <StatCard label="Health" value="..." />
        </KpiGrid>
      ) : (
        <KpiGrid>
          <StatCard
            label="Status"
            value={status?.running ? 'Running' : 'Stopped'}
          />
          <StatCard
            label="Feedback Recorded"
            value={String(status?.stats?.feedback_recorded ?? 0)}
          />
          <StatCard
            label="Auto-train Steps"
            value={String(status?.stats?.auto_train_steps ?? 0)}
          />
          <StatCard
            label="Workflow Runs"
            value={String(status?.stats?.workflow_runs ?? 0)}
          />
        </KpiGrid>
      )}

      {triggerMsg && (
        <div className="rounded-md bg-primary/10 border border-primary/20 px-4 py-3 text-sm text-primary">
          {triggerMsg}
          <button className="ml-2 underline" onClick={() => setTriggerMsg(null)}>Dismiss</button>
        </div>
      )}

      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-base">Status</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center gap-3">
            <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium ${
              status?.running
                ? 'bg-success/15 text-success'
                : 'bg-muted text-muted-foreground'
            }`}>
              <span className={`h-1.5 w-1.5 rounded-full ${status?.running ? 'bg-success' : 'bg-muted-foreground/40'}`} />
              {status?.running ? 'Running' : 'Stopped'}
            </span>
            <Button size="sm" onClick={handleToggle} disabled={toggling}>
              {toggling ? '...' : status?.running ? 'Stop' : 'Start'}
            </Button>
          </div>
        </CardContent>
      </Card>

      <WorkflowHealthCard status={status} />

      <WorkflowPipeline status={status} />

      {status?.config && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Configuration</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: 'Aggregate', value: `${status.config.aggregate_interval_minutes} min` },
                { label: 'Prune', value: `${status.config.prune_interval_minutes} min` },
                { label: 'Export', value: `${status.config.export_interval_hours} hr` },
                { label: 'Health Check', value: `${status.config.health_check_interval_seconds}s` },
              ].map(c => (
                <div key={c.label} className="rounded-md bg-muted/30 p-3 text-center">
                  <div className="text-xs text-muted-foreground">{c.label}</div>
                  <div className="text-sm font-mono font-medium">{c.value}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {status?.stats && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Stats</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-md bg-muted/30 p-3 text-center">
                <div className="text-xs text-muted-foreground">Feedback Recorded</div>
                <div className="text-lg font-mono font-medium">{status.stats.feedback_recorded ?? 0}</div>
              </div>
              <div className="rounded-md bg-muted/30 p-3 text-center">
                <div className="text-xs text-muted-foreground">Auto-train Steps</div>
                <div className="text-lg font-mono font-medium">{status.stats.auto_train_steps ?? 0}</div>
              </div>
              <div className="rounded-md bg-muted/30 p-3 text-center">
                <div className="text-xs text-muted-foreground">Workflow Runs</div>
                <div className="text-lg font-mono font-medium">{status.stats.workflow_runs ?? 0}</div>
              </div>
              <div className="rounded-md bg-muted/30 p-3 text-center">
                <div className="text-xs text-muted-foreground">DPO Train Steps</div>
                <div className="text-lg font-mono font-medium">{status.stats.dpo_train_steps ?? 0}</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Manual Triggers</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            {['aggregate', 'prune', 'export'].map(action => (
              <Button
                key={action}
                size="sm"
                variant="outline"
                onClick={() => handleTrigger(action)}
                disabled={triggering}
              >
                {action.charAt(0).toUpperCase() + action.slice(1)}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>
    </>
  )
}
