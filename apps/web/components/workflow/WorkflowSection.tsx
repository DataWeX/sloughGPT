'use client'

import { useState, useEffect } from 'react'
import { cn, ActionCard, Card, CardHeader, CardTitle, CardContent, Button, StatCard, KpiGrid } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { SectionLabel } from '@/components/composed/SectionLabel'
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
      addToast('Could not load workflow status', 'error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    let active = true
    const load = async () => {
      setLoading(true)
      try {
        const s = await workflowController.status()
        if (active) setStatus(s)
      } catch {
        if (active) addToast('Could not load workflow status', 'error')
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    return () => { active = false }
  }, [])

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
      addToast('Could not toggle workflow', 'error')
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
      <div className="flex items-center justify-between border-b border-border/30 pb-1.5 pt-0.5">
        <SectionLabel>Feedback Pipeline</SectionLabel>
        <Button size="sm" variant="ghost" className="h-6 w-6 p-0" onClick={fetchStatus} aria-label="Refresh">
          <IconRefresh className="h-3 w-3" />
        </Button>
      </div>

      {loading ? (
        <KpiGrid>
          <StatCard label="Status" value="" loading />
          <StatCard label="Feedback" value="" loading />
          <StatCard label="Adapters" value="" loading />
          <StatCard label="Health" value="" loading />
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
        <div className="rounded-lg bg-primary/10 border border-primary/20 px-2.5 py-1.5 text-[11px] text-primary">
          {triggerMsg}
          <button type="button" className="ml-1.5 text-[10px] underline text-primary/70 hover:text-primary" onClick={() => setTriggerMsg(null)}>Dismiss</button>
        </div>
      )}

      <ActionCard
        title="Status"
        contentClassName="space-y-2"
      >
          <div className="flex items-center gap-2">
            <span className={cn('inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[10px] font-medium', status?.running
                ? 'bg-success/15 text-success'
                : 'bg-muted text-muted-foreground')}>
              <span className={cn('h-1.5 w-1.5 rounded-full', status?.running ? 'bg-success' : 'bg-muted-foreground/40')} />
              {status?.running ? 'Running' : 'Stopped'}
            </span>
            <Button size="sm" className="h-6 text-[10px]" onClick={handleToggle} disabled={toggling}>
              {toggling ? '...' : status?.running ? 'Stop' : 'Start'}
            </Button>
          </div>
      </ActionCard>

      <WorkflowHealthCard status={status} />

      <WorkflowPipeline status={status} />

      {status?.config && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">Configuration</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-2">
              {[
                { label: 'Aggregate', value: `${status.config.aggregate_interval_minutes} min` },
                { label: 'Prune', value: `${status.config.prune_interval_minutes} min` },
                { label: 'Export', value: `${status.config.export_interval_hours} hr` },
                { label: 'Health Check', value: `${status.config.health_check_interval_seconds}s` },
              ].map(c => (
                <div key={c.label} className="rounded-lg bg-muted/30 p-2 text-center">
                  <div className="text-[10px] text-muted-foreground/60">{c.label}</div>
                  <div className="text-[11px] font-mono font-medium tabular-nums">{c.value}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {status?.stats && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">Stats</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-2">
              <div className="rounded-lg bg-muted/30 p-2 text-center">
                <div className="text-[10px] text-muted-foreground/60">Feedback Recorded</div>
                <div className="text-[13px] font-mono font-medium tabular-nums">{status.stats.feedback_recorded ?? 0}</div>
              </div>
              <div className="rounded-lg bg-muted/30 p-2 text-center">
                <div className="text-[10px] text-muted-foreground/60">Auto-train Steps</div>
                <div className="text-[13px] font-mono font-medium tabular-nums">{status.stats.auto_train_steps ?? 0}</div>
              </div>
              <div className="rounded-lg bg-muted/30 p-2 text-center">
                <div className="text-[10px] text-muted-foreground/60">Workflow Runs</div>
                <div className="text-[13px] font-mono font-medium tabular-nums">{status.stats.workflow_runs ?? 0}</div>
              </div>
              <div className="rounded-lg bg-muted/30 p-2 text-center">
                <div className="text-[10px] text-muted-foreground/60">DPO Train Steps</div>
                <div className="text-[13px] font-mono font-medium tabular-nums">{status.stats.dpo_train_steps ?? 0}</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-xs">Manual Triggers</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-1.5">
            {['aggregate', 'prune', 'export'].map(action => (
              <Button
                key={action}
                size="sm"
                variant="outline"
                className="h-6 text-[10px]"
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
