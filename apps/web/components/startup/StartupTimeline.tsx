'use client'

import { useLiveStatus, type HookStatus } from '@/hooks/useLiveStatus'
import { cn } from '@/lib/utils'

const HOOK_LABELS: Record<string, string> = {
  db_pool: 'Database',
  model_load: 'AI Model',
  core_routers: 'API Routes',
  model_ready: 'Model Warmup',
  training_restore: 'Training',
  wandb: 'Experiment Tracking',
  multimodal: 'Multimodal',
  metrics: 'Metrics',
  autotrainer: 'Auto Trainer',
  rag_ingest: 'RAG Ingestion',
}

const STAGE_COLORS: Record<string, string> = {
  init: 'hsl(var(--muted-foreground))',
  critical: 'hsl(var(--warning))',
  ready: 'hsl(var(--info))',
  background: 'hsl(var(--success))',
}

interface TimelineBarProps {
  hook: HookStatus
  maxDuration: number
  startOffset: number
}

function TimelineBar({ hook, maxDuration, startOffset }: TimelineBarProps) {
  const width = maxDuration > 0 ? (hook.duration_seconds / maxDuration) * 100 : 0
  const left = maxDuration > 0 ? (startOffset / maxDuration) * 100 : 0

  const statusColors: Record<string, string> = {
    pending: 'hsl(var(--muted))',
    running: 'hsl(var(--warning))',
    ok: 'hsl(var(--success))',
    timeout: 'hsl(var(--destructive))',
    error: 'hsl(var(--destructive))',
  }

  return (
    <div className="flex items-center gap-3 py-1">
      <div className="w-24 text-[10px] text-muted-foreground truncate text-right">
        {HOOK_LABELS[hook.name] ?? hook.name}
      </div>
      <div className="flex-1 h-4 bg-muted rounded relative overflow-hidden">
        <div
          className="absolute h-full rounded transition-all duration-300"
          style={{
            left: `${left}%`,
            width: `${Math.max(2, width)}%`,
            backgroundColor: statusColors[hook.status] ?? 'hsl(var(--muted))',
          }}
        />
      </div>
      <div className="w-16 text-[10px] font-mono text-muted-foreground text-right">
        {hook.duration_seconds > 0 ? `${hook.duration_seconds}s` : '—'}
      </div>
    </div>
  )
}

export function StartupTimeline() {
  const { startupHooks, startupElapsed } = useLiveStatus()

  const hooks = Object.values(startupHooks)
    .filter((h: HookStatus) => h.duration_seconds > 0 || h.status === 'running')
    .sort((a: HookStatus, b: HookStatus) => a.duration_seconds - b.duration_seconds)

  if (hooks.length === 0) return null

  const maxDuration = Math.max(
    ...hooks.map((h: HookStatus) => h.duration_seconds),
    startupElapsed || 1,
  )

  // Group by stage
  const stages = ['critical', 'ready', 'background']
  const grouped = stages.map(stage => ({
    stage,
    hooks: hooks.filter((h: HookStatus) => h.stage === stage),
  })).filter(g => g.hooks.length > 0)

  return (
    <div className="rounded-xl border border-border/10 bg-card overflow-hidden">
      <div className="h-9 px-4 bg-muted/50 border-b border-border/10 flex items-center justify-between">
        <span className="text-[11px] font-medium text-muted-foreground">Startup Timeline</span>
        <span className="text-[10px] text-muted-foreground font-mono">{startupElapsed.toFixed(1)}s total</span>
      </div>
      <div className="p-4 space-y-4">
        {grouped.map(({ stage, hooks: stageHooks }) => (
          <div key={stage}>
            <div className="flex items-center gap-2 mb-2">
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: STAGE_COLORS[stage] }}
              />
              <span className="text-[10px] text-muted-foreground uppercase tracking-wider">{stage}</span>
            </div>
            <div className="space-y-0.5">
              {stageHooks.map((hook: HookStatus) => (
                <TimelineBar
                  key={hook.name}
                  hook={hook}
                  maxDuration={maxDuration}
                  startOffset={0}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
