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
  init: '#636366',
  critical: '#febc2e',
  ready: '#0a7aff',
  background: '#28c840',
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
    pending: '#2c2c2e',
    running: '#febc2e',
    ok: '#28c840',
    timeout: '#ff5f57',
    error: '#ff5f57',
  }

  return (
    <div className="flex items-center gap-3 py-1">
      <div className="w-24 text-[10px] text-[#8e8e93] truncate text-right">
        {HOOK_LABELS[hook.name] ?? hook.name}
      </div>
      <div className="flex-1 h-4 bg-[#1c1c1e] rounded relative overflow-hidden">
        <div
          className="absolute h-full rounded transition-all duration-300"
          style={{
            left: `${left}%`,
            width: `${Math.max(2, width)}%`,
            backgroundColor: statusColors[hook.status] ?? '#2c2c2e',
          }}
        />
      </div>
      <div className="w-16 text-[10px] font-mono text-[#636366] text-right">
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
    <div className="rounded-xl border border-white/[0.06] bg-[#0a0a0a] overflow-hidden">
      <div className="h-9 px-4 bg-[#1c1c1e] border-b border-white/[0.06] flex items-center justify-between">
        <span className="text-[11px] font-medium text-[#8e8e93]">Startup Timeline</span>
        <span className="text-[10px] text-[#636366] font-mono">{startupElapsed.toFixed(1)}s total</span>
      </div>
      <div className="p-4 space-y-4">
        {grouped.map(({ stage, hooks: stageHooks }) => (
          <div key={stage}>
            <div className="flex items-center gap-2 mb-2">
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: STAGE_COLORS[stage] }}
              />
              <span className="text-[10px] text-[#8e8e93] uppercase tracking-wider">{stage}</span>
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
