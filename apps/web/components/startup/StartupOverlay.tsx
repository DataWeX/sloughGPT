'use client'

import { useEffect, useState } from 'react'
import { useLiveStatus, type StartupStage, type HookStatus } from '@/hooks/useLiveStatus'
import { cn } from '@/lib/utils'

const STAGE_LABELS: Record<StartupStage, string> = {
  init: 'Initializing',
  critical: 'Starting core services',
  ready: 'Loading AI model',
  background: 'Almost ready',
  unknown: 'Connecting',
}

const STAGE_ORDER: StartupStage[] = ['init', 'critical', 'ready', 'background']

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

export function StartupOverlay() {
  const { startupStage, startupModelProgress, startupModelProgressMessage, startupHooks, connected } = useLiveStatus()
  const [visible, setVisible] = useState(true)
  const [fadeOut, setFadeOut] = useState(false)

  const isReady = startupStage === 'background' || startupStage === 'ready'

  useEffect(() => {
    if (isReady && connected) {
      setFadeOut(true)
      const timer = setTimeout(() => setVisible(false), 600)
      return () => clearTimeout(timer)
    }
  }, [isReady, connected])

  if (!visible) return null

  const stageIndex = STAGE_ORDER.indexOf(startupStage)
  const progress = startupModelProgress > 0
    ? startupModelProgress
    : stageIndex >= 0
      ? (stageIndex + 1) / STAGE_ORDER.length
      : 0.1

  // Get active hooks (running or recently completed)
  const activeHooks = Object.values(startupHooks)
    .filter((h: HookStatus) => h.status === 'running' || (h.status === 'ok' && h.duration_seconds < 2))
    .slice(0, 3)

  return (
    <div
      className={cn(
        'fixed inset-0 z-[9999] flex flex-col items-center justify-center bg-[#0a0a0a] transition-opacity duration-500',
        fadeOut ? 'opacity-0 pointer-events-none' : 'opacity-100',
      )}
    >
      {/* Logo */}
      <div className="mb-8">
        <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-[#0a7aff] to-[#5856d6] flex items-center justify-center shadow-lg shadow-[#0a7aff]/20">
          <span className="text-white font-bold text-lg">S</span>
        </div>
      </div>

      {/* Stage label */}
      <h2 className="text-[14px] font-medium text-[#c7c7cc] mb-6">
        {STAGE_LABELS[startupStage] || 'Starting up'}
      </h2>

      {/* Progress bar */}
      <div className="w-48 h-1 rounded-full bg-[#1c1c1e] overflow-hidden mb-3">
        <div
          className="h-full rounded-full bg-gradient-to-r from-[#0a7aff] to-[#5856d6] transition-all duration-300 ease-out"
          style={{ width: `${Math.min(100, progress * 100)}%` }}
        />
      </div>

      {/* Progress details */}
      <div className="flex items-center gap-2 text-[11px] text-[#636366] mb-4">
        {startupModelProgress > 0 && (
          <span className="font-mono">{Math.round(startupModelProgress * 100)}%</span>
        )}
        {startupModelProgressMessage && (
          <span className="max-w-48 truncate">{startupModelProgressMessage}</span>
        )}
      </div>

      {/* Active hooks */}
      {activeHooks.length > 0 && (
        <div className="flex flex-col items-center gap-1.5 mb-6">
          {activeHooks.map((hook: HookStatus) => (
            <div key={hook.name} className="flex items-center gap-2 text-[10px]">
              <span className={cn(
                'w-1 h-1 rounded-full',
                hook.status === 'running' ? 'bg-[#febc2e] animate-pulse' : 'bg-[#28c840]',
              )} />
              <span className="text-[#8e8e93]">{HOOK_LABELS[hook.name] ?? hook.name}</span>
              {hook.status === 'ok' && (
                <span className="text-[#28c840] font-mono">{hook.duration_seconds}s</span>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Stage indicators */}
      <div className="flex items-center gap-1.5">
        {STAGE_ORDER.map((stage, i) => (
          <div
            key={stage}
            className={cn(
              'w-1.5 h-1.5 rounded-full transition-all duration-300',
              i < stageIndex
                ? 'bg-[#28c840]'
                : i === stageIndex
                  ? 'bg-[#0a7aff] scale-125'
                  : 'bg-[#2c2c2e]',
            )}
          />
        ))}
      </div>
    </div>
  )
}
