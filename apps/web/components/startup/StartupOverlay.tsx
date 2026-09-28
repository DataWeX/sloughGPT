'use client'

import { useEffect, useState } from 'react'
import { useLiveStatus, type StartupStage, type HookStatus } from '@/hooks/useLiveStatus'
import { ManMark } from '@/components/brand/ManMark'
import { logStateEvent } from '@/lib/state-events'
import { cn } from '@/lib/utils'

const STAGE_LABELS: Record<StartupStage, string> = {
  init: 'Initializing',
  critical: 'Starting core services',
  ready: 'Loading AI model',
  background: 'Almost ready',
  unknown: 'Connecting',
}

const STAGE_ORDER: StartupStage[] = ['init', 'critical', 'ready', 'background']

const STUCK_TIMEOUT_MS = 8_000

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
  const {
    startupStage,
    startupModelProgress,
    startupModelProgressMessage,
    startupHooks,
    startupElapsed,
    connected,
  } = useLiveStatus()
  const [visible, setVisible] = useState(true)
  const [fadeOut, setFadeOut] = useState(false)
  const [showDetails, setShowDetails] = useState(false)
  const [stuck, setStuck] = useState(false)

  const stageIndex = STAGE_ORDER.indexOf(startupStage)
  const isReady = startupStage === 'background' || startupStage === 'ready'

  useEffect(() => {
    logStateEvent('overlay_shown', {
      kind: 'overlay',
      message: `overlay_shown stage=${startupStage}`,
      data: { stage: startupStage },
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Stuck-connecting watchdog: stage never leaves "unknown" (no health payload).
  useEffect(() => {
    if (isReady || stageIndex >= 0) {
      setStuck(false)
      return
    }
    const timer = setTimeout(() => {
      setStuck(true)
      logStateEvent('overlay_stuck', {
        kind: 'overlay',
        message: `overlay_stuck stage=${startupStage} after ${STUCK_TIMEOUT_MS}ms`,
        data: { stage: startupStage, timeout_ms: STUCK_TIMEOUT_MS },
      })
    }, STUCK_TIMEOUT_MS)
    return () => clearTimeout(timer)
  }, [isReady, stageIndex, startupStage])

  useEffect(() => {
    if (isReady && connected) {
      setFadeOut(true)
      logStateEvent('overlay_hidden', {
        kind: 'overlay',
        message: `overlay_hidden stage=${startupStage}`,
        data: { stage: startupStage },
      })
      const timer = setTimeout(() => setVisible(false), 600)
      return () => clearTimeout(timer)
    }
  }, [isReady, connected, startupStage])

  if (!visible) return null

  const isIndeterminate = stageIndex < 0 && startupModelProgress <= 0
  const progress =
    startupModelProgress > 0
      ? startupModelProgress
      : stageIndex >= 0
        ? (stageIndex + 1) / STAGE_ORDER.length
        : 0.1
  const progressPct = Math.min(100, Math.round(progress * 100))

  // Get completed hooks for timing breakdown
  const completedHooks = Object.values(startupHooks)
    .filter((h: HookStatus) => h.status === 'ok' && h.duration_seconds > 0)
    .sort((a: HookStatus, b: HookStatus) => b.duration_seconds - a.duration_seconds)
    .slice(0, 5)

  // Get active hooks (running)
  const activeHooks = Object.values(startupHooks)
    .filter((h: HookStatus) => h.status === 'running')
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
        <ManMark className="h-12 w-12 rounded-2xl text-lg font-bold shadow-lg shadow-primary/20" />
      </div>

      {/* Stage label */}
      <h2 className="text-[14px] font-medium text-[#c7c7cc] mb-6">
        {stuck ? 'Still connecting' : STAGE_LABELS[startupStage] || 'Starting up'}
      </h2>

      {/* Progress bar */}
      <div
        role="progressbar"
        aria-label="Startup progress"
        aria-valuemin={0}
        aria-valuemax={100}
        {...(isIndeterminate ? {} : { 'aria-valuenow': progressPct })}
        className="w-48 h-1 rounded-full bg-[#1c1c1e] overflow-hidden mb-3"
      >
        {isIndeterminate ? (
          <div className="sl-bar-shimmer h-full w-1/3 rounded-full bg-gradient-to-r from-[#0a7aff] to-[#5856d6]" />
        ) : (
          <div
            className="h-full rounded-full bg-gradient-to-r from-[#0a7aff] to-[#5856d6] transition-all duration-300 ease-out"
            style={{ width: `${progressPct}%` }}
          />
        )}
      </div>

      {/* Progress details */}
      <div className="flex items-center gap-2 text-[11px] text-[#636366] mb-4" aria-live="polite">
        {!isIndeterminate && startupModelProgress > 0 && (
          <span className="font-mono">{Math.round(startupModelProgress * 100)}%</span>
        )}
        {startupModelProgressMessage && (
          <span className="max-w-48 truncate">{startupModelProgressMessage}</span>
        )}
        {startupElapsed > 0 && <span className="font-mono">{startupElapsed.toFixed(1)}s</span>}
      </div>

      {/* Stuck-connecting recovery */}
      {stuck && (
        <div role="alert" className="flex flex-col items-center gap-2 mb-4">
          <p className="text-[11px] text-[#febc2e]">
            No response from the server yet — check that the backend is running
          </p>
          <button
            type="button"
            onClick={() => {
              logStateEvent('overlay_retry', {
                kind: 'overlay',
                message: 'overlay_retry reload requested',
                data: { stage: startupStage },
              })
              window.location.reload()
            }}
            className="px-4 py-1.5 rounded-full bg-[#0a7aff] text-white text-[12px] font-medium transition-all duration-200 hover:bg-[#0a7aff]/90 active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#0a7aff] focus-visible:ring-offset-2 focus-visible:ring-offset-[#0a0a0a]"
          >
            Retry
          </button>
        </div>
      )}

      {/* Active hooks */}
      {activeHooks.length > 0 && (
        <div className="flex flex-col items-center gap-1.5 mb-4">
          {activeHooks.map((hook: HookStatus) => (
            <div key={hook.name} className="flex items-center gap-2 text-[10px]">
              <span className="w-1 h-1 rounded-full bg-[#febc2e] animate-pulse" />
              <span className="text-[#8e8e93]">{HOOK_LABELS[hook.name] ?? hook.name}</span>
            </div>
          ))}
        </div>
      )}

      {/* Timing breakdown (click to toggle) */}
      {completedHooks.length > 0 && (
        <button
          type="button"
          onClick={() => setShowDetails(!showDetails)}
          className="text-[9px] text-[#636366] hover:text-[#8e8e93] transition-colors mb-4"
        >
          {showDetails ? 'Hide details' : 'Show timing'}
        </button>
      )}

      {showDetails && completedHooks.length > 0 && (
        <div className="w-48 space-y-1 mb-4">
          {completedHooks.map((hook: HookStatus) => (
            <div key={hook.name} className="flex items-center justify-between text-[9px]">
              <span className="text-[#8e8e93] truncate">{HOOK_LABELS[hook.name] ?? hook.name}</span>
              <span className="text-[#636366] font-mono ml-2">{hook.duration_seconds}s</span>
            </div>
          ))}
        </div>
      )}

      {/* Stage indicators */}
      <div role="group" aria-label="Startup stages" className="flex items-center gap-1.5">
        {STAGE_ORDER.map((stage, i) => {
          const state =
            stageIndex < 0
              ? 'unknown'
              : i < stageIndex
                ? 'done'
                : i === stageIndex
                  ? 'active'
                  : 'pending'
          const stateLabel =
            state === 'done'
              ? 'done'
              : state === 'active'
                ? 'active'
                : state === 'unknown'
                  ? 'connecting'
                  : 'pending'
          return (
            <div
              key={stage}
              role="img"
              aria-label={`Stage ${i + 1} of ${STAGE_ORDER.length}: ${STAGE_LABELS[stage]} (${stateLabel})`}
              className={cn(
                'w-1.5 h-1.5 rounded-full transition-all duration-300',
                state === 'done' && 'bg-[#28c840]',
                state === 'active' && 'bg-[#0a7aff] scale-125 sl-dot-pulse',
                state === 'pending' && 'bg-[#2c2c2e]',
                state === 'unknown' && 'bg-[#0a7aff]/60 sl-dot-wave',
              )}
              style={state === 'unknown' ? { animationDelay: `${i * 0.15}s` } : undefined}
            />
          )
        })}
      </div>
      <span className="sr-only" role="status">
        {stageIndex < 0
          ? 'Connecting'
          : `Stage ${stageIndex + 1} of ${STAGE_ORDER.length}: ${STAGE_LABELS[startupStage]}`}
      </span>
    </div>
  )
}
