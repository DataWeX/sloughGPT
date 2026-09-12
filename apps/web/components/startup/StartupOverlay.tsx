'use client'

import { useEffect, useState } from 'react'
import { useLiveStatus, type StartupStage } from '@/hooks/useLiveStatus'
import { cn } from '@/lib/utils'

const STAGE_LABELS: Record<StartupStage, string> = {
  init: 'Initializing',
  critical: 'Starting core services',
  ready: 'Loading AI model',
  background: 'Almost ready',
  unknown: 'Connecting',
}

const STAGE_ORDER: StartupStage[] = ['init', 'critical', 'ready', 'background']

export function StartupOverlay() {
  const { startupStage, startupModelProgress, startupModelProgressMessage, connected } = useLiveStatus()
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
      <div className="flex items-center gap-2 text-[11px] text-[#636366]">
        {startupModelProgress > 0 && (
          <span className="font-mono">{Math.round(startupModelProgress * 100)}%</span>
        )}
        {startupModelProgressMessage && (
          <span className="max-w-48 truncate">{startupModelProgressMessage}</span>
        )}
      </div>

      {/* Stage indicators */}
      <div className="flex items-center gap-1.5 mt-6">
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
