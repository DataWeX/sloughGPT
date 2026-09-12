'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { cn } from '@sloughgpt/strui'
import { useV86 } from '@/hooks/useV86'

export interface V86TerminalPanelProps {
  className?: string
  imageUrl?: string
  imageSize?: number
  memoryMb?: number
}

/**
 * v86-based terminal panel that runs a real Linux VM in the browser.
 *
 * @example
 * ```tsx
 * <V86TerminalPanel className="h-96" />
 * ```
 */
export function V86TerminalPanel({
  className,
  imageUrl,
  imageSize,
  memoryMb,
}: V86TerminalPanelProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const { isBooted, error, init, reset } = useV86({ imageUrl, imageSize, memoryMb })
  const [initStarted, setInitStarted] = useState(false)

  // Initialize v86 when container is ready
  useEffect(() => {
    if (containerRef.current && !initStarted) {
      setInitStarted(true)
      init(containerRef.current)
    }
  }, [init, initStarted])

  return (
    <div className={cn(
      'flex flex-col rounded-xl overflow-hidden',
      'border border-white/[0.08] shadow-2xl shadow-black/40',
      'bg-[#0a0a0a]',
      className,
    )}>
      {/* macOS title bar */}
      <div className="flex items-center h-11 px-4 bg-[#1c1c1e] border-b border-white/[0.06]">
        {/* Traffic lights */}
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-[#ff5f57] shadow-[inset_0_-1px_1px_rgba(0,0,0,0.15)] ring-1 ring-black/10" />
          <span className="w-3 h-3 rounded-full bg-[#febc2e] shadow-[inset_0_-1px_1px_rgba(0,0,0,0.15)] ring-1 ring-black/10" />
          <span className="w-3 h-3 rounded-full bg-[#28c840] shadow-[inset_0_-1px_1px_rgba(0,0,0,0.15)] ring-1 ring-black/10" />
        </div>
        {/* Title */}
        <div className="flex-1 flex items-center justify-center">
          <span className="text-[11px] font-medium text-[#8e8e93] select-none">v86 linux</span>
        </div>
        {/* Status + restart */}
        <div className="flex items-center gap-2">
          {isBooted && (
            <button
              type="button"
              onClick={reset}
              className="text-[10px] text-[#8e8e93] hover:text-[#c7c7cc] transition-colors"
            >
              Restart
            </button>
          )}
          <span className={cn(
            'w-1.5 h-1.5 rounded-full',
            isBooted ? 'bg-[#28c840]' : error ? 'bg-[#ff5f57]' : 'bg-[#febc2e] animate-pulse',
          )} />
        </div>
      </div>

      {/* VM screen */}
      <div
        ref={containerRef}
        className="flex-1 overflow-hidden bg-black"
        data-testid="v86-screen"
      />

      {/* Error display */}
      {error && (
        <div className="flex items-center gap-2 px-5 py-2.5 border-t border-white/[0.04] bg-[#ff5f57]/[0.08] text-[#ff5f57] text-[11px]">
          <span className="text-[10px] font-bold">!</span>
          {error}
        </div>
      )}
    </div>
  )
}
