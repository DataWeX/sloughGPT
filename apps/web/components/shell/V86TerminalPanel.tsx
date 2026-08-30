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
      'flex flex-col rounded-lg border border-border bg-background',
      className,
    )}>
      {/* Status bar */}
      <div className="flex items-center gap-2 border-b border-border px-3 py-1.5">
        <div className={cn(
          'h-2 w-2 rounded-full',
          isBooted ? 'bg-success' : error ? 'bg-destructive' : 'bg-warning animate-pulse',
        )} />
        <span className="text-xs text-muted-foreground">
          {isBooted ? 'Linux VM Running' : error ? 'VM Error' : 'Booting...'}
        </span>
        {isBooted && (
          <button
            type="button"
            onClick={reset}
            className="ml-auto text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            Restart
          </button>
        )}
      </div>

      {/* VM screen */}
      <div
        ref={containerRef}
        className="flex-1 overflow-hidden bg-black"
        data-testid="v86-screen"
      />

      {/* Error display */}
      {error && (
        <div className="border-t border-border px-3 py-2 text-xs text-destructive">
          {error}
        </div>
      )}
    </div>
  )
}
