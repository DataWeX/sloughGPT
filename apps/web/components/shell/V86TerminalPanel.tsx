'use client'

import { useState, useRef, useEffect } from 'react'
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
  const serialRef = useRef<HTMLTextAreaElement>(null)
  const { isBooted, error, init, reset } = useV86({ imageUrl, imageSize, memoryMb })
  const [initStarted, setInitStarted] = useState(false)

  // Initialize v86 when container is ready
  useEffect(() => {
    if (containerRef.current && !initStarted) {
      setInitStarted(true)
      init(containerRef.current, serialRef.current ?? undefined)
    }
  }, [init, initStarted])

  return (
    <div className={cn('flex flex-col rounded-lg border border-border bg-background', className)}>
      {/* Status bar */}
      <div className="flex items-center gap-2 border-b border-border px-3 py-1.5">
        <div
          className={cn(
            'h-2 w-2 rounded-full',
            isBooted ? 'bg-success' : error ? 'bg-destructive' : 'bg-warning animate-pulse',
          )}
        />
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

      {/* VGA screen (boot output). The demo v86 build renders into a
          pre-existing monospace text div (visible) and a hidden canvas
          (VGA/screenshot); the npm build ignores the div and creates its
          own canvas. Once the kernel switches its console to the serial
          port, this goes quiet and the shell moves to the terminal below. */}
      <div
        ref={containerRef}
        className="h-40 shrink-0 overflow-hidden bg-black"
        data-testid="v86-screen"
      >
        <div
          data-v86-text
          style={{ whiteSpace: 'pre', font: '14px monospace', lineHeight: '14px' }}
        />
        <canvas style={{ display: 'none' }} />
      </div>

      {/* Serial console (COM1 / ttyS0). The Buildroot ISO's interactive
          shell lives here — the login prompt and shell I/O are wired by the
          v86 serial adapter into this textarea. */}
      <textarea
        ref={serialRef}
        spellCheck={false}
        className="flex-1 min-h-0 w-full resize-none bg-black p-2 font-mono text-xs text-green-400 outline-none"
        style={{ whiteSpace: 'pre', lineHeight: '14px' }}
        data-testid="v86-serial"
        aria-label="VM serial console"
      />

      {/* Error display */}
      {error && (
        <div className="border-t border-border px-3 py-2 text-xs text-destructive">{error}</div>
      )}
    </div>
  )
}
