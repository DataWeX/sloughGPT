'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { cn } from '@sloughgpt/strui'
import { useV86, probeImage, IMAGE_URLS } from '@/hooks/useV86'
import type { V86BootMedia, V86ImageKind } from '@/hooks/useV86'

export interface V86TerminalPanelProps {
  className?: string
  imageUrl?: string
  imageSize?: number
  memoryMb?: number
}

const MEDIA_OPTIONS: readonly { value: V86BootMedia; label: string }[] = [
  { value: 'auto', label: 'auto' },
  { value: 'hda', label: 'hda' },
  { value: 'kernel', label: 'kernel' },
  { value: 'iso', label: 'iso' },
]

async function probeKind(kind: V86ImageKind): Promise<boolean> {
  const probes = await Promise.all(IMAGE_URLS[kind].map(probeImage))
  return probes.some((p) => p.available)
}

/**
 * v86-based terminal panel that runs a real Linux VM in the browser.
 *
 * The status-bar boot media selector picks hda / kernel / iso (auto keeps the
 * default resolution order). Availability is probed with header-only HEAD
 * requests; unavailable kinds are disabled. Switching media on a running VM
 * tears it down and reboots with the new image.
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
  const [media, setMedia] = useState<V86BootMedia>('auto')
  const mediaRef = useRef<V86BootMedia>('auto')
  const [availability, setAvailability] = useState<Partial<Record<V86ImageKind, boolean>>>({})
  const { isBooted, error, init, reboot, reset } = useV86({
    imageUrl,
    imageSize,
    memoryMb,
    bootMedia: media === 'auto' ? undefined : media,
  })
  const [initStarted, setInitStarted] = useState(false)

  // Probe kind availability once, for the selector's disabled states.
  useEffect(() => {
    let active = true
    Promise.all(
      (Object.keys(IMAGE_URLS) as V86ImageKind[]).map(
        async (kind) => [kind, await probeKind(kind)] as const,
      ),
    ).then((entries) => {
      if (active) setAvailability(Object.fromEntries(entries))
    })
    return () => {
      active = false
    }
  }, [])

  // Initialize v86 when container is ready
  useEffect(() => {
    if (containerRef.current && !initStarted) {
      setInitStarted(true)
      init(containerRef.current)
    }
  }, [init, initStarted])

  // Media switch on an already-starting/running VM: reboot into the new image.
  // Runs post-commit so `reboot` sees the latest bootMedia closure.
  useEffect(() => {
    if (mediaRef.current === media) return
    mediaRef.current = media
    if (initStarted) void reboot()
  }, [media, reboot, initStarted])

  const handleMediaChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setMedia(e.target.value as V86BootMedia)
  }, [])

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
        <label className="ml-auto flex items-center gap-1.5 text-xs text-muted-foreground">
          Boot media
          <select
            value={media}
            onChange={handleMediaChange}
            aria-label="Boot media"
            className="px-2 py-0.5 text-xs border rounded bg-background"
          >
            {MEDIA_OPTIONS.map(({ value, label }) => (
              <option
                key={value}
                value={value}
                disabled={value !== 'auto' && availability[value] === false}
              >
                {label}
                {value !== 'auto' && availability[value] === false ? ' (n/a)' : ''}
              </option>
            ))}
          </select>
        </label>
        {isBooted && (
          <button
            type="button"
            onClick={reset}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
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
        <div className="border-t border-border px-3 py-2 text-xs text-destructive">{error}</div>
      )}
    </div>
  )
}
