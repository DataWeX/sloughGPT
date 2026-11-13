'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { cn } from '@sloughgpt/strui'
import { useV86, probeImage, IMAGE_URLS } from '@/hooks/useV86'
import type { V86BootMedia, V86ImageKind } from '@/hooks/useV86'
import { GUI_STATE_PRESETS } from '@/lib/v86-gui-states'

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
 * v86-based terminal panel that runs a real OS in the browser.
 *
 * The status-bar boot media selector picks hda / kernel / iso (auto keeps the
 * default resolution order). Availability is probed with header-only HEAD
 * requests; unavailable kinds are disabled. Switching media on a running VM
 * tears it down and reboots with the new image.
 *
 * The "Desktop states" selector boots vendored copy.sh GUI snapshots
 * (initial_state restores, see lib/v86-gui-states.ts). Closing the panel
 * suspends the machine instead of destroying it — reopening shows a
 * power-on standby button; the session only ends on an explicit reboot.
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
  const [guiState, setGuiState] = useState('')
  const guiRef = useRef('')
  const [availability, setAvailability] = useState<Partial<Record<V86ImageKind, boolean>>>({})
  const { isBooted, isSuspended, error, init, reboot, reset, resume } = useV86({
    imageUrl,
    imageSize,
    memoryMb,
    bootMedia: media === 'auto' ? undefined : media,
    statePresetId: guiState || undefined,
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

  // Desktop gallery pick: same reboot-on-change contract as the media selector.
  useEffect(() => {
    if (guiRef.current === guiState) return
    guiRef.current = guiState
    if (initStarted) void reboot()
  }, [guiState, reboot, initStarted])

  const handleMediaChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setMedia(e.target.value as V86BootMedia)
  }, [])

  const handleGuiChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setGuiState(e.target.value)
  }, [])

  return (
    <div className={cn('flex flex-col rounded-lg border border-border bg-background', className)}>
      {/* Status bar */}
      <div className="flex items-center gap-2 border-b border-border px-3 py-1.5">
        <div
          className={cn(
            'h-2 w-2 rounded-full',
            isBooted && !isSuspended
              ? 'bg-success'
              : isBooted
                ? 'bg-muted-foreground'
                : error
                  ? 'bg-destructive'
                  : 'bg-warning animate-pulse',
          )}
        />
        <span className="text-xs text-muted-foreground">
          {isBooted
            ? isSuspended
              ? 'VM Suspended'
              : 'VM Running'
            : error
              ? 'VM Error'
              : 'Booting...'}
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
        <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
          Desktop states
          <select
            value={guiState}
            onChange={handleGuiChange}
            aria-label="Desktop states"
            className="px-2 py-0.5 text-xs border rounded bg-background"
          >
            <option value="">none</option>
            {GUI_STATE_PRESETS.map(({ id, name }) => (
              <option key={id} value={id}>
                {name}
              </option>
            ))}
          </select>
        </label>
        {isBooted && !isSuspended && (
          <button
            type="button"
            onClick={reset}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            Restart
          </button>
        )}
      </div>

      {/* VM screen. v86 appends a canvas (graphics mode) and a div of row
          divs (text mode); rows are block divs of self-colored spans, so the
          container only needs the mono font, preserved spaces, and
          positioning so v86's absolutely-positioned cursor anchors here. The
          hidden-tab rAF→timer fallback lives in the controller: without it
          the adapter's repaint loop is suspended while the document is
          hidden and the screen never paints. The screen div itself is
          controller-owned (moved in/out on mount/unmount) so closing the
          panel never orphans the adapter — the standby overlay covers it
          while the machine is suspended. */}
      <div className="relative flex-1 overflow-hidden">
        <div
          ref={containerRef}
          className="absolute inset-0 overflow-hidden bg-black font-mono text-sm [&_div]:whitespace-pre"
          data-testid="v86-screen"
        />
        {isBooted && isSuspended && (
          <div
            className="absolute inset-0 z-10 flex items-center justify-center bg-background/85"
            data-testid="v86-standby"
          >
            <button
              type="button"
              onClick={resume}
              aria-label="Power on VM"
              className="flex items-center gap-2 rounded-md border border-border bg-card px-3 py-2 text-xs text-foreground transition-colors hover:bg-accent/20"
            >
              <svg
                aria-hidden="true"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth={2}
                strokeLinecap="round"
                className="h-3.5 w-3.5"
              >
                <path d="M12 2v10" />
                <path d="M18.4 6.6a9 9 0 1 1-12.77.04" />
              </svg>
              Power on
            </button>
          </div>
        )}
      </div>

      {/* Error display */}
      {error && (
        <div className="border-t border-border px-3 py-2 text-xs text-destructive">{error}</div>
      )}
    </div>
  )
}
