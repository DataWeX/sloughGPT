'use client'

import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { cn } from '@sloughgpt/strui'

export type OverlayTone = 'default' | 'primary' | 'success' | 'warning' | 'destructive'

/**
 * One tenant of the floating dock.
 *
 * The dock owns the corner — position, stacking, and open-at-a-time — the
 * way a window manager owns its overlay layer (Android heads-up model).
 * Tenants register an icon, an optional count badge, a severity ring and a
 * panel; they never position themselves. Consciousness quick-actions,
 * DebugOverlay and OutputPanel are grandfathered at their current spots and
 * migrate as tenants when next touched (AGENTS.md Page Rule: no retroactive
 * purge).
 */
export interface OverlayWidget {
  /** Stable key, also the data-testid suffix. */
  id: string
  /** Button title + aria-label and the panel dialog label. */
  label: string
  icon: ReactNode
  /** Count chip shown when truthy (worst severity count, findings, …). */
  badge?: string | null
  /** Severity ring around the button — status-LED color, tokens only. */
  tone?: OverlayTone
  /**
   * Operator surfaces are tagged, never mainstreamed (AGENTS.md Page Rule
   * #3): infra/system widgets carry role="operator" on button and panel.
   */
  role?: string
  /** Panel body. `close` dismisses the panel (in-panel actions call it). */
  render: (close: () => void) => ReactNode
}

/** Ring recipe per tone — semantic tokens only, no raw hex. */
const TONE_RING: Record<OverlayTone, string> = {
  default: 'ring-border',
  primary: 'ring-primary/70',
  success: 'ring-success/70',
  warning: 'ring-warning/80',
  destructive: 'ring-destructive',
}

/**
 * Badge chip recipe per tone — borrowed verbatim from the contrast-safe
 * severity chips in components/mole/severity.ts (proven on both themes).
 */
const TONE_BADGE: Record<OverlayTone, string> = {
  default: 'border-border bg-muted text-muted-foreground',
  primary: 'border-primary/30 bg-primary/10 text-primary',
  success: 'border-success/30 bg-success/10 text-success',
  warning: 'border-warning/30 bg-warning/10 text-warning',
  destructive: 'border-destructive/30 bg-destructive/10 text-destructive',
}

interface OverlayDockProps {
  /** Registered tenants, top-to-bottom. One panel open at a time. */
  widgets: OverlayWidget[]
  /** Landmark label for the dock region. */
  ariaLabel?: string
}

/**
 * Floating overlay dock — the single owner of the bottom-right corner.
 *
 * Mounted once by AppLayout (both the Next and Vite shells render it), so
 * every route shares one host instead of stacking independent
 * `fixed bottom-right z-[9999]` elements.
 */
export function OverlayDock({ widgets, ariaLabel = 'System overlays' }: OverlayDockProps) {
  const [openId, setOpenId] = useState<string | null>(null)
  const dockRef = useRef<HTMLDivElement>(null)
  const close = useCallback(() => setOpenId(null), [])

  // Esc closes the open panel.
  useEffect(() => {
    if (!openId) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [openId, close])

  // Outside click closes — button and panel are one dock region.
  useEffect(() => {
    if (!openId) return
    const onDown = (e: MouseEvent) => {
      if (dockRef.current && !dockRef.current.contains(e.target as Node)) close()
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [openId, close])

  if (widgets.length === 0) return null
  const open = widgets.find((w) => w.id === openId) ?? null

  return (
    <div
      ref={dockRef}
      data-testid="overlay-dock"
      aria-label={ariaLabel}
      className="fixed bottom-20 right-6 z-[9999] flex flex-col items-end gap-2"
    >
      {open ? (
        <div
          role="dialog"
          aria-label={open.label}
          data-testid="overlay-dock-panel"
          data-role={open.role}
          className="max-h-[70vh] w-[22rem] max-w-[calc(100vw-3rem)] overflow-y-auto rounded-xl border border-border/70 bg-card p-3 text-foreground shadow-2xl"
        >
          {open.render(close)}
        </div>
      ) : null}

      <div className="flex flex-col items-end gap-2">
        {widgets.map((w) => {
          const isOpen = w.id === openId
          const tone = w.tone ?? 'default'
          return (
            <div key={w.id} className="relative" data-role={w.role}>
              <button
                type="button"
                data-testid={`overlay-widget-${w.id}`}
                aria-haspopup="dialog"
                aria-expanded={isOpen}
                aria-label={w.label}
                title={w.label}
                onClick={() => setOpenId(isOpen ? null : w.id)}
                className={cn(
                  'relative flex h-11 w-11 cursor-pointer items-center justify-center rounded-full border bg-card text-foreground shadow-lg transition-all',
                  'hover:scale-105 active:scale-95',
                  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
                  TONE_RING[tone],
                )}
              >
                {w.icon}
              </button>
              {w.badge ? (
                <span
                  data-testid={`overlay-badge-${w.id}`}
                  className={cn(
                    'pointer-events-none absolute -right-1.5 -top-1.5 rounded-full border px-1.5 text-[10px] font-semibold leading-4',
                    TONE_BADGE[tone],
                  )}
                >
                  {w.badge}
                </span>
              ) : null}
            </div>
          )
        })}
      </div>
    </div>
  )
}
