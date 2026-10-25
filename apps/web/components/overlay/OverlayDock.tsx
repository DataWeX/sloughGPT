'use client'

import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { cn } from '@sloughgpt/strui'

export type OverlayTone = 'default' | 'primary' | 'success' | 'warning' | 'destructive'

/**
 * One tenant of the overlay rail (Android system-UI model: the rail owns
 * the right edge — position, stacking, LED status lamp, open-at-a-time —
 * tenants bring their own key glyph, badge and panel).
 *
 * Panels open beside the rail, never over the corner: the bottom-right
 * status bar is the only thing left down there (the old pill + the
 * consciousness FAB both migrated in — consciousness as of v2).
 */
export interface OverlayWidget {
  /** Stable key, also the data-testid suffix. */
  id: string
  /** Button title + aria-label and the panel dialog label. */
  label: string
  icon: ReactNode
  /** Count chip shown when truthy (worst severity count, findings, …). */
  badge?: string | null
  /** Severity ring around the key — status-LED color, tokens only. */
  tone?: OverlayTone
  /**
   * Operator surfaces are tagged, never mainstreamed (AGENTS.md Page Rule
   * #3): infra/system widgets carry role="operator" on button and panel.
   */
  role?: string
  /** Panel body. `close` dismisses the panel (in-panel actions call it). */
  render: (close: () => void) => ReactNode
}

/**
 * Ask the dock to open a widget by id — how keyboard shortcuts (e.g.
 * Ctrl+Shift+Q for consciousness) reach the panel now that the host owns
 * open state. Same event-bridge pattern AppLayout uses for
 * 'toggle-output-panel' / 'toggle-whatsnew'.
 */
export const OVERLAY_DOCK_OPEN_EVENT = 'overlay-dock:open'

export function requestOverlayDockOpen(id: string): void {
  if (typeof window === 'undefined') return
  window.dispatchEvent(new CustomEvent(OVERLAY_DOCK_OPEN_EVENT, { detail: { id } }))
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

/** LED lamp per tone — pulsing while actionable, steady when healthy. */
const TONE_LED: Record<OverlayTone, string> = {
  default: 'bg-muted',
  primary: 'bg-primary',
  success: 'bg-success',
  warning: 'bg-warning animate-pulse',
  destructive: 'bg-destructive animate-pulse',
}

/** Worst tone wins — the lamp reports the rail's overall state. */
const TONE_RANK: Record<OverlayTone, number> = {
  default: 0,
  primary: 1,
  success: 2,
  warning: 3,
  destructive: 4,
}

function worstTone(widgets: OverlayWidget[]): OverlayTone {
  return widgets.reduce<OverlayTone>(
    (acc, w) => (TONE_RANK[w.tone ?? 'default'] > TONE_RANK[acc] ? (w.tone ?? 'default') : acc),
    'default',
  )
}

interface OverlayDockProps {
  /** Registered tenants, top-to-bottom. One panel open at a time. */
  widgets: OverlayWidget[]
  /** Landmark label for the dock region. */
  ariaLabel?: string
}

/**
 * The overlay rail — the single owner of the right edge.
 *
 * Slim vertical glass column at mid-right (game-HUD style): status LED on
 * top, extruded tenant keys below, badges as chips. The former
 * bottom-right pill and the consciousness FAB are tenants now, so the
 * corner holds only the status bar. Mounted once by AppLayout (both the
 * Next and Vite shells render it).
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

  // Outside click closes — rail and its panel are one dock region.
  useEffect(() => {
    if (!openId) return
    const onDown = (e: MouseEvent) => {
      if (dockRef.current && !dockRef.current.contains(e.target as Node)) close()
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [openId, close])

  // Keyboard shortcuts (Ctrl+Shift+Q, …) open their widget by id.
  useEffect(() => {
    const onRequest = (e: Event) => {
      const id = (e as CustomEvent<{ id?: string }>).detail?.id
      if (id && widgets.some((w) => w.id === id)) setOpenId(id)
    }
    window.addEventListener(OVERLAY_DOCK_OPEN_EVENT, onRequest)
    return () => window.removeEventListener(OVERLAY_DOCK_OPEN_EVENT, onRequest)
  }, [widgets])

  if (widgets.length === 0) return null
  const open = widgets.find((w) => w.id === openId) ?? null
  const led = worstTone(widgets)

  return (
    <div
      ref={dockRef}
      data-testid="overlay-dock"
      aria-label={ariaLabel}
      className="fixed right-3 top-1/2 z-[9999] -translate-y-1/2"
    >
      {open ? (
        <div
          role="dialog"
          aria-label={open.label}
          data-testid="overlay-dock-panel"
          data-role={open.role}
          className="absolute right-full top-1/2 mr-3 max-h-[80vh] w-[22rem] max-w-[calc(100vw-6rem)] -translate-y-1/2 overflow-y-auto rounded-2xl border border-border/70 bg-card/90 p-3 text-foreground shadow-2xl backdrop-blur-xl"
        >
          {open.render(close)}
        </div>
      ) : null}

      <div
        data-testid="overlay-rail"
        className="flex flex-col items-center gap-1.5 rounded-full border border-border/70 bg-card/70 p-1.5 shadow-2xl backdrop-blur-md"
      >
        <span
          data-testid="overlay-led"
          aria-hidden
          title="Rail status"
          className={cn('mb-0.5 h-2.5 w-2.5 rounded-full', TONE_LED[led])}
        />
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
                  'relative flex h-10 w-10 cursor-pointer items-center justify-center rounded-full border bg-card text-foreground transition-all',
                  'shadow-[0_2px_0_0_rgb(var(--border))] hover:scale-105',
                  'active:translate-y-0.5 active:shadow-[0_1px_0_0_rgb(var(--border))]',
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
