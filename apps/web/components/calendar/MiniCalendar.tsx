'use client'

import { useEffect, useRef, useState } from 'react'
import { cn } from '@sloughgpt/strui'

/**
 * MiniCalendar — compact session-only month view (kanban card
 * 20261101_102135_minicalendar-widget).
 *
 * No API, no persistence: selection is reported upward via `onSelect` and the
 * view month / collapse state live in component state only. The isometric
 * art direction is carried by (a) the extruded plate shadow on the card and
 * (b) the rotate-45 diamond selection tile — both purely decorative, so
 * pointer geometry and hit targets stay on the normal untransformed layer.
 */

interface MiniCalendarProps {
  /** Selected day. Controls both the diamond tile and the month the view follows. */
  selected?: Date | null
  /** Fires with a local-midnight Date when a day is clicked or Enter-pressed. */
  onSelect?: (date: Date) => void
  className?: string
}

interface MonthYear {
  year: number
  month: number
}

const WEEKDAY_INITIALS = ['S', 'M', 'T', 'W', 'T', 'F', 'S']

const MONTH_LABELS = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
]

/**
 * 6x7 grid (Sun-first) for a month: leading/trailing cells are null so the
 * day numbers never leave the current month — adjacent-month peeks would
 * cost density and clutter the compact layout.
 */
export function buildMonthGrid(year: number, month: number): (Date | null)[][] {
  const firstWeekday = new Date(year, month, 1).getDay()
  const daysInMonth = new Date(year, month + 1, 0).getDate()
  const cells: (Date | null)[] = []
  for (let i = 0; i < firstWeekday; i++) cells.push(null)
  for (let day = 1; day <= daysInMonth; day++) cells.push(new Date(year, month, day))
  while (cells.length % 7 !== 0 || cells.length < 42) cells.push(null)
  const weeks: (Date | null)[][] = []
  for (let i = 0; i < cells.length; i += 7) weeks.push(cells.slice(i, i + 7))
  return weeks
}

function sameDay(a: Date, b: Date): boolean {
  return (
    a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate()
  )
}

function shiftMonth({ year, month }: MonthYear, offset: number): MonthYear {
  const target = new Date(year, month + offset, 1)
  return { year: target.getFullYear(), month: target.getMonth() }
}

export function MiniCalendar({ selected, onSelect, className }: MiniCalendarProps) {
  // Null until mount — the page may SSR this widget with a server "today"
  // that differs from the client clock (same React #418 class the calendar
  // page itself guards against).
  const [view, setView] = useState<MonthYear | null>(null)
  const [expanded, setExpanded] = useState(true)
  const cellRefs = useRef<(HTMLButtonElement | null)[]>([])
  // Last selected day we synced the view to. Ref (not effect-dep on `view`):
  // re-running the sync on every manual month flip would yank the view back
  // to the selected month, making prev/next navigation impossible.
  const lastSelectedRef = useRef<Date | null>(null)

  useEffect(() => {
    const anchor = selected ?? new Date()
    const prev = lastSelectedRef.current
    lastSelectedRef.current = anchor
    // Same calendar day (incl. mount, where prev is null): don't reset the
    // view — the user may be browsing other months. Different day (e.g. the
    // page's prev/next-day buttons crossing a month boundary): follow it.
    if (prev && sameDay(prev, anchor)) return
    setView({ year: anchor.getFullYear(), month: anchor.getMonth() })
  }, [selected])

  const weeks = view ? buildMonthGrid(view.year, view.month) : []
  const flatDays = weeks.flat()
  const today = new Date()
  // Roving tabindex: one tab stop for the whole grid, arrows move focus.
  const rovingIndex = flatDays.findIndex(
    (d) => d !== null && (selected ? sameDay(d, selected) : sameDay(d, today)),
  )

  const handleGridKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const key = event.key
    const deltas: Record<string, number> = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -7, ArrowDown: 7 }
    let next: number | null = null
    if (key in deltas) next = rovingIndex + deltas[key]
    else if (key === 'Home') next = 0
    else if (key === 'End') next = flatDays.length - 1
    if (next === null) return
    // Skip blank padding cells in the direction of travel.
    while (next >= 0 && next < flatDays.length && flatDays[next] === null) {
      next += key === 'ArrowLeft' || key === 'ArrowUp' ? -1 : 1
    }
    if (next < 0 || next >= flatDays.length || flatDays[next] === null) return
    event.preventDefault()
    cellRefs.current[next]?.focus()
  }

  const label = view ? `${MONTH_LABELS[view.month]} ${view.year}` : '—'

  return (
    <div
      className={cn(
        // Extruded plate shadow = the isometric "depth" cue; decorative only.
        'inline-block overflow-hidden rounded-lg border border-border/40 bg-card shadow-[2px_2px_0_0_rgb(var(--border))]',
        className,
      )}
      data-testid="mini-calendar"
    >
      <div className="flex items-center justify-between gap-1 border-b border-border/30 bg-muted/20 px-2 py-1">
        <span
          className="text-[10px] font-medium text-muted-foreground/60"
          data-testid="mini-calendar-label"
        >
          {label}
        </span>
        <div className="flex items-center gap-0.5">
          <button
            type="button"
            aria-label="Previous month"
            onClick={() => setView((v) => (v ? shiftMonth(v, -1) : v))}
            className="flex h-5 w-5 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring motion-reduce:transition-none"
          >
            <svg
              className="h-3 w-3"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              aria-hidden="true"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <button
            type="button"
            aria-label="Next month"
            onClick={() => setView((v) => (v ? shiftMonth(v, 1) : v))}
            className="flex h-5 w-5 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring motion-reduce:transition-none"
          >
            <svg
              className="h-3 w-3"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              aria-hidden="true"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
          <button
            type="button"
            aria-label={expanded ? 'Collapse calendar' : 'Expand calendar'}
            aria-expanded={expanded}
            onClick={() => setExpanded((e) => !e)}
            className="flex h-5 w-5 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring motion-reduce:transition-none"
          >
            <svg
              className={cn(
                'h-3 w-3 transition-transform motion-reduce:transition-none',
                expanded ? '' : 'rotate-180',
              )}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              aria-hidden="true"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>
      </div>

      {expanded && view && (
        <div role="group" aria-label={`${label} calendar`}>
          <div className="grid grid-cols-7 gap-1 px-2 pb-1 pt-1.5" aria-hidden="true">
            {WEEKDAY_INITIALS.map((initial, i) => (
              <span
                key={i}
                className="flex h-4 items-center justify-center text-[10px] font-medium text-muted-foreground/60"
              >
                {initial}
              </span>
            ))}
          </div>
          <div className="grid grid-cols-7 gap-1 px-2 pb-2" onKeyDown={handleGridKeyDown}>
            {flatDays.map((day, index) => {
              if (day === null) {
                return <span key={index} className="h-6 w-6" aria-hidden="true" />
              }
              const isSelected = selected ? sameDay(day, selected) : false
              const isToday = sameDay(day, today)
              return (
                <button
                  key={index}
                  type="button"
                  ref={(el) => {
                    cellRefs.current[index] = el
                  }}
                  tabIndex={index === rovingIndex ? 0 : -1}
                  aria-pressed={isSelected}
                  aria-label={day.toLocaleDateString('en-US', {
                    month: 'long',
                    day: 'numeric',
                    year: 'numeric',
                  })}
                  onClick={() => onSelect?.(new Date(day))}
                  className={cn(
                    'relative flex h-6 w-6 items-center justify-center rounded text-[10px] transition-colors motion-reduce:transition-none',
                    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset',
                    isSelected
                      ? 'font-medium text-primary-foreground'
                      : isToday
                        ? 'border border-primary/50 font-medium text-primary hover:bg-muted'
                        : 'text-foreground hover:bg-muted',
                  )}
                >
                  {isSelected && (
                    // The isometric diamond: rotated tile under an upright
                    // numeral — flavor without touching hit geometry.
                    <span
                      aria-hidden="true"
                      className="absolute inset-[3px] rotate-45 rounded-[2px] bg-primary motion-reduce:transition-none"
                    />
                  )}
                  <span className="relative">{day.getDate()}</span>
                </button>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
