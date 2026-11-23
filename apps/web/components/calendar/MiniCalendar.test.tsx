import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import { MiniCalendar, buildMonthGrid } from './MiniCalendar'

// Deterministic "now": 2026-06-15 local. Tests pin the view month through
// the `selected` prop wherever possible so they don't depend on the clock;
// the fallback path (no selected) is exercised via fake timers.
const SELECTED = new Date(2026, 5, 15)

beforeEach(() => {
  vi.useFakeTimers({ now: new Date(2026, 5, 15, 12, 0, 0) })
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

describe('buildMonthGrid', () => {
  it('produces a 6x7 grid of exactly the month days', () => {
    const weeks = buildMonthGrid(2026, 5) // June 2026: 30 days, starts Monday
    expect(weeks).toHaveLength(6)
    for (const week of weeks) expect(week).toHaveLength(7)
    const days = weeks.flat().filter((d): d is Date => d !== null)
    expect(days).toHaveLength(30)
    expect(days[0].getDate()).toBe(1)
    expect(days[29].getDate()).toBe(30)
    for (const d of days) {
      expect(d.getMonth()).toBe(5)
      expect(d.getFullYear()).toBe(2026)
    }
  })

  it('pads leading cells so day 1 sits on its weekday', () => {
    const weeks = buildMonthGrid(2026, 5)
    const first = weeks[0].findIndex((d) => d !== null)
    expect(first).toBe(new Date(2026, 5, 1).getDay())
  })

  it('handles February in a non-leap year (28 days)', () => {
    const days = buildMonthGrid(2026, 1).flat().filter((d): d is Date => d !== null)
    expect(days).toHaveLength(28)
  })
})

describe('MiniCalendar', () => {
  it('renders the month label for the selected date', () => {
    render(<MiniCalendar selected={SELECTED} />)
    expect(screen.getByTestId('mini-calendar-label')).toHaveTextContent('June 2026')
  })

  it('marks the selected day with aria-pressed', () => {
    render(<MiniCalendar selected={SELECTED} />)
    const fifteen = screen.getByRole('button', { name: 'June 15, 2026' })
    expect(fifteen).toHaveAttribute('aria-pressed', 'true')
    const sixteen = screen.getByRole('button', { name: 'June 16, 2026' })
    expect(sixteen).toHaveAttribute('aria-pressed', 'false')
  })

  it('calls onSelect with the clicked local-midnight date', () => {
    const onSelect = vi.fn()
    render(<MiniCalendar selected={SELECTED} onSelect={onSelect} />)
    fireEvent.click(screen.getByRole('button', { name: 'June 21, 2026' }))
    expect(onSelect).toHaveBeenCalledTimes(1)
    const picked = onSelect.mock.calls[0][0] as Date
    expect(picked.getFullYear()).toBe(2026)
    expect(picked.getMonth()).toBe(5)
    expect(picked.getDate()).toBe(21)
    expect(picked.getHours()).toBe(0)
  })

  it('navigates months via the prev/next buttons', () => {
    render(<MiniCalendar selected={SELECTED} />)
    fireEvent.click(screen.getByRole('button', { name: 'Next month' }))
    expect(screen.getByTestId('mini-calendar-label')).toHaveTextContent('July 2026')
    fireEvent.click(screen.getByRole('button', { name: 'Previous month' }))
    fireEvent.click(screen.getByRole('button', { name: 'Previous month' }))
    expect(screen.getByTestId('mini-calendar-label')).toHaveTextContent('May 2026')
  })

  it('follows the selected date across a month boundary', () => {
    const { rerender } = render(<MiniCalendar selected={SELECTED} />)
    expect(screen.getByTestId('mini-calendar-label')).toHaveTextContent('June 2026')
    rerender(<MiniCalendar selected={new Date(2026, 6, 2)} />)
    expect(screen.getByTestId('mini-calendar-label')).toHaveTextContent('July 2026')
    expect(screen.getByRole('button', { name: 'July 2, 2026' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
  })

  it('collapses to header-only and expands back', () => {
    render(<MiniCalendar selected={SELECTED} />)
    const toggle = screen.getByRole('button', { name: 'Collapse calendar' })
    expect(toggle).toHaveAttribute('aria-expanded', 'true')
    fireEvent.click(toggle)
    expect(screen.queryByRole('button', { name: 'June 15, 2026' })).not.toBeInTheDocument()
    const expand = screen.getByRole('button', { name: 'Expand calendar' })
    expect(expand).toHaveAttribute('aria-expanded', 'false')
    fireEvent.click(expand)
    expect(screen.getByRole('button', { name: 'June 15, 2026' })).toBeInTheDocument()
  })

  it('renders without crashing when no date is selected (SSR-safe null)', () => {
    render(<MiniCalendar />)
    expect(screen.getByTestId('mini-calendar')).toBeInTheDocument()
    // View month falls back to the client clock's month.
    act(() => {
      vi.advanceTimersByTime(0)
    })
    expect(screen.getByTestId('mini-calendar-label')).toHaveTextContent('June 2026')
  })

  it('uses a roving tabindex: exactly one day is tabbable', () => {
    render(<MiniCalendar selected={SELECTED} />)
    // Day buttons carry full-date aria-labels ("June 15, 2026"); nav buttons
    // do not, so this filter isolates the day cells.
    const isDayButton = (b: HTMLElement) =>
      /^[A-Z][a-z]+ \d+, \d{4}$/.test(b.getAttribute('aria-label') ?? '')
    const tabbable = screen
      .getAllByRole('button')
      .filter(isDayButton)
      .filter((b) => b.getAttribute('tabindex') === '0')
    expect(tabbable).toHaveLength(1)
    expect(tabbable[0]).toHaveAttribute('aria-label', 'June 15, 2026')
  })

  it('makes no network calls (session-only contract)', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    render(<MiniCalendar selected={SELECTED} onSelect={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: 'June 21, 2026' }))
    fireEvent.click(screen.getByRole('button', { name: 'Next month' }))
    expect(fetchSpy).not.toHaveBeenCalled()
    fetchSpy.mockRestore()
  })
})
