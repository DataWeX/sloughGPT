import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  Spinner: () => <span>loading-spinner</span>,
  // Mirrors the real primitive: native <details> collapsed by default.
  FoldSection: ({ heading, children }: any) => (
    <details>
      <summary>{heading}</summary>
      {children}
    </details>
  ),
}))

vi.mock('@/lib/http-client', () => ({
  apiGet: vi.fn(),
}))

import { TestRunsCard } from './TestRunsCard'
import { apiGet } from '@/lib/http-client'

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

const GREEN = {
  file: 'green_journey_results.json',
  status: 'passed',
  total: 2,
  passed: 2,
  failed: 0,
  mtime: 1767000000,
  entries: [
    { test: 'dashboard_loads', passed: true, detail: 'ok' },
    { test: 'chat_streams', passed: true, detail: 'ok' },
  ],
}

const RED = {
  file: 'red_journey_results.json',
  status: 'failed',
  total: 2,
  passed: 1,
  failed: 1,
  mtime: 1767000600,
  entries: [
    { test: 'training_config', passed: true, detail: 'ok' },
    { test: 'checkpoint_input', passed: false, detail: 'missing element' },
  ],
}

describe('TestRunsCard', () => {
  it('renders a loading state while fetching', () => {
    vi.mocked(apiGet).mockReturnValue(new Promise(() => {}))
    const { container } = render(<TestRunsCard />)
    expect(container.textContent).toContain('loading-spinner')
  })

  it('shows per-file summaries with counts', async () => {
    vi.mocked(apiGet).mockResolvedValue({ data: { files: [GREEN, RED] } })
    render(<TestRunsCard />)
    expect(await screen.findByText('green_journey_results.json')).toBeTruthy()
    expect(screen.getByText('red_journey_results.json')).toBeTruthy()
    // summary carries the counts — the decision-relevant numbers up front
    expect(screen.getByText(/2\/2/)).toBeTruthy()
    expect(screen.getByText(/1\/2/)).toBeTruthy()
  })

  it('makes a failing run obvious in the summary row', async () => {
    vi.mocked(apiGet).mockResolvedValue({ data: { files: [GREEN, RED] } })
    render(<TestRunsCard />)
    const badges = await screen.findAllByText('failed')
    expect(badges.length).toBeGreaterThan(0)
    // the failed row is the one carrying the destructive badge
    expect(badges[0].className).toContain('destructive')
  })

  it('keeps raw entries collapsed by default, one click away', async () => {
    vi.mocked(apiGet).mockResolvedValue({ data: { files: [RED] } })
    const { container } = render(<TestRunsCard />)
    await screen.findByText('red_journey_results.json')

    const details = container.querySelector('details')!
    expect(details.open).toBe(false)

    fireEvent.click(details.querySelector('summary')!)
    expect(details.open).toBe(true)
    // raw detail is wired in behind the fold
    expect(screen.getByText('checkpoint_input')).toBeTruthy()
    expect(screen.getByText('missing element')).toBeTruthy()
  })

  it('shows an empty state when no runs are stored', async () => {
    vi.mocked(apiGet).mockResolvedValue({ data: { files: [] } })
    render(<TestRunsCard />)
    expect(await screen.findByText(/no test runs/i)).toBeTruthy()
  })

  it('surfaces a fetch failure instead of rendering stale data', async () => {
    vi.mocked(apiGet).mockRejectedValue(new Error('boom'))
    render(<TestRunsCard />)
    expect(await screen.findByText(/could not load test runs/i)).toBeTruthy()
  })
})
