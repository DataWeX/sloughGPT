import { describe, it, expect, afterEach, vi, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'
import { TestRunsCard } from './TestRunsCard'
import { useEndpoint } from '@/lib/protocol/useEndpoint'

vi.mock('@/lib/protocol/useEndpoint', () => ({
  useEndpoint: vi.fn(),
}))

vi.mock('@/lib/time-ago', () => ({ timeAgo: () => '3h ago' }))

const refetch = vi.fn()

type EndpointResult = ReturnType<typeof useEndpoint>

const runsData = {
  suites: [
    {
      file: 'user_journey_results.json',
      mtime: '2026-11-11T12:55:35+00:00',
      total: 3,
      passed: 1,
      failed: 1,
      unknown: 1,
      items: [
        { test: 'login works', passed: true, detail: 'ok' },
        { test: 'logout works', passed: false, detail: 'session stuck' },
        { test: 'flaky check', passed: null, detail: 'inconclusive' },
      ],
    },
    {
      file: 'e2e_training_trigger_results.json',
      mtime: '2026-11-11T12:50:00+00:00',
      total: 1,
      passed: 1,
      failed: 0,
      unknown: 0,
      items: [{ test: 'training trigger', passed: true, detail: 'ok' }],
    },
  ],
  total: 4,
  passed: 2,
  failed: 1,
  unknown: 1,
}

function mockEndpoint(partial: Partial<EndpointResult>) {
  vi.mocked(useEndpoint).mockReturnValue({
    data: undefined,
    status: 'idle',
    error: undefined,
    refetch,
    ...partial,
  } as EndpointResult)
}

describe('TestRunsCard', () => {
  beforeEach(() => {
    refetch.mockClear()
  })

  afterEach(cleanup)

  it('reads through the test_results.list contract, never a raw URL', () => {
    mockEndpoint({ status: 'loading' })
    render(<TestRunsCard />)
    expect(useEndpoint).toHaveBeenCalledWith('test_results.list')
  })

  it('shows an explicit loading skeleton while status is loading', () => {
    mockEndpoint({ status: 'loading' })
    const { container } = render(<TestRunsCard />)
    expect(container.querySelector('.animate-pulse')).toBeTruthy()
  })

  it('shows the error with a retry that refetches', () => {
    mockEndpoint({ status: 'error', error: 'boom' })
    render(<TestRunsCard />)
    expect(screen.getByText('boom')).toBeDefined()
    fireEvent.click(screen.getByRole('button', { name: /retry/i }))
    expect(refetch).toHaveBeenCalledTimes(1)
  })

  it('shows an empty state when no runs exist', () => {
    mockEndpoint({
      status: 'success',
      data: { suites: [], total: 0, passed: 0, failed: 0, unknown: 0 },
    })
    render(<TestRunsCard />)
    expect(screen.getByText('No test runs recorded yet')).toBeDefined()
  })

  it('renders totals per outcome', () => {
    mockEndpoint({ status: 'success', data: runsData })
    render(<TestRunsCard />)
    expect(screen.getAllByText('2').length).toBeGreaterThan(0) // passed
    expect(screen.getAllByText('1').length).toBeGreaterThanOrEqual(2) // failed + unknown KPIs
    expect(screen.getAllByText('4').length).toBeGreaterThan(0) // total
  })

  it('lists each suite with its outcome labels', () => {
    mockEndpoint({ status: 'success', data: runsData })
    render(<TestRunsCard />)
    expect(screen.getByText('user_journey_results.json')).toBeDefined()
    expect(screen.getByText('e2e_training_trigger_results.json')).toBeDefined()
    // KPI labels and per-item badges share these words — any occurrence proves both
    expect(screen.getAllByText('Passed').length).toBeGreaterThanOrEqual(2)
    expect(screen.getAllByText('Failed').length).toBeGreaterThanOrEqual(2)
    expect(screen.getAllByText('Unknown').length).toBeGreaterThanOrEqual(2)
  })

  it('renders the relative mtime of the newest run', () => {
    mockEndpoint({ status: 'success', data: runsData })
    render(<TestRunsCard />)
    expect(screen.getAllByText(/3h ago/).length).toBeGreaterThan(0)
  })

  it('keeps the raw per-test list collapsed by default', () => {
    mockEndpoint({ status: 'success', data: runsData })
    const { container } = render(<TestRunsCard />)
    const details = container.querySelector('details')
    expect(details).toBeTruthy()
    expect(details!.open).toBe(false)
    expect(details!.textContent).toContain('logout works')
  })
})
