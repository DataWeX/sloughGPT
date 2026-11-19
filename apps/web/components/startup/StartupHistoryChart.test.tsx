import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { StartupHistoryChart, STARTUP_HISTORY_KEY } from './StartupHistoryChart'
import { revalidateAll, getQueryState } from '@/lib/cache'
import { useQueryStore } from '@/lib/cache/client'

const record = (total: number, i: number) => ({
  timestamp: 1700000000000 + i,
  total_duration: total,
  stage_durations: {},
  hook_durations: {},
  success: true,
})

const payload = (runs: number) => ({
  json: async () => ({
    data: { records: Array.from({ length: runs }, (_, i) => record(10 + i, i)) },
  }),
})

describe('StartupHistoryChart', () => {
  beforeEach(() => {
    // The query store is module-global: without a reset, a failed read in one
    // test would inherit the previous test's cached records (and the error
    // branch — which only fires when nothing is cached — would never run).
    useQueryStore.setState({ cache: {}, fetchingKeys: new Set(), refreshVersion: 0 })
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it('reads records through the shared query cache', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve(payload(1))),
    )
    const { container } = render(<StartupHistoryChart />)
    await waitFor(() => expect(screen.getByText('1 runs')).toBeTruthy())
    expect(container.querySelectorAll('.flex-1.flex.flex-col').length).toBe(1)
  })

  it('re-reads records after revalidateAll() without remounting', async () => {
    let reads = 0
    vi.stubGlobal(
      'fetch',
      vi.fn(() => {
        reads += 1
        return Promise.resolve(payload(reads === 1 ? 1 : 2))
      }),
    )

    const { container } = render(<StartupHistoryChart />)
    await waitFor(() => expect(screen.getByText('1 runs')).toBeTruthy())
    const bars = () => container.querySelectorAll('.flex-1.flex.flex-col')
    expect(bars().length).toBe(1)
    const header = screen.getByText('1 runs').closest('div')!

    revalidateAll()

    await waitFor(() => expect(screen.getByText('2 runs')).toBeTruthy())
    expect(bars().length).toBe(2)
    // same DOM node — the component was never remounted
    expect(screen.getByText('2 runs').closest('div')).toBe(header)
  })

  it('gives each column a definite height so bar percentages can resolve', async () => {
    // jsdom performs no layout, so a collapsed chart is invisible to it — this
    // asserts the contract that makes the bar's `height: N%` resolvable instead.
    //
    // Without a definite height on the column, the bar's percentage has no
    // containing block to resolve against: the column is content-sized, its only
    // in-flow child is that same bar (the tooltip is `absolute`), so the height
    // resolves to 0 and the chart renders empty while reporting "10 runs".
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve(payload(3))),
    )

    const { container } = render(<StartupHistoryChart />)
    await waitFor(() => expect(screen.getByText('3 runs')).toBeTruthy())

    const columns = container.querySelectorAll('.flex-1.flex.flex-col')
    expect(columns.length).toBe(3)
    for (const column of columns) {
      expect(column.classList.contains('h-full')).toBe(true)
      // the row is `items-end`; once the column stops shrink-wrapping it must
      // push its own content to the baseline itself
      expect(column.classList.contains('justify-end')).toBe(true)
    }

    // and every bar carries a percentage height to resolve against it
    for (const bar of container.querySelectorAll('[style*="height"]')) {
      expect(bar.getAttribute('style')).toMatch(/height:\s*[\d.]+%/)
    }
  })

  it('shows Loading... while the read is in flight, not the empty state', async () => {
    // Never-settling fetch: the read stays in flight for the whole test.
    vi.stubGlobal(
      'fetch',
      vi.fn(() => new Promise(() => {})),
    )

    render(<StartupHistoryChart />)
    await waitFor(() => expect(screen.getByText('Loading...')).toBeTruthy())
    expect(screen.queryByText('No startup records yet')).toBeNull()
    expect(screen.queryByText('Failed to load startup history')).toBeNull()
  })

  it('keeps "No startup records yet" for a genuinely zero read', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve(payload(0))),
    )

    render(<StartupHistoryChart />)
    await waitFor(() => expect(getQueryState(STARTUP_HISTORY_KEY).status).toBe('success'))
    expect(screen.getByText('No startup records yet')).toBeTruthy()
    expect(screen.queryByText('Failed to load startup history')).toBeNull()
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
  })

  it('renders a failed read as an error state with a working Retry, not as empty', async () => {
    const user = userEvent.setup()
    let reads = 0
    vi.stubGlobal(
      'fetch',
      vi.fn(() => {
        reads += 1
        return reads === 1
          ? Promise.reject(new Error('network unreachable'))
          : Promise.resolve(payload(2))
      }),
    )

    const { container } = render(<StartupHistoryChart />)
    const card = container.firstChild

    // read #1 fails → explicit failed-read state, NEVER the empty copy
    await waitFor(() => expect(screen.getByText('Failed to load startup history')).toBeTruthy())
    expect(screen.queryByText('No startup records yet')).toBeNull()
    expect(reads).toBe(1)

    // Retry → read #2 succeeds → records render in place (no remount)
    await user.click(screen.getByRole('button', { name: 'Retry' }))
    await waitFor(() => expect(screen.getByText('2 runs')).toBeTruthy())
    expect(container.firstChild).toBe(card)
    expect(screen.queryByText('Failed to load startup history')).toBeNull()
    expect(reads).toBe(2)
  })
})
