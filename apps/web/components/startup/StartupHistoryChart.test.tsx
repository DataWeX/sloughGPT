import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'

import { StartupHistoryChart } from './StartupHistoryChart'
import { revalidateAll } from '@/lib/cache'

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
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it('reads records through the shared query cache', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(payload(1))))
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
})
