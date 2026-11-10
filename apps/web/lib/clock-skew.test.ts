import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fetchQuery, isStale, useQueryStore } from './cache/client'
import { resetServerClock, sampleServerTime } from '../../lib/clock'

const KEY = 'clock-skew-test'
const STALE_TIME = 60_000

function resetStore() {
  useQueryStore.setState({ cache: {}, fetchingKeys: new Set() })
}

function dateHeaders(offsetMs: number): Headers {
  return new Headers({ date: new Date(Date.now() + offsetMs).toUTCString() })
}

beforeEach(() => {
  resetStore()
  resetServerClock()
})

afterEach(() => {
  vi.restoreAllMocks()
  resetStore()
  resetServerClock()
})

describe('cache TTL under server clock skew', () => {
  it('serves a fresh entry without refetching (offset 0 sanity)', async () => {
    const first = vi.fn().mockResolvedValue('v1')
    const second = vi.fn().mockResolvedValue('v2')

    await fetchQuery(KEY, first, { staleTime: STALE_TIME })
    const again = await fetchQuery(KEY, second, { staleTime: STALE_TIME })

    expect(first).toHaveBeenCalledTimes(1)
    expect(second).not.toHaveBeenCalled()
    expect(again).toBe('v1')
    expect(isStale(KEY)).toBe(false)
  })

  it('expires the entry when the server clock runs 90s ahead of the write (no LATE expiry)', async () => {
    const first = vi.fn().mockResolvedValue('v1')
    const second = vi.fn().mockResolvedValue('v2')

    // Written at offset 0.
    await fetchQuery(KEY, first, { staleTime: STALE_TIME })
    expect(isStale(KEY)).toBe(false)

    // Next response says server time jumped +90s: server-side age is now ~90s
    // > 60s TTL. Local-only logic would call it fresh (serving 90s-stale data).
    sampleServerTime(dateHeaders(90_000))
    expect(isStale(KEY)).toBe(true)

    const refetched = await fetchQuery(KEY, second, { staleTime: STALE_TIME })
    expect(second).toHaveBeenCalledTimes(1)
    expect(refetched).toBe('v2')
  })

  it('keeps the entry fresh when the server clock corrects 90s backward (no EARLY expiry)', async () => {
    const first = vi.fn().mockResolvedValue('v1')
    const second = vi.fn().mockResolvedValue('v2')

    // Written while the server clock reads +90s (updatedAt = local + 90s).
    sampleServerTime(dateHeaders(90_000))
    await fetchQuery(KEY, first, { staleTime: STALE_TIME })

    // Server corrects back to local time: server-side age is ~-90s, nowhere
    // near the 60s TTL. A mixed-clock read (serverNow vs a local write) would
    // compute ~+90s and wrongly expire the entry.
    sampleServerTime(dateHeaders(0))
    expect(isStale(KEY)).toBe(false)

    const again = await fetchQuery(KEY, second, { staleTime: STALE_TIME })
    expect(second).not.toHaveBeenCalled()
    expect(again).toBe('v1')
  })
})
