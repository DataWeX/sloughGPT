import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import {
  useQueryStore,
  fetchQuery,
  invalidateQuery,
  subscribeQuery,
  getQueryState,
  isStale,
} from './client'
import { serializeKey } from './types'

function resetStore() {
  useQueryStore.setState({ cache: {}, fetchingKeys: new Set() })
}

describe('serializeKey', () => {
  it('passes string keys through unchanged', () => {
    expect(serializeKey('models')).toBe('models')
  })

  it('JSON-stringifies array keys', () => {
    expect(serializeKey(['a', 1, true])).toBe('["a",1,true]')
  })
})

describe('fetchQuery', () => {
  beforeEach(resetStore)

  it('fetches and stores successful data', async () => {
    const fetcher = vi.fn().mockResolvedValue('data')
    await expect(fetchQuery('k', fetcher)).resolves.toBe('data')
    expect(fetcher).toHaveBeenCalledTimes(1)
    expect(getQueryState('k')).toMatchObject({
      data: 'data',
      status: 'success',
      isFetching: false,
      error: null,
    })
  })

  it('returns fresh cached data without refetching', async () => {
    const fetcher = vi.fn().mockResolvedValue('cached')
    await fetchQuery('k', fetcher, { staleTime: 100000 })
    await expect(fetchQuery('k', fetcher, { staleTime: 100000 })).resolves.toBe('cached')
    expect(fetcher).toHaveBeenCalledTimes(1)
  })

  it('refetches when data is stale', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce('v1').mockResolvedValue('v2')
    await fetchQuery('k', fetcher)
    await expect(fetchQuery('k', fetcher)).resolves.toBe('v2')
    expect(fetcher).toHaveBeenCalledTimes(2)
  })

  it('deduplicates concurrent in-flight requests for the same key', async () => {
    let resolveFn!: (v: string) => void
    const fetcher = vi.fn(() => new Promise<string>(r => { resolveFn = r }))
    const p1 = fetchQuery('k', fetcher)
    const p2 = fetchQuery('k', fetcher)
    resolveFn('done')
    await expect(p1).resolves.toBe('done')
    await expect(p2).resolves.toBe('done')
    expect(fetcher).toHaveBeenCalledTimes(1)
  })

  it('stores errors and rethrows them', async () => {
    const fetcher = vi.fn().mockRejectedValue(new Error('boom'))
    await expect(fetchQuery('k', fetcher)).rejects.toThrow('boom')
    const state = getQueryState('k')
    expect(state.status).toBe('error')
    expect(state.error).toBeInstanceOf(Error)
    expect(state.isFetching).toBe(false)
  })

  it('retries on failure and eventually succeeds', async () => {
    const fetcher = vi.fn()
      .mockRejectedValueOnce(new Error('fail1'))
      .mockRejectedValueOnce(new Error('fail2'))
      .mockResolvedValue('ok')
    await expect(fetchQuery('k', fetcher, { retry: 2, retryDelay: 1 })).resolves.toBe('ok')
    expect(fetcher).toHaveBeenCalledTimes(3)
  })

  it('throws the last error once retries are exhausted', async () => {
    const fetcher = vi.fn().mockRejectedValue(new Error('down'))
    await expect(fetchQuery('k', fetcher, { retry: 1, retryDelay: 1 })).rejects.toThrow('down')
    expect(fetcher).toHaveBeenCalledTimes(2)
  })

  it('tracks the key in fetchingKeys while in flight', async () => {
    let resolveFn!: (v: string) => void
    const fetcher = vi.fn(() => new Promise<string>(r => { resolveFn = r }))
    const promise = fetchQuery('k', fetcher)
    expect(useQueryStore.getState().fetchingKeys.has('k')).toBe(true)
    resolveFn('x')
    await promise
    expect(useQueryStore.getState().fetchingKeys.has('k')).toBe(false)
  })
})

describe('invalidateQuery', () => {
  beforeEach(resetStore)

  it('is a no-op for an unknown key', () => {
    expect(() => invalidateQuery('ghost')).not.toThrow()
  })

  it('resets updatedAt so the next read is stale', async () => {
    const fetcher = vi.fn().mockResolvedValue('x')
    await fetchQuery('k', fetcher, { staleTime: 100000 })
    expect(isStale('k')).toBe(false)
    invalidateQuery('k')
    expect(isStale('k')).toBe(true)
  })
})

describe('isStale / getQueryState', () => {
  beforeEach(resetStore)

  it('treats missing keys as stale with idle state', () => {
    expect(isStale('ghost')).toBe(true)
    expect(getQueryState('ghost')).toEqual({
      data: undefined,
      error: null,
      status: 'idle',
      isFetching: false,
      updatedAt: 0,
    })
  })
})

describe('subscribeQuery garbage collection', () => {
  beforeEach(() => {
    resetStore()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('deletes the entry after the last subscriber unsubscribes', () => {
    const unsub = subscribeQuery('k')
    unsub()
    vi.advanceTimersByTime(5 * 60 * 1000 + 1)
    expect(useQueryStore.getState().cache['k']).toBeUndefined()
  })

  it('keeps the entry while subscribers remain', () => {
    const unsubA = subscribeQuery('k')
    subscribeQuery('k')
    unsubA()
    vi.advanceTimersByTime(10 * 60 * 1000)
    expect(useQueryStore.getState().cache['k']).toBeDefined()
  })

  it('cancels a pending GC when resubscribed', () => {
    subscribeQuery('k')()
    subscribeQuery('k')
    vi.advanceTimersByTime(10 * 60 * 1000)
    expect(useQueryStore.getState().cache['k']).toBeDefined()
  })
})
