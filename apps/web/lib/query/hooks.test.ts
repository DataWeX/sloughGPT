import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act, waitFor, cleanup } from '@testing-library/react'
import { useQueryStore, invalidateQuery, isStale } from './client'
import { useQuery, useMutation, useInvalidate, useIsFetching } from './hooks'

function resetStore() {
  useQueryStore.setState({ cache: {}, fetchingKeys: new Set() })
}

describe('useQuery', () => {
  beforeEach(() => {
    resetStore()
    vi.clearAllMocks()
  })

  afterEach(cleanup)

  it('fetches on mount and exposes the data', async () => {
    const fetcher = vi.fn().mockResolvedValue('hello')
    const { result } = renderHook(() => useQuery('k', fetcher))
    expect(result.current.isLoading).toBe(true)
    await waitFor(() => expect(result.current.data).toBe('hello'))
    expect(result.current.status).toBe('success')
    expect(result.current.isFetching).toBe(false)
    expect(fetcher).toHaveBeenCalledTimes(1)
  })

  it('does not fetch when disabled', async () => {
    const fetcher = vi.fn().mockResolvedValue('x')
    renderHook(() => useQuery('k', fetcher, { enabled: false }))
    await new Promise(r => setTimeout(r, 20))
    expect(fetcher).not.toHaveBeenCalled()
  })

  it('calls onSuccess with the fetched data', async () => {
    const onSuccess = vi.fn()
    renderHook(() => useQuery('k', () => Promise.resolve('v'), { onSuccess }))
    await waitFor(() => expect(onSuccess).toHaveBeenCalledWith('v'))
  })

  it('calls onError when the fetch fails', async () => {
    const onError = vi.fn()
    const { result } = renderHook(() =>
      useQuery('k', () => Promise.reject(new Error('nope')), { onError }),
    )
    await waitFor(() => expect(result.current.error).toBeInstanceOf(Error))
    expect(onError).toHaveBeenCalledTimes(1)
  })

  it('refetch() re-runs the fetcher', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce('a').mockResolvedValue('b')
    const { result } = renderHook(() => useQuery('k', fetcher))
    await waitFor(() => expect(result.current.data).toBe('a'))
    await act(async () => {
      await result.current.refetch()
    })
    expect(result.current.data).toBe('b')
    expect(fetcher).toHaveBeenCalledTimes(2)
  })

  it('refetches when remounted after invalidation', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce('v1').mockResolvedValue('v2')
    const { result, unmount } = renderHook(() => useQuery('k', fetcher))
    await waitFor(() => expect(result.current.data).toBe('v1'))
    unmount()
    invalidateQuery('k')
    const { result: result2 } = renderHook(() => useQuery('k', fetcher))
    await waitFor(() => expect(result2.current.data).toBe('v2'))
    expect(fetcher).toHaveBeenCalledTimes(2)
  })
})

describe('useMutation', () => {
  beforeEach(() => {
    resetStore()
    vi.clearAllMocks()
  })

  afterEach(cleanup)

  it('runs mutateAsync and calls onSuccess with data and vars', async () => {
    const fn = vi.fn().mockResolvedValue('ok')
    const onSuccess = vi.fn()
    const { result } = renderHook(() => useMutation(fn, { onSuccess }))
    await act(async () => {
      await result.current.mutateAsync('vars')
    })
    expect(fn).toHaveBeenCalledWith('vars')
    expect(onSuccess).toHaveBeenCalledWith('ok', 'vars')
    expect(result.current.isLoading).toBe(false)
    expect(result.current.error).toBeNull()
  })

  it('sets error state and calls onError/onSettled on failure', async () => {
    const onError = vi.fn()
    const onSettled = vi.fn()
    const { result } = renderHook(() =>
      useMutation(() => Promise.reject(new Error('boom')), { onError, onSettled }),
    )
    await act(async () => {
      await expect(result.current.mutateAsync('v')).rejects.toThrow('boom')
    })
    expect(result.current.error).toBeInstanceOf(Error)
    expect(onError).toHaveBeenCalledWith(expect.any(Error), 'v')
    expect(onSettled).toHaveBeenCalledWith(undefined, expect.any(Error), 'v')
  })

  it('invalidates listed keys after success', async () => {
    const fetcher = vi.fn().mockResolvedValue('base')
    renderHook(() => useQuery('models', fetcher))
    await waitFor(() => expect(fetcher).toHaveBeenCalled())

    const { result } = renderHook(() =>
      useMutation(() => Promise.resolve('ok'), { invalidateKeys: ['models'] }),
    )
    await act(async () => {
      await result.current.mutateAsync(undefined)
    })
    expect(useQueryStore.getState().cache['models']!.updatedAt).toBe(0)
  })

  it('mutate swallows errors while recording the error state', async () => {
    const fn = vi.fn().mockRejectedValue(new Error('bad'))
    const { result } = renderHook(() => useMutation(fn))
    await act(async () => {
      result.current.mutate('v')
    })
    expect(result.current.error).toBeInstanceOf(Error)
  })

  it('reset clears the error state', async () => {
    const { result } = renderHook(() =>
      useMutation<string, string>(() => Promise.reject(new Error('x'))),
    )
    await act(async () => {
      await result.current.mutateAsync('v').catch(() => {})
    })
    expect(result.current.error).toBeInstanceOf(Error)
    act(() => result.current.reset())
    expect(result.current.error).toBeNull()
  })
})

describe('useInvalidate / useIsFetching', () => {
  beforeEach(() => {
    resetStore()
    vi.clearAllMocks()
  })

  afterEach(cleanup)

  it('useInvalidate makes the given key stale', async () => {
    const fetcher = vi.fn().mockResolvedValue('x')
    renderHook(() => useQuery('k', fetcher))
    await waitFor(() => expect(fetcher).toHaveBeenCalled())
    const { result } = renderHook(() => useInvalidate())
    act(() => result.current('k'))
    expect(isStale('k')).toBe(true)
  })

  it('useIsFetching counts in-flight fetches', async () => {
    let resolveFn!: (v: string) => void
    const fetcher = vi.fn(() => new Promise<string>(r => { resolveFn = r }))
    const { result } = renderHook(() => useIsFetching())
    expect(result.current).toBe(0)
    renderHook(() => useQuery('k', fetcher))
    await waitFor(() => expect(result.current).toBe(1))
    await act(async () => {
      resolveFn('done')
    })
    await waitFor(() => expect(result.current).toBe(0))
  })
})
