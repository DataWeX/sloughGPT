import { describe, expect, it, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router-dom'
import { useRouter } from './navigation'
import { useQueryStore, fetchQuery, isStale } from '../../lib/cache/client'

// NOTE: named .tsx so vitest's `vite/**/*.test.tsx` → jsdom glob picks it up
// (the `.ts` sibling pattern would run in the default `node` environment).

const wrapper = ({ children }: { children: ReactNode }) => <MemoryRouter>{children}</MemoryRouter>

describe('useRouter().refresh', () => {
  beforeEach(() => {
    useQueryStore.setState({ cache: {}, fetchingKeys: new Set(), refreshVersion: 0 })
  })

  it('revalidates client state in place instead of reloading the document', async () => {
    await fetchQuery('navigation-test', async () => 'payload', { staleTime: 60_000 })
    expect(isStale('navigation-test')).toBe(false)

    const { result } = renderHook(() => useRouter(), { wrapper })

    // jsdom's window.location.reload is non-configurable, so replace the whole
    // location object for the duration of the call to observe any reload.
    const locationDescriptor = Object.getOwnPropertyDescriptor(window, 'location')!
    const reload = vi.fn()
    Object.defineProperty(window, 'location', { configurable: true, value: { reload } })

    const versionBefore = useQueryStore.getState().refreshVersion
    try {
      act(() => {
        result.current.refresh()
      })

      // No document reload…
      expect(reload).not.toHaveBeenCalled()

      // …but the cache really was revalidated: version bumped, every entry
      // marked stale (a staleTime of 60s proves it is the mark, not the clock),
      // and the previous data kept so views do not flash.
      expect(useQueryStore.getState().refreshVersion).toBe(versionBefore + 1)
      expect(isStale('navigation-test')).toBe(true)
      expect(useQueryStore.getState().cache['navigation-test']?.data).toBe('payload')
    } finally {
      Object.defineProperty(window, 'location', locationDescriptor)
    }
  })

  it('keeps the router callback identity stable across re-renders', () => {
    const { result, rerender } = renderHook(() => useRouter(), { wrapper })
    const first = result.current.refresh
    rerender()
    expect(result.current.refresh).toBe(first)
  })
})
