/**
 * Hook-level guards for useVmPage that the component suite in page.test.tsx
 * cannot express: the hydration race (an edit made while the chatDB read is
 * still in flight must win over the persisted value). Poll/ persistence/
 * clamping behaviour is covered end-to-end by page.test.tsx — not repeated
 * here (Test Rule: nothing verified twice).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, waitFor, act } from '@testing-library/react'
import { chatDB } from '@/lib/db'

const kvStore = new Map<string, unknown>()

vi.mock('@/lib/db', () => ({
  chatDB: {
    getKV: vi.fn(async (key: string) => kvStore.get(key) ?? null),
    setKV: vi.fn(async (key: string, value: unknown) => {
      kvStore.set(key, value)
    }),
  },
}))

vi.mock('@/lib/vm-controller', () => ({
  vmController: {
    run: vi.fn(),
    trainingJob: vi.fn(),
  },
}))

vi.mock('@/lib/dataset-controller', () => ({
  datasetController: {
    list: vi.fn(async () => []),
  },
}))

vi.mock('@/hooks/useRefreshShortcut', () => ({
  useRefreshShortcut: vi.fn(),
}))

import { useVmPage } from './useVmPage'

describe('useVmPage hydration', () => {
  beforeEach(() => {
    kvStore.clear()
    vi.mocked(chatDB.getKV).mockImplementation(async (key: string) => kvStore.get(key) ?? null)
    vi.mocked(chatDB.setKV).mockImplementation(async (key: string, value: unknown) => {
      kvStore.set(key, value)
    })
  })
  afterEach(() => vi.clearAllMocks())

  it('does not let hydration clobber an edit made while it was loading', async () => {
    kvStore.set('vm-source', '; persisted program')
    kvStore.set('vm-role', 'admin')

    const { result } = renderHook(() => useVmPage())
    // Edit before the async load lands. The touched flag is set synchronously,
    // so the load that resolves later must skip the source field…
    act(() => {
      result.current.setSource('; my edit')
    })

    // useVmPage does not expose `hydrated`; role is loaded by the same
    // Promise.allSettled batch as source, so it is the hydration barrier.
    await waitFor(() => expect(result.current.role).toBe('admin'))

    expect(result.current.source).toBe('; my edit')
    // …while untouched fields still pick up the fresh server value.
    expect(result.current.role).toBe('admin')
  })
})
