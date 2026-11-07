import { renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { apiGet } from '@/lib/http-client'

import { protocolStore } from './store'
import { useEndpoint } from './useEndpoint'

vi.mock('@/lib/http-client', () => ({
  apiGet: vi.fn(),
}))

const apiGetMock = vi.mocked(apiGet)

beforeEach(() => {
  protocolStore.getState().invalidate()
  apiGetMock.mockReset()
  apiGetMock.mockResolvedValue({ history: [{ step: 0, voice: { warmth: 0.5 } }] })
})

describe('useEndpoint — components name a capability, never a URL', () => {
  it('reads through the descriptor and lands in success', async () => {
    const { result } = renderHook(() =>
      useEndpoint<{ history: unknown[] }>('consciousness.personality_history'),
    )

    expect(result.current.status).toBe('loading')
    await waitFor(() => expect(result.current.status).toBe('success'))

    expect(result.current.data?.history).toHaveLength(1)
    expect(apiGetMock).toHaveBeenCalledWith('/consciousness/personality/history', undefined)
    expect(result.current.error).toBeUndefined()
  })

  it('refetch re-reads past the idempotent cache', async () => {
    const { result } = renderHook(() =>
      useEndpoint('consciousness.personality_history'),
    )
    await waitFor(() => expect(result.current.status).toBe('success'))

    apiGetMock.mockResolvedValue({ history: [] })
    result.current.refetch()
    await waitFor(() => expect(apiGetMock).toHaveBeenCalledTimes(2))
    expect(result.current.data).toEqual({ history: [] })
  })

  it('keeps the previous data while a forced re-read is in flight', async () => {
    const { result } = renderHook(() =>
      useEndpoint('consciousness.personality_history'),
    )
    await waitFor(() => expect(result.current.status).toBe('success'))

    apiGetMock.mockReturnValue(new Promise(() => {})) // never settles
    result.current.refetch()
    await waitFor(() =>
      expect(protocolStore.getState().reads['consciousness.personality_history@1#']?.status).toBe(
        'loading',
      ),
    )
    expect(result.current.data).toEqual({ history: [{ step: 0, voice: { warmth: 0.5 } }] })
  })

  it('distinct params produce distinct slices', async () => {
    const { result, rerender } = renderHook(
      ({ prefix }: { prefix: string }) => useEndpoint('contracts.list', { prefix }),
      { initialProps: { prefix: '/a' } },
    )
    await waitFor(() => expect(result.current.status).toBe('success'))

    apiGetMock.mockResolvedValue({ contracts: [], count: 0 })
    rerender({ prefix: '/b' })
    await waitFor(() =>
      expect(apiGetMock).toHaveBeenCalledWith('/contracts', { prefix: '/b' }),
    )
  })
})
