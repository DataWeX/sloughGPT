import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useConsciousnessBatch } from './useConsciousnessBatch'
import type { ConsciousnessBatchOperation } from './useConsciousnessBatch'

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    seedData: vi.fn(),
    submitFeedback: vi.fn(),
    reflect: vi.fn(),
    updateConfig: vi.fn(),
  },
}))

import { consciousnessController } from '@/lib/consciousness-controller'
const mockCtrl = vi.mocked(consciousnessController)

describe('useConsciousnessBatch', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('returns initial idle state', () => {
    const { result } = renderHook(() => useConsciousnessBatch())
    expect(result.current.loading).toBe(false)
    expect(result.current.results).toBeNull()
    expect(result.current.error).toBeNull()
  })

  it('executeBatch posts operations and returns results', async () => {
    mockCtrl.seedData.mockResolvedValue({ seeded: true } as any)
    mockCtrl.reflect.mockResolvedValue({ reflection: 'test' } as any)

    const { result } = renderHook(() => useConsciousnessBatch())
    const ops: ConsciousnessBatchOperation[] = [
      { type: 'seed', payload: { text: 'hello' } },
      { type: 'reflect', payload: {} },
    ]

    await act(async () => {
      await result.current.executeBatch(ops)
    })

    expect(result.current.results).toHaveLength(2)
    expect(result.current.results[0].status).toBe('fulfilled')
    expect(result.current.results[1].status).toBe('fulfilled')
    expect(result.current.error).toBeNull()
  })

  it('sets loading during execution', async () => {
    let resolve: (v: any) => void
    mockCtrl.updateConfig.mockImplementation(
      () => new Promise((r) => { resolve = r })
    )

    const { result } = renderHook(() => useConsciousnessBatch())

    act(() => {
      result.current.executeBatch([{ type: 'config', payload: {} }])
    })

    expect(result.current.loading).toBe(true)

    await act(async () => {
      resolve!({ data: {} })
    })

    expect(result.current.loading).toBe(false)
  })

  it('handles individual operation rejection via Promise.allSettled', async () => {
    mockCtrl.seedData.mockRejectedValue(new Error('offline'))

    const { result } = renderHook(() => useConsciousnessBatch())

    await act(async () => {
      await result.current.executeBatch([{ type: 'seed', payload: {} }])
    })

    expect(result.current.results).toEqual([
      { status: 'rejected', reason: expect.objectContaining({ message: 'offline' }) },
    ])
    expect(result.current.error).toBeNull()
  })

  it('returns raw results when data key missing', async () => {
    mockCtrl.reflect.mockResolvedValue({ reflection: 'ok' })

    const { result } = renderHook(() => useConsciousnessBatch())

    await act(async () => {
      await result.current.executeBatch([{ type: 'reflect', payload: {} }])
    })

    expect(result.current.results).toEqual([{ status: 'fulfilled', value: { reflection: 'ok' } }])
  })
})
