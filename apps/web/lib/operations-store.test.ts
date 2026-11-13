'use client'

import { describe, it, expect, beforeEach, vi } from 'vitest'

vi.mock('@/lib/http-client', () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
}))

import { operationsStore, useOperationsStore } from './operations-store'
import { apiGet, apiPost } from '@/lib/http-client'

const mockGet = vi.mocked(apiGet)
const mockPost = vi.mocked(apiPost)

beforeEach(() => {
  operationsStore.setState({
    operations: [],
    counts: {},
    loading: false,
    error: null,
  })
  vi.clearAllMocks()
})

describe('operationsStore', () => {
  it('fetch loads operations', async () => {
    mockGet.mockResolvedValueOnce({
      operations: [
        { id: 'op1', type: 'training', label: 'shakespeare', status: 'running', created_at: 1, started_at: 1, finished_at: null, elapsed_s: 5, error: null, meta: {} },
      ],
      counts: { running: 1 },
    })
    await operationsStore.getState().fetch()
    const s = operationsStore.getState()
    expect(s.operations).toHaveLength(1)
    expect(s.operations[0].id).toBe('op1')
    expect(s.counts).toEqual({ running: 1 })
    expect(s.loading).toBe(false)
  })

  it('fetch handles error', async () => {
    mockGet.mockRejectedValueOnce(new Error('network'))
    await operationsStore.getState().fetch()
    expect(operationsStore.getState().error).toBe('network')
    expect(operationsStore.getState().loading).toBe(false)
  })

  it('cancel calls POST /cancel/{id} and refreshes', async () => {
    mockPost.mockResolvedValueOnce({ id: 'op1', status: 'cancelled' })
    mockGet.mockResolvedValueOnce({ operations: [], counts: {} })
    const ok = await operationsStore.getState().cancel('op1')
    expect(ok).toBe(true)
    expect(mockPost).toHaveBeenCalledWith('/cancel/op1')
    expect(mockGet).toHaveBeenCalled()
  })

  it('cancel returns false on error', async () => {
    mockPost.mockRejectedValueOnce(new Error('fail'))
    const ok = await operationsStore.getState().cancel('op1')
    expect(ok).toBe(false)
  })

  it('cancelAll calls POST /cancel-all', async () => {
    mockPost.mockResolvedValueOnce({ cancelled: ['a', 'b'], count: 2 })
    mockGet.mockResolvedValueOnce({ operations: [], counts: {} })
    const n = await operationsStore.getState().cancelAll('training')
    expect(n).toBe(2)
    expect(mockPost).toHaveBeenCalledWith('/cancel-all?type=training')
  })

  it('cancelAll without type omits query string', async () => {
    mockPost.mockResolvedValueOnce({ cancelled: [], count: 0 })
    mockGet.mockResolvedValueOnce({ operations: [], counts: {} })
    await operationsStore.getState().cancelAll()
    expect(mockPost).toHaveBeenCalledWith('/cancel-all')
  })

  it('isAnyActive returns true when running ops exist', () => {
    operationsStore.setState({
      operations: [
        { id: 'op1', type: 'training', label: 'test', status: 'running', created_at: 1, started_at: 1, finished_at: null, elapsed_s: 0, error: null, meta: {} },
      ],
    })
    expect(operationsStore.getState().isAnyActive()).toBe(true)
    expect(operationsStore.getState().isAnyActive('training')).toBe(true)
    expect(operationsStore.getState().isAnyActive('inference')).toBe(false)
  })

  it('isAnyActive returns false when all completed', () => {
    operationsStore.setState({
      operations: [
        { id: 'op1', type: 'training', label: 'test', status: 'completed', created_at: 1, started_at: 1, finished_at: 2, elapsed_s: 1, error: null, meta: {} },
      ],
    })
    expect(operationsStore.getState().isAnyActive()).toBe(false)
  })

  it('activeByType filters correctly', () => {
    operationsStore.setState({
      operations: [
        { id: 'op1', type: 'training', label: 'a', status: 'running', created_at: 1, started_at: 1, finished_at: null, elapsed_s: 0, error: null, meta: {} },
        { id: 'op2', type: 'inference', label: 'b', status: 'running', created_at: 1, started_at: 1, finished_at: null, elapsed_s: 0, error: null, meta: {} },
        { id: 'op3', type: 'training', label: 'c', status: 'completed', created_at: 1, started_at: 1, finished_at: 2, elapsed_s: 1, error: null, meta: {} },
      ],
    })
    expect(operationsStore.getState().activeByType('training')).toHaveLength(1)
    expect(operationsStore.getState().activeByType('training')[0].id).toBe('op1')
  })

  it('hasActive checks type', () => {
    operationsStore.setState({
      operations: [
        { id: 'op1', type: 'download', label: 'file', status: 'registered', created_at: 1, started_at: null, finished_at: null, elapsed_s: 0, error: null, meta: {} },
      ],
    })
    expect(operationsStore.getState().hasActive('download')).toBe(true)
    expect(operationsStore.getState().hasActive('training')).toBe(false)
  })

  it('startPolling sets up interval and fetches immediately', async () => {
    vi.useFakeTimers()
    mockGet.mockResolvedValue({ operations: [], counts: {} })
    operationsStore.getState().startPolling(1000)
    expect(mockGet).toHaveBeenCalledTimes(1)
    vi.advanceTimersByTime(1000)
    expect(mockGet).toHaveBeenCalledTimes(2)
    operationsStore.getState().stopPolling()
    vi.useRealTimers()
  })

  it('stopPolling clears interval', () => {
    vi.useFakeTimers()
    mockGet.mockResolvedValue({ operations: [], counts: {} })
    operationsStore.getState().startPolling(1000)
    operationsStore.getState().stopPolling()
    expect(operationsStore.getState()._pollTimer).toBeNull()
    vi.useRealTimers()
  })
})
