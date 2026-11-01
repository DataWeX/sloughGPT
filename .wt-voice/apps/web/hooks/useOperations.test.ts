// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest'
import { renderHook, cleanup } from '@testing-library/react'
import { useOperations } from './useOperations'
import { operationsStore, useActiveOperations, useHasActiveOperations } from '@/lib/operations-store'

afterEach(cleanup)

beforeEach(() => {
  operationsStore.setState({
    operations: [],
    counts: {},
    loading: false,
    error: null,
    _pollTimer: null,
  })
  vi.clearAllMocks()
  vi.useRealTimers()
})

describe('useOperations', () => {
  it('returns empty state initially', () => {
    const { result } = renderHook(() => useOperations())
    expect(result.current.operations).toEqual([])
    expect(result.current.isActive).toBe(false)
  })

  it('isActive reflects running ops', () => {
    operationsStore.setState({
      operations: [
        { id: 'op1', type: 'training', label: 't', status: 'running', created_at: 1, started_at: 1, finished_at: null, elapsed_s: 0, error: null, meta: {} },
      ],
    })
    const { result } = renderHook(() => useOperations())
    expect(result.current.isActive).toBe(true)
    expect(result.current.hasTraining).toBe(true)
    expect(result.current.hasInference).toBe(false)
  })

  it('activeOps filters by type', () => {
    operationsStore.setState({
      operations: [
        { id: 'op1', type: 'training', label: 'a', status: 'running', created_at: 1, started_at: 1, finished_at: null, elapsed_s: 0, error: null, meta: {} },
        { id: 'op2', type: 'inference', label: 'b', status: 'running', created_at: 1, started_at: 1, finished_at: null, elapsed_s: 0, error: null, meta: {} },
      ],
    })
    const { result } = renderHook(() => useOperations('training'))
    expect(result.current.activeOps).toHaveLength(1)
    expect(result.current.activeOps[0].type).toBe('training')
  })
})

describe('useActiveOperations', () => {
  it('returns running ops', () => {
    operationsStore.setState({
      operations: [
        { id: 'a', type: 'training', label: 'a', status: 'running', created_at: 1, started_at: 1, finished_at: null, elapsed_s: 0, error: null, meta: {} },
        { id: 'b', type: 'training', label: 'b', status: 'completed', created_at: 1, started_at: 1, finished_at: 2, elapsed_s: 1, error: null, meta: {} },
      ],
    })
    const { result } = renderHook(() => useActiveOperations())
    expect(result.current).toHaveLength(1)
  })
})

describe('useHasActiveOperations', () => {
  it('returns boolean for type check', () => {
    operationsStore.setState({
      operations: [
        { id: 'a', type: 'download', label: 'dl', status: 'registered', created_at: 1, started_at: null, finished_at: null, elapsed_s: 0, error: null, meta: {} },
      ],
    })
    const { result } = renderHook(() => useHasActiveOperations('download'))
    expect(result.current).toBe(true)
  })
})
