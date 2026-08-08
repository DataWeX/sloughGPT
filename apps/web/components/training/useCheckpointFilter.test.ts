// @vitest-environment jsdom
import { describe, it, expect } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useCheckpointFilter } from './useCheckpointFilter'
import type { Checkpoint } from '@/lib/souls-controller'

function mkCp(overrides: Partial<Checkpoint> = {}): Checkpoint {
  return { name: 'test', soul: 'test', ...overrides }
}

describe('useCheckpointFilter', () => {
  it('returns all checkpoints by default', () => {
    const cps = [mkCp({ name: 'a' }), mkCp({ name: 'b' })]
    const { result } = renderHook(() => useCheckpointFilter(cps))
    expect(result.current.filtered).toHaveLength(2)
    expect(result.current.types).toEqual(['unknown'])
  })

  it('extracts unique types', () => {
    const cps = [
      mkCp({ model_type: 'slonet' }),
      mkCp({ model_type: 'lora' }),
      mkCp({ model_type: 'slonet' }),
    ]
    const { result } = renderHook(() => useCheckpointFilter(cps))
    expect(result.current.types).toEqual(['lora', 'slonet'])
  })

  it('filters by type', () => {
    const cps = [
      mkCp({ name: 'a', model_type: 'slonet' }),
      mkCp({ name: 'b', model_type: 'lora' }),
    ]
    const { result } = renderHook(() => useCheckpointFilter(cps))
    act(() => result.current.setTypeFilter('slonet'))
    expect(result.current.filtered).toHaveLength(1)
    expect(result.current.filtered[0].name).toBe('a')
  })

  it('filters by max loss', () => {
    const cps = [
      mkCp({ name: 'a', loss: 1.0 }),
      mkCp({ name: 'b', loss: 3.0 }),
    ]
    const { result } = renderHook(() => useCheckpointFilter(cps))
    act(() => result.current.setLossMax('2.0'))
    expect(result.current.filtered).toHaveLength(1)
    expect(result.current.filtered[0].name).toBe('a')
  })

  it('filters by both type and loss', () => {
    const cps = [
      mkCp({ name: 'a', model_type: 'slonet', loss: 1.0 }),
      mkCp({ name: 'b', model_type: 'lora', loss: 1.0 }),
      mkCp({ name: 'c', model_type: 'slonet', loss: 3.0 }),
    ]
    const { result } = renderHook(() => useCheckpointFilter(cps))
    act(() => { result.current.setTypeFilter('slonet'); result.current.setLossMax('2.0') })
    expect(result.current.filtered).toHaveLength(1)
    expect(result.current.filtered[0].name).toBe('a')
  })

  it('uses lineage when model_type missing', () => {
    const cps = [
      mkCp({ name: 'a', lineage: 'custom' }),
      mkCp({ name: 'b' }),
    ]
    const { result } = renderHook(() => useCheckpointFilter(cps))
    expect(result.current.types).toContain('custom')
    expect(result.current.types).toContain('unknown')
  })
})
