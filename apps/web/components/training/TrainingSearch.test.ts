// @vitest-environment jsdom
import { describe, it, expect } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useTrainingSearch } from './TrainingSearch'
import type { Checkpoint } from '@/lib/souls-controller'

function mkCp(overrides: Partial<Checkpoint> = {}): Checkpoint {
  return { name: 'test', soul: 'test', ...overrides }
}

describe('useTrainingSearch', () => {
  it('returns all checkpoints when query is empty', () => {
    const cps = [mkCp({ name: 'a' }), mkCp({ name: 'b' })]
    const { result } = renderHook(() => useTrainingSearch(cps))
    expect(result.current.filtered).toHaveLength(2)
    expect(result.current.query).toBe('')
  })

  it('filters by name', () => {
    const cps = [mkCp({ name: 'alpha' }), mkCp({ name: 'beta' })]
    const { result } = renderHook(() => useTrainingSearch(cps))
    act(() => result.current.setQuery('alpha'))
    expect(result.current.filtered).toHaveLength(1)
    expect(result.current.filtered[0].name).toBe('alpha')
  })

  it('filters by soul', () => {
    const cps = [mkCp({ soul: 'friendly' }), mkCp({ soul: 'formal' })]
    const { result } = renderHook(() => useTrainingSearch(cps))
    act(() => result.current.setQuery('friendly'))
    expect(result.current.filtered).toHaveLength(1)
  })

  it('filters by model_type', () => {
    const cps = [
      mkCp({ name: 'a', model_type: 'slonet' }),
      mkCp({ name: 'b', model_type: 'lora' }),
    ]
    const { result } = renderHook(() => useTrainingSearch(cps))
    act(() => result.current.setQuery('slonet'))
    expect(result.current.filtered).toHaveLength(1)
  })

  it('filters by training_dataset', () => {
    const cps = [
      mkCp({ name: 'a', training_dataset: 'shakespeare' }),
      mkCp({ name: 'b', training_dataset: 'code' }),
    ]
    const { result } = renderHook(() => useTrainingSearch(cps))
    act(() => result.current.setQuery('shakespeare'))
    expect(result.current.filtered).toHaveLength(1)
  })

  it('filters by loss value', () => {
    const cps = [
      mkCp({ name: 'a', loss: 1.5 }),
      mkCp({ name: 'b', loss: 3.0 }),
    ]
    const { result } = renderHook(() => useTrainingSearch(cps))
    act(() => result.current.setQuery('1.5'))
    expect(result.current.filtered).toHaveLength(1)
    expect(result.current.filtered[0].name).toBe('a')
  })

  it('case-insensitive search', () => {
    const cps = [mkCp({ name: 'Alpha' })]
    const { result } = renderHook(() => useTrainingSearch(cps))
    act(() => result.current.setQuery('alpha'))
    expect(result.current.filtered).toHaveLength(1)
  })

  it('empty query returns all', () => {
    const cps = [mkCp({ name: 'a' }), mkCp({ name: 'b' })]
    const { result } = renderHook(() => useTrainingSearch(cps))
    act(() => result.current.setQuery('xyz'))
    expect(result.current.filtered).toHaveLength(0)
    act(() => result.current.setQuery(''))
    expect(result.current.filtered).toHaveLength(2)
  })
})
