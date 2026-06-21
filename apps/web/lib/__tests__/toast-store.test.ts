import { describe, expect, it, beforeEach, vi, afterEach } from 'vitest'

import { useToastStore } from '../toast-store'

describe('useToastStore', () => {
  beforeEach(() => { useToastStore.getState().clearToasts() })

  it('starts empty', () => {
    expect(useToastStore.getState().toasts).toEqual([])
  })

  it('addToast adds a toast with generated id', () => {
    const id = useToastStore.getState().addToast('Hello')
    const toasts = useToastStore.getState().toasts
    expect(toasts).toHaveLength(1)
    expect(toasts[0].message).toBe('Hello')
    expect(toasts[0].type).toBe('info')
    expect(id).toMatch(/^toast_\d+_/)
  })

  it('addToast accepts type and verbose', () => {
    useToastStore.getState().addToast('Error!', 'error', 'Stack trace here')
    const t = useToastStore.getState().toasts[0]
    expect(t.type).toBe('error')
    expect(t.verbose).toBe('Stack trace here')
  })

  it('addToast defaults to info type', () => {
    useToastStore.getState().addToast('Just so you know')
    expect(useToastStore.getState().toasts[0].type).toBe('info')
  })

  it('dismissToast removes by id', () => {
    const id = useToastStore.getState().addToast('Test')
    expect(useToastStore.getState().toasts).toHaveLength(1)
    useToastStore.getState().dismissToast(id)
    expect(useToastStore.getState().toasts).toHaveLength(0)
  })

  it('clearToasts removes all', () => {
    useToastStore.getState().addToast('A')
    useToastStore.getState().addToast('B')
    useToastStore.getState().clearToasts()
    expect(useToastStore.getState().toasts).toEqual([])
  })

  it('auto-dismisses after 6 seconds', () => {
    vi.useFakeTimers()
    useToastStore.getState().addToast('Auto dismiss')
    expect(useToastStore.getState().toasts).toHaveLength(1)
    vi.advanceTimersByTime(6000)
    expect(useToastStore.getState().toasts).toHaveLength(0)
    vi.useRealTimers()
  })

  it('does not auto-dismiss if manually dismissed first', () => {
    vi.useFakeTimers()
    const id = useToastStore.getState().addToast('Manual first')
    expect(useToastStore.getState().toasts).toHaveLength(1)
    useToastStore.getState().dismissToast(id)
    expect(useToastStore.getState().toasts).toHaveLength(0)
    vi.advanceTimersByTime(6000)
    expect(useToastStore.getState().toasts).toHaveLength(0)
    vi.useRealTimers()
  })

  it('returns id from addToast', () => {
    const id = useToastStore.getState().addToast('Return id')
    expect(typeof id).toBe('string')
    expect(id.startsWith('toast_')).toBe(true)
  })
})
