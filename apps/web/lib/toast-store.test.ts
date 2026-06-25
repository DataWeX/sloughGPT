// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { useToastStore } from './toast-store'

describe('toast-store', () => {
  beforeEach(() => {
    useToastStore.getState().clearToasts()
    vi.useFakeTimers()
  })
  afterEach(() => {
    vi.useRealTimers()
    useToastStore.getState().clearToasts()
  })

  it('starts with empty toasts', () => {
    expect(useToastStore.getState().toasts).toEqual([])
  })

  it('addToast creates a toast with given message and type', () => {
    const id = useToastStore.getState().addToast('Hello', 'success')
    const toast = useToastStore.getState().toasts.find(t => t.id === id)
    expect(toast?.message).toBe('Hello')
    expect(toast?.type).toBe('success')
  })

  it('addToast defaults type to info', () => {
    const id = useToastStore.getState().addToast('Hello')
    const toast = useToastStore.getState().toasts.find(t => t.id === id)
    expect(toast?.type).toBe('info')
  })

  it('addToast stores verbose message', () => {
    const id = useToastStore.getState().addToast('Hello', 'error', 'Details here')
    expect(useToastStore.getState().toasts.find(t => t.id === id)?.verbose).toBe('Details here')
  })

  it('dismissToast removes toast', () => {
    const id = useToastStore.getState().addToast('Hello')
    expect(useToastStore.getState().toasts.length).toBe(1)
    useToastStore.getState().dismissToast(id)
    expect(useToastStore.getState().toasts.length).toBe(0)
  })

  it('clearToasts removes all toasts', () => {
    useToastStore.getState().addToast('A')
    useToastStore.getState().addToast('B')
    useToastStore.getState().clearToasts()
    expect(useToastStore.getState().toasts).toEqual([])
  })

  it('auto-removes toast after 6 seconds', () => {
    const id = useToastStore.getState().addToast('Hello')
    expect(useToastStore.getState().toasts.length).toBe(1)
    vi.advanceTimersByTime(6000)
    expect(useToastStore.getState().toasts.length).toBe(0)
  })

  it('generates unique IDs for each toast', () => {
    const id1 = useToastStore.getState().addToast('A')
    const id2 = useToastStore.getState().addToast('B')
    expect(id1).not.toBe(id2)
  })
})
