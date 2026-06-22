/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi, afterEach, beforeEach } from 'vitest'
import { renderHook, cleanup } from '@testing-library/react'

const mockPush = vi.fn()

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: mockPush }),
}))

import { useGlobalShortcuts } from './useGlobalShortcuts'

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

beforeEach(() => {
  mockPush.mockReset()
})

function keydown(key: string, mods: { ctrl?: boolean; meta?: boolean; shift?: boolean; alt?: boolean } = {}) {
  const ev = new KeyboardEvent('keydown', {
    key,
    ctrlKey: mods.ctrl ?? false,
    metaKey: mods.meta ?? false,
    shiftKey: mods.shift ?? false,
    altKey: mods.alt ?? false,
    bubbles: true,
  })
  window.dispatchEvent(ev)
}

describe('useGlobalShortcuts', () => {
  it('Ctrl+1 navigates to /chat', () => {
    renderHook(() => useGlobalShortcuts())
    keydown('1', { ctrl: true })
    expect(mockPush).toHaveBeenCalledWith('/chat')
  })

  it('Ctrl+2 navigates to /models', () => {
    renderHook(() => useGlobalShortcuts())
    keydown('2', { ctrl: true })
    expect(mockPush).toHaveBeenCalledWith('/models')
  })

  it('Ctrl+3 navigates to /knowledge', () => {
    renderHook(() => useGlobalShortcuts())
    keydown('3', { ctrl: true })
    expect(mockPush).toHaveBeenCalledWith('/knowledge')
  })

  it('Ctrl+4 navigates to /training', () => {
    renderHook(() => useGlobalShortcuts())
    keydown('4', { ctrl: true })
    expect(mockPush).toHaveBeenCalledWith('/training')
  })

  it('Ctrl+5 navigates to /datasets', () => {
    renderHook(() => useGlobalShortcuts())
    keydown('5', { ctrl: true })
    expect(mockPush).toHaveBeenCalledWith('/datasets')
  })

  it('Ctrl+n dispatches new-chat event', () => {
    const fn = vi.fn()
    window.addEventListener('new-chat', fn)
    renderHook(() => useGlobalShortcuts())
    keydown('n', { ctrl: true })
    expect(fn).toHaveBeenCalledTimes(1)
    window.removeEventListener('new-chat', fn)
  })

  it('Ctrl+Shift+A navigates to /monitoring', () => {
    renderHook(() => useGlobalShortcuts())
    keydown('A', { ctrl: true, shift: true })
    expect(mockPush).toHaveBeenCalledWith('/monitoring')
  })

  it('Ctrl+Shift+F dispatches search-conversations event', () => {
    const fn = vi.fn()
    window.addEventListener('search-conversations', fn)
    renderHook(() => useGlobalShortcuts())
    keydown('F', { ctrl: true, shift: true })
    expect(fn).toHaveBeenCalledTimes(1)
    window.removeEventListener('search-conversations', fn)
  })

  it('Ctrl+Shift+C dispatches copy-last-response event', () => {
    const fn = vi.fn()
    window.addEventListener('copy-last-response', fn)
    renderHook(() => useGlobalShortcuts())
    keydown('C', { ctrl: true, shift: true })
    expect(fn).toHaveBeenCalledTimes(1)
    window.removeEventListener('copy-last-response', fn)
  })

  it('? (no modifier) dispatches toggle-shortcuts event', () => {
    const fn = vi.fn()
    window.addEventListener('toggle-shortcuts', fn)
    renderHook(() => useGlobalShortcuts())
    keydown('?', {})
    expect(fn).toHaveBeenCalledTimes(1)
    window.removeEventListener('toggle-shortcuts', fn)
  })

  it('does not trigger ? when inside an input element', () => {
    const fn = vi.fn()
    window.addEventListener('toggle-shortcuts', fn)
    const input = document.createElement('input')
    document.body.appendChild(input)
    renderHook(() => useGlobalShortcuts())

    const ev = new KeyboardEvent('keydown', { key: '?', bubbles: true })
    input.dispatchEvent(ev)

    expect(fn).not.toHaveBeenCalled()
    document.body.removeChild(input)
  })

  it('does not trigger ? with ctrl modifier', () => {
    const fn = vi.fn()
    window.addEventListener('toggle-shortcuts', fn)
    renderHook(() => useGlobalShortcuts())
    keydown('?', { ctrl: true })
    expect(fn).not.toHaveBeenCalled()
    window.removeEventListener('toggle-shortcuts', fn)
  })

  it('does not navigate for unknown key', () => {
    renderHook(() => useGlobalShortcuts())
    keydown('9', { ctrl: true })
    expect(mockPush).not.toHaveBeenCalled()
  })

  it('removes event listener on unmount', () => {
    const fn = vi.fn()
    window.addEventListener('toggle-shortcuts', fn)
    const { unmount } = renderHook(() => useGlobalShortcuts())
    unmount()
    keydown('?', {})
    expect(fn).not.toHaveBeenCalled()
    window.removeEventListener('toggle-shortcuts', fn)
  })
})
