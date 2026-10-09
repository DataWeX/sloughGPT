// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { renderHook } from '@testing-library/react'
import { useRefreshShortcut } from './useRefreshShortcut'

describe('useRefreshShortcut', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('calls onRefresh when r is pressed', () => {
    const onRefresh = vi.fn()
    renderHook(() => useRefreshShortcut(onRefresh))
    
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'r' }))
    expect(onRefresh).toHaveBeenCalledTimes(1)
  })

  it('ignores r when meta key is held', () => {
    const onRefresh = vi.fn()
    renderHook(() => useRefreshShortcut(onRefresh))
    
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'r', metaKey: true }))
    expect(onRefresh).not.toHaveBeenCalled()
  })

  it('ignores r when ctrl key is held', () => {
    const onRefresh = vi.fn()
    renderHook(() => useRefreshShortcut(onRefresh))
    
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'r', ctrlKey: true }))
    expect(onRefresh).not.toHaveBeenCalled()
  })

  it('ignores r when target is an input element', () => {
    const onRefresh = vi.fn()
    renderHook(() => useRefreshShortcut(onRefresh))
    
    const input = document.createElement('input')
    document.body.appendChild(input)
    input.focus()
    
    // Dispatch with target set to input
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'r', bubbles: true }))
    expect(onRefresh).not.toHaveBeenCalled()
    
    document.body.removeChild(input)
  })

  it('ignores non-r keys', () => {
    const onRefresh = vi.fn()
    renderHook(() => useRefreshShortcut(onRefresh))
    
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'a' }))
    expect(onRefresh).not.toHaveBeenCalled()
  })

  it('cleans up event listener on unmount', () => {
    const onRefresh = vi.fn()
    const { unmount } = renderHook(() => useRefreshShortcut(onRefresh))
    
    unmount()
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'r' }))
    expect(onRefresh).not.toHaveBeenCalled()
  })
})
