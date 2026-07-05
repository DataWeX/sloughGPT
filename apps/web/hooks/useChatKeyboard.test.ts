import { renderHook, cleanup } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { createRef } from 'react'

import { useChatKeyboard } from './useChatKeyboard'

function fireKey(key: string, options?: { ctrlKey?: boolean; metaKey?: boolean; shiftKey?: boolean }) {
  window.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, ...options }))
}

describe('useChatKeyboard', () => {
  const loadingRef = { current: new AbortController() }
  const newChatRef = { current: vi.fn() }
  const handleRegenerateRef = { current: vi.fn() }
  const setToolPanelOpen = vi.fn()
  const setShowSettings = vi.fn()
  const setLoading = vi.fn()
  const setCurrentError = vi.fn()

  beforeEach(() => {
    vi.restoreAllMocks()
    vi.spyOn(loadingRef.current, 'abort')
  })
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('aborts loading on Escape when loading', () => {
    renderHook(() => useChatKeyboard({
      loading: true, currentError: null, showSettings: false,
      setToolPanelOpen, setShowSettings, setLoading, setCurrentError,
      loadingRef, newChatRef, handleRegenerateRef,
    }))
    fireKey('Escape')
    expect(loadingRef.current.abort).toHaveBeenCalled()
    expect(setLoading).toHaveBeenCalledWith(false)
  })

  it('clears error on Escape when error present', () => {
    renderHook(() => useChatKeyboard({
      loading: false, currentError: 'error', showSettings: false,
      setToolPanelOpen, setShowSettings, setLoading, setCurrentError,
      loadingRef, newChatRef, handleRegenerateRef,
    }))
    fireKey('Escape')
    expect(setCurrentError).toHaveBeenCalledWith(null)
  })

  it('toggles settings on Escape when settings open', () => {
    renderHook(() => useChatKeyboard({
      loading: false, currentError: null, showSettings: true,
      setToolPanelOpen, setShowSettings, setLoading, setCurrentError,
      loadingRef, newChatRef, handleRegenerateRef,
    }))
    fireKey('Escape')
    expect(setShowSettings).toHaveBeenCalled()
  })

  it('toggles tool panel on Ctrl+K', () => {
    renderHook(() => useChatKeyboard({
      loading: false, currentError: null, showSettings: false,
      setToolPanelOpen, setShowSettings, setLoading, setCurrentError,
      loadingRef, newChatRef, handleRegenerateRef,
    }))
    fireKey('k', { ctrlKey: true })
    expect(setToolPanelOpen).toHaveBeenCalled()
  })

  it('toggles tool panel on Meta+K', () => {
    renderHook(() => useChatKeyboard({
      loading: false, currentError: null, showSettings: false,
      setToolPanelOpen, setShowSettings, setLoading, setCurrentError,
      loadingRef, newChatRef, handleRegenerateRef,
    }))
    fireKey('k', { metaKey: true })
    expect(setToolPanelOpen).toHaveBeenCalled()
  })

  it('toggles tool panel on Ctrl+Shift+B', () => {
    renderHook(() => useChatKeyboard({
      loading: false, currentError: null, showSettings: false,
      setToolPanelOpen, setShowSettings, setLoading, setCurrentError,
      loadingRef, newChatRef, handleRegenerateRef,
    }))
    fireKey('b', { ctrlKey: true, shiftKey: true })
    expect(setToolPanelOpen).toHaveBeenCalled()
  })

  it('toggles tool panel on Meta+Shift+B', () => {
    renderHook(() => useChatKeyboard({
      loading: false, currentError: null, showSettings: false,
      setToolPanelOpen, setShowSettings, setLoading, setCurrentError,
      loadingRef, newChatRef, handleRegenerateRef,
    }))
    fireKey('b', { metaKey: true, shiftKey: true })
    expect(setToolPanelOpen).toHaveBeenCalled()
  })

  it('toggles settings on Ctrl+?', () => {
    renderHook(() => useChatKeyboard({
      loading: false, currentError: null, showSettings: false,
      setToolPanelOpen, setShowSettings, setLoading, setCurrentError,
      loadingRef, newChatRef, handleRegenerateRef,
    }))
    fireKey('?', { ctrlKey: true })
    expect(setShowSettings).toHaveBeenCalled()
  })

  it('calls newChatRef on Ctrl+N', () => {
    renderHook(() => useChatKeyboard({
      loading: false, currentError: null, showSettings: false,
      setToolPanelOpen, setShowSettings, setLoading, setCurrentError,
      loadingRef, newChatRef, handleRegenerateRef,
    }))
    fireKey('n', { ctrlKey: true })
    expect(newChatRef.current).toHaveBeenCalled()
  })

  it('calls handleRegenerateRef on Ctrl+R', () => {
    renderHook(() => useChatKeyboard({
      loading: false, currentError: null, showSettings: false,
      setToolPanelOpen, setShowSettings, setLoading, setCurrentError,
      loadingRef, newChatRef, handleRegenerateRef,
    }))
    fireKey('r', { ctrlKey: true })
    expect(handleRegenerateRef.current).toHaveBeenCalled()
  })

  it('does not react to unrelated keys', () => {
    renderHook(() => useChatKeyboard({
      loading: false, currentError: null, showSettings: false,
      setToolPanelOpen, setShowSettings, setLoading, setCurrentError,
      loadingRef, newChatRef, handleRegenerateRef,
    }))
    fireKey('a')
    expect(setToolPanelOpen).not.toHaveBeenCalled()
    expect(setShowSettings).not.toHaveBeenCalled()
    expect(newChatRef.current).not.toHaveBeenCalled()
    expect(handleRegenerateRef.current).not.toHaveBeenCalled()
  })

  it('cleans up event listener on unmount', () => {
    const { unmount } = renderHook(() => useChatKeyboard({
      loading: false, currentError: null, showSettings: false,
      setToolPanelOpen, setShowSettings, setLoading, setCurrentError,
      loadingRef, newChatRef, handleRegenerateRef,
    }))
    unmount()
    fireKey('Escape')
    expect(setLoading).not.toHaveBeenCalled()
  })
})
