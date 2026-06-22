/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { renderHook, act, cleanup } from '@testing-library/react'
import { useChatUI } from './useChatUI'

afterEach(cleanup)

describe('useChatUI', () => {
  it('returns default state values', () => {
    const { result } = renderHook(() => useChatUI())
    expect(result.current.showSettings).toBe(false)
    expect(result.current.showConversationViewer).toBe(false)
    expect(result.current.searchQuery).toBe('')
    expect(result.current.toolPanelOpen).toBe(true)
    expect(result.current.voiceMode).toBe(false)
    expect(result.current.sidebarOpen).toBe(false)
    expect(result.current.chatScreenRef).toBeDefined()
  })

  it('toggleSettings flips showSettings', () => {
    const { result } = renderHook(() => useChatUI())
    act(() => result.current.toggleSettings())
    expect(result.current.showSettings).toBe(true)
    act(() => result.current.toggleSettings())
    expect(result.current.showSettings).toBe(false)
  })

  it('handleSearchChange updates query and resets matchIndex', () => {
    const { result } = renderHook(() => useChatUI())
    act(() => result.current.setMatchIndex(5))
    act(() => result.current.handleSearchChange('hello'))
    expect(result.current.searchQuery).toBe('hello')
    expect(result.current.matchIndex).toBe(0)
  })

  it('handleSearchClear clears query', () => {
    const { result } = renderHook(() => useChatUI())
    act(() => result.current.setSearchQuery('test'))
    act(() => result.current.handleSearchClear())
    expect(result.current.searchQuery).toBe('')
  })

  it('setter functions update individual states', () => {
    const { result } = renderHook(() => useChatUI())
    act(() => result.current.setShowSettings(true))
    expect(result.current.showSettings).toBe(true)
    act(() => result.current.setVoiceMode(true))
    expect(result.current.voiceMode).toBe(true)
    act(() => result.current.setSidebarOpen(true))
    expect(result.current.sidebarOpen).toBe(true)
    act(() => result.current.setToolPanelOpen(false))
    expect(result.current.toolPanelOpen).toBe(false)
  })
})
