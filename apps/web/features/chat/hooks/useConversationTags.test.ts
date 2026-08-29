import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act, cleanup } from '@testing-library/react'
import { useConversationTags } from './useConversationTags'

afterEach(cleanup)

describe('useConversationTags', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('initializes with empty tags', () => {
    const { result } = renderHook(() => useConversationTags())
    expect(result.current.tags).toEqual({})
    expect(result.current.loading).toBe(false)
  })

  it('adds a tag', () => {
    const { result } = renderHook(() => useConversationTags())
    
    act(() => {
      result.current.addTag('session1', 'important')
    })

    expect(result.current.getTags('session1')).toEqual(['important'])
  })

  it('does not add duplicate tags', () => {
    const { result } = renderHook(() => useConversationTags())
    
    act(() => {
      result.current.addTag('session1', 'important')
      result.current.addTag('session1', 'important')
    })

    expect(result.current.getTags('session1')).toEqual(['important'])
  })

  it('removes a tag', () => {
    const { result } = renderHook(() => useConversationTags())
    
    act(() => {
      result.current.addTag('session1', 'important')
      result.current.removeTag('session1', 'important')
    })

    expect(result.current.getTags('session1')).toEqual([])
  })

  it('checks if session has tag', () => {
    const { result } = renderHook(() => useConversationTags())
    
    act(() => {
      result.current.addTag('session1', 'important')
    })

    expect(result.current.hasTag('session1', 'important')).toBe(true)
    expect(result.current.hasTag('session1', 'other')).toBe(false)
  })

  it('gets all unique tags', () => {
    const { result } = renderHook(() => useConversationTags())
    
    act(() => {
      result.current.addTag('session1', 'important')
      result.current.addTag('session2', 'work')
      result.current.addTag('session1', 'work')
    })

    expect(result.current.getAllTags()).toEqual(['important', 'work'])
  })

  it('trims and lowercases tags', () => {
    const { result } = renderHook(() => useConversationTags())
    
    act(() => {
      result.current.addTag('session1', '  Important  ')
    })

    expect(result.current.getTags('session1')).toEqual(['important'])
  })

  it('persists to localStorage', () => {
    const { result } = renderHook(() => useConversationTags())
    
    act(() => {
      result.current.addTag('session1', 'important')
    })

    const stored = JSON.parse(localStorage.getItem('chat-conversation-tags') || '{}')
    expect(stored.session1).toEqual(['important'])
  })
})