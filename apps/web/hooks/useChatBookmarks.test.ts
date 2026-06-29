// @vitest-environment jsdom
import { describe, it, expect, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useChatBookmarks } from './useChatBookmarks'

describe('useChatBookmarks', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('starts with empty bookmarks', () => {
    const { result } = renderHook(() => useChatBookmarks())
    expect(result.current.bookmarks).toEqual([])
  })

  it('adds a bookmark', () => {
    const { result } = renderHook(() => useChatBookmarks())
    act(() => result.current.addBookmark({ id: '1', content: 'hello', role: 'user' }))
    expect(result.current.bookmarks).toHaveLength(1)
    expect(result.current.bookmarks[0].id).toBe('1')
    expect(result.current.bookmarks[0].content).toBe('hello')
  })

  it('does not add duplicate', () => {
    const { result } = renderHook(() => useChatBookmarks())
    act(() => result.current.addBookmark({ id: '1', content: 'hello', role: 'user' }))
    act(() => result.current.addBookmark({ id: '1', content: 'hello', role: 'user' }))
    expect(result.current.bookmarks).toHaveLength(1)
  })

  it('prepends new bookmarks', () => {
    const { result } = renderHook(() => useChatBookmarks())
    act(() => result.current.addBookmark({ id: '1', content: 'first', role: 'user' }))
    act(() => result.current.addBookmark({ id: '2', content: 'second', role: 'user' }))
    expect(result.current.bookmarks[0].id).toBe('2')
  })

  it('checks isBookmarked', () => {
    const { result } = renderHook(() => useChatBookmarks())
    expect(result.current.isBookmarked('1')).toBe(false)
    act(() => result.current.addBookmark({ id: '1', content: 'x', role: 'user' }))
    expect(result.current.isBookmarked('1')).toBe(true)
  })

  it('removes a bookmark', () => {
    const { result } = renderHook(() => useChatBookmarks())
    act(() => result.current.addBookmark({ id: '1', content: 'x', role: 'user' }))
    act(() => result.current.removeBookmark('1'))
    expect(result.current.bookmarks).toHaveLength(0)
    expect(result.current.isBookmarked('1')).toBe(false)
  })

  it('clears all bookmarks', () => {
    const { result } = renderHook(() => useChatBookmarks())
    act(() => result.current.addBookmark({ id: '1', content: 'a', role: 'user' }))
    act(() => result.current.addBookmark({ id: '2', content: 'b', role: 'user' }))
    act(() => result.current.clearAll())
    expect(result.current.bookmarks).toEqual([])
  })

  it('persists to localStorage', () => {
    const { result } = renderHook(() => useChatBookmarks())
    act(() => result.current.addBookmark({ id: '1', content: 'stored', role: 'user' }))
    const raw = localStorage.getItem('chat:bookmarks')
    expect(raw).toBeTruthy()
    const parsed = JSON.parse(raw!)
    expect(parsed).toHaveLength(1)
    expect(parsed[0].id).toBe('1')
  })

  it('loads existing bookmarks from localStorage', () => {
    localStorage.setItem('chat:bookmarks', JSON.stringify([{ id: '3', content: 'saved', role: 'assistant', timestamp: 100 }]))
    const { result } = renderHook(() => useChatBookmarks())
    expect(result.current.bookmarks).toHaveLength(1)
    expect(result.current.bookmarks[0].id).toBe('3')
    expect(result.current.isBookmarked('3')).toBe(true)
  })

  it('handles corrupted localStorage gracefully', () => {
    localStorage.setItem('chat:bookmarks', 'not-json')
    const { result } = renderHook(() => useChatBookmarks())
    expect(result.current.bookmarks).toEqual([])
  })
})
