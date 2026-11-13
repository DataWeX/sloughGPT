import { describe, it, expect, beforeEach, vi } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'

const { mockBookmarks, mockChatDB } = vi.hoisted(() => {
  const mockBookmarks = new Map<string, any>()
  const mockChatDB = {
    getBookmarks: vi.fn(async () => [...mockBookmarks.values()]),
    addBookmark: vi.fn(async (bm: any) => { mockBookmarks.set(bm.id, bm) }),
    removeBookmark: vi.fn(async (id: string) => { mockBookmarks.delete(id) }),
    clearBookmarks: vi.fn(async () => { mockBookmarks.clear() }),
  }
  return { mockBookmarks, mockChatDB }
})

vi.mock('@/lib/db', () => ({
  chatDB: mockChatDB,
}))

import { useChatBookmarks } from './useChatBookmarks'

beforeEach(() => {
  mockBookmarks.clear()
  vi.clearAllMocks()
  mockChatDB.getBookmarks.mockImplementation(async () => [...mockBookmarks.values()])
  mockChatDB.addBookmark.mockImplementation(async (bm: any) => { mockBookmarks.set(bm.id, bm) })
  mockChatDB.removeBookmark.mockImplementation(async (id: string) => { mockBookmarks.delete(id) })
  mockChatDB.clearBookmarks.mockImplementation(async () => { mockBookmarks.clear() })
})

describe('useChatBookmarks', () => {
  it('loads bookmarks from chatDB on mount', async () => {
    const { result } = renderHook(() => useChatBookmarks())
    await waitFor(() => expect(result.current.bookmarks).toEqual([]))
    expect(mockChatDB.getBookmarks).toHaveBeenCalled()
  })

  it('adds a bookmark', async () => {
    const { result } = renderHook(() => useChatBookmarks())
    await waitFor(() => expect(result.current.bookmarks).toEqual([]))

    await act(async () => { await result.current.addBookmark({ id: '1', content: 'hello', role: 'user' }) })
    expect(result.current.bookmarks).toHaveLength(1)
    expect(result.current.bookmarks[0].id).toBe('1')
    expect(result.current.bookmarks[0].content).toBe('hello')
    expect(mockChatDB.addBookmark).toHaveBeenCalled()
  })

  it('does not add duplicate', async () => {
    const { result } = renderHook(() => useChatBookmarks())
    await waitFor(() => expect(result.current.bookmarks).toEqual([]))

    await act(async () => { await result.current.addBookmark({ id: '1', content: 'hello', role: 'user' }) })
    await act(async () => { await result.current.addBookmark({ id: '1', content: 'hello', role: 'user' }) })
    expect(result.current.bookmarks).toHaveLength(1)
  })

  it('prepends new bookmarks', async () => {
    const { result } = renderHook(() => useChatBookmarks())
    await waitFor(() => expect(result.current.bookmarks).toEqual([]))

    await act(async () => { await result.current.addBookmark({ id: '1', content: 'first', role: 'user' }) })
    await act(async () => { await result.current.addBookmark({ id: '2', content: 'second', role: 'user' }) })
    expect(result.current.bookmarks[0].id).toBe('2')
  })

  it('checks isBookmarked', async () => {
    const { result } = renderHook(() => useChatBookmarks())
    await waitFor(() => expect(result.current.bookmarks).toEqual([]))

    expect(result.current.isBookmarked('1')).toBe(false)
    await act(async () => { await result.current.addBookmark({ id: '1', content: 'x', role: 'user' }) })
    expect(result.current.isBookmarked('1')).toBe(true)
  })

  it('removes a bookmark', async () => {
    const { result } = renderHook(() => useChatBookmarks())
    await waitFor(() => expect(result.current.bookmarks).toEqual([]))

    await act(async () => { await result.current.addBookmark({ id: '1', content: 'x', role: 'user' }) })
    await act(async () => { await result.current.removeBookmark('1') })
    expect(result.current.bookmarks).toHaveLength(0)
    expect(result.current.isBookmarked('1')).toBe(false)
    expect(mockChatDB.removeBookmark).toHaveBeenCalledWith('1')
  })

  it('clears all bookmarks', async () => {
    const { result } = renderHook(() => useChatBookmarks())
    await waitFor(() => expect(result.current.bookmarks).toEqual([]))

    await act(async () => { await result.current.addBookmark({ id: '1', content: 'a', role: 'user' }) })
    await act(async () => { await result.current.addBookmark({ id: '2', content: 'b', role: 'user' }) })
    await act(async () => { await result.current.clearAll() })
    expect(result.current.bookmarks).toEqual([])
    expect(mockChatDB.clearBookmarks).toHaveBeenCalled()
  })

  it('loads pre-existing bookmarks from chatDB', async () => {
    const existing = { id: '3', content: 'saved', role: 'assistant', timestamp: 100 }
    mockChatDB.getBookmarks.mockResolvedValueOnce([existing])
    const { result } = renderHook(() => useChatBookmarks())
    await waitFor(() => {
      expect(result.current.bookmarks).toHaveLength(1)
    })
    expect(result.current.bookmarks[0].id).toBe('3')
    expect(result.current.isBookmarked('3')).toBe(true)
  })
})
