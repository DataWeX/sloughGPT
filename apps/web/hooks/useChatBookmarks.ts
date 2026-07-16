'use client'

import { useState, useCallback, useEffect } from 'react'
import { chatDB, type BookmarkedMessage } from '@/lib/db'

export type { BookmarkedMessage }

export function useChatBookmarks() {
  const [bookmarks, setBookmarks] = useState<BookmarkedMessage[]>([])

  useEffect(() => {
    chatDB.getBookmarks().then(items => setBookmarks(items)).catch(() => {})
  }, [])

  const addBookmark = useCallback(async (msg: { id: string; content: string; role: 'user' | 'assistant'; timestamp?: number }, sessionTitle?: string) => {
    const bm: BookmarkedMessage = {
      id: msg.id,
      content: msg.content,
      role: msg.role,
      timestamp: msg.timestamp ?? Date.now(),
      sessionTitle,
    }
    await chatDB.addBookmark(bm)
    setBookmarks(prev => {
      if (prev.some(b => b.id === msg.id)) return prev
      return [bm, ...prev]
    })
  }, [])

  const removeBookmark = useCallback(async (id: string) => {
    await chatDB.removeBookmark(id)
    setBookmarks(prev => prev.filter(b => b.id !== id))
  }, [])

  const isBookmarked = useCallback((id: string) => {
    return bookmarks.some(b => b.id === id)
  }, [bookmarks])

  const clearAll = useCallback(async () => {
    await chatDB.clearBookmarks()
    setBookmarks([])
  }, [])

  return { bookmarks, addBookmark, removeBookmark, isBookmarked, clearAll }
}
