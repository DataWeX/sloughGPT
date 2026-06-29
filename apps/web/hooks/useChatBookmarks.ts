'use client'

import { useState, useCallback, useEffect } from 'react'

export interface BookmarkedMessage {
  id: string
  content: string
  role: 'user' | 'assistant'
  timestamp: number
  sessionTitle?: string
}

const STORAGE_KEY = 'chat:bookmarks'

function loadBookmarks(): BookmarkedMessage[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveBookmarks(bookmarks: BookmarkedMessage[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(bookmarks))
  } catch {}
}

export function useChatBookmarks() {
  const [bookmarks, setBookmarks] = useState<BookmarkedMessage[]>(loadBookmarks)

  useEffect(() => {
    saveBookmarks(bookmarks)
  }, [bookmarks])

  const addBookmark = useCallback((msg: { id: string; content: string; role: 'user' | 'assistant'; timestamp?: number }, sessionTitle?: string) => {
    setBookmarks(prev => {
      if (prev.some(b => b.id === msg.id)) return prev
      return [{
        id: msg.id,
        content: msg.content,
        role: msg.role,
        timestamp: msg.timestamp ?? Date.now(),
        sessionTitle,
      }, ...prev]
    })
  }, [])

  const removeBookmark = useCallback((id: string) => {
    setBookmarks(prev => prev.filter(b => b.id !== id))
  }, [])

  const isBookmarked = useCallback((id: string) => {
    return bookmarks.some(b => b.id === id)
  }, [bookmarks])

  const clearAll = useCallback(() => {
    setBookmarks([])
  }, [])

  return { bookmarks, addBookmark, removeBookmark, isBookmarked, clearAll }
}
