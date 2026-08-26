'use client'

import { useState, useCallback, useEffect } from 'react'

interface ConversationTags {
  [sessionId: string]: string[]
}

interface UseConversationTagsReturn {
  tags: ConversationTags
  getTags: (sessionId: string) => string[]
  addTag: (sessionId: string, tag: string) => void
  removeTag: (sessionId: string, tag: string) => void
  hasTag: (sessionId: string, tag: string) => boolean
  getAllTags: () => string[]
  loading: boolean
}

const STORAGE_KEY = 'chat-conversation-tags'

export function useConversationTags(): UseConversationTagsReturn {
  const [tags, setTags] = useState<ConversationTags>({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) {
        setTags(JSON.parse(stored))
      }
    } catch {
      setTags({})
    } finally {
      setLoading(false)
    }
  }, [])

  const saveTags = useCallback((newTags: ConversationTags) => {
    setTags(newTags)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(newTags))
  }, [])

  const getTags = useCallback((sessionId: string) => {
    return tags[sessionId] || []
  }, [tags])

  const addTag = useCallback((sessionId: string, tag: string) => {
    const trimmed = tag.trim().toLowerCase()
    if (!trimmed) return

    setTags(prev => {
      const current = prev[sessionId] || []
      if (current.includes(trimmed)) return prev
      const updated = { ...prev, [sessionId]: [...current, trimmed] }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated))
      return updated
    })
  }, [])

  const removeTag = useCallback((sessionId: string, tag: string) => {
    setTags(prev => {
      const current = prev[sessionId] || []
      const updated = { ...prev, [sessionId]: current.filter(t => t !== tag) }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated))
      return updated
    })
  }, [])

  const hasTag = useCallback((sessionId: string, tag: string) => {
    return (tags[sessionId] || []).includes(tag)
  }, [tags])

  const getAllTags = useCallback(() => {
    const allTags = new Set<string>()
    for (const sessionTags of Object.values(tags)) {
      for (const tag of sessionTags) {
        allTags.add(tag)
      }
    }
    return Array.from(allTags).sort()
  }, [tags])

  return {
    tags,
    getTags,
    addTag,
    removeTag,
    hasTag,
    getAllTags,
    loading,
  }
}