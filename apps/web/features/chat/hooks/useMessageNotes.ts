'use client'

import { useState, useCallback, useEffect } from 'react'
import { chatDB, type MessageNote } from '@/lib/db'

export type { MessageNote }

interface UseMessageNotesOptions {
  sessionId: string | null
}

interface UseMessageNotesReturn {
  notes: Record<string, string>
  getNote: (messageId: string) => string | undefined
  setNote: (messageId: string, content: string) => Promise<void>
  removeNote: (messageId: string) => Promise<void>
  hasNote: (messageId: string) => boolean
  loading: boolean
}

export function useMessageNotes({ sessionId }: UseMessageNotesOptions): UseMessageNotesReturn {
  const [notes, setNotes] = useState<Record<string, string>>({})
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!sessionId) {
      setNotes({})
      return
    }

    let ignore = false
    setLoading(true)

    chatDB.getMessageNotes(sessionId)
      .then(items => {
        if (ignore) return
        const notesMap: Record<string, string> = {}
        for (const item of items) {
          notesMap[item.messageId] = item.content
        }
        setNotes(notesMap)
      })
      .catch(() => {
        if (!ignore) setNotes({})
      })
      .finally(() => {
        if (!ignore) setLoading(false)
      })

    return () => { ignore = true }
  }, [sessionId])

  const setNote = useCallback(async (messageId: string, content: string) => {
    if (!sessionId) return

    const note: MessageNote = {
      id: `${sessionId}:${messageId}`,
      sessionId,
      messageId,
      content,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    }

    await chatDB.saveMessageNote(note)
    setNotes(prev => ({ ...prev, [messageId]: content }))
  }, [sessionId])

  const removeNote = useCallback(async (messageId: string) => {
    if (!sessionId) return

    await chatDB.removeMessageNote(sessionId, messageId)
    setNotes(prev => {
      const next = { ...prev }
      delete next[messageId]
      return next
    })
  }, [sessionId])

  const getNote = useCallback((messageId: string) => {
    return notes[messageId]
  }, [notes])

  const hasNote = useCallback((messageId: string) => {
    return messageId in notes
  }, [notes])

  return {
    notes,
    getNote,
    setNote,
    removeNote,
    hasNote,
    loading,
  }
}