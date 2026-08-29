'use client'

import { useState, useCallback, useMemo } from 'react'
import type { ChatMessage } from '@/lib/chat-utils'

interface Thread {
  id: string
  parentMessageId: string
  messages: ChatMessage[]
  createdAt: number
}

interface UseMessageThreadsOptions {
  messages: ChatMessage[]
}

interface UseMessageThreadsReturn {
  threads: Record<string, Thread>
  getThread: (messageId: string) => Thread | undefined
  createThread: (parentMessageId: string) => string
  addToThread: (threadId: string, message: ChatMessage) => void
  getThreadMessages: (threadId: string) => ChatMessage[]
  hasThread: (messageId: string) => boolean
  threadCount: number
}

export function useMessageThreads({ messages }: UseMessageThreadsOptions): UseMessageThreadsReturn {
  const [threads, setThreads] = useState<Record<string, Thread>>({})

  const getThread = useCallback((messageId: string) => {
    return threads[messageId]
  }, [threads])

  const createThread = useCallback((parentMessageId: string) => {
    const threadId = `thread-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
    const thread: Thread = {
      id: threadId,
      parentMessageId,
      messages: [],
      createdAt: Date.now(),
    }
    setThreads(prev => ({ ...prev, [parentMessageId]: thread }))
    return threadId
  }, [])

  const addToThread = useCallback((threadId: string, message: ChatMessage) => {
    setThreads(prev => {
      const entries = Object.entries(prev)
      for (const [key, thread] of entries) {
        if (thread.id === threadId) {
          return {
            ...prev,
            [key]: {
              ...thread,
              messages: [...thread.messages, message],
            },
          }
        }
      }
      return prev
    })
  }, [])

  const getThreadMessages = useCallback((threadId: string) => {
    for (const thread of Object.values(threads)) {
      if (thread.id === threadId) {
        return thread.messages
      }
    }
    return []
  }, [threads])

  const hasThread = useCallback((messageId: string) => {
    return messageId in threads
  }, [threads])

  const threadCount = useMemo(() => Object.keys(threads).length, [threads])

  return {
    threads,
    getThread,
    createThread,
    addToThread,
    getThreadMessages,
    hasThread,
    threadCount,
  }
}