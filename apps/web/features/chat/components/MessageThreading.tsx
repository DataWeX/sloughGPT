'use client'

import { useState, useCallback, useMemo, memo } from 'react'
import { Button, IconX, IconCheck } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import type { ChatMessage } from '@/lib/chat-utils'

interface Thread {
  id: string
  parentMessageId: string
  replies: ChatMessage[]
  createdAt: number
}

interface MessageThreadingProps {
  messages: ChatMessage[]
  onJumpToMessage: (messageId: string) => void
  onReply: (parentMessageId: string, content: string) => void
  className?: string
}

export const MessageThreading = memo(function MessageThreading({
  messages,
  onJumpToMessage,
  onReply,
  className,
}: MessageThreadingProps) {
  const [threads, setThreads] = useState<Thread[]>([])
  const [selectedThread, setSelectedThread] = useState<string | null>(null)
  const [replyDraft, setReplyDraft] = useState('')
  const [replyingTo, setReplyingTo] = useState<string | null>(null)

  const threadsWithMessages = useMemo(() => {
    return threads.map(thread => ({
      ...thread,
      parentMessage: messages.find(m => m.id === thread.parentMessageId),
    }))
  }, [threads, messages])

  const handleStartReply = useCallback((messageId: string) => {
    setReplyingTo(messageId)
    setReplyDraft('')
  }, [])

  const handleCancelReply = useCallback(() => {
    setReplyingTo(null)
    setReplyDraft('')
  }, [])

  const handleSubmitReply = useCallback(() => {
    if (!replyingTo || !replyDraft.trim()) return

    const newReply: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: replyDraft.trim(),
      timestamp: new Date(),
    }

    setThreads(prev => {
      const existing = prev.find(t => t.parentMessageId === replyingTo)
      if (existing) {
        return prev.map(t =>
          t.parentMessageId === replyingTo
            ? { ...t, replies: [...t.replies, newReply] }
            : t
        )
      } else {
        return [...prev, {
          id: crypto.randomUUID(),
          parentMessageId: replyingTo,
          replies: [newReply],
          createdAt: Date.now(),
        }]
      }
    })

    onReply(replyingTo, replyDraft.trim())
    setReplyingTo(null)
    setReplyDraft('')
  }, [replyingTo, replyDraft, onReply])

  const handleDeleteThread = useCallback((threadId: string) => {
    setThreads(prev => prev.filter(t => t.id !== threadId))
    if (selectedThread === threadId) {
      setSelectedThread(null)
    }
  }, [selectedThread])

  const getThreadForMessage = useCallback((messageId: string) => {
    return threads.find(t => t.parentMessageId === messageId)
  }, [threads])

  const messagesWithThreads = useMemo(() => {
    return messages.filter(m => getThreadForMessage(m.id))
  }, [messages, getThreadForMessage])

  return (
    <div className={cn('border rounded-lg bg-card overflow-hidden', className)}>
      <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/30">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium">Threads</span>
          <span className="text-[10px] text-muted-foreground">({threads.length})</span>
        </div>
      </div>

      <div className="max-h-[400px] overflow-y-auto">
        {threads.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">
            No threads yet. Click the reply icon on any message to start a thread.
          </p>
        ) : (
          <div className="divide-y">
            {threadsWithMessages.map(thread => (
              <div key={thread.id} className="px-3 py-2 hover:bg-muted/30">
                <div className="flex items-start justify-between gap-2">
                  <button
                    type="button"
                    className="flex-1 text-left min-w-0"
                    onClick={() => setSelectedThread(
                      selectedThread === thread.id ? null : thread.id
                    )}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary">
                        Thread
                      </span>
                      <span className="text-[10px] text-muted-foreground">
                        {thread.replies.length} {thread.replies.length === 1 ? 'reply' : 'replies'}
                      </span>
                    </div>
                    {thread.parentMessage && (
                      <p className="text-xs truncate text-muted-foreground">
                        {thread.parentMessage.content.slice(0, 60)}
                      </p>
                    )}
                  </button>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    className="h-5 w-5 opacity-0 group-hover:opacity-100 shrink-0"
                    onClick={() => handleDeleteThread(thread.id)}
                    title="Delete thread"
                  >
                    <IconX className="h-3 w-3" />
                  </Button>
                </div>

                {selectedThread === thread.id && (
                  <div className="mt-2 space-y-2 pl-4 border-l-2 border-primary/20">
                    {thread.replies.map(reply => (
                      <div key={reply.id} className="text-xs">
                        <span className="text-[10px] text-muted-foreground mr-1">
                          {reply.role === 'user' ? 'You' : 'AI'}:
                        </span>
                        {reply.content.slice(0, 80)}
                      </div>
                    ))}

                    {replyingTo === thread.parentMessageId ? (
                      <div className="space-y-1">
                        <input
                          type="text"
                          value={replyDraft}
                          onChange={(e) => setReplyDraft(e.target.value)}
                          placeholder="Reply to thread..."
                          className="w-full text-xs bg-transparent border rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-primary/50"
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleSubmitReply()
                            if (e.key === 'Escape') handleCancelReply()
                          }}
                          autoFocus
                        />
                        <div className="flex gap-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="text-[10px] h-5"
                            onClick={handleSubmitReply}
                            disabled={!replyDraft.trim()}
                          >
                            <IconCheck className="h-3 w-3 mr-1" />
                            Reply
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="text-[10px] h-5"
                            onClick={handleCancelReply}
                          >
                            Cancel
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-[10px] h-5"
                        onClick={() => handleStartReply(thread.parentMessageId)}
                      >
                        Reply to thread
                      </Button>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {messagesWithThreads.length > 0 && (
        <div className="px-3 py-2 border-t text-[10px] text-muted-foreground">
          {messagesWithThreads.length} messages with threads
        </div>
      )}
    </div>
  )
})