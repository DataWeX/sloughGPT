'use client'

import { useEffect, useState } from 'react'
import { IconX, IconTrash, IconThumbUp, IconThumbDown, IconChat } from '@/components/ui'
import { cn } from '@/lib/cn'
import { Button } from '@/components/ui/button'

interface ViewerMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  feedback?: 'positive' | 'negative'
}

interface ConversationViewerProps {
  isOpen: boolean
  onClose: () => void
  messages?: ViewerMessage[]
  title?: string
  onExport?: (format: 'md' | 'json') => void
  onDelete?: () => void
}

function formatTimestamp(timestamp: number): string {
  const date = new Date(timestamp)
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

export function ConversationViewer({
  isOpen,
  onClose,
  messages: initialMessages,
  title = 'Conversation',
  onExport,
  onDelete,
}: ConversationViewerProps) {
  const [messages, setMessages] = useState<ViewerMessage[]>(initialMessages || [])

  useEffect(() => {
    if (isOpen && initialMessages) {
      setMessages(initialMessages)
    }
  }, [isOpen, initialMessages])

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    if (isOpen) {
      document.addEventListener('keydown', handleEscape)
      return () => document.removeEventListener('keydown', handleEscape)
    }
  }, [isOpen, onClose])

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden'
      return () => { document.body.style.overflow = '' }
    }
  }, [isOpen])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center" role="dialog" aria-modal="true" aria-labelledby="viewer-title">
      <div
        className="absolute inset-0 bg-foreground/20 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden="true"
      />
      <div className="relative z-10 flex h-[90dvh] w-[90vw] max-w-3xl flex-col rounded-lg border border-border bg-background shadow-2xl">
        <div className="flex shrink-0 items-center justify-between border-b border-border px-4 py-3">
          <div className="min-w-0 flex-1">
            <h2 id="viewer-title" className="truncate text-base font-semibold text-foreground">{title}</h2>
            <p className="text-xs text-muted-foreground">
              {messages.length} message{messages.length !== 1 ? 's' : ''}
            </p>
          </div>
          <div className="flex items-center gap-1">
{onDelete && (
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={onDelete}
                className="text-destructive hover:text-destructive"
                aria-label="Delete conversation"
              >
                <IconTrash className="h-4 w-4" />
              </Button>
            )}
            <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close dialog">
              <IconX className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <div className="mb-3 rounded-full bg-muted/50 p-4">
                <IconChat className="h-8 w-8 text-muted-foreground/50" />
              </div>
              <p className="text-sm text-muted-foreground">No messages in this conversation</p>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((message, index) => (
                <div
                  key={message.id || index}
                  className={cn(
                    'group rounded-lg p-3',
                    message.role === 'user'
                      ? 'bg-primary/5 border border-primary/10'
                      : 'bg-muted/30 border border-border/50'
                  )}
                >
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <span className={cn(
                      'text-xs font-medium',
                      message.role === 'user' ? 'text-primary' : 'text-muted-foreground'
                    )}>
                      {message.role === 'user' ? 'You' : message.role === 'assistant' ? 'Assistant' : 'System'}
                    </span>
                    <div className="flex items-center gap-2">
                      {message.feedback && (
                        <span className={cn(
                          'flex items-center gap-0.5 text-xs',
                          message.feedback === 'positive' ? 'text-success' : 'text-destructive'
                        )}>
                          {message.feedback === 'positive' ? (
                            <IconThumbUp className="h-3 w-3" />
                          ) : (
                            <IconThumbDown className="h-3 w-3" />
                          )}
                        </span>
                      )}
                      <span className="text-xs text-muted-foreground/60">
                        {formatTimestamp(message.timestamp)}
                      </span>
                    </div>
                  </div>
                  <div className="text-sm text-foreground whitespace-pre-wrap">
                    {message.content}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="shrink-0 border-t border-border px-4 py-3">
          <p className="text-xs text-muted-foreground text-center">
            Press <kbd className="mx-1 rounded bg-muted px-1.5 py-0.5 font-mono text-xs">Esc</kbd> to close
          </p>
        </div>
      </div>
    </div>
  )
}

export type { ViewerMessage, ConversationViewerProps }