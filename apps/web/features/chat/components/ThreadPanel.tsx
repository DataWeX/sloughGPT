'use client'

import { useState, memo, useRef, useEffect } from 'react'
import { Button, IconX, IconSend } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import type { ChatMessage } from '@/lib/chat-utils'
import { MessageBubble } from './messages/MessageBubble'

interface ThreadPanelProps {
  parentMessage: ChatMessage
  threadMessages: ChatMessage[]
  onSend: (content: string) => void
  onClose: () => void
  className?: string
}

export const ThreadPanel = memo(function ThreadPanel({
  parentMessage,
  threadMessages,
  onSend,
  onClose,
  className,
}: ThreadPanelProps) {
  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView?.({ behavior: 'smooth' })
  }, [threadMessages])

  const handleSend = () => {
    if (input.trim()) {
      onSend(input.trim())
      setInput('')
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className={cn('flex flex-col h-full border-l border-border/50 bg-background', className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-border/50 shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Thread</span>
          <span className="text-[10px] text-muted-foreground">
            {threadMessages.length} {threadMessages.length === 1 ? 'reply' : 'replies'}
          </span>
        </div>
        <Button variant="ghost" size="icon" className="h-6 w-6" onClick={onClose} aria-label="Close thread">
          <IconX className="h-3.5 w-3.5" />
        </Button>
      </div>

      {/* Parent message */}
      <div className="px-3 py-2 border-b border-border/30 bg-muted/20">
        <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Original</div>
        <div className="text-xs text-foreground/80 line-clamp-2">{parentMessage.content}</div>
      </div>

      {/* Thread messages */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-3">
        {threadMessages.length === 0 ? (
          <div className="text-xs text-muted-foreground text-center py-4">
            No replies yet. Start the thread below.
          </div>
        ) : (
          threadMessages.map(msg => (
            <MessageBubble
              key={msg.id}
              content={msg.content}
              role={msg.role}
              timestamp={new Date(msg.timestamp)}
              showTimestamp
            />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="px-3 py-2 border-t border-border/50 shrink-0">
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Reply in thread..."
            className="flex-1 text-xs bg-transparent border border-border/30 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-primary/50"
          />
          <Button
            variant="ghost"
            size="icon-sm"
            className="h-7 w-7"
            onClick={handleSend}
            disabled={!input.trim()}
            aria-label="Send reply"
          >
            <IconSend className="h-3 w-3" />
          </Button>
        </div>
      </div>
    </div>
  )
})