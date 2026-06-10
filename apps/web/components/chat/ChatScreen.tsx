'use client'

import { forwardRef, useState, useEffect } from 'react'
import { MessageBubble } from './MessageBubble'
import { EmptyState } from './EmptyState'
import { SystemBanner } from './SystemBanner'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'
import type { ChatMessage } from './ChatMessages'
import { cn } from '@/lib/cn'

interface ChatScreenProps {
  messages: ChatMessage[]
  loading: boolean
  health: ApiHealthSnapshot
  onRefreshHealth: () => void
  onCopy: (text: string) => void
  onRegenerate?: () => void
  onThumbsUp?: (messageId: string) => void
  onThumbsDown?: (messageId: string) => void
  onEdit?: (messageId: string, newContent: string) => void
  searchQuery?: string
  onSuggestionClick?: (text: string) => void
  className?: string
}

export const ChatScreen = forwardRef<HTMLDivElement, ChatScreenProps>(
  function ChatScreen({ messages, loading, health, onRefreshHealth, onCopy, onRegenerate, onThumbsUp, onThumbsDown, onEdit, searchQuery, onSuggestionClick, className }, ref) {
    const isOffline = health === 'offline'
    const hasModel = health !== null && health !== 'offline' && health.model_loaded
    const [emptyFading, setEmptyFading] = useState(false)

    useEffect(() => {
      if (messages.length > 0) {
        setEmptyFading(true)
        const timer = setTimeout(() => setEmptyFading(false), 300)
        return () => clearTimeout(timer)
      } else {
        setEmptyFading(false)
      }
    }, [messages.length])

    const filteredMessages = searchQuery
      ? messages.filter(m => m.content.toLowerCase().includes(searchQuery.toLowerCase()))
      : messages

    return (
      <div className={cn("flex flex-col", className)}>
        {isOffline && (
          <SystemBanner
            type="offline"
            title="API Server Offline"
            message="The API server is not responding. Make sure it is running."
            actionLabel="Check Again"
            onAction={onRefreshHealth}
          />
        )}
        
        {messages.length === 0 && !isOffline && (
          <div className={cn("transition-all duration-300", emptyFading && "opacity-0 scale-95")}>
            <EmptyState hasModel={hasModel} onSuggestionClick={onSuggestionClick} />
          </div>
        )}

        {searchQuery && filteredMessages.length === 0 && messages.length > 0 && (
          <p className="text-center text-sm text-muted-foreground py-4">
            No messages match &quot;{searchQuery}&quot;
          </p>
        )}
        
        <div 
          className="mx-auto w-full max-w-2xl space-y-3 sm:space-y-4 px-3 sm:px-4 pb-2"
          role="feed"
          aria-label="Message history"
          aria-busy={loading}
        >
          {filteredMessages.map((message, index) => {
            const originalIndex = messages.findIndex(m => m.id === message.id)
            const isLast = originalIndex === messages.length - 1
            const showRegenerate = isLast && message.role === 'assistant' && onRegenerate
            const isStreaming = loading && isLast && message.role === 'assistant'
            
            return (
              <MessageBubble
                key={message.id}
                messageId={message.id}
                content={message.content}
                role={message.role}
                timestamp={message.timestamp}
                showTimestamp={true}
                images={message.images}
                onCopy={onCopy}
                onThumbsUp={onThumbsUp}
                onThumbsDown={onThumbsDown}
                onEdit={onEdit}
                onRegenerate={showRegenerate ? onRegenerate : undefined}
                searchQuery={searchQuery}
                isStreaming={isStreaming}
              />
            )
          })}
          
          <div ref={ref} />
        </div>
      </div>
    )
  }
)
