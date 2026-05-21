'use client'

import { forwardRef, useEffect } from 'react'
import { MessageBubble } from './MessageBubble'
import { EmptyState } from './EmptyState'
import { LoadingIndicator } from './LoadingIndicator'
import { SystemBanner } from './SystemBanner'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'
import type { ImageAttachment } from './ImageUpload'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  images?: ImageAttachment[]
}

interface ChatMessagesProps {
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
}

export const ChatMessages = forwardRef<HTMLDivElement, ChatMessagesProps>(
  function ChatMessages({ messages, loading, health, onRefreshHealth, onCopy, onRegenerate, onThumbsUp, onThumbsDown, onEdit }, ref) {
    const isOffline = health === 'offline'
    const hasModel = health !== null && health !== 'offline' && health.model_loaded

    return (
      <section 
        className="flex-1 min-h-0 overflow-y-auto"
        aria-label="Chat messages"
        aria-roledescription="chat thread"
      >
        <div className="mx-auto max-w-2xl px-3 py-4 sm:px-4 sm:py-6">
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
            <EmptyState hasModel={hasModel} />
          )}
          
          <div 
            className="space-y-3 sm:space-y-4"
            role="feed"
            aria-label="Message history"
            aria-busy={loading}
          >
            {messages.map((msg, idx) => {
              const isLast = idx === messages.length - 1
              const isGeneratingThis = isLast && loading
              const hasContent = msg.content.length > 0
              const isAssistantWithContent = msg.role === 'assistant' && hasContent && !isGeneratingThis
              return (
                <MessageBubble
                  key={msg.id}
                  content={msg.content}
                  role={msg.role}
                  timestamp={msg.timestamp}
                  showTimestamp={isAssistantWithContent}
                  isStreaming={isGeneratingThis}
                  messageId={msg.id}
                  images={msg.images}
                  onCopy={msg.role === 'assistant' ? onCopy : undefined}
                  onRegenerate={msg.role === 'assistant' && isLast ? onRegenerate : undefined}
                  onThumbsUp={msg.role === 'assistant' ? onThumbsUp : undefined}
                  onThumbsDown={msg.role === 'assistant' ? onThumbsDown : undefined}
                  onEdit={msg.role === 'user' ? onEdit : undefined}
                />
              )
            })}
            
            {loading && (
              <LoadingIndicator />
            )}
          </div>
          
          <div ref={ref} className="h-4" />
        </div>
      </section>
    )
  }
)
