'use client'

import React, { forwardRef, useState, useEffect } from 'react'
import { MessageBubble } from './MessageBubble'
import { EmptyState } from './EmptyState'
import { SystemBanner } from './SystemBanner'
import { ReasoningPanel } from './ReasoningPanel'
import { ToolCallPanel } from './ToolCallPanel'
import type { ToolCallEvent } from '@/lib/stream-chat-response'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'
import type { ChatMessage } from './types'
import { cn } from '@/lib/cn'

function formatDateLabel(date: Date): string {
  const now = new Date()
  const input = new Date(date)
  const diffDays = Math.floor((now.getTime() - input.getTime()) / 86400000)
  if (diffDays === 0) return 'Today'
  if (diffDays === 1) return 'Yesterday'
  if (diffDays < 7) return input.toLocaleDateString(undefined, { weekday: 'long' })
  return input.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}

function isDifferentDay(a: Date | string | number, b: Date | string | number): boolean {
  const da = new Date(a)
  const db = new Date(b)
  return da.getFullYear() !== db.getFullYear() || da.getMonth() !== db.getMonth() || da.getDate() !== db.getDate()
}

interface ChatScreenProps {
  messages: ChatMessage[]
  loading: boolean
  sessionLoading?: boolean
  health: ApiHealthSnapshot
  suggestions?: { text: string; icon: string }[]
  toolEvents?: ToolCallEvent[]
  onRefreshHealth: () => void
  onCopy: (text: string) => void
  onRegenerate?: () => void
  onThumbsUp?: (messageId: string) => void
  onThumbsDown?: (messageId: string) => void
  onEdit?: (messageId: string, newContent: string) => void
  searchQuery?: string
  onSuggestionClick?: (text: string) => void
  className?: string
  model?: string
  isBookmarked?: (id: string) => boolean
  onBookmark?: (messageId: string) => void
  onDelete?: (messageId: string) => void
  collapsibleLength?: number
}

export const ChatScreen = forwardRef<HTMLDivElement, ChatScreenProps>(
  function ChatScreen({ messages, loading, sessionLoading, health, suggestions, toolEvents, onRefreshHealth, onCopy, onRegenerate, onThumbsUp, onThumbsDown, onEdit, searchQuery, onSuggestionClick, className, model, isBookmarked, onBookmark, onDelete, collapsibleLength }, ref) {
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

        {sessionLoading && (
          <div className="mx-auto w-full max-w-2xl px-3 sm:px-4 py-8 space-y-4" role="status" aria-busy="true" aria-label="Loading messages">
            <span className="sr-only">Loading conversation...</span>
            {[1, 2, 3].map(i => (
              <div key={i} className="animate-pulse space-y-2" aria-hidden="true">
                <div className={cn("h-8 rounded-lg", i % 2 === 0 ? "ml-12 w-3/4" : "mr-12 w-2/3 ml-auto")} style={{ backgroundColor: 'hsl(var(--muted))' }} />
                <div className={cn("h-4 rounded", i % 2 === 0 ? "ml-12 w-1/2" : "mr-12 w-1/3 ml-auto")} style={{ backgroundColor: 'hsl(var(--muted))' }} />
              </div>
            ))}
          </div>
        )}

        {messages.length === 0 && !isOffline && !sessionLoading && (
          <div className={cn("transition-all duration-300", emptyFading && "opacity-0 scale-95")}>
            <EmptyState hasModel={hasModel} suggestions={suggestions} onSuggestionClick={onSuggestionClick} />
          </div>
        )}

        {toolEvents && toolEvents.length > 0 && (
          <div className="mx-auto w-full max-w-2xl px-3 sm:px-4 pb-1">
            <ToolCallPanel events={toolEvents} />
          </div>
        )}

        <div
          id="chat-messages"
          className="mx-auto w-full max-w-2xl space-y-3 sm:space-y-4 px-3 sm:px-4 pb-2"
          role="feed"
          aria-label="Message history"
          aria-busy={loading}
        >
          {messages.map((message, index) => {
            const originalIndex = index
            const isLast = originalIndex === messages.length - 1
            const showRegenerate = isLast && message.role === 'assistant' && onRegenerate
            const isStreaming = loading && isLast && message.role === 'assistant'
            const prevMsg = index > 0 ? messages[index - 1] : null
            const showDateDivider = !prevMsg || isDifferentDay(prevMsg.timestamp, message.timestamp)

            return (
              <React.Fragment key={message.id}>
              {showDateDivider && (
                <div className="relative flex items-center py-2" role="separator" aria-label={formatDateLabel(new Date(message.timestamp))}>
                  <div className="flex-1 border-t border-border/30" />
                  <span className="mx-3 text-[10px] font-medium text-muted-foreground/60 uppercase tracking-wider px-2 py-0.5 rounded-full bg-muted/30">
                    {formatDateLabel(new Date(message.timestamp))}
                  </span>
                  <div className="flex-1 border-t border-border/30" />
                </div>
              )}
              <MessageBubble
                key={message.id}
                messageId={message.id}
                content={message.content}
                role={message.role}
                timestamp={message.timestamp}
                showTimestamp={true}
                model={model}
                images={message.images}
                onCopy={onCopy}
                onThumbsUp={onThumbsUp}
                onThumbsDown={onThumbsDown}
                onEdit={onEdit}
                onRegenerate={showRegenerate ? onRegenerate : undefined}
                onSuggestionClick={onSuggestionClick}
                searchQuery={searchQuery}
                isStreaming={isStreaming}
                isError={message.isError}
                aria-live={isStreaming ? 'polite' : undefined}
                isBookmarked={isBookmarked?.(message.id)}
                onBookmark={onBookmark}
                onDelete={onDelete}
                collapsibleLength={collapsibleLength}
              />
              </React.Fragment>
            )
          })}

          {!loading && messages.length > 0 && messages[messages.length - 1].role === 'assistant' && onSuggestionClick && (() => {
            const last = messages[messages.length - 1].content.toLowerCase()
            const secondToLast = messages.length > 1 ? messages[messages.length - 2].content.toLowerCase() : ''
            const hasCode = last.includes('```')
            const isFileUpload = secondToLast.includes('📎') || secondToLast.includes('uploaded')
            const isSummary = last.length > 200 && !hasCode && !isFileUpload
            let suggestions: string[]
            if (isFileUpload) {
              suggestions = ['Summarize this', 'What are the key points?', 'Explain in simple terms', 'What does this mean for me?']
            } else if (hasCode) {
              suggestions = ['Explain this code', 'Simplify this', 'How do I test this?']
            } else if (isSummary) {
              suggestions = ['Summarize this', 'Explain like I\'m 5', 'Tell me more', 'Give an example']
            } else {
              suggestions = ['Tell me more', 'Give an example', 'Why is that?']
            }
            return (
              <div className="flex flex-wrap gap-1.5 px-3 sm:px-4" role="group" aria-label="Suggested follow-ups">
                {suggestions.map(s => (
                  <button
                    key={s}
                    onClick={() => onSuggestionClick(s)}
                    className="px-2.5 py-1 rounded-full border border-border/50 bg-muted/30 text-[11px] text-muted-foreground hover:bg-muted/50 hover:text-foreground transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )
          })()}

          {loading && messages.length > 0 && messages[messages.length - 1].role !== 'assistant' && (
            <ReasoningPanel isThinking={true} className="py-1" />
          )}

          <div ref={ref} />
        </div>
      </div>
    )
  }
)
