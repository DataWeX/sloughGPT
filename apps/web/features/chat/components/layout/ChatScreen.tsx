'use client'

import React, { forwardRef, memo, useState, useEffect, useCallback, useMemo } from 'react'
import { MessageBubble } from './../messages/MessageBubble'
import { EmptyState } from './../messages/EmptyState'
import { SystemBanner } from './../messages/SystemBanner'
import { ReasoningPanel } from './../messages/ReasoningPanel'
import { ToolCallPanel } from './../messages/ToolCallPanel'
import type { ToolCallEvent } from '@/lib/stream-chat-response'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'
import type { ChatMessage } from './../types'
import { cn } from '@sloughgpt/strui'
import { IconPin, IconX } from '@sloughgpt/strui'
import { MS_PER_DAY } from '@/lib/format-bytes'

function formatDateLabel(date: Date): string {
  const now = new Date()
  const input = new Date(date)
  const diffDays = Math.floor((now.getTime() - input.getTime()) / MS_PER_DAY)
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
  ragVerification?: {
    confidence: number
    is_verified: boolean
    hallucination_rate: number
    citations: string
    grounded_claims: number
    hallucinated_claims: number
  } | null
  onRefreshHealth: () => void
  onCopy: (text: string) => void
  onRegenerate?: (messageId: string) => void
  onRegenerateWithOptions?: (messageId: string, options: { temperature?: number; maxTokens?: number }) => void
  onThumbsUp?: (messageId: string) => void
  onThumbsDown?: (messageId: string) => void
  onEdit?: (messageId: string, newContent: string) => void
  onReact?: (messageId: string, emoji: string) => void
  searchQuery?: string
  onSuggestionClick?: (text: string) => void
  className?: string
  model?: string
  isBookmarked?: (id: string) => boolean
  onBookmark?: (messageId: string) => void
  onDelete?: (messageId: string) => void
  onSaveToKnowledge?: (messageId: string, content: string) => void
  collapsibleLength?: number
  temperature?: number
  contextLayers?: Array<{ type: 'knowledge' | 'memory' | 'rag' | 'tool' | 'soul' | 'system'; label: string; detail?: string }>
  noteMap?: Record<string, string>
  onAddNote?: (messageId: string) => void
  onPin?: (messageId: string) => void
  selectionMode?: boolean
  selectedMessageIds?: Set<string>
  onToggleSelection?: (messageId: string) => void
  hasThread?: (id: string) => boolean
  onThread?: (messageId: string) => void
  onForward?: (content: string) => void
  onExportMessageAsMarkdown?: (messageId: string, content: string, role: string, timestamp: string | number) => void
  conversationSearchQuery?: string
  setConversationSearchQuery?: (query: string) => void
  conversationSearchOpen?: boolean
  setConversationSearchOpen?: (open: boolean) => void
  onQuickReply?: (messageId: string) => void
}

export const ChatScreen = memo(forwardRef<HTMLDivElement, ChatScreenProps>(
  function ChatScreen({ messages, loading, sessionLoading, health, suggestions, toolEvents, ragVerification, onRefreshHealth, onCopy, onRegenerate, onRegenerateWithOptions, onThumbsUp, onThumbsDown, onEdit, onReact, onPin, searchQuery, onSuggestionClick, className, model, isBookmarked, onBookmark, onDelete, onSaveToKnowledge, collapsibleLength, temperature, contextLayers, noteMap, onAddNote, selectionMode, selectedMessageIds, onToggleSelection, hasThread, onThread, onForward, onExportMessageAsMarkdown, conversationSearchQuery, setConversationSearchQuery, conversationSearchOpen, setConversationSearchOpen, onQuickReply }, ref) {
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

    // Memoize stable callback references for MessageBubble
    const stableOnCopy = useCallback((text: string) => onCopy(text), [onCopy])
    const stableOnThumbsUp = useCallback((id: string) => onThumbsUp?.(id), [onThumbsUp])
    const stableOnThumbsDown = useCallback((id: string) => onThumbsDown?.(id), [onThumbsDown])
    const stableOnEdit = useCallback((id: string, content: string) => onEdit?.(id, content), [onEdit])
    const stableOnReact = useCallback((id: string, emoji: string) => onReact?.(id, emoji), [onReact])
    const stableOnPin = useCallback((id: string) => onPin?.(id), [onPin])
    const stableOnBookmark = useCallback((id: string) => onBookmark?.(id), [onBookmark])
    const stableOnDelete = useCallback((id: string) => onDelete?.(id), [onDelete])
    const stableOnSaveToKnowledge = useCallback((id: string, content: string) => onSaveToKnowledge?.(id, content), [onSaveToKnowledge])
    const stableOnThread = useCallback((id: string) => onThread?.(id), [onThread])
    const stableOnToggleSelection = useCallback((id: string) => onToggleSelection?.(id), [onToggleSelection])
    const stableOnSuggestionClick = useCallback((text: string) => onSuggestionClick?.(text), [onSuggestionClick])

    return (
      <div className={cn("flex flex-col", className)}>
        {isOffline && (
          <SystemBanner
            type="offline"
            title="Service Unavailable"
            message="The service is not responding. Please try again."
            actionLabel="Check Again"
            onAction={onRefreshHealth}
          />
        )}

        {sessionLoading && (
          <div className="mx-auto w-full max-w-3xl px-4 sm:px-6 py-4 space-y-2" role="status" aria-busy="true" aria-label="Loading messages">
            <span className="sr-only">Loading conversation...</span>
            {[1, 2, 3].map(i => (
                <div key={i} className="animate-pulse space-y-1" aria-hidden="true">
                <div className={cn("h-5 rounded-lg bg-muted/40", i % 2 === 0 ? "ml-8 w-2/3" : "mr-8 w-1/2 ml-auto")} />
                <div className={cn("h-2.5 rounded bg-muted/30", i % 2 === 0 ? "ml-8 w-1/3" : "mr-8 w-1/4 ml-auto")} />
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
          <div className="mx-auto w-full max-w-3xl px-4 sm:px-6 pb-1">
            <ToolCallPanel events={toolEvents} />
          </div>
        )}

        {ragVerification && (
          <div className="mx-auto w-full max-w-3xl px-4 sm:px-6 pb-1">
            <div className="flex items-center gap-2 text-[10px] text-muted-foreground rounded-md border border-border/40 bg-muted/30 px-3 py-1.5" role="status" aria-live="polite">
              <span className={cn(
                "inline-block h-1.5 w-1.5 rounded-full",
                ragVerification.is_verified ? "bg-success" : ragVerification.confidence > 0.5 ? "bg-warning" : "bg-destructive"
              )} />
              <span>
                RAG: {ragVerification.is_verified ? 'Verified' : 'Unverified'} ({(ragVerification.confidence * 100).toFixed(0)}% confidence)
              </span>
              {ragVerification.grounded_claims > 0 && (
                <span className="text-success">{ragVerification.grounded_claims} grounded</span>
              )}
              {ragVerification.hallucinated_claims > 0 && (
                <span className="text-destructive">{ragVerification.hallucinated_claims} hallucinated</span>
              )}
              {ragVerification.citations && (
                <span className="truncate max-w-[200px] opacity-60" title={ragVerification.citations}>
                  {ragVerification.citations.slice(0, 60)}{ragVerification.citations.length > 60 ? '...' : ''}
                </span>
              )}
            </div>
          </div>
        )}

        <div
          id="chat-messages"
          className="mx-auto w-full max-w-3xl space-y-1.5 sm:space-y-2 px-4 sm:px-6 pb-4"
          role="feed"
          aria-label="Message history"
          aria-busy={loading}
        >
          {messages.map((message, index) => {
            const isLast = index === messages.length - 1
            const showRegenerate = message.role === 'assistant' && onRegenerate && !loading
            const isStreaming = loading && isLast && message.role === 'assistant'
            const prevMsg = index > 0 ? messages[index - 1] : null
            const showDateDivider = !prevMsg || isDifferentDay(prevMsg.timestamp, message.timestamp)

            return (
              <React.Fragment key={message.id}>
              {showDateDivider && (
                <div className="relative flex items-center py-2" role="separator" aria-label={formatDateLabel(new Date(message.timestamp))}>
                  <div className="flex-1 border-t border-border/40" />
                  <span className="mx-3 text-[10px] font-medium text-muted-foreground/60 uppercase tracking-widest">
                    {formatDateLabel(new Date(message.timestamp))}
                  </span>
                  <div className="flex-1 border-t border-border/40" />
                </div>
              )}
              <MessageBubble
                messageId={message.id}
                content={message.content}
                role={message.role}
                timestamp={message.timestamp}
                showTimestamp={true}
                model={model}
                images={message.images}
                audio={message.audio}
                reactions={message.reactions}
                onCopy={stableOnCopy}
                onThumbsUp={stableOnThumbsUp}
                onThumbsDown={stableOnThumbsDown}
                onEdit={stableOnEdit}
                onReact={stableOnReact}
                onPin={stableOnPin}
                onRegenerate={showRegenerate ? onRegenerate : undefined}
                onRegenerateWithOptions={showRegenerate ? onRegenerateWithOptions : undefined}
                onSuggestionClick={stableOnSuggestionClick}
                searchQuery={searchQuery}
                isStreaming={isStreaming}
                isError={message.isError}
                isPinned={message.pinned}
                aria-live={isStreaming ? 'polite' : undefined}
                isBookmarked={isBookmarked?.(message.id)}
                onBookmark={stableOnBookmark}
                onDelete={stableOnDelete}
                onSaveToKnowledge={stableOnSaveToKnowledge}
                collapsibleLength={collapsibleLength}
                temperature={temperature}
                hasNote={!!noteMap?.[message.id]}
                note={noteMap?.[message.id]}
                onAddNote={onAddNote}
                hasThread={hasThread?.(message.id)}
                onThread={stableOnThread}
                onForward={onForward}
                onExportMessageAsMarkdown={onExportMessageAsMarkdown}
                onQuickReply={onQuickReply}
                selectionMode={selectionMode}
                isSelected={selectedMessageIds?.has(message.id)}
                onToggleSelection={stableOnToggleSelection}
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
              <div className="flex flex-wrap gap-1 px-4 sm:px-6" role="group" aria-label="Suggested follow-ups">
                {suggestions.map(s => (
                  <button
                    key={s}
                    type="button"
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
            <ReasoningPanel isThinking={true} contextLayers={contextLayers} className="py-1" />
          )}

          <div ref={ref} />
        </div>
      </div>
    )
  }
))
