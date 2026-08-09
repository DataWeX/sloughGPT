'use client'

import { memo, useEffect, useRef, useState } from 'react'
import { cn } from '@sloughgpt/strui'
import { MS_PER_MINUTE } from '@/lib/format-bytes'
import { MessageActions } from './MessageActions'
import { MessageContextMenu } from './MessageContextMenu'
import { MessageImages } from './MessageImages'
import { MessageContent } from './MessageContent'
import type { ImageAttachment } from './../input/ImageUpload'

export interface MessageBubbleProps {
  content: string
  role: 'user' | 'assistant'
  timestamp: Date | string
  showTimestamp: boolean
  images?: ImageAttachment[]
  onCopy?: (text: string) => void
  onRegenerate?: () => void
  onThumbsUp?: (messageId: string) => void
  onThumbsDown?: (messageId: string) => void
  onEdit?: (messageId: string, newContent: string) => void
  onSuggestionClick?: (text: string) => void
  messageId?: string
  isStreaming?: boolean
  isError?: boolean
  searchQuery?: string
  model?: string
  isBookmarked?: boolean
  onBookmark?: (messageId: string) => void
  onDelete?: (messageId: string) => void
  onSaveToKnowledge?: (messageId: string, content: string) => void
  collapsibleLength?: number
  'aria-live'?: 'polite' | 'assertive' | 'off'
}

function formatTime(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date
  const now = new Date()
  const diffMs = now.getTime() - d.getTime()
  const diffMins = Math.floor(diffMs / MS_PER_MINUTE)
  if (diffMins < 1) return 'just now'
  if (diffMins < 60) return `${diffMins}m ago`
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export const MessageBubble = memo(function MessageBubble({
  content,
  role,
  timestamp,
  showTimestamp,
  images,
  onCopy,
  onRegenerate,
  onThumbsUp,
  onThumbsDown,
  onEdit,
  onSuggestionClick,
  messageId,
  isStreaming = false,
  isError = false,
  searchQuery,
  model,
  isBookmarked = false,
  onBookmark,
  onDelete,
  onSaveToKnowledge,
  collapsibleLength = 0,
  'aria-live': ariaLive,
}: MessageBubbleProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const bubbleRef = useRef<HTMLDivElement>(null)

  useEffect(() => { setIsVisible(true) }, [])



  const hasContent = content && content.trim().length > 0
  const showActions = role === 'assistant' && hasContent && !isStreaming && !isError
  const id = messageId || 'msg'

  return (
    <MessageContextMenu
      messageId={id}
      content={content}
      role={role}
      isBookmarked={isBookmarked}
      onCopy={onCopy}
      onEdit={onEdit ? () => setIsEditing(true) : undefined}
      onBookmark={onBookmark}
      onRegenerate={showActions ? onRegenerate : undefined}
      onDelete={onDelete}
      onSaveToKnowledge={onSaveToKnowledge}
    >
    <div
      id={messageId ? `msg-${messageId}` : undefined}
      ref={bubbleRef}
      role="article"
      tabIndex={0}
      aria-label={`Message from ${role === 'user' ? 'You' : 'Assistant'}`}
      aria-live={isStreaming ? 'polite' : ariaLive}
      onKeyDown={(e) => {
        if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
          e.preventDefault()
          const feed = document.getElementById('chat-messages')
          if (!feed) return
          const articles = Array.from(feed.querySelectorAll<HTMLElement>('[role="article"]'))
          const idx = articles.indexOf(e.currentTarget as HTMLElement)
          const next = e.key === 'ArrowUp' ? idx - 1 : idx + 1
          if (next >= 0 && next < articles.length) articles[next].focus()
        }
      }}
      className={cn(
        "group flex flex-col transition-all duration-300 ease-out",
        isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-3",
        role === 'user' ? 'items-end' : 'items-start'
      )}
    >
      <div
        className={cn(
          "relative rounded-2xl px-4 py-2.5 sm:px-4 sm:py-3 max-w-[88%] sm:max-w-[75%] lg:max-w-[65%] transition-all duration-200 leading-relaxed",
          role === 'user'
            ? 'bg-primary text-primary-foreground rounded-br-md shadow-md'
            : 'bg-card text-foreground rounded-bl-md border border-border/50 shadow-sm',
          isStreaming && role === 'assistant' && "ring-1 ring-primary/20 animate-pulse",
          isError && role === 'assistant' && "ring-1 ring-destructive/40 border-destructive/30"
        )}
      >
        <span className={cn(
          "text-[10px] font-semibold tracking-wider uppercase mb-1.5 block",
          role === 'user' ? 'text-primary-foreground/70 text-right' : 'text-muted-foreground/60'
        )}>
          {role === 'user' ? 'You' : 'Assistant'}
          {isBookmarked && (
            <span className="ml-1.5 text-warning" aria-label="Bookmarked">
              <svg className="h-2.5 w-2.5 inline fill-current" viewBox="0 0 24 24"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
            </span>
          )}
          {role === 'assistant' && model && !isError && (
            <span className="ml-1.5 text-[9px] font-mono text-muted-foreground/40 group-hover:opacity-100 opacity-0 transition-opacity">
              {model}
            </span>
          )}
          {isError && (
            <span className="inline-flex items-center gap-1 ml-1.5 px-1.5 py-0.5 rounded-full bg-destructive/10 text-destructive text-[10px] font-medium">
              Interrupted
            </span>
          )}
        </span>

        {images && images.length > 0 && <MessageImages images={images} role={role} />}

        <MessageContent
          content={content}
          role={role}
          searchQuery={searchQuery}
          messageId={messageId}
          isStreaming={isStreaming}
          isError={isError}
          collapsibleLength={collapsibleLength}
          isEditing={isEditing}
          onEdit={onEdit}
          onEditStart={() => setIsEditing(true)}
          onEditCancel={() => setIsEditing(false)}
        />

        {showTimestamp && (
          <p className={cn(
            "mt-2 text-[10px] font-normal leading-none opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity",
            role === 'user' ? 'text-primary-foreground/50 text-right' : 'text-muted-foreground/40'
          )}>
            {formatTime(timestamp)}
          </p>
        )}
      </div>

      {(showActions || isError) && (
        <MessageActions
          content={content}
          messageId={id}
          role={role}
          onCopy={onCopy}
          onRegenerate={onRegenerate}
          onThumbsUp={onThumbsUp}
          onThumbsDown={onThumbsDown}
          onSuggestionClick={onSuggestionClick}
          isBookmarked={isBookmarked}
          onBookmark={onBookmark}
          onDelete={onDelete}
          onSaveToKnowledge={onSaveToKnowledge}
        />
      )}

      {role === 'user' && hasContent && !isEditing && (
        <MessageActions
          content={content}
          messageId={id}
          role={role}
          onCopy={onCopy}
          onEdit={() => setIsEditing(true)}
          onSuggestionClick={onSuggestionClick}
          onDelete={onDelete}
        />
      )}
    </div>
    </MessageContextMenu>
  )
})
