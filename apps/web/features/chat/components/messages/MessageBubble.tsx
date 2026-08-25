'use client'

import { memo, useCallback, useEffect, useRef, useState } from 'react'
import { cn, IconStar } from '@sloughgpt/strui'
import { timeAgo } from '@/lib/time-ago'
import { MessageActions } from './MessageActions'
import { MessageContextMenu } from './MessageContextMenu'
import { MessageImages } from './MessageImages'
import { MessageContent } from './MessageContent'
import { MessageReactions } from './MessageReactions'
import type { ImageAttachment } from './../input/ImageUpload'
import type { AudioAttachment } from '@/lib/chat-utils'

export interface MessageBubbleProps {
  content: string
  role: 'user' | 'assistant'
  timestamp: Date | string
  showTimestamp: boolean
  images?: ImageAttachment[]
  audio?: AudioAttachment
  reactions?: Record<string, number>
  onCopy?: (text: string) => void
  onRegenerate?: (messageId: string) => void
  onRegenerateWithOptions?: (messageId: string, options: { temperature?: number; maxTokens?: number }) => void
  onThumbsUp?: (messageId: string) => void
  onThumbsDown?: (messageId: string) => void
  onEdit?: (messageId: string, newContent: string) => void
  onReact?: (messageId: string, emoji: string) => void
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
  temperature?: number
  'aria-live'?: 'polite' | 'assertive' | 'off'
}

export const MessageBubble = memo(function MessageBubble({
  content,
  role,
  timestamp,
  showTimestamp,
  images,
  audio,
  reactions,
  onCopy,
  onRegenerate,
  onRegenerateWithOptions,
  onThumbsUp,
  onThumbsDown,
  onEdit,
  onReact,
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
  temperature,
  'aria-live': ariaLive,
}: MessageBubbleProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const bubbleRef = useRef<HTMLDivElement>(null)

  useEffect(() => { setIsVisible(true) }, [])

  const handleEditStart = useCallback(() => setIsEditing(true), [])
  const handleEditCancel = useCallback(() => setIsEditing(false), [])

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
      onEdit={onEdit ? handleEditStart : undefined}
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
      {/* Role label — outside bubble for clear separation */}
      <span className={cn(
        "text-[10px] font-semibold tracking-wider uppercase mb-0.5 block",
        role === 'user' ? 'text-primary/70 text-right' : 'text-muted-foreground/70'
      )}>
        {role === 'user' ? 'You' : 'Assistant'}
        {isBookmarked && (
          <span className="ml-1.5 text-warning" aria-label="Bookmarked">
            <IconStar className="h-2.5 w-2.5 inline" filled aria-hidden="true" />
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

      <div
        className={cn(
          "relative rounded-2xl px-3.5 py-2 sm:px-4 sm:py-2.5 max-w-[90%] sm:max-w-[80%] lg:max-w-[72%] transition-all duration-200 leading-relaxed",
          role === 'user'
            ? 'bg-primary text-primary-foreground rounded-br-md shadow-md'
            : 'bg-card text-foreground rounded-bl-md border border-border/40 shadow-sm',
          isStreaming && role === 'assistant' && "ring-1 ring-primary/20 animate-pulse",
          isError && role === 'assistant' && "ring-1 ring-destructive/40 border-destructive/30"
        )}
      >
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
          audio={audio}
          onEdit={onEdit}
          onEditStart={handleEditStart}
          onEditCancel={handleEditCancel}
        />

        {showTimestamp && (
          <p className={cn(
            "mt-1 text-[10px] font-normal leading-none opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity",
            role === 'user' ? 'text-primary-foreground/50 text-right' : 'text-muted-foreground/40'
          )}>
            {timeAgo(timestamp)}
          </p>
        )}
        {reactions && Object.keys(reactions).length > 0 && (
          <MessageReactions
            reactions={reactions}
            onReact={(emoji) => onReact?.(id, emoji)}
            className="mt-1"
          />
        )}
        {isStreaming && role === 'assistant' && hasContent && (
          <div className="flex items-center gap-1.5 mt-1" aria-live="polite">
            <div className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
            <span className="text-[10px] text-muted-foreground/50 font-mono">
              {content.length} chars
            </span>
          </div>
        )}
      </div>

      {(showActions || isError) && (
        <MessageActions
          content={content}
          messageId={id}
          role={role}
          onCopy={onCopy}
          onRegenerate={onRegenerate}
          onRegenerateWithOptions={onRegenerateWithOptions}
          onThumbsUp={onThumbsUp}
          onThumbsDown={onThumbsDown}
          onSuggestionClick={onSuggestionClick}
          isBookmarked={isBookmarked}
          onBookmark={onBookmark}
          onDelete={onDelete}
          onSaveToKnowledge={onSaveToKnowledge}
          temperature={temperature}
        />
      )}

      {role === 'user' && hasContent && !isEditing && (
        <MessageActions
          content={content}
          messageId={id}
          role={role}
          onCopy={onCopy}
          onEdit={handleEditStart}
          onSuggestionClick={onSuggestionClick}
          onDelete={onDelete}
        />
      )}
    </div>
    </MessageContextMenu>
  )
})
