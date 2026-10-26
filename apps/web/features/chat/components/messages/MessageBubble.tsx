'use client'

import { memo, useCallback, useEffect, useRef, useState } from 'react'
import dynamic from 'next/dynamic'
import { cn, IconStar, StatusDot } from '@sloughgpt/strui'
import { formatRelativeTime } from '@/lib/time-format'
import { ConsciousnessMessageBadge } from '../ConsciousnessMessageBadge'

import { MessageImages } from './MessageImages'
import { MessageContent } from './MessageContent'
import type { ImageAttachment } from './../input/ImageUpload'

const MessageActions = dynamic(() => import('./MessageActions').then((m) => m.MessageActions), {
  ssr: false,
})
const MessageContextMenu = dynamic(
  () => import('./MessageContextMenu').then((m) => m.MessageContextMenu),
  { ssr: false },
)

export interface MessageBubbleProps {
  content: string
  role: 'user' | 'assistant'
  timestamp: Date | string
  showTimestamp: boolean
  images?: ImageAttachment[]
  onCopy?: (text: string) => void
  onRegenerate?: (messageId: string) => void
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
  onReact?: (messageId: string, emoji: string) => void
  onPin?: (messageId: string) => void
  hasNote?: boolean
  onAddNote?: (messageId: string) => void
  hasThread?: boolean
  onThread?: (messageId: string) => void
  /** Position of this turn in the transcript (1-based). Renders in the gutter. */
  turnNumber?: number
  /** Per-turn stream metrics for the margin (tok/s, token count, latency). */
  metrics?: { tokens?: number; elapsedMs?: number }
  'aria-live'?: 'polite' | 'assertive' | 'off'
}

function formatMetrics(m?: { tokens?: number; elapsedMs?: number }): string | null {
  if (!m || m.tokens == null || m.elapsedMs == null || m.elapsedMs <= 0) return null
  const tps = m.tokens / (m.elapsedMs / 1000)
  const secs = (m.elapsedMs / 1000).toFixed(1)
  return `${tps.toFixed(1)} tok/s · ${m.tokens} tok · ${secs}s`
}

/**
 * Transcript turn — the log-entry primitive (see DESIGN_SYSTEM.md "The Log Entry").
 *
 * Anatomy: gutter (turn number, monospace) | content (document-like: assistant
 * prose flows with no bubble chrome, user turns are compact tinted cards) |
 * margin (living metadata: role, model, per-turn stream metrics, StatusDot).
 */
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
  onReact,
  onPin,
  hasNote,
  onAddNote,
  hasThread,
  onThread,
  turnNumber,
  metrics,
  'aria-live': ariaLive,
}: MessageBubbleProps) {
  const [isEditing, setIsEditing] = useState(false)
  const bubbleRef = useRef<HTMLDivElement>(null)
  const articlesRef = useRef<HTMLElement[]>([])

  const hasContent = content && content.trim().length > 0
  const showActions = role === 'assistant' && hasContent && !isStreaming && !isError
  const id = messageId || 'msg'
  const metricsText = formatMetrics(metrics)

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      e.preventDefault()
      if (articlesRef.current.length === 0) return
      const idx = articlesRef.current.indexOf(e.currentTarget as HTMLElement)
      const next = e.key === 'ArrowUp' ? idx - 1 : idx + 1
      if (next >= 0 && next < articlesRef.current.length) articlesRef.current[next].focus()
    }
  }, [])

  const handleEditStart = useCallback(() => setIsEditing(true), [])
  const handleEditCancel = useCallback(() => setIsEditing(false), [])

  useEffect(() => {
    const feed = document.getElementById('chat-messages')
    if (!feed) return
    articlesRef.current = Array.from(feed.querySelectorAll<HTMLElement>('[role="article"]'))
  })

  return (
    <MessageContextMenu
      messageId={id}
      content={content}
      role={role}
      isBookmarked={isBookmarked}
      isPinned={false}
      hasNote={hasNote}
      hasThread={hasThread}
      onCopy={onCopy}
      onEdit={onEdit ? handleEditStart : undefined}
      onBookmark={onBookmark}
      onPin={onPin}
      onRegenerate={showActions ? onRegenerate : undefined}
      onDelete={onDelete}
      onSaveToKnowledge={onSaveToKnowledge}
      onAddNote={onAddNote}
      onThread={onThread}
    >
      <div
        id={messageId ? `msg-${messageId}` : undefined}
        ref={bubbleRef}
        role="article"
        tabIndex={0}
        aria-label={`Message from ${role === 'user' ? 'You' : 'Assistant'}`}
        aria-live={isStreaming ? 'polite' : ariaLive}
        onKeyDown={handleKeyDown}
        className={cn(
          'group grid transition-all duration-300 ease-out',
          'grid-cols-[minmax(0,1fr)] sm:grid-cols-[2.5rem_minmax(0,1fr)_auto]',
          'items-start',
        )}
      >
        {/* Gutter — stable turn number, like a transcript or lab log */}
        <span
          className="hidden sm:block pt-1 text-right font-mono text-[10px] leading-5 text-muted-foreground/50 select-none"
          aria-hidden="true"
        >
          {turnNumber != null ? String(turnNumber).padStart(2, '0') : ''}
        </span>

        {/* Content — the words are the interface */}
        <div className={cn('min-w-0', role === 'user' && !isEditing && 'flex justify-end')}>
          <div
            className={cn(
              'transition-all duration-200 leading-relaxed',
              role === 'user'
                ? isEditing
                  ? 'w-full bg-muted text-foreground rounded-lg px-4 py-2.5'
                  : 'bg-primary text-primary-foreground rounded-lg px-4 py-2.5 max-w-[85%] sm:max-w-[75%] lg:max-w-[65%] shadow-sm'
                : 'text-foreground max-w-[70ch] py-0.5',
              isError && role === 'assistant' && 'text-destructive',
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
              onEdit={onEdit}
              onEditStart={handleEditStart}
              onEditCancel={handleEditCancel}
            />
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
              onReact={onReact}
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

        {/* Margin — living metadata in monospace: the log's instrumentation */}
        <div
          className={cn(
            'hidden sm:flex flex-col items-end gap-0.5 pt-1 pl-4',
            'font-mono text-[10px] leading-5 text-muted-foreground/60 select-none',
          )}
        >
          <span className="flex items-center gap-1.5">
            {isStreaming && <StatusDot tone="primary" pulse aria-hidden />}
            {isError && <StatusDot tone="destructive" aria-hidden />}
            {role === 'user' ? 'You' : 'Assistant'}
            {isBookmarked && (
              <span className="text-warning" aria-label="Bookmarked">
                <IconStar className="h-2.5 w-2.5 inline" filled />
              </span>
            )}
          </span>
          {role === 'assistant' && model && !isError && <span>{model}</span>}
          {metricsText && <span>{metricsText}</span>}
          {isError && <span className="text-destructive">Interrupted</span>}
          {showTimestamp && (
            <span className="flex items-center gap-1.5">
              {formatRelativeTime(timestamp)}
              {role === 'assistant' && messageId && (
                <ConsciousnessMessageBadge messageId={messageId} />
              )}
            </span>
          )}
        </div>
      </div>
    </MessageContextMenu>
  )
})
