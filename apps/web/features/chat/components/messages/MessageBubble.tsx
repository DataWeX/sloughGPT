'use client'

import { memo, useCallback, useEffect, useRef, useState } from 'react'
import dynamic from '@/vite/next-compat/dynamic'
import { cn, IconStar, StatusDot } from '@sloughgpt/strui'
import { formatRelativeTime } from '@/lib/time-format'
import { ConsciousnessMessageBadge } from '../ConsciousnessMessageBadge'

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
  /** Per-turn stream metrics for the frame footer (tok/s, token count, latency). */
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
 * Transcript turn — composes MessageFrame (Direction B, ruling 2026-11-03).
 *
 * Anatomy: the shared frame (user = right-aligned tinted surface, assistant =
 * left-aligned card surface, both rounded-2xl) with the metadata footer inside
 * it; this file owns only the surrounding behavior — context menu, actions,
 * aria/live wiring. Chrome is never re-decided here.
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

  // Metadata footer — inside the frame (Direction B), never a detached gutter.
  const footer = (
    <>
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
      {role === 'assistant' && model && !isError && hasContent && <span>{model}</span>}
      {metricsText && <span>{metricsText}</span>}
      {isError && <span className="text-destructive">Interrupted</span>}
      {/* Timestamp = the entry-closed marker; don't claim a finished entry
          before the first token lands. Role label + StatusDot still signal
          the in-progress turn. */}
      {showTimestamp && (hasContent || role === 'user') && (
        <span className="flex items-center gap-1.5">
          {formatRelativeTime(timestamp)}
          {role === 'assistant' && messageId && <ConsciousnessMessageBadge messageId={messageId} />}
        </span>
      )}
    </>
  )

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
        className={cn('group transition-all duration-300 ease-out')}
      >
        {/* Content — the frame carries the chrome; actions sit outside it.
            This wrapper must stay a plain block: MessageFrame already owns
            `justify-end` for user turns, and adding it here right-aligned
            [frame][hover toolbar] as a PAIR — the toolbar is opacity-0 but still
            occupies ~200px, so the bubble stopped short of the right edge and
            read as floating mid-row. */}
        <div className="min-w-0">
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
            images={images}
            footer={footer}
          />

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
            /* Column + items-end: MessageActions is a fragment (toolbar, then
               reactions), so it must stack vertically while its buttons sit
               flush with the right-aligned bubble above. */
            <div className="flex flex-col items-end">
              <MessageActions
                content={content}
                messageId={id}
                role={role}
                onCopy={onCopy}
                onEdit={handleEditStart}
                onSuggestionClick={onSuggestionClick}
                onDelete={onDelete}
              />
            </div>
          )}
        </div>
      </div>
    </MessageContextMenu>
  )
})
