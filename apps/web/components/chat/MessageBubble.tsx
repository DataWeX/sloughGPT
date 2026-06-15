'use client'

import { memo, useEffect, useState, useRef } from 'react'
import { cn } from '@/lib/cn'
import { Button } from '@/components/ui/button'
import { Markdown } from './Markdown'
import { Textarea } from '@/components/ui/textarea'
import { MessageActions } from './MessageActions'
import type { ImageAttachment } from './ImageUpload'

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
  messageId?: string
  isStreaming?: boolean
  searchQuery?: string
  model?: string
  'aria-live'?: 'polite' | 'assertive' | 'off'
}

function formatTime(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date
  const now = new Date()
  const diffMs = now.getTime() - d.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  if (diffMins < 1) return 'just now'
  if (diffMins < 60) return `${diffMins}m ago`
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function highlightText(text: string, query: string): (string | JSX.Element)[] {
  if (!query) return [text]
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const parts = text.split(new RegExp(`(${escaped})`, 'gi'))
  return parts.map((part, i) =>
    part.toLowerCase() === query.toLowerCase()
      ? <mark key={i} className="bg-primary/20 rounded px-0.5 text-inherit">{part}</mark>
      : part
  )
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
  messageId,
  isStreaming = false,
  searchQuery,
  model,
  'aria-live': ariaLive,
}: MessageBubbleProps) {
  const [displayContent, setDisplayContent] = useState(content)
  const [isVisible, setIsVisible] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editContent, setEditContent] = useState(content)
  const bubbleRef = useRef<HTMLDivElement>(null)
  
  useEffect(() => {
    setIsVisible(true)
  }, [])
  
  useEffect(() => {
    setDisplayContent(content)
    setEditContent(content)
  }, [content])

  // Auto-scroll streaming content into view (throttled to ~200ms)
  const lastScrollRef = useRef(0)
  useEffect(() => {
    if (isStreaming && bubbleRef.current) {
      const now = Date.now()
      if (now - lastScrollRef.current > 200) {
        lastScrollRef.current = now
        bubbleRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' })
      }
    }
  }, [content, isStreaming])

  const handleEditSave = () => {
    if (editContent.trim() && onEdit && messageId) {
      onEdit(messageId, editContent.trim())
      setIsEditing(false)
    }
  }

  const handleEditCancel = () => {
    setEditContent(content)
    setIsEditing(false)
  }

  const hasContent = displayContent && displayContent.trim().length > 0
  const showActions = role === 'assistant' && hasContent && !isStreaming
  const id = messageId || 'msg'

  return (
    <div
      id={messageId ? `msg-${messageId}` : undefined}
      ref={bubbleRef}
      role="article"
      aria-label={`Message from ${role === 'user' ? 'You' : 'Assistant'}`}
      aria-live={isStreaming ? 'polite' : ariaLive}
      className={cn(
        "group flex flex-col transition-all duration-300 ease-out",
        isVisible 
          ? "opacity-100 translate-y-0" 
          : "opacity-0 translate-y-3",
        role === 'user' ? 'items-end' : 'items-start'
      )}
    >
      <div
        className={cn(
          "relative rounded-2xl px-3 py-2.5 sm:px-4 sm:py-3 max-w-[85%] sm:max-w-[70%] lg:max-w-[60%] transition-all duration-200 leading-relaxed",
          role === 'user'
            ? 'bg-primary text-primary-foreground rounded-br-sm shadow-sm'
            : 'bg-card text-foreground rounded-bl-sm border border-border/40 shadow-sm',
          isStreaming && role === 'assistant' && "ring-1 ring-primary/10"
        )}
      >
        {/* role indicator */}
        <span className={cn(
          "text-[10px] font-medium tracking-wide uppercase mb-1 block",
          role === 'user' ? 'text-primary-foreground/60 text-right' : 'text-muted-foreground/50'
        )}>
          {role === 'user' ? 'You' : 'Assistant'}
          {role === 'assistant' && model && (
            <span className="ml-1.5 text-[9px] font-mono text-muted-foreground/40 group-hover:opacity-100 opacity-0 transition-opacity">
              {model}
            </span>
          )}
        </span>

        {images && images.length > 0 && (
          <div className={cn(
            "flex gap-2 mb-3 flex-wrap",
            role === 'user' && "flex-row-reverse"
          )}>
            {images.map((img) => (
              <img
                key={img.id}
                src={img.dataUrl}
                alt={img.name}
                className="h-24 w-24 rounded-xl object-cover border border-current/20 shadow-sm hover:shadow-md transition-shadow"
              />
            ))}
          </div>
        )}
        
        {hasContent && (
          role === 'assistant' ? (
            searchQuery && content.toLowerCase().includes(searchQuery.toLowerCase()) ? (
              <p className="whitespace-pre-wrap break-words leading-relaxed text-sm">{highlightText(content, searchQuery)}</p>
            ) : (
              <article className="leading-relaxed text-sm" aria-label={`${role} message`}>
                <Markdown content={displayContent} />
                {isStreaming && (
                  <span className="inline-block ml-0.5 animate-pulse text-primary" aria-hidden="true">▊</span>
                )}
              </article>
            )
          ) : isEditing ? (
            <form 
              className="space-y-2"
              onSubmit={(e) => {
                e.preventDefault()
                handleEditSave()
              }}
              aria-label="Edit message form"
            >
              <label className="sr-only" htmlFor={`edit-${id}`}>Edit message</label>
              <Textarea
                id={`edit-${id}`}
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                className="w-full min-h-[60px] text-sm bg-background text-foreground"
                rows={3}
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === 'Escape') {
                    handleEditCancel()
                  }
                }}
                aria-describedby={`edit-hint-${id}`}
              />
              <p id={`edit-hint-${id}`} className="sr-only">Press Enter to save, Escape to cancel</p>
              <div className="flex justify-end gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleEditCancel}
                  aria-label="Cancel editing"
                >
                  Cancel
                </Button>
                <Button
                  size="sm"
                  aria-label="Save and resend message"
                >
                  Resend
                </Button>
              </div>
            </form>
          ) : (
            <p className="whitespace-pre-wrap break-words leading-relaxed text-sm">
              {searchQuery ? highlightText(displayContent, searchQuery) : displayContent}
            </p>
          )
        )}
        
        {!hasContent && role === 'assistant' && (
          <div className="flex gap-1 items-center h-5" aria-hidden="true">
            <span className="w-2 h-2 bg-foreground/30 rounded-full animate-bounce [animation-delay:0ms]" />
            <span className="w-2 h-2 bg-foreground/30 rounded-full animate-bounce [animation-delay:150ms]" />
            <span className="w-2 h-2 bg-foreground/30 rounded-full animate-bounce [animation-delay:300ms]" />
          </div>
        )}
        
        {showTimestamp && (
          <p className={cn(
            "mt-1.5 text-[10px] font-normal leading-none opacity-0 group-hover:opacity-100 transition-opacity",
            role === 'user' ? 'text-primary-foreground/50 text-right' : 'text-muted-foreground/40'
          )}>
            {formatTime(timestamp)}
          </p>
        )}
      </div>

      {showActions && (
        <MessageActions
          content={content}
          messageId={id}
          onCopy={onCopy}
          onRegenerate={onRegenerate}
          onThumbsUp={onThumbsUp}
          onThumbsDown={onThumbsDown}
        />
      )}
      
      {role === 'user' && hasContent && !isEditing && onEdit && (
        <MessageActions
          content={content}
          messageId={id}
          onEdit={() => setIsEditing(true)}
        />
      )}
    </div>
  )
})
