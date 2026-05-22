'use client'

import { useEffect, useState } from 'react'
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
}

function formatTime(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date
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

export function MessageBubble({ 
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
}: MessageBubbleProps) {
  const [displayContent, setDisplayContent] = useState(content)
  const [isVisible, setIsVisible] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editContent, setEditContent] = useState(content)
  
  useEffect(() => {
    setIsVisible(true)
  }, [])
  
  useEffect(() => {
    setDisplayContent(content)
    setEditContent(content)
  }, [content])

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
      className={cn(
        "flex flex-col transition-all duration-300 ease-out group",
        isVisible 
          ? "opacity-100 translate-y-0" 
          : "opacity-0 translate-y-2",
        role === 'user' ? 'items-end' : 'items-start'
      )}
    >
      <div
        className={cn(
          "relative rounded-2xl px-3 py-2.5 text-xs sm:px-4 sm:py-3 max-w-[40%] transition-all duration-200 leading-relaxed",
          role === 'user'
            ? 'bg-primary text-primary-foreground rounded-br-sm shadow-md'
            : 'bg-card text-foreground rounded-bl-sm border border-border/60 shadow-sm hover:shadow-md',
          isStreaming && role === 'assistant' && "cursor-wait"
        )}
      >
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
              <p className="whitespace-pre-wrap break-words leading-relaxed">{highlightText(content, searchQuery)}</p>
            ) : (
              <article className="leading-relaxed" aria-label={`${role} message`}>
                <Markdown content={displayContent} />
                {isStreaming && (
                  <span className="inline-block ml-1 animate-pulse" aria-hidden="true">▊</span>
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
                className="w-full min-h-[60px]"
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
            <p className="whitespace-pre-wrap break-words leading-relaxed">
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
            "mt-1 text-xs opacity-40 font-normal",
            role === 'user' ? 'text-right' : 'text-left'
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
}
