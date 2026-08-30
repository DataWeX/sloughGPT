'use client'

import { useState, useMemo } from 'react'
import { Button, Textarea } from '@sloughgpt/strui'
import { Markdown } from './Markdown'

interface MessageContentProps {
  content: string
  role: 'user' | 'assistant'
  searchQuery?: string
  messageId?: string
  isStreaming?: boolean
  isError?: boolean
  collapsibleLength?: number
  isEditing?: boolean
  onEdit?: (messageId: string, newContent: string) => void
  onEditStart?: () => void
  onEditCancel?: () => void
}

function highlightText(text: string, query: string): (string | React.JSX.Element)[] {
  if (!query) return [text]
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const parts = text.split(new RegExp(`(${escaped})`, 'gi'))
  return parts.map((part, i) =>
    part.toLowerCase() === query.toLowerCase()
      ? <mark key={i} className="bg-primary/20 rounded px-0.5 text-inherit">{part}</mark>
      : part
  )
}

export function MessageContent({
  content,
  role,
  searchQuery,
  messageId,
  isStreaming = false,
  isError = false,
  collapsibleLength = 0,
  isEditing = false,
  onEdit,
  onEditStart,
  onEditCancel,
}: MessageContentProps) {
  const [editContent, setEditContent] = useState(content)
  const [isCollapsed, setIsCollapsed] = useState(true)

  const id = messageId || 'msg'
  const hasContent = content && content.trim().length > 0
  const isCollapsible = collapsibleLength > 0 && content.length > collapsibleLength && !isEditing && !isStreaming
  const visibleContent = isCollapsible && isCollapsed ? content.slice(0, collapsibleLength) : content
  const highlightedContent = useMemo(() => {
    if (!searchQuery || !content) return null
    return highlightText(content, searchQuery)
  }, [content, searchQuery])

  if (!hasContent) {
    if (role === 'assistant') {
      return (
        <div className="flex gap-1 items-center h-5" aria-hidden="true">
          <span className="w-2 h-2 bg-foreground/30 rounded-full animate-bounce [animation-delay:0ms]" />
          <span className="w-2 h-2 bg-foreground/30 rounded-full animate-bounce [animation-delay:150ms]" />
          <span className="w-2 h-2 bg-foreground/30 rounded-full animate-bounce [animation-delay:300ms]" />
        </div>
      )
    }
    return null
  }

  if (role === 'assistant') {
    if (searchQuery && content.toLowerCase().includes(searchQuery.toLowerCase())) {
      return <p className="whitespace-pre-wrap break-words leading-relaxed text-sm">{highlightedContent}</p>
    }
    return (
      <article className="leading-relaxed text-sm" aria-label={`${role} message`}>
        <Markdown content={visibleContent} />
        {isCollapsible && isCollapsed && (
          <span className="text-muted-foreground/40 select-none">…</span>
        )}
        {isStreaming && (
          <span className="inline-block ml-0.5 animate-pulse text-primary" aria-hidden="true">▊</span>
        )}
        {isCollapsible && (
          <button
            type="button"
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="block mt-1.5 text-xs font-medium text-primary hover:text-primary/80 transition-colors"
            aria-expanded={!isCollapsed}
          >
            {isCollapsed ? `Show more (${content.length - collapsibleLength} more)` : 'Show less'}
          </button>
        )}
      </article>
    )
  }

  // User role
  if (isEditing) {
    const handleSave = () => {
      if (editContent.trim() && onEdit && messageId) {
        onEdit(messageId, editContent.trim())
        onEditCancel?.()
      }
    }
    return (
      <form className="space-y-2" onSubmit={(e) => { e.preventDefault(); handleSave() }} aria-label="Edit message form">
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
              setEditContent(content)
              onEditCancel?.()
            }
          }}
          aria-describedby={`edit-hint-${id}`}
        />
        <p id={`edit-hint-${id}`} className="sr-only">Press Enter to save, Escape to cancel</p>
        <div className="flex justify-end gap-2">
          <Button variant="outline" size="sm" onClick={() => { setEditContent(content); onEditCancel?.() }} aria-label="Cancel editing">
            Cancel
          </Button>
          <Button size="sm" aria-label="Save and resend message">
            Resend
          </Button>
        </div>
      </form>
    )
  }

  return (
    <div>
      <p className="whitespace-pre-wrap break-words leading-relaxed text-sm">
        {searchQuery ? highlightText(visibleContent, searchQuery) : visibleContent}
      </p>
      {isCollapsible && isCollapsed && (
        <span className="text-primary-foreground/50 select-none text-sm">…</span>
      )}
      {isCollapsible && (
        <button
          type="button"
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="mt-1 text-xs font-medium text-primary-foreground/70 hover:text-primary-foreground transition-colors"
          aria-expanded={!isCollapsed}
        >
          {isCollapsed ? `Show more (${content.length - collapsibleLength} more)` : 'Show less'}
        </button>
      )}
    </div>
  )
}
