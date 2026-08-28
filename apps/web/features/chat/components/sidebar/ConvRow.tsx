'use client'

import { memo, useEffect, useRef, useState } from 'react'
import { cn } from '@sloughgpt/strui'
import { IconPin, IconStar, IconDot, IconDotOutline, IconDownload, IconDocument, IconCopy, IconFolder, IconX } from '@sloughgpt/strui'
import type { Conversation } from '@/lib/session-controller'
import { formatDate, truncateMessage } from '@/lib/conversations-utils'

export interface ConvRowProps {
  conversation: Conversation
  isActive: boolean
  onSelect: () => void
  onDelete?: (e: React.MouseEvent) => void
  onStar?: (e: React.MouseEvent) => void
  onPin?: (e: React.MouseEvent) => void
  onArchive?: (e: React.MouseEvent) => void
  onRename?: (name: string) => void
  onExport?: (e: React.MouseEvent, format?: 'json' | 'markdown') => void
  onDuplicate?: (e: React.MouseEvent) => void
  onToggleUnread?: (e: React.MouseEvent) => void
  searchQuery?: string
}

export const ConvRow = memo(function ConvRow({
  conversation: c,
  isActive,
  onSelect,
  onDelete,
  onStar,
  onPin,
  onArchive,
  onRename,
  onExport,
  onDuplicate,
  onToggleUnread,
  searchQuery,
}: ConvRowProps) {
  const [editing, setEditing] = useState(false)
  const [editValue, setEditValue] = useState(c.name)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus()
      inputRef.current.select()
    }
  }, [editing])

  const handleFinishEdit = () => {
    const trimmed = editValue.trim()
    if (trimmed && trimmed !== c.name) {
      onRename?.(trimmed)
    }
    setEditing(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleFinishEdit()
    } else if (e.key === 'Escape') {
      setEditValue(c.name)
      setEditing(false)
    }
  }

  const msgCount = c.messages?.length ?? c.message_count ?? 0
  const lastMsg = c.messages?.[c.messages.length - 1]?.content || ''

  const highlightMatch = (text: string, query: string): React.ReactNode => {
    if (!query) return text
    const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    const parts = text.split(new RegExp(`(${escaped})`, 'gi'))
    return parts.map((part, i) =>
      part.toLowerCase() === query.toLowerCase()
        ? <mark key={i} className="bg-primary/20 rounded px-0.5 text-inherit">{part}</mark>
        : part
    )
  }

  return (
    <div
      className={cn(
        "group flex items-start gap-2 rounded-md px-2 py-1.5 cursor-pointer transition-colors",
        isActive ? "bg-primary/10" : "hover:bg-muted/40",
        c.unread && !isActive && "bg-primary/5"
      )}
      onClick={!editing ? onSelect : undefined}
      role="button"
      tabIndex={0}
      onFocus={(e) => {
        const buttons = e.currentTarget.querySelectorAll<HTMLElement>('.sm\\:opacity-0')
        buttons.forEach(btn => btn.classList.remove('sm:opacity-0'))
      }}
      onBlur={(e) => {
        const buttons = e.currentTarget.querySelectorAll<HTMLElement>('.sm\\:opacity-0')
        buttons.forEach(btn => btn.classList.add('sm:opacity-0'))
      }}
      onKeyDown={(e) => {
        if (e.key === 'Enter' && !editing) { e.preventDefault(); onSelect(); return }
        if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
          e.preventDefault()
          const scrollable = e.currentTarget.closest('.overflow-y-auto') || e.currentTarget.parentElement?.parentElement?.parentElement
          if (!scrollable) return
          const items = Array.from(scrollable.querySelectorAll<HTMLElement>('[role="button"]'))
          const idx = items.indexOf(e.currentTarget)
          const next = e.key === 'ArrowDown' ? idx + 1 : idx - 1
          if (next >= 0 && next < items.length) items[next].focus()
        }
      }}
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={onPin}
            className="h-7 w-7 flex items-center justify-center rounded hover:bg-muted/60 shrink-0 -ml-0.5 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 focus-within:opacity-100 transition-opacity"
            aria-label={c.pinned ? 'Unpin' : 'Pin'}
          >
            <IconPin className={cn("h-2.5 w-2.5", c.pinned ? "text-primary" : "text-muted-foreground/40")} />
          </button>
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); onToggleUnread?.(e) }}
            className={cn(
              "h-7 w-7 flex items-center justify-center rounded hover:bg-muted/60 shrink-0",
              c.unread ? "opacity-100 text-primary" : "opacity-100 sm:opacity-0 sm:group-hover:opacity-100 focus-within:opacity-100 text-muted-foreground/40"
            )}
            aria-label={c.unread ? 'Mark as read' : 'Mark as unread'}
          >
            {c.unread ? (
              <IconDot className="h-2.5 w-2.5" />
            ) : (
              <IconDotOutline className="h-2.5 w-2.5" />
            )}
          </button>
          <button
            type="button"
            onClick={onStar}
            className="h-7 w-7 flex items-center justify-center rounded hover:bg-muted/60 shrink-0 -ml-0.5 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 focus-within:opacity-100 transition-opacity"
            aria-label={c.starred ? 'Unstar' : 'Star'}
          >
            <IconStar className={cn("h-2.5 w-2.5", c.starred ? "text-warning" : "text-muted-foreground/40")} filled={c.starred} />
          </button>
          {editing ? (
            <input
              ref={inputRef}
              type="text"
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              onBlur={handleFinishEdit}
              onKeyDown={handleKeyDown}
              onClick={(e) => e.stopPropagation()}
              className="flex-1 min-w-0 h-5 text-xs font-medium bg-muted/60 rounded-sm px-1 outline-none ring-1 ring-primary/40"
              aria-label="Rename conversation"
            />
          ) : (
            <p
              className={cn(
                "text-xs truncate text-foreground",
                c.unread ? "font-semibold" : "font-medium"
              )}
              onDoubleClick={(e) => { e.stopPropagation(); setEditValue(c.name); setEditing(true) }}
            >
              {highlightMatch(c.name, searchQuery || '')}
            </p>
          )}
        </div>
        {lastMsg && !editing && (
          <p className="text-[11px] text-muted-foreground/70 mt-0.5 line-clamp-1">
            {searchQuery ? highlightMatch(truncateMessage(lastMsg, 36), searchQuery) : truncateMessage(lastMsg, 36)}
          </p>
        )}
        {!editing && (
          <div className="flex items-center gap-1.5 mt-0.5">
            <span className="text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
              {msgCount}
            </span>
            <span className="text-xs text-muted-foreground/50">
              {formatDate(c.updated_at || c.updatedAt)}
            </span>
            {c.pinned && <span className="text-xs text-primary">📌</span>}
            {c.starred && <span className="text-xs">★</span>}
          </div>
        )}
      </div>
      <div className="hidden sm:flex items-center gap-0.5 shrink-0 mt-0.5 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
        {onExport && !editing && (
          <>
            <button
              type="button"
              onClick={(e) => onExport(e, 'json')}
              className="h-7 w-7 flex items-center justify-center rounded hover:bg-muted/60 text-muted-foreground hover:text-foreground"
              aria-label="Export as JSON"
              title="Export as JSON"
            >
              <IconDownload className="h-2.5 w-2.5" />
            </button>
            <button
              type="button"
              onClick={(e) => onExport(e, 'markdown')}
              className="h-7 w-7 flex items-center justify-center rounded hover:bg-muted/60 text-muted-foreground hover:text-foreground"
              aria-label="Export as Markdown"
              title="Export as Markdown"
            >
              <IconDocument className="h-2.5 w-2.5" />
            </button>
          </>
        )}
        {onDuplicate && !editing && (
          <button
            type="button"
            onClick={onDuplicate}
            className="h-4 w-4 flex items-center justify-center rounded hover:bg-muted/60 text-muted-foreground hover:text-foreground"
            aria-label={`Duplicate ${c.name}`}
            title="Duplicate conversation"
          >
            <IconCopy className="h-2.5 w-2.5" />
          </button>
        )}
        {onArchive && !editing && (
          <button
            type="button"
            onClick={onArchive}
            className="h-7 w-7 flex items-center justify-center rounded hover:bg-muted/60 text-muted-foreground hover:text-warning"
            aria-label="Archive"
          >
            <IconFolder className="h-2.5 w-2.5" />
          </button>
        )}
        {onDelete && !editing && (
          <button
            type="button"
            onClick={onDelete}
            className="h-7 w-7 flex items-center justify-center rounded hover:bg-muted/60 text-muted-foreground hover:text-destructive"
            aria-label={`Delete ${c.name}`}
          >
            <IconX className="h-2.5 w-2.5" />
          </button>
        )}
      </div>
    </div>
  )
})
