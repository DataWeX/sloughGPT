'use client'

import { memo, useEffect, useRef, useState } from 'react'
import { cn } from '@sloughgpt/strui'
import {
  IconPin,
  IconStar,
  IconChat,
  IconDot,
  IconDotOutline,
  IconDownload,
  IconDocument,
  IconCopy,
  IconFolder,
  IconX,
} from '@sloughgpt/strui'
import type { Conversation } from '@/lib/session-controller'
import { formatDate, truncateMessage } from '@/lib/conversations-utils'

/** Keyboard-focus ring shared by the row container and its action buttons. */
const FOCUS_CLS =
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset'

export interface ConvRowProps {
  conversation: Conversation
  isActive: boolean
  onSelect: (id: string) => void
  onDelete?: (e: React.MouseEvent, id: string) => void
  onStar?: (e: React.MouseEvent, id: string, starred: boolean) => void
  onPin?: (e: React.MouseEvent, id: string, pinned: boolean) => void
  onArchive?: (e: React.MouseEvent, id: string, archive: boolean) => void
  onRename?: (id: string, name: string) => void
  onExport?: (e: React.MouseEvent, conversation: Conversation, format?: 'json' | 'markdown') => void
  onDuplicate?: (e: React.MouseEvent, id: string, name: string) => void
  onToggleUnread?: (e: React.MouseEvent, id: string, unread: boolean) => void
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
      onRename?.(c.id, trimmed)
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
      part.toLowerCase() === query.toLowerCase() ? (
        <mark key={i} className="bg-primary/20 rounded px-0.5 text-inherit">
          {part}
        </mark>
      ) : (
        part
      ),
    )
  }

  return (
    <div
      className={cn(
        'group relative flex items-start gap-2 rounded-md px-2 py-1.5 cursor-pointer transition-colors',
        FOCUS_CLS,
        'after:absolute after:left-0 after:top-1/2 after:h-4 after:w-0.5 after:-translate-y-1/2 after:rounded-full after:content-[""]',
        isActive ? 'bg-primary/10 after:bg-primary' : 'hover:bg-muted/40 after:bg-transparent',
        c.unread && !isActive && 'bg-primary/5',
      )}
      onClick={!editing ? () => onSelect(c.id) : undefined}
      role="button"
      tabIndex={0}
      aria-current={isActive ? 'true' : undefined}
      onFocus={(e) => {
        const buttons = e.currentTarget.querySelectorAll<HTMLElement>('.sm\\:opacity-0')
        buttons.forEach((btn) => btn.classList.remove('sm:opacity-0'))
      }}
      onBlur={(e) => {
        const buttons = e.currentTarget.querySelectorAll<HTMLElement>('.sm\\:opacity-0')
        buttons.forEach((btn) => btn.classList.add('sm:opacity-0'))
      }}
      onKeyDown={(e) => {
        if ((e.key === 'Enter' || e.key === ' ') && !editing) {
          e.preventDefault()
          onSelect(c.id)
          return
        }
        if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
          e.preventDefault()
          const scrollable =
            e.currentTarget.closest('.overflow-y-auto') ||
            e.currentTarget.parentElement?.parentElement?.parentElement
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
            onClick={(e) => onPin?.(e, c.id, !c.pinned)}
            className="active:bg-muted h-7 w-7 flex items-center justify-center rounded hover:bg-muted/60 shrink-0 -ml-0.5 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 focus-within:opacity-100 transition-opacity focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset"
            aria-label={c.pinned ? 'Unpin' : 'Pin'}
          >
            <IconPin
              className={cn('h-2.5 w-2.5', c.pinned ? 'text-primary' : 'text-muted-foreground/40')}
            />
          </button>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              onToggleUnread?.(e, c.id, !c.unread)
            }}
            className={cn(
              'h-7 w-7 flex items-center justify-center rounded hover:bg-muted/60 shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset',
              c.unread
                ? 'opacity-100 text-primary'
                : 'opacity-100 sm:opacity-0 sm:group-hover:opacity-100 focus-within:opacity-100 transition-opacity duration-150 text-muted-foreground/40',
              'active:bg-muted',
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
            onClick={(e) => onStar?.(e, c.id, !c.starred)}
            className="active:bg-muted h-7 w-7 flex items-center justify-center rounded hover:bg-muted/60 shrink-0 -ml-0.5 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 focus-within:opacity-100 transition-opacity focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset"
            aria-label={c.starred ? 'Unstar' : 'Star'}
          >
            <IconStar
              className={cn('h-2.5 w-2.5', c.starred ? 'text-warning' : 'text-muted-foreground/40')}
              filled={c.starred}
            />
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
                'text-xs truncate text-foreground',
                c.unread ? 'font-semibold' : 'font-medium',
              )}
              onDoubleClick={(e) => {
                e.stopPropagation()
                setEditValue(c.name)
                setEditing(true)
              }}
            >
              {highlightMatch(c.name, searchQuery || '')}
            </p>
          )}
        </div>
        {lastMsg && !editing && (
          <p className="text-[11px] text-muted-foreground/70 mt-0.5 line-clamp-1">
            {searchQuery
              ? highlightMatch(truncateMessage(lastMsg, 36), searchQuery)
              : truncateMessage(lastMsg, 36)}
          </p>
        )}
        {!editing && (
          <div className="mt-0.5 flex items-center gap-1 text-[11px] text-muted-foreground tabular-nums">
            <IconChat className="h-2.5 w-2.5 shrink-0" aria-hidden="true" />
            <span>{msgCount}</span>
            <IconDot className="h-2 w-2 shrink-0" aria-hidden="true" />
            <span className="text-muted-foreground/50 truncate">
              {formatDate(c.updated_at || c.updatedAt)}
            </span>
          </div>
        )}
      </div>
      <div className="hidden sm:flex items-center gap-0.5 shrink-0 mt-0.5 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
        {onExport && !editing && (
          <>
            <button
              type="button"
              onClick={(e) => onExport?.(e, c, 'json')}
              className="active:bg-muted h-7 w-7 flex items-center justify-center rounded hover:bg-muted/60 text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset"
              aria-label="Export as JSON"
              title="Export as JSON"
            >
              <IconDownload className="h-2.5 w-2.5" />
            </button>
            <button
              type="button"
              onClick={(e) => onExport?.(e, c, 'markdown')}
              className="active:bg-muted h-7 w-7 flex items-center justify-center rounded hover:bg-muted/60 text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset"
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
            onClick={(e) => onDuplicate?.(e, c.id, c.name)}
            className="active:bg-muted h-7 w-7 flex items-center justify-center rounded hover:bg-muted/60 text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset"
            aria-label={`Duplicate ${c.name}`}
            title="Duplicate conversation"
          >
            <IconCopy className="h-2.5 w-2.5" />
          </button>
        )}
        {onArchive && !editing && (
          <button
            type="button"
            onClick={(e) => onArchive?.(e, c.id, false)}
            className="active:bg-muted h-7 w-7 flex items-center justify-center rounded hover:bg-muted/60 text-muted-foreground hover:text-warning focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset"
            aria-label="Archive"
            title="Archive conversation"
          >
            <IconFolder className="h-2.5 w-2.5" />
          </button>
        )}
        {onDelete && !editing && (
          <button
            type="button"
            onClick={(e) => onDelete?.(e, c.id)}
            className="active:bg-muted h-7 w-7 flex items-center justify-center rounded hover:bg-muted/60 text-muted-foreground hover:text-destructive focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset"
            aria-label={`Delete ${c.name}`}
            title="Delete conversation"
          >
            <IconX className="h-2.5 w-2.5" />
          </button>
        )}
      </div>
    </div>
  )
})
