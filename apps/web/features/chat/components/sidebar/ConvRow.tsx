'use client'

import { memo, useEffect, useRef, useState } from 'react'
import { cn } from '@sloughgpt/strui'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  IconMore,
  IconPin,
  IconStar,
  IconDownload,
  IconDocument,
  IconCopy,
  IconFolder,
  IconX,
} from '@sloughgpt/strui'
import type { Conversation } from '@/lib/session-controller'
import { formatDate, truncateMessage } from '@/lib/conversations-utils'

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

  // Every action lives behind one overflow affordance so the row never spends
  // its width on controls: content owns the leftover space, actions cost 24px.
  const hasActions = Boolean(
    onPin || onToggleUnread || onStar || onExport || onDuplicate || onArchive || onDelete,
  )

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
      onClick={!editing ? () => onSelect(c.id) : undefined}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' && !editing) { e.preventDefault(); onSelect(c.id); return }
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
        <div className="flex items-center gap-1.5 min-w-0">
          {c.unread && !editing && (
            <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-primary" aria-hidden="true" />
          )}
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
      {hasActions && !editing && (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              type="button"
              aria-label={`Actions for ${c.name}`}
              aria-haspopup="menu"
              title="Conversation actions"
              onClick={(e) => e.stopPropagation()}
              onDoubleClick={(e) => e.stopPropagation()}
              className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-muted/60 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <IconMore className="h-3.5 w-3.5" aria-hidden="true" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="min-w-[176px]">
            {onPin && (
              <DropdownMenuItem onSelect={(e) => { e.stopPropagation(); onPin(e, c.id, !c.pinned) }}>
                <IconPin className={cn('mr-2 h-3.5 w-3.5', c.pinned ? 'text-primary' : 'text-muted-foreground/40')} />
                <span className="flex-1">{c.pinned ? 'Unpin' : 'Pin'}</span>
              </DropdownMenuItem>
            )}
            {onToggleUnread && (
              <DropdownMenuItem onSelect={(e) => { e.stopPropagation(); onToggleUnread(e, c.id, !c.unread) }}>
                <span className="mr-2 flex h-3.5 w-3.5 items-center justify-center" aria-hidden="true">
                  <span
                    className={cn(
                      'h-1.5 w-1.5 rounded-full',
                      c.unread ? 'bg-primary' : 'border border-muted-foreground/40',
                    )}
                  />
                </span>
                <span className="flex-1">{c.unread ? 'Mark as read' : 'Mark as unread'}</span>
              </DropdownMenuItem>
            )}
            {onStar && (
              <DropdownMenuItem onSelect={(e) => { e.stopPropagation(); onStar(e, c.id, !c.starred) }}>
                <IconStar className={cn('mr-2 h-3.5 w-3.5', c.starred ? 'text-warning' : 'text-muted-foreground/40')} filled={c.starred} />
                <span className="flex-1">{c.starred ? 'Unstar' : 'Star'}</span>
              </DropdownMenuItem>
            )}
            {(onExport || onDuplicate || onArchive || onDelete) && <DropdownMenuSeparator />}
            {onExport && (
              <>
                <DropdownMenuItem onSelect={(e) => { e.stopPropagation(); onExport(e, c, 'json') }}>
                  <IconDownload className="mr-2 h-3.5 w-3.5 text-muted-foreground/40" />
                  <span className="flex-1">Export as JSON</span>
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={(e) => { e.stopPropagation(); onExport(e, c, 'markdown') }}>
                  <IconDocument className="mr-2 h-3.5 w-3.5 text-muted-foreground/40" />
                  <span className="flex-1">Export as Markdown</span>
                </DropdownMenuItem>
              </>
            )}
            {onDuplicate && (
              <DropdownMenuItem onSelect={(e) => { e.stopPropagation(); onDuplicate(e, c.id, c.name) }}>
                <IconCopy className="mr-2 h-3.5 w-3.5 text-muted-foreground/40" />
                <span className="flex-1">Duplicate conversation</span>
              </DropdownMenuItem>
            )}
            {onArchive && (
              <DropdownMenuItem onSelect={(e) => { e.stopPropagation(); onArchive(e, c.id, false) }}>
                <IconFolder className="mr-2 h-3.5 w-3.5 text-muted-foreground/40" />
                <span className="flex-1">Archive</span>
              </DropdownMenuItem>
            )}
            {onDelete && (
              <DropdownMenuItem destructive onSelect={(e) => { e.stopPropagation(); onDelete(e, c.id) }}>
                <IconX className="mr-2 h-3.5 w-3.5" />
                <span className="flex-1">Delete conversation</span>
              </DropdownMenuItem>
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      )}
    </div>
  )
})
