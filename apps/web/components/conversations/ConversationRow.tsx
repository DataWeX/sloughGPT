'use client'

import { useState } from 'react'
import { cn, Button } from '@sloughgpt/strui'
import { IconStar, IconPin, IconChat, IconTrash, IconEdit, IconDownload, IconMore, IconFolder } from '@sloughgpt/strui'
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger,
} from '@sloughgpt/strui'
import { type Conversation } from '@/lib/session-controller'
import { formatDate, truncateMessage } from '@/lib/conversations-utils'

interface ConversationRowProps {
  conversation: Conversation
  selected: boolean
  onToggleSelect: () => void
  onSelect: () => void
  onPin: (pinned: boolean) => void
  onStar: (starred: boolean) => void
  onArchive: (archived: boolean) => void
  onDelete: () => void
  onRename: () => void
  onExport: (format: 'md' | 'json') => void
}

export default function ConversationRow({
  conversation: c,
  selected,
  onToggleSelect,
  onSelect,
  onPin,
  onStar,
  onArchive,
  onDelete,
  onRename,
  onExport,
}: ConversationRowProps) {
  const msgCount = c.messages?.length ?? c.message_count ?? 0
  const lastMsg = c.messages?.[c.messages.length - 1]?.content || ''

  return (
    <div
      className={cn(
        "group flex items-start gap-2 rounded-md border border-border/40 bg-card/50 p-2.5 cursor-pointer transition-all",
        selected ? "border-primary/40 bg-primary/5" : "hover:bg-secondary/30 hover:border-border/60"
      )}
    >
      <div
        className="flex items-center justify-center pt-1 shrink-0"
        onClick={(e) => { e.stopPropagation(); onToggleSelect() }}
        role="checkbox"
        aria-checked={selected}
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onToggleSelect() } }}
      >
        <input
          type="checkbox"
          className="h-3.5 w-3.5 rounded border-border accent-primary cursor-pointer"
          checked={selected}
          onChange={onToggleSelect}
          onClick={(e) => e.stopPropagation()}
        />
      </div>
      <div className="flex-1 min-w-0" onClick={onSelect} role="button" tabIndex={0} onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); onSelect() } }}>
        <div className="flex items-center gap-1.5">
          {c.pinned && <IconPin className="h-3 w-3 text-primary shrink-0" />}
          {c.starred && <IconStar className="h-3 w-3 text-warning shrink-0" filled />}
          {c.archived && <span className="text-[10px] text-muted-foreground/60 border border-border/40 rounded px-1 shrink-0">Archived</span>}
          <p className="text-sm font-medium truncate text-foreground">{c.name}</p>
        </div>
        {lastMsg && (
          <p className="text-xs text-muted-foreground/70 mt-0.5 line-clamp-1">
            {truncateMessage(lastMsg)}
          </p>
        )}
        <div className="flex items-center gap-1.5 mt-1">
          <IconChat className="h-3 w-3 text-muted-foreground/60 shrink-0" />
          <span className="text-xs text-muted-foreground/60">
            {msgCount} messages · {formatDate(c.updated_at || c.updatedAt)}
          </span>
        </div>
      </div>

      <div className="shrink-0 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 focus-within:opacity-100 transition-opacity pt-0.5">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="icon-sm"
              className="h-6 w-6 p-0 hover:bg-transparent"
              onClick={(e) => e.stopPropagation()}
              aria-label="More options"
            >
              <IconMore className="h-3.5 w-3.5 text-muted-foreground" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-36 text-xs">
            <DropdownMenuItem onSelect={(e) => { e.preventDefault(); onRename() }} className="text-xs py-1.5">
              <IconEdit className="mr-2 h-3 w-3" /> Rename
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={(e) => { e.preventDefault(); onStar(!c.starred) }} className="text-xs py-1.5">
              <IconStar className="mr-2 h-3 w-3" filled={c.starred} />
              {c.starred ? 'Unstar' : 'Star'}
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={(e) => { e.preventDefault(); onPin(!c.pinned) }} className="text-xs py-1.5">
              <IconPin className="mr-2 h-3 w-3" />
              {c.pinned ? 'Unpin' : 'Pin to top'}
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={(e) => { e.preventDefault(); onExport('md') }} className="text-xs py-1.5">
              <IconDownload className="mr-2 h-3 w-3" /> Export MD
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={(e) => { e.preventDefault(); onExport('json') }} className="text-xs py-1.5">
              <IconDownload className="mr-2 h-3 w-3" /> Export JSON
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={(e) => { e.preventDefault(); onArchive(!c.archived) }} className="text-xs py-1.5">
              <IconFolder className="mr-2 h-3 w-3" />
              {c.archived ? 'Restore' : 'Archive'}
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={(e) => { e.preventDefault(); onDelete() }} className="text-destructive focus:text-destructive text-xs py-1.5">
              <IconTrash className="mr-2 h-3 w-3" /> Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  )
}
