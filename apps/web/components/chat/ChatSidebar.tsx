'use client'

import { useState, useCallback, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { IconChat, IconPin, IconStar, IconMore, IconEdit, IconCopy, IconTrash } from '@/components/ui'
import { ChatSidebarHeader } from './ChatSidebarHeader'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { cn } from '@/lib/cn'
import type { Conversation } from '@/lib/session-controller'

interface ChatSidebarProps {
  conversations: Conversation[]
  currentConversationId?: string
  onLoadConversation: (id: string) => void
  onDeleteConversation: (id: string) => void
  onStarConversation?: (id: string, starred: boolean) => void
  onPinConversation?: (id: string, pinned: boolean) => void
  onRenameConversation?: (id: string, name: string) => void
  onDuplicateConversation?: (id: string) => void
  onNewChat: () => void
  collapsed?: boolean
  onToggleCollapse?: () => void
  onOpenConversationSearch?: () => void
}

function formatDate(dateStr: string | undefined): string {
  if (!dateStr) return ''
  const date = new Date(dateStr)
  if (isNaN(date.getTime())) return ''
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)
  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m`
  if (diffHours < 24) return `${diffHours}h`
  if (diffDays < 7) return `${diffDays}d`
  return date.toLocaleDateString()
}

function truncateMessage(content: string, maxLen = 45): string {
  if (!content) return 'Empty conversation'
  const firstLine = content.split('\n')[0]
  return firstLine.length > maxLen ? firstLine.slice(0, maxLen) + '…' : firstLine
}

function ConvItem({
  c,
  isActive,
  onAction,
  onClick,
  onRename,
}: {
  c: Conversation
  isActive: boolean
  onAction: (action: string, conv: Conversation) => void
  onClick: () => void
  onRename: (id: string, name: string) => void
}) {
  const [menuOpen, setMenuOpen] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editValue, setEditValue] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const msgCount = c.messages?.length ?? c.message_count ?? 0

  const handleEditKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      if (editValue.trim()) onRename(c.id, editValue.trim())
      setIsEditing(false)
    } else if (e.key === 'Escape') {
      setIsEditing(false)
    }
  }

  if (isEditing) {
    return (
      <div className="flex items-center rounded-md border border-primary/40 bg-card/60 py-2 pr-1 pl-2">
        <Input
          ref={inputRef}
          value={editValue}
          onChange={e => setEditValue(e.target.value)}
          onKeyDown={handleEditKeyDown}
          onBlur={() => {
            if (editValue.trim()) onRename(c.id, editValue.trim())
            setIsEditing(false)
          }}
          className="h-6 text-xs py-0 px-1.5"
          autoFocus
        />
      </div>
    )
  }

  return (
    <div
      className={cn(
        "group relative flex items-start rounded-md transition-all cursor-pointer border border-border/40 bg-card/60",
        isActive ? "bg-secondary/50 border-primary/30" : "hover:bg-secondary/30 hover:border-border/60"
      )}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick() } }}
      aria-label={c.name || 'Conversation'}
    >
      <div className="flex-1 min-w-0 py-2 pr-1 pl-2">
        <div className="flex items-center gap-1.5">
          {c.pinned && <IconPin className="h-2.5 w-2.5 text-primary shrink-0" />}
          {c.starred && <IconStar className="h-2.5 w-2.5 text-warning shrink-0" filled />}
          <p className="text-xs font-medium truncate text-foreground">{c.name}</p>
        </div>
        <p className="text-xs text-muted-foreground/70 mt-0.5 line-clamp-1">
          {truncateMessage(c.messages?.[c.messages.length - 1]?.content || '')}
        </p>
        <div className="flex items-center gap-1 mt-0.5">
          <IconChat className="h-2.5 w-2.5 text-muted-foreground/60 shrink-0" />
          <p className="text-xs text-muted-foreground/70">
            {msgCount} · {formatDate(c.updated_at || c.updatedAt)}
          </p>
        </div>
      </div>

      <div className={cn("transition-opacity pr-1 pt-1.5", menuOpen ? "opacity-100" : "opacity-0 group-hover:opacity-100")}>
        <DropdownMenu open={menuOpen} onOpenChange={setMenuOpen}>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="icon-sm"
              className="h-5 w-5 p-0 hover:bg-transparent"
              onClick={(e) => e.stopPropagation()}
            >
              <IconMore className="h-3.5 w-3.5 text-muted-foreground" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-36 text-xs">
            <DropdownMenuItem onSelect={() => { setIsEditing(true); setEditValue(c.name) }} className="text-xs py-1.5">
              <IconEdit className="mr-2 h-3 w-3" />
              Rename
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => onAction('pin', c)} className="text-xs py-1.5">
              <IconPin className="mr-2 h-3 w-3" />
              {c.pinned ? 'Unpin' : 'Pin to top'}
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => onAction('star', c)} className="text-xs py-1.5">
              <IconStar className="mr-2 h-3 w-3" filled={c.starred} />
              {c.starred ? 'Remove from Starred' : 'Add to Starred'}
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => onAction('duplicate', c)} className="text-xs py-1.5">
              <IconCopy className="mr-2 h-3 w-3" />
              Duplicate
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={() => onAction('delete', c)} className="text-destructive focus:text-destructive text-xs py-1.5">
              <IconTrash className="mr-2 h-3 w-3" />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  )
}

export function ChatSidebar({
  conversations,
  currentConversationId,
  onLoadConversation,
  onDeleteConversation,
  onStarConversation,
  onPinConversation,
  onRenameConversation,
  onDuplicateConversation,
  onNewChat,
  collapsed = false,
  onToggleCollapse,
  onOpenConversationSearch,
}: ChatSidebarProps) {
  const [searchQuery, setSearchQuery] = useState('')

  const starredCount = conversations.filter(c => c.starred).length

  const filtered = [...conversations.filter(c =>
    c.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    c.messages?.some(m => m.content.toLowerCase().includes(searchQuery.toLowerCase()))
  )].sort((a, b) => {
    if (a.pinned && !b.pinned) return -1
    if (!a.pinned && b.pinned) return 1
    if (a.starred && !b.starred) return -1
    if (!a.starred && b.starred) return 1
    return new Date(b.updated_at || b.updatedAt || 0).getTime() - new Date(a.updated_at || a.updatedAt || 0).getTime()
  })

  const handleAction = useCallback((action: string, conv: Conversation) => {
    switch (action) {
      case 'star': onStarConversation?.(conv.id, !conv.starred); break
      case 'pin': onPinConversation?.(conv.id, !conv.pinned); break
      case 'duplicate': onDuplicateConversation?.(conv.id); break
      case 'delete': onDeleteConversation?.(conv.id); break
    }
  }, [onStarConversation, onPinConversation, onDuplicateConversation, onDeleteConversation])

  return (
    <aside className="hidden lg:flex flex-col w-56 sm:w-64 lg:w-72 border-r border-border bg-background h-full overflow-hidden">
      <ChatSidebarHeader
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        collapsed={collapsed}
        onToggleCollapse={onToggleCollapse}
        onNewChat={onNewChat}
        starredCount={starredCount}
        onOpenConversationSearch={onOpenConversationSearch}
        className="sticky top-0 z-10"
      />

      <div className="flex-1 min-h-0 overflow-y-auto p-1.5 space-y-0.5">
        {filtered.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-6 px-2">
            {searchQuery ? 'No matches' : 'No conversations yet'}
          </p>
        ) : filtered.map(c => (
          <ConvItem key={c.id} c={c} isActive={c.id === currentConversationId}
            onAction={handleAction} onClick={() => c.id !== currentConversationId && onLoadConversation?.(c.id)}
            onRename={(id, name) => onRenameConversation?.(id, name)} />
        ))}
      </div>
    </aside>
  )
}
