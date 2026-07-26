'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { cn, Button } from '@sloughgpt/strui'
import { IconPlus, IconStar, IconPin, IconChat, IconX, IconSearch, IconFolder, IconSort, IconCheck, IconChevronLeft } from '@sloughgpt/strui'
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from '@sloughgpt/strui'
import type { Conversation } from '@/lib/session-controller'
import { formatDate, truncateMessage } from '@/lib/conversations-utils'

interface ConversationSidebarProps {
  conversations: Conversation[]
  currentConversationId?: string
  onLoadConversation: (id: string) => void
  onNewChat: () => void
  onDeleteConversation?: (id: string) => void
  onStarConversation?: (id: string, starred: boolean) => void
  onPinConversation?: (id: string, pinned: boolean) => void
  onArchiveConversation?: (id: string, archived: boolean) => void
  archivedCount?: number
  onRenameConversation?: (id: string, name: string) => void
  open: boolean
  onClose: () => void
  collapsed?: boolean
  onToggleCollapse?: () => void
}

function SidebarContent({
  conversations,
  currentConversationId,
  onLoadConversation,
  onNewChat,
  onDeleteConversation,
  onStarConversation,
  onPinConversation,
  onArchiveConversation,
  archivedCount,
  onRenameConversation,
  onClose,
  isDrawer,
  onToggleCollapse,
}: {
  conversations: Conversation[]
  currentConversationId?: string
  onLoadConversation: (id: string) => void
  onNewChat: () => void
  onDeleteConversation?: (id: string) => void
  onStarConversation?: (id: string, starred: boolean) => void
  onPinConversation?: (id: string, pinned: boolean) => void
  onArchiveConversation?: (id: string, archived: boolean) => void
  archivedCount?: number
  onRenameConversation?: (id: string, name: string) => void
  onClose?: () => void
  isDrawer?: boolean
  onToggleCollapse?: () => void
}) {
  const SORT_KEY = 'sloughgpt:sidebar-sort'

  const [search, setSearch] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<{ id: string; name: string } | null>(null)
  const [sortMode, setSortMode] = useState<'updated' | 'name' | 'messages'>(() => {
    if (typeof window === 'undefined') return 'updated'
    const saved = localStorage.getItem(SORT_KEY)
    if (saved === 'name' || saved === 'messages') return saved
    return 'updated'
  })
  const [sortOpen, setSortOpen] = useState(false)

  useEffect(() => {
    localStorage.setItem(SORT_KEY, sortMode)
  }, [sortMode])

  const sorted = useMemo(() => {
    return [...conversations].sort((a, b) => {
      if (sortMode === 'name') {
        return (a.name || '').localeCompare(b.name || '')
      }
      if (sortMode === 'messages') {
        return (b.message_count ?? b.messages?.length ?? 0) - (a.message_count ?? a.messages?.length ?? 0)
      }
      return new Date(b.updated_at || b.updatedAt || 0).getTime() - new Date(a.updated_at || a.updatedAt || 0).getTime()
    })
  }, [conversations, sortMode])

  const q = search.toLowerCase().trim()
  const filtered = useMemo(() => {
    if (!q) return sorted
    return sorted.filter(c =>
      c.name?.toLowerCase().includes(q) ||
      c.messages?.some(m => m.content?.toLowerCase().includes(q))
    )
  }, [sorted, q])

  const handleDelete = (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    const conv = sorted.find(c => c.id === id)
    setDeleteTarget({ id, name: conv?.name || 'this conversation' })
  }

  const confirmDelete = () => {
    if (deleteTarget) {
      onDeleteConversation?.(deleteTarget.id)
      setDeleteTarget(null)
    }
  }

  const starred = filtered.filter(c => c.starred).slice(0, 10)
  const unstarred = filtered.filter(c => !c.starred)

  function recencyGroup(dateStr: string | undefined): string {
    if (!dateStr) return 'Older'
    const diff = Date.now() - new Date(dateStr).getTime()
    const days = diff / 86400000
    if (days < 1) return 'Today'
    if (days < 2) return 'Yesterday'
    if (days < 7) return 'Last 7 days'
    return 'Older'
  }

  const recencyGroups = useMemo(() => {
    const groups: { label: string; conversations: Conversation[] }[] = []
    const seen = new Set<string>()
    for (const c of unstarred) {
      const label = recencyGroup(c.updated_at || c.updatedAt)
      if (!seen.has(label)) {
        seen.add(label)
        groups.push({ label, conversations: [] })
      }
      const group = groups.find(g => g.label === label)!
      if (group.conversations.length < 15) group.conversations.push(c)
    }
    const order = ['Today', 'Yesterday', 'Last 7 days', 'Older']
    return groups.sort((a, b) => order.indexOf(a.label) - order.indexOf(b.label))
  }, [unstarred])

  const handleSelect = (id: string) => {
    onLoadConversation(id)
    onClose?.()
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between px-3 py-2 border-b border-border/50 shrink-0">
        <div className="flex items-center gap-1">
          {!isDrawer && onToggleCollapse && (
            <button
              onClick={onToggleCollapse}
              className="h-5 w-5 flex items-center justify-center rounded hover:bg-muted/60 text-muted-foreground hover:text-foreground transition-colors shrink-0"
              aria-label="Collapse sidebar"
              title="Collapse"
            >
              <IconChevronLeft className="h-3 w-3" />
            </button>
          )}
          <span className="text-xs font-medium text-foreground">Conversations</span>
          <div className="relative">
            <button
              onClick={() => setSortOpen(!sortOpen)}
              className={cn(
                "h-5 w-5 flex items-center justify-center rounded hover:bg-muted/60 transition-colors",
                sortMode !== 'updated' ? "text-primary" : "text-muted-foreground/60"
              )}
              aria-label="Sort conversations"
              title={`Sort by: ${sortMode === 'updated' ? 'Last updated' : sortMode === 'name' ? 'Name' : 'Messages'}`}
            >
              <IconSort className="h-3 w-3" />
            </button>
            {sortOpen && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setSortOpen(false)} />
                <div className="absolute left-0 top-full mt-1 z-50 bg-background border border-border/60 rounded-lg shadow-lg py-1 min-w-[140px]">
                  {([
                    { value: 'updated', label: 'Last updated' },
                    { value: 'name', label: 'Name' },
                    { value: 'messages', label: 'Message count' },
                  ] as const).map(opt => (
                    <button
                      key={opt.value}
                      onClick={() => { setSortMode(opt.value); setSortOpen(false) }}
                      className={cn(
                        "w-full text-left px-3 py-1.5 text-xs hover:bg-muted/40 transition-colors flex items-center gap-2",
                        sortMode === opt.value && "text-primary font-medium"
                      )}
                    >
                      {sortMode === opt.value && <IconCheck className="h-3 w-3 shrink-0" />}
                      <span className={sortMode !== opt.value ? "ml-5" : ""}>{opt.label}</span>
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => { onNewChat(); onClose?.() }}
            className="text-xs h-6 px-2 gap-1 text-muted-foreground hover:text-foreground"
          >
            <IconPlus className="h-3 w-3" />
            New
          </Button>
          {isDrawer && onClose && (
            <button
              onClick={onClose}
              className="flex items-center justify-center h-6 w-6 rounded-md hover:bg-muted/60 text-muted-foreground hover:text-foreground transition-colors"
              aria-label="Close sidebar"
            >
              <IconX className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>

      {conversations.length > 0 && (
        <div className="relative px-2 pt-1.5 pb-1">
          <IconSearch className="absolute left-3.5 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground/60 pointer-events-none" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search conversations…"
            className="w-full h-7 rounded-md bg-muted/50 pl-7 pr-2 text-xs outline-none placeholder:text-muted-foreground/40 focus:bg-muted/80 focus:ring-1 focus:ring-primary/30 transition-colors"
            aria-label="Search conversations"
          />
        </div>
      )}
      <div className="flex-1 overflow-y-auto overscroll-contain px-1.5 py-1">
        {conversations.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-6 px-3 gap-2">
            <IconChat className="h-5 w-5 text-muted-foreground/30" />
            <p className="text-[11px] text-muted-foreground/60 text-center leading-relaxed">
              No conversations yet
            </p>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => { onNewChat(); onClose?.() }}
              className="text-[11px] h-6 px-2 gap-1 text-primary hover:text-primary/80"
            >
              <IconPlus className="h-3 w-3" />
              Start chatting
            </Button>
          </div>
        ) : q && filtered.length === 0 ? (
          <div className="text-center py-6 px-3 space-y-3">
            <p className="text-xs text-muted-foreground">
              No conversations match &ldquo;{q}&rdquo;
            </p>
            <button
              onClick={() => window.dispatchEvent(new CustomEvent('search-conversations'))}
              className="text-xs text-primary underline underline-offset-2 hover:text-primary/80 transition-colors"
            >
              Search all conversations
            </button>
          </div>
        ) : (
          <>
            {starred.length > 0 && (
              <div className="mb-1">
                <p className="text-[10px] font-medium text-muted-foreground/60 uppercase tracking-wider px-2 py-1">
                  Starred
                </p>
                <div className="space-y-0.5">
                    {starred.map(c => (
                      <ConvRow
                        key={c.id}
                        conversation={c}
                        isActive={c.id === currentConversationId}
                        onSelect={() => handleSelect(c.id)}
                        onDelete={(e) => handleDelete(e, c.id)}
                        onStar={(e) => { e.stopPropagation(); onStarConversation?.(c.id, !c.starred) }}
                        onPin={(e) => { e.stopPropagation(); onPinConversation?.(c.id, !c.pinned) }}
                        onArchive={(e) => { e.stopPropagation(); onArchiveConversation?.(c.id, true) }}
                        onRename={(name) => onRenameConversation?.(c.id, name)}
                      />
                    ))}
                  </div>
                </div>
              )}

              {recencyGroups.map(group => (
                <div key={group.label}>
                  <p className="text-[10px] font-medium text-muted-foreground/60 uppercase tracking-wider px-2 py-1">
                    {group.label}
                  </p>
                  <div className="space-y-0.5">
                    {group.conversations.map(c => (
                      <ConvRow
                        key={c.id}
                        conversation={c}
                        isActive={c.id === currentConversationId}
                        onSelect={() => handleSelect(c.id)}
                        onDelete={(e) => handleDelete(e, c.id)}
                        onStar={(e) => { e.stopPropagation(); onStarConversation?.(c.id, !c.starred) }}
                        onPin={(e) => { e.stopPropagation(); onPinConversation?.(c.id, !c.pinned) }}
                        onArchive={(e) => { e.stopPropagation(); onArchiveConversation?.(c.id, true) }}
                        onRename={(name) => onRenameConversation?.(c.id, name)}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </>
          )}
        </div>

      {/* Delete confirmation */}
      <AlertDialog open={deleteTarget !== null} onOpenChange={(open) => { if (!open) setDeleteTarget(null) }}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete &ldquo;{deleteTarget?.name}&rdquo;?</AlertDialogTitle>
            <AlertDialogDescription>This cannot be undone.</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={confirmDelete} className="bg-destructive text-destructive-foreground">Delete</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}

export function ConversationSidebar({ collapsed, onToggleCollapse, ...props }: ConversationSidebarProps) {
  const { open, onClose } = props

  return (
    <>
      {/* Desktop: collapsible aside */}
      <aside
        className={cn(
          "hidden lg:flex shrink-0 flex-col border-r border-border/40 bg-background/80 transition-all duration-200 ease-in-out overflow-hidden",
          collapsed ? "w-0 border-r-0" : "w-64"
        )}
      >
        <div className={cn("min-w-[256px] h-full", collapsed && "pointer-events-none")}>
          <SidebarContent {...props} isDrawer={false} onToggleCollapse={onToggleCollapse} />
        </div>
      </aside>

      {/* Mobile: drawer overlay */}
      {open && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div
            className="absolute inset-0 bg-black/40 backdrop-blur-sm"
            onClick={onClose}
          />
          <aside className="absolute inset-y-0 left-0 w-72 flex flex-col bg-background shadow-xl">
            <SidebarContent {...props} isDrawer={true} />
          </aside>
        </div>
      )}
    </>
  )
}

function ConvRow({
  conversation: c,
  isActive,
  onSelect,
  onDelete,
  onStar,
  onPin,
  onArchive,
  onRename,
}: {
  conversation: Conversation
  isActive: boolean
  onSelect: () => void
  onDelete?: (e: React.MouseEvent) => void
  onStar?: (e: React.MouseEvent) => void
  onPin?: (e: React.MouseEvent) => void
  onArchive?: (e: React.MouseEvent) => void
  onRename?: (name: string) => void
}) {
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

  return (
    <div
      className={cn(
        "group flex items-start gap-2 rounded-md px-2 py-1.5 cursor-pointer transition-colors",
        isActive ? "bg-primary/10" : "hover:bg-muted/40"
      )}
      onClick={!editing ? onSelect : undefined}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter' && !editing) { e.preventDefault(); onSelect() } }}
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1">
          <button
            onClick={onPin}
            className="h-4 w-4 flex items-center justify-center rounded hover:bg-muted/60 shrink-0 -ml-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
            aria-label={c.pinned ? 'Unpin' : 'Pin'}
          >
            <IconPin className={cn("h-2.5 w-2.5", c.pinned ? "text-primary" : "text-muted-foreground/40")} />
          </button>
          <button
            onClick={onStar}
            className="h-4 w-4 flex items-center justify-center rounded hover:bg-muted/60 shrink-0 -ml-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
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
              className="text-xs font-medium truncate text-foreground"
              onDoubleClick={(e) => { e.stopPropagation(); setEditValue(c.name); setEditing(true) }}
            >
              {c.name}
            </p>
          )}
        </div>
        {lastMsg && !editing && (
          <p className="text-[11px] text-muted-foreground/70 mt-0.5 line-clamp-1">
            {truncateMessage(lastMsg, 36)}
          </p>
        )}
        {!editing && (
          <div className="flex items-center gap-1 mt-0.5">
            <IconChat className="h-2.5 w-2.5 text-muted-foreground/50 shrink-0" />
            <span className="text-[10px] text-muted-foreground/50">
              {msgCount} msgs · {formatDate(c.updated_at || c.updatedAt)}
            </span>
          </div>
        )}
      </div>
      <div className="flex items-center gap-0.5 shrink-0 mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
        {onArchive && !editing && (
          <button
            onClick={onArchive}
            className="h-5 w-5 flex items-center justify-center rounded hover:bg-muted/60 text-muted-foreground hover:text-warning"
            aria-label="Archive"
          >
            <IconFolder className="h-3 w-3" />
          </button>
        )}
        {onDelete && !editing && (
          <button
            onClick={onDelete}
            className="h-5 w-5 flex items-center justify-center rounded hover:bg-muted/60 text-muted-foreground hover:text-destructive"
            aria-label={`Delete ${c.name}`}
          >
            <IconX className="h-3 w-3" />
          </button>
        )}
      </div>
    </div>
  )
}
