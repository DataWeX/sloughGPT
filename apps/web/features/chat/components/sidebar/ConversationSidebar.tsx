'use client'

import { useEffect, useMemo, useRef, useState, useDeferredValue, useCallback } from 'react'
import { cn, Button } from '@sloughgpt/strui'
import { IconPlus, IconStar, IconPin, IconChat, IconX, IconSearch, IconFolder, IconSort, IconCheck, IconChevronLeft, IconChevronRight, IconDot, IconDotOutline } from '@sloughgpt/strui'
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from '@sloughgpt/strui'
import type { Conversation } from '@/lib/session-controller'
import { ConvRow } from './ConvRow'
import { downloadJson, downloadMarkdown } from '@/lib/download-utils'
import { MS_PER_DAY } from '@/lib/format-bytes'
import { chatDB } from '@/lib/db'

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
  onToggleUnreadConversation?: (id: string, unread: boolean) => void
  onDuplicateConversation?: (id: string, name: string) => void
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
  onToggleUnreadConversation,
  onDuplicateConversation,
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
  onToggleUnreadConversation?: (id: string, unread: boolean) => void
  onDuplicateConversation?: (id: string, name: string) => void
  onClose?: () => void
  isDrawer?: boolean
  onToggleCollapse?: () => void
}) {
  const SORT_KEY = 'sloughgpt:sidebar-sort'

  const [search, setSearch] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<{ id: string; name: string } | null>(null)
  const [sortMode, setSortMode] = useState<'updated' | 'name' | 'messages'>('updated')
  const deferredSearch = useDeferredValue(search)

  useEffect(() => {
    let cancelled = false
    chatDB.getKV<string>(SORT_KEY).then(saved => {
      if (!cancelled && (saved === 'name' || saved === 'messages')) {
        setSortMode(saved)
      }
    })
    return () => { cancelled = true }
  }, [])
  const unreadCount = useMemo(() => conversations.filter(c => c.unread).length, [conversations])
  const [archivedExpanded, setArchivedExpanded] = useState(false)
  const [archivedConversations, setArchivedConversations] = useState<Conversation[]>([])
  const [archivedLoading, setArchivedLoading] = useState(false)
  const [sortOpen, setSortOpen] = useState(false)

  useEffect(() => {
    chatDB.setKV(SORT_KEY, sortMode).catch(() => {})
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

  const q = deferredSearch.toLowerCase().trim()
  const filtered = useMemo(() => {
    if (!q) return sorted
    return sorted.filter(c =>
      c.name?.toLowerCase().includes(q) ||
      c.messages?.some(m => m.content?.toLowerCase().includes(q))
    )
  }, [sorted, q])

  const handleDelete = useCallback((e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    const conv = sorted.find(c => c.id === id)
    setDeleteTarget({ id, name: conv?.name || 'this conversation' })
  }, [sorted])

  const confirmDelete = () => {
    if (deleteTarget) {
      onDeleteConversation?.(deleteTarget.id)
      setDeleteTarget(null)
    }
  }

  const handleSelect = useCallback((id: string) => {
    onLoadConversation(id)
    onClose?.()
  }, [onLoadConversation, onClose])

  const handleStar = useCallback((id: string, starred: boolean) => {
    onStarConversation?.(id, starred)
  }, [onStarConversation])

  const handlePin = useCallback((id: string, pinned: boolean) => {
    onPinConversation?.(id, pinned)
  }, [onPinConversation])

  const handleArchive = useCallback((id: string, archive: boolean) => {
    onArchiveConversation?.(id, archive)
  }, [onArchiveConversation])

  const handleRename = useCallback((id: string, name: string) => {
    onRenameConversation?.(id, name)
  }, [onRenameConversation])

  const handleDuplicate = useCallback((id: string, name: string) => {
    onDuplicateConversation?.(id, name)
  }, [onDuplicateConversation])

  const handleToggleUnread = useCallback((id: string, unread: boolean) => {
    onToggleUnreadConversation?.(id, unread)
  }, [onToggleUnreadConversation])

  const starred = useMemo(() => filtered.filter(c => c.starred).slice(0, 10), [filtered])
  const unstarred = useMemo(() => filtered.filter(c => !c.starred), [filtered])
  const pinned = useMemo(() => unstarred.filter(c => c.pinned), [unstarred])
  const unpinned = useMemo(() => unstarred.filter(c => !c.pinned), [unstarred])

  function recencyGroup(dateStr: string | undefined): string {
    if (!dateStr) return 'Older'
    const diff = Date.now() - new Date(dateStr).getTime()
    const days = diff / MS_PER_DAY
    if (days < 1) return 'Today'
    if (days < 2) return 'Yesterday'
    if (days < 7) return 'Last 7 days'
    return 'Older'
  }

  const recencyGroups = useMemo(() => {
    const groups: { label: string; conversations: Conversation[] }[] = []
    const seen = new Set<string>()
    for (const c of unpinned) {
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
  }, [unpinned])

  const handleExport = useCallback((e: React.MouseEvent, c: Conversation, format: 'json' | 'markdown' = 'json') => {
    e.stopPropagation()
    const safeName = (c.name || 'conversation').replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 50)
    if (format === 'markdown') {
      const messages = c.messages || []
      const md = [
        `# ${c.name}`,
        '',
        `*Exported ${new Date().toLocaleString()}*`,
        '',
        ...messages.map(m => {
          const role = m.role === 'user' ? '**You**' : '**Assistant**'
          const ts = m.timestamp ? ` _${new Date(m.timestamp).toLocaleString()}_` : ''
          return `### ${role}${ts}\n\n${m.content}`
        }),
      ].join('\n\n---\n\n')
      downloadMarkdown(md, `${safeName}.md`)
    } else {
      const data = {
        id: c.id,
        name: c.name,
        messages: c.messages || [],
        created_at: c.created_at || c.createdAt,
        updated_at: c.updated_at || c.updatedAt,
      }
      downloadJson(data, `${safeName}.json`)
    }
  }, [])

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center justify-between px-3 py-2 border-b border-border/50 shrink-0">
        <div className="flex items-center gap-1">
          {!isDrawer && onToggleCollapse && (
            <button
              onClick={onToggleCollapse}
              className="h-5 w-5 flex items-center justify-center rounded hover:bg-muted/60 text-muted-foreground hover:text-foreground transition-colors shrink-0"
              aria-label="Collapse sidebar"
              title="Collapse"
            >
              <IconChevronLeft className="h-4 w-4" />
            </button>
          )}
          <span className="text-xs font-medium text-foreground">Conversations</span>
          {unreadCount > 0 && (
            <>
              <span className="h-4 min-w-[16px] flex items-center justify-center rounded-full bg-primary text-primary-foreground text-[9px] font-bold px-1">
                {unreadCount}
              </span>
              <button
                onClick={() => { conversations.filter(c => c.unread).forEach(c => onToggleUnreadConversation?.(c.id, false)) }}
                className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                aria-label="Mark all as read"
              >
                Mark read
              </button>
            </>
          )}
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
              <IconSort className="h-4 w-4" />
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
                      {sortMode === opt.value && <IconCheck className="h-4 w-4 shrink-0" />}
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
            <IconPlus className="h-4 w-4" />
            New
          </Button>
          {isDrawer && onClose && (
            <button
              onClick={onClose}
              className="flex items-center justify-center h-6 w-6 rounded-md hover:bg-muted/60 text-muted-foreground hover:text-foreground transition-colors"
              aria-label="Close sidebar"
            >
              <IconX className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {conversations.length > 0 && (
        <div className="relative px-2 pt-1.5 pb-1">
          <IconSearch className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground/60 pointer-events-none" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search conversations…"
            className="w-full h-8 rounded-md bg-muted/50 pl-7 pr-2 text-xs outline-none placeholder:text-muted-foreground/40 focus:bg-muted/80 focus:ring-2 focus:ring-primary/30 transition-colors"
            aria-label="Search conversations"
          />
        </div>
      )}
      <div className="flex-1 overflow-y-auto overscroll-contain px-1.5 py-1 pb-4">
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
              <IconPlus className="h-4 w-4" />
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
                <p className="text-xs font-medium text-muted-foreground/60 uppercase tracking-wider px-2 py-1 flex items-center gap-1.5">
                  Starred
                  <span className="text-muted-foreground/30 font-mono">{starred.length}</span>
                </p>
                <div className="space-y-0.5">
                    {starred.map(c => (
                      <ConvRow
                        key={c.id}
                        conversation={c}
                        isActive={c.id === currentConversationId}
                        onSelect={handleSelect}
                        onDelete={handleDelete}
                        onStar={(_e, id, starred) => handleStar(id, starred)}
                        onPin={(_e, id, pinned) => handlePin(id, pinned)}
                        onArchive={(_e, id, archive) => handleArchive(id, archive)}
                        onRename={handleRename}
                        onExport={handleExport}
                        onDuplicate={(_e, id, name) => handleDuplicate(id, name)}
                        onToggleUnread={(_e, id, unread) => handleToggleUnread(id, unread)}
                        searchQuery={q}
                      />
                    ))}
                  </div>
                </div>
              )}

              {pinned.length > 0 && (
                <div className="mb-1">
                  <p className="text-xs font-medium text-muted-foreground/60 uppercase tracking-wider px-2 py-1 flex items-center gap-1.5">
                    Pinned
                    <span className="text-muted-foreground/30 font-mono">{pinned.length}</span>
                  </p>
                  <div className="space-y-0.5">
                    {pinned.map(c => (
                      <ConvRow
                        key={c.id}
                        conversation={c}
                        isActive={c.id === currentConversationId}
                        onSelect={handleSelect}
                        onDelete={handleDelete}
                        onStar={(_e, id, starred) => handleStar(id, starred)}
                        onPin={(_e, id, pinned) => handlePin(id, pinned)}
                        onArchive={(_e, id, archive) => handleArchive(id, archive)}
                        onRename={handleRename}
                        onExport={handleExport}
                        onDuplicate={(_e, id, name) => handleDuplicate(id, name)}
                        onToggleUnread={(_e, id, unread) => handleToggleUnread(id, unread)}
                        searchQuery={q}
                      />
                    ))}
                  </div>
                </div>
              )}

              {recencyGroups.map(group => (
                <div key={group.label}>
                  <p className="text-xs font-medium text-muted-foreground/60 uppercase tracking-wider px-2 py-1 flex items-center gap-1.5">
                    {group.label}
                    <span className="text-muted-foreground/30 font-mono">{group.conversations.length}</span>
                  </p>
                  <div className="space-y-0.5">
                    {group.conversations.map(c => (
                      <ConvRow
                        key={c.id}
                        conversation={c}
                        isActive={c.id === currentConversationId}
                        onSelect={handleSelect}
                        onDelete={handleDelete}
                        onStar={(_e, id, starred) => handleStar(id, starred)}
                        onPin={(_e, id, pinned) => handlePin(id, pinned)}
                        onArchive={(_e, id, archive) => handleArchive(id, archive)}
                        onRename={handleRename}
                        onExport={handleExport}
                        onDuplicate={(_e, id, name) => handleDuplicate(id, name)}
                        onToggleUnread={(_e, id, unread) => handleToggleUnread(id, unread)}
                        searchQuery={q}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </>
          )}

          {(archivedCount ?? 0) > 0 && (
            <div className="mt-2 border-t border-border/30 pt-2">
              <button
                onClick={async () => {
                  const next = !archivedExpanded
                  setArchivedExpanded(next)
                  if (next && archivedConversations.length === 0) {
                    setArchivedLoading(true)
                    try {
                      const { sessionController } = await import('@/lib/session-controller')
                      const sessions = await sessionController.list(true)
                      setArchivedConversations(sessions.map(s => ({
                        id: s.id,
                        name: s.name,
                        session_id: s.id,
                        created_at: s.created_at,
                        updated_at: s.updated_at,
                        pinned: s.pinned ?? false,
                        starred: s.starred ?? false,
                        archived: true,
                        message_count: s.messages?.length ?? 0,
                      })))
                    } catch { /* ignore */ }
                    setArchivedLoading(false)
                  }
                }}
                className="flex items-center gap-2 w-full px-2 py-1 text-left group"
                aria-expanded={archivedExpanded}
              >
                <IconFolder className={cn("h-4 w-4 transition-colors", archivedExpanded ? "text-primary" : "text-muted-foreground/60")} />
                <span className="text-xs font-medium text-muted-foreground/60 uppercase tracking-wider flex-1">
                  Archived
                </span>
                <span className="text-xs text-muted-foreground/40 font-mono">
                  {archivedCount}
                </span>
                <IconChevronRight className={cn("h-4 w-4 text-muted-foreground/40 transition-transform", archivedExpanded && "rotate-90")} />
              </button>
              {archivedExpanded && (
                <div className="space-y-0.5 mt-1">
                  {archivedLoading ? (
                    <div className="text-[11px] text-muted-foreground/50 px-2 py-2">Loading…</div>
                  ) : archivedConversations.length === 0 ? (
                    <div className="text-[11px] text-muted-foreground/50 px-2 py-2">No archived conversations</div>
                  ) : (
                    archivedConversations.map(c => (
                      <ConvRow
                        key={c.id}
                        conversation={c}
                        isActive={c.id === currentConversationId}
                        onSelect={handleSelect}
                        onDelete={handleDelete}
                        onStar={(_e, id, starred) => handleStar(id, starred)}
                        onPin={(_e, id, pinned) => handlePin(id, pinned)}
                        onArchive={(_e, id, _archive) => handleArchive(id, false)}
                        onRename={handleRename}
                        onExport={handleExport}
                        onDuplicate={(_e, id, name) => handleDuplicate(id, name)}
                        onToggleUnread={(_e, id, unread) => handleToggleUnread(id, unread)}
                        searchQuery={q}
                      />
                    ))
                  )}
                </div>
              )}
            </div>
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
          collapsed ? "w-0 border-r-0" : "w-[var(--conv-sidebar-width)]"
        )}
      >
        <div className={cn("min-w-[var(--conv-sidebar-width)] h-full", collapsed && "pointer-events-none")}>
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
          <aside className="absolute inset-y-0 left-0 w-[var(--conv-sidebar-width)] flex flex-col bg-background shadow-xl">
            <SidebarContent {...props} isDrawer={true} />
          </aside>
        </div>
      )}
    </>
  )
}

