'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import { useRouter } from 'next/navigation'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { SearchInput } from '@/components/ui/input'
import {
  IconSearch, IconStar, IconPin, IconChat, IconTrash, IconEdit,
  IconDownload, IconPlus, IconMore,
} from '@/components/ui'
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { cn } from '@/lib/cn'
import { sessionController, type Conversation } from '@/lib/session-controller'
import { useToastStore } from '@/lib/toast-store'

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

function truncateMessage(content: string, maxLen = 60): string {
  if (!content) return ''
  const firstLine = content.split('\n')[0]
  return firstLine.length > maxLen ? firstLine.slice(0, maxLen) + '…' : firstLine
}

export default function ConversationsPage() {
  const router = useRouter()
  const addToast = useToastStore(s => s.addToast)
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [deleting, setDeleting] = useState(false)

  const fetchAll = useCallback(async () => {
    setLoading(true)
    try {
      const sessions = await sessionController.list()
      const mapped: Conversation[] = sessions.map(s => ({
        id: s.id,
        name: s.name,
        session_id: s.id,
        created_at: s.created_at,
        updated_at: s.updated_at,
        pinned: s.pinned || false,
        starred: s.starred || false,
        message_count: s.messages?.length || 0,
        messages: (s.messages || []).map(m => ({
          id: m.id || '',
          role: m.role,
          content: m.content,
          timestamp: m.timestamp,
        })),
      }))
      setConversations(mapped)
    } catch (e) {
      addToast('Failed to load conversations', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => { fetchAll() }, [fetchAll])

  const filtered = useMemo(() => {
    if (!search.trim()) return conversations
    const q = search.toLowerCase()
    return conversations.filter(c =>
      c.name.toLowerCase().includes(q) ||
      c.messages?.some(m => m.content.toLowerCase().includes(q))
    )
  }, [conversations, search])

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      if (a.pinned && !b.pinned) return -1
      if (!a.pinned && b.pinned) return 1
      if (a.starred && !b.starred) return -1
      if (!a.starred && b.starred) return 1
      return new Date(b.updated_at || 0).getTime() - new Date(a.updated_at || 0).getTime()
    })
  }, [filtered])

  const handleNavigate = (id: string) => {
    router.push(`/chat?session=${id}`)
  }

  const handlePin = async (id: string, pinned: boolean) => {
    try {
      await sessionController.update(id, { pinned })
      setConversations(prev => prev.map(c => c.id === id ? { ...c, pinned } : c))
    } catch { addToast('Failed to update pin', 'error') }
  }

  const handleStar = async (id: string, starred: boolean) => {
    try {
      await sessionController.update(id, { starred })
      setConversations(prev => prev.map(c => c.id === id ? { ...c, starred } : c))
    } catch { addToast('Failed to update star', 'error') }
  }

  const handleDelete = async (id: string) => {
    try {
      await sessionController.delete(id)
      setConversations(prev => prev.filter(c => c.id !== id))
    } catch { addToast('Failed to delete conversation', 'error') }
  }

  const handleRename = async (id: string) => {
    const conv = conversations.find(c => c.id === id)
    if (!conv) return
    const name = window.prompt('Rename conversation:', conv.name)
    const trimmed = name?.trim()
    if (trimmed && trimmed !== conv.name) {
      try {
        await sessionController.update(id, { name: trimmed })
        setConversations(prev => prev.map(c => c.id === id ? { ...c, name: trimmed } : c))
      } catch { addToast('Failed to rename', 'error') }
    }
  }

  const handleBatchDelete = async () => {
    if (selectedIds.size === 0) return
    if (!confirm(`Delete ${selectedIds.size} conversations?`)) return
    setDeleting(true)
    let deleted = 0
    for (const id of selectedIds) {
      try {
        await sessionController.delete(id)
        deleted++
      } catch { /* skip failed */ }
    }
    setConversations(prev => prev.filter(c => !selectedIds.has(c.id)))
    setSelectedIds(new Set())
    setDeleting(false)
    addToast(`Deleted ${deleted} conversation${deleted !== 1 ? 's' : ''}`, deleted > 0 ? 'info' : 'error')
  }

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const toggleSelectAll = () => {
    if (selectedIds.size === sorted.length) setSelectedIds(new Set())
    else setSelectedIds(new Set(sorted.map(c => c.id)))
  }

  const handleExport = (conv: Conversation, format: 'md' | 'json') => {
    const data = format === 'md'
      ? `# ${conv.name}\n\n${(conv.messages || []).map(m => `**${m.role}**: ${m.content}`).join('\n\n')}`
      : JSON.stringify(conv, null, 2)
    const blob = new Blob([data], { type: format === 'md' ? 'text/markdown' : 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${conv.name || 'conversation'}.${format}`
    a.click()
    URL.revokeObjectURL(url)
  }

  const starred = sorted.filter(c => c.starred)
  const pinned = sorted.filter(c => c.pinned && !c.starred)
  const rest = sorted.filter(c => !c.pinned && !c.starred)

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="sm" onClick={() => router.push('/chat')} className="h-7 px-1.5 text-xs text-muted-foreground hover:text-foreground">
              ← Back
            </Button>
            <AppRouteHeaderLead title="Conversations" />
          </div>
        }
      />

      <div className="space-y-4">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <CardTitle className="text-base">All conversations</CardTitle>
                {sorted.length > 0 && (
                  <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer select-none">
                    <input
                      type="checkbox"
                      className="h-3.5 w-3.5 rounded border-border accent-primary"
                      checked={selectedIds.size === sorted.length && sorted.length > 0}
                      onChange={toggleSelectAll}
                    />
                    {selectedIds.size > 0 && selectedIds.size < sorted.length
                      ? `${selectedIds.size} selected`
                      : 'Select all'}
                  </label>
                )}
              </div>
              <Button
                size="sm"
                onClick={() => router.push('/chat')}
              >
                <IconPlus className="h-3.5 w-3.5 mr-1" />
                New chat
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <SearchInput
              value={search}
              onChange={setSearch}
              placeholder="Search conversations..."
            />

            {selectedIds.size > 0 && (
              <div className="flex items-center justify-between rounded-md border border-border/60 bg-muted/30 px-3 py-2">
                <span className="text-xs text-muted-foreground">{selectedIds.size} selected</span>
                <div className="flex items-center gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 text-xs"
                    onClick={() => setSelectedIds(new Set())}
                  >
                    Cancel
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 text-xs text-destructive hover:text-destructive border-destructive/30"
                    onClick={handleBatchDelete}
                    disabled={deleting}
                  >
                    <IconTrash className="h-3 w-3 mr-1" />
                    {deleting ? 'Deleting...' : 'Delete selected'}
                  </Button>
                </div>
              </div>
            )}

            {loading ? (
              <div className="space-y-2">
                {Array.from({ length: 5 }).map((_, i) => (
                  <div key={i} className="h-14 rounded-md bg-muted/40 animate-pulse" />
                ))}
              </div>
            ) : sorted.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-8">
                {search ? 'No matching conversations' : 'No conversations yet'}
              </p>
            ) : (
              <div className="space-y-4">
                {starred.length > 0 && (
                  <Section label="Starred">
                    {starred.map(c => (
                      <Row
                        key={c.id}
                        conversation={c}
                        selected={selectedIds.has(c.id)}
                        onToggleSelect={() => toggleSelect(c.id)}
                        onSelect={() => handleNavigate(c.id)}
                        onPin={(p) => handlePin(c.id, p)}
                        onStar={(s) => handleStar(c.id, s)}
                        onDelete={() => handleDelete(c.id)}
                        onRename={() => handleRename(c.id)}
                        onExport={(f) => handleExport(c, f)}
                      />
                    ))}
                  </Section>
                )}

                {pinned.length > 0 && (
                  <Section label="Pinned">
                    {pinned.map(c => (
                      <Row
                        key={c.id}
                        conversation={c}
                        selected={selectedIds.has(c.id)}
                        onToggleSelect={() => toggleSelect(c.id)}
                        onSelect={() => handleNavigate(c.id)}
                        onPin={(p) => handlePin(c.id, p)}
                        onStar={(s) => handleStar(c.id, s)}
                        onDelete={() => handleDelete(c.id)}
                        onRename={() => handleRename(c.id)}
                        onExport={(f) => handleExport(c, f)}
                      />
                    ))}
                  </Section>
                )}

                {rest.length > 0 && (
                  <Section label="Recent">
                    {rest.map(c => (
                      <Row
                        key={c.id}
                        conversation={c}
                        selected={selectedIds.has(c.id)}
                        onToggleSelect={() => toggleSelect(c.id)}
                        onSelect={() => handleNavigate(c.id)}
                        onPin={(p) => handlePin(c.id, p)}
                        onStar={(s) => handleStar(c.id, s)}
                        onDelete={() => handleDelete(c.id)}
                        onRename={() => handleRename(c.id)}
                        onExport={(f) => handleExport(c, f)}
                      />
                    ))}
                  </Section>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1.5 px-1">
        {label}
      </p>
      <div className="space-y-1">{children}</div>
    </div>
  )
}

function Row({
  conversation: c,
  selected,
  onToggleSelect,
  onSelect,
  onPin,
  onStar,
  onDelete,
  onRename,
  onExport,
}: {
  conversation: Conversation
  selected: boolean
  onToggleSelect: () => void
  onSelect: () => void
  onPin: (pinned: boolean) => void
  onStar: (starred: boolean) => void
  onDelete: () => void
  onRename: () => void
  onExport: (format: 'md' | 'json') => void
}) {
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

      <div className="shrink-0 opacity-0 group-hover:opacity-100 transition-opacity pt-0.5">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="icon-sm"
              className="h-6 w-6 p-0 hover:bg-transparent"
              onClick={(e) => e.stopPropagation()}
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
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={(e) => { e.preventDefault(); onExport('md') }} className="text-xs py-1.5">
              <IconDownload className="mr-2 h-3 w-3" /> Export MD
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={(e) => { e.preventDefault(); onExport('json') }} className="text-xs py-1.5">
              <IconDownload className="mr-2 h-3 w-3" /> Export JSON
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
