'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { SearchInput } from '@/components/ui/input'
import {
  IconSearch, IconStar, IconPin, IconChat, IconTrash, IconEdit,
  IconDownload, IconPlus, IconMore, IconFolder,
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

function parseConversationJSON(data: any): { name: string; messages: { role: string; content: string }[] }[] {
  const arr = Array.isArray(data) ? data : [data]
  return arr.flatMap((item: any) => {
    if (!item.messages || !Array.isArray(item.messages)) return []
    return [{
      name: item.name || item.id || `Imported ${new Date().toLocaleDateString()}`,
      messages: item.messages.map((m: any) => ({
        role: m.role === 'user' || m.role === 'assistant' ? m.role : 'user',
        content: typeof m.content === 'string' ? m.content : '',
      })),
    }]
  })
}

function parseConversationMD(text: string): { name: string; messages: { role: string; content: string }[] }[] {
  const blocks = text.split(/(?=^# )/m)
  return blocks.filter(b => b.trim()).map(block => {
    const lines = block.split('\n')
    const name = lines[0].replace(/^#\s*/, '').trim() || 'Imported'
    const messages: { role: string; content: string }[] = []
    let currentRole: 'user' | 'assistant' | null = null
    let currentContent: string[] = []
    for (const line of lines.slice(1)) {
      const userMatch = line.match(/^\*\*(user|User)\*\*:\s*(.*)/)
      const asstMatch = line.match(/^\*\*(assistant|Assistant)\*\*:\s*(.*)/)
      if (userMatch || asstMatch) {
        if (currentRole && currentContent.length > 0) {
          messages.push({ role: currentRole, content: currentContent.join('\n').trim() })
        }
        currentRole = userMatch ? 'user' : 'assistant'
        currentContent = [userMatch ? userMatch[2] : asstMatch![2]]
      } else if (currentRole) {
        currentContent.push(line)
      }
    }
    if (currentRole && currentContent.length > 0) {
      messages.push({ role: currentRole, content: currentContent.join('\n').trim() })
    }
    return { name, messages }
  }).filter(c => c.messages.length > 0)
}

export default function ConversationsPage() {
  const router = useRouter()
  const addToast = useToastStore(s => s.addToast)
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [deleting, setDeleting] = useState(false)
  const [archiving, setArchiving] = useState(false)
  const [filter, setFilter] = useState<'all' | 'active' | 'archived'>('active')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [importing, setImporting] = useState(false)
  const [importResult, setImportResult] = useState<{ ok: number; fail: number; names: string[] } | null>(null)

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
        archived: s.archived || false,
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
    let list = conversations
    if (filter === 'active') list = list.filter(c => !c.archived)
    else if (filter === 'archived') list = list.filter(c => c.archived)
    if (!search.trim()) return list
    const q = search.toLowerCase()
    return list.filter(c =>
      c.name.toLowerCase().includes(q) ||
      c.messages?.some(m => m.content.toLowerCase().includes(q))
    )
  }, [conversations, search, filter])

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

  const handleArchive = async (id: string, archived: boolean) => {
    try {
      await sessionController.update(id, { archived })
      setConversations(prev => prev.map(c => c.id === id ? { ...c, archived } : c))
    } catch { addToast('Failed to archive conversation', 'error') }
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

  const handleBatchArchive = async () => {
    if (selectedIds.size === 0) return
    setArchiving(true)
    let archived = 0
    for (const id of selectedIds) {
      try {
        await sessionController.update(id, { archived: true })
        archived++
      } catch { /* skip failed */ }
    }
    setConversations(prev => prev.map(c => selectedIds.has(c.id) ? { ...c, archived: true } : c))
    setSelectedIds(new Set())
    setArchiving(false)
    addToast(`Archived ${archived} conversation${archived !== 1 ? 's' : ''}`, archived > 0 ? 'info' : 'error')
  }

  const handleFileImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setImporting(true)
    setImportResult(null)
    try {
      const text = await file.text()
      let parsed: { name: string; messages: { role: string; content: string }[] }[]

      if (file.name.endsWith('.json')) {
        const data = JSON.parse(text)
        parsed = parseConversationJSON(data)
      } else if (file.name.endsWith('.md')) {
        parsed = parseConversationMD(text)
      } else {
        addToast('Unsupported file format — use .json or .md', 'error')
        setImporting(false)
        return
      }

      if (parsed.length === 0) {
        addToast('No conversations found in file', 'error')
        setImporting(false)
        return
      }

      let ok = 0; let fail = 0; const names: string[] = []
      for (const conv of parsed) {
        try {
          await sessionController.create(conv.name)
          ok++
          names.push(conv.name)
        } catch {
          fail++
        }
      }
      setImportResult({ ok, fail, names })
      addToast(`Imported ${ok} of ${parsed.length} conversations`, fail > 0 ? 'info' : 'success')
      if (ok > 0) void fetchAll()
    } catch (err: any) {
      addToast(`Import failed: ${err.message}`, 'error')
    } finally {
      setImporting(false)
      e.target.value = ''
    }
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

  const activeList = sorted.filter(c => !c.archived)
  const starred = activeList.filter(c => c.starred)
  const pinned = activeList.filter(c => c.pinned && !c.starred)
  const rest = activeList.filter(c => !c.pinned && !c.starred)
  const archivedList = sorted.filter(c => c.archived)

  return (
    <>
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
                <div className="flex items-center gap-2">
                  <CardTitle className="text-base">Conversations</CardTitle>
                  <div className="flex items-center rounded-md border border-border/40 bg-muted/20 p-0.5">
                    {(['active', 'all', 'archived'] as const).map(f => (
                      <button
                        key={f}
                        onClick={() => setFilter(f)}
                        className={cn(
                          "text-[11px] px-2 py-0.5 rounded-sm transition-colors",
                          filter === f
                            ? "bg-background shadow-sm text-foreground font-medium"
                            : "text-muted-foreground hover:text-foreground"
                        )}
                      >
                        {f === 'active' ? 'Active' : f === 'archived' ? 'Archived' : 'All'}
                      </button>
                    ))}
                  </div>
                </div>
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
              <div className="flex items-center gap-1">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".json,.md"
                  className="hidden"
                  onChange={handleFileImport}
                />
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-7 text-xs"
                  disabled={importing}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <IconDownload className="h-3.5 w-3.5 mr-1 rotate-180" />
                  Import
                </Button>
                <Button
                  size="sm"
                  onClick={() => router.push('/chat')}
                >
                  <IconPlus className="h-3.5 w-3.5 mr-1" />
                  New chat
                </Button>
              </div>
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
                    className="h-7 text-xs"
                    onClick={handleBatchArchive}
                    disabled={archiving}
                  >
                    <IconFolder className="h-3 w-3 mr-1" />
                    {archiving ? 'Archiving...' : 'Archive selected'}
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
              <div className="text-center py-8 space-y-3">
                <p className="text-sm text-muted-foreground">
                  {search ? 'No matching conversations' : 'No conversations yet'}
                </p>
                {!search && (
                  <Button size="sm" onClick={() => router.push('/chat')}>
                    <IconPlus className="w-3.5 h-3.5 mr-1" /> Start chatting
                  </Button>
                )}
              </div>
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
                        onArchive={(a) => handleArchive(c.id, a)}
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
                        onArchive={(a) => handleArchive(c.id, a)}
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
                        onArchive={(a) => handleArchive(c.id, a)}
                        onDelete={() => handleDelete(c.id)}
                        onRename={() => handleRename(c.id)}
                        onExport={(f) => handleExport(c, f)}
                      />
                    ))}
                  </Section>
                )}

                {archivedList.length > 0 && (
                  <Section label="Archived">
                    {archivedList.map(c => (
                      <Row
                        key={c.id}
                        conversation={c}
                        selected={selectedIds.has(c.id)}
                        onToggleSelect={() => toggleSelect(c.id)}
                        onSelect={() => handleNavigate(c.id)}
                        onPin={(p) => handlePin(c.id, p)}
                        onStar={(s) => handleStar(c.id, s)}
                        onArchive={(a) => handleArchive(c.id, a)}
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

      {importResult && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={() => setImportResult(null)}>
          <div className="bg-card rounded-lg border shadow-lg w-full max-w-md mx-4 p-4" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium">Import complete</h3>
              <button className="text-muted-foreground hover:text-foreground text-xs" onClick={() => setImportResult(null)}>Close</button>
            </div>
            <div className="text-xs text-muted-foreground mb-2">
              {importResult.ok} imported, {importResult.fail} failed
            </div>
            {importResult.names.length > 0 && (
              <div className="space-y-1 max-h-60 overflow-y-auto">
                {importResult.names.map((name, i) => (
                  <div key={i} className="text-xs bg-muted/40 rounded px-2 py-1 truncate">{name}</div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </>
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
  onArchive,
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
  onArchive: (archived: boolean) => void
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
