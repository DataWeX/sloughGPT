'use client'
export const dynamic = 'force-dynamic'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { cn, Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { SearchInput } from '@sloughgpt/strui'
import { IconTrash, IconDownload, IconPlus, IconFolder } from '@sloughgpt/strui'
import { sessionController, type Conversation } from '@/lib/session-controller'
import { useToastStore } from '@/lib/toast-store'
import { parseConversationJSON, parseConversationMD } from '@/lib/conversations-utils'
import ConversationSection from '@/components/conversations/ConversationSection'
import ImportResultModal from '@/components/conversations/ImportResultModal'

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

  const handleNavigate = (id: string) => router.push(`/chat?session=${id}`)

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
      try { await sessionController.delete(id); deleted++ } catch {}
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
      try { await sessionController.update(id, { archived: true }); archived++ } catch {}
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

      if (file.name.endsWith('.json')) { parsed = parseConversationJSON(JSON.parse(text)) }
      else if (file.name.endsWith('.md')) { parsed = parseConversationMD(text) }
      else { addToast('Unsupported file format — use .json or .md', 'error'); setImporting(false); return }

      if (parsed.length === 0) { addToast('No conversations found in file', 'error'); setImporting(false); return }

      let ok = 0; let fail = 0; const names: string[] = []
      for (const conv of parsed) {
        try { await sessionController.create(conv.name); ok++; names.push(conv.name) } catch { fail++ }
      }
      setImportResult({ ok, fail, names })
      addToast(`Imported ${ok} of ${parsed.length} conversations`, fail > 0 ? 'info' : 'success')
      if (ok > 0) void fetchAll()
    } catch (err: any) { addToast(`Import failed: ${err.message}`, 'error')
    } finally { setImporting(false); e.target.value = '' }
  }

  const toggleSelect = (id: string) => setSelectedIds(prev => {
    const next = new Set(prev)
    if (next.has(id)) next.delete(id); else next.add(id)
    return next
  })

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

  const sections = useMemo(() => {
    const active = sorted.filter(c => !c.archived)
    return {
      starred: active.filter(c => c.starred),
      pinned: active.filter(c => c.pinned && !c.starred),
      recent: active.filter(c => !c.pinned && !c.starred),
      archived: sorted.filter(c => c.archived),
    }
  }, [sorted])

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
                <Button size="sm" onClick={() => router.push('/chat')}>
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
                  <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => setSelectedIds(new Set())}>
                    Cancel
                  </Button>
                  <Button variant="outline" size="sm" className="h-7 text-xs" onClick={handleBatchArchive} disabled={archiving}>
                    <IconFolder className="h-3 w-3 mr-1" />
                    {archiving ? 'Archiving...' : 'Archive selected'}
                  </Button>
                  <Button variant="outline" size="sm" className="h-7 text-xs text-destructive hover:text-destructive border-destructive/30" onClick={handleBatchDelete} disabled={deleting}>
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
                <ConversationSection label="Starred" conversations={sections.starred} selectedIds={selectedIds} onToggleSelect={toggleSelect} onSelect={handleNavigate} onPin={handlePin} onStar={handleStar} onArchive={handleArchive} onDelete={handleDelete} onRename={handleRename} onExport={handleExport} />
                <ConversationSection label="Pinned" conversations={sections.pinned} selectedIds={selectedIds} onToggleSelect={toggleSelect} onSelect={handleNavigate} onPin={handlePin} onStar={handleStar} onArchive={handleArchive} onDelete={handleDelete} onRename={handleRename} onExport={handleExport} />
                <ConversationSection label="Recent" conversations={sections.recent} selectedIds={selectedIds} onToggleSelect={toggleSelect} onSelect={handleNavigate} onPin={handlePin} onStar={handleStar} onArchive={handleArchive} onDelete={handleDelete} onRename={handleRename} onExport={handleExport} />
                <ConversationSection label="Archived" conversations={sections.archived} selectedIds={selectedIds} onToggleSelect={toggleSelect} onSelect={handleNavigate} onPin={handlePin} onStar={handleStar} onArchive={handleArchive} onDelete={handleDelete} onRename={handleRename} onExport={handleExport} />
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {importResult && <ImportResultModal result={importResult} onClose={() => setImportResult(null)} />}
    </div>
  )
}
