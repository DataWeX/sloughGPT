'use client'
export const dynamic = 'force-dynamic'

import { useCallback, useEffect, useRef, useState } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from '@sloughgpt/strui'
import { Card, CardContent, CardHeader, CardTitle, EmptyCard } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import { Chip } from '@sloughgpt/strui'
import { IconRefresh, IconPlus, IconTrash, IconSearch, IconCheck, IconX } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { knowledgeController, type KnowledgeItem, type KnowledgeStats, type TopicCount } from '@/lib/knowledge-controller'
import { downloadJson } from '@/lib/download-utils'

export default function KnowledgePage() {
  const addToast = useToastStore(s => s.addToast)
  const [items, setItems] = useState<KnowledgeItem[]>([])
  const [stats, setStats] = useState<KnowledgeStats | null>(null)
  const [topics, setTopics] = useState<TopicCount[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [searchResults, setSearchResults] = useState<KnowledgeItem[] | null>(null)
  const [searching, setSearching] = useState(false)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [pendingDelete, setPendingDelete] = useState<KnowledgeItem | null>(null)
  const [pendingBatchDelete, setPendingBatchDelete] = useState(false)
  const [showAdd, setShowAdd] = useState(false)
  const [newContent, setNewContent] = useState('')
  const [newTopic, setNewTopic] = useState('general')
  const [activeTopic, setActiveTopic] = useState<string | null>(null)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editContent, setEditContent] = useState('')
  const [editTopic, setEditTopic] = useState('')
  const [importing, setImporting] = useState(false)
  const [sortBy, setSortBy] = useState<'date' | 'importance' | 'topic'>('date')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const [itemsResult, statsResult, topicsResult] = await Promise.all([
        knowledgeController.list(),
        knowledgeController.stats(),
        knowledgeController.topics(),
      ])
      setItems(itemsResult)
      setStats(statsResult)
      setTopics(topicsResult.topics || [])
    } catch {
      addToast('Failed to load knowledge', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => { fetchData() }, [fetchData])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return
      if (e.key === 'n' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault()
        setShowAdd(true)
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
        e.preventDefault()
        document.querySelector<HTMLInputElement>('[placeholder="Search knowledge..."]')?.focus()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const handleSearch = useCallback(async () => {
    if (!search.trim()) {
      setSearchResults(null)
      return
    }
    setSearching(true)
    try {
      const result = await knowledgeController.search(search)
      setSearchResults(result.results || [])
    } catch {
      addToast('Search failed', 'error')
    } finally {
      setSearching(false)
    }
  }, [search, addToast])

  useEffect(() => {
    const timer = setTimeout(() => {
      if (search.trim()) handleSearch()
      else setSearchResults(null)
    }, 300)
    return () => clearTimeout(timer)
  }, [search, handleSearch])

  const displayItems = (searchResults ?? (activeTopic
    ? items.filter(i => i.topic === activeTopic)
    : items)).slice().sort((a, b) => {
    if (sortBy === 'importance') return b.importance - a.importance
    if (sortBy === 'topic') return (a.topic || '').localeCompare(b.topic || '')
    return b.timestamp - a.timestamp
  })

  const handleDelete = async () => {
    if (!pendingDelete) return
    try {
      await knowledgeController.delete(pendingDelete.id)
      setItems(prev => prev.filter(i => i.id !== pendingDelete.id))
      setStats(prev => prev ? { ...prev, total_items: prev.total_items - 1 } : prev)
      addToast('Deleted', 'info')
    } catch {
      addToast('Delete failed', 'error')
    } finally {
      setPendingDelete(null)
    }
  }

  const handleBatchDelete = async () => {
    if (selectedIds.size === 0) return
    try {
      await knowledgeController.batchDelete(Array.from(selectedIds))
      setItems(prev => prev.filter(i => !selectedIds.has(i.id)))
      setStats(prev => prev ? { ...prev, total_items: prev.total_items - selectedIds.size } : prev)
      addToast(`Deleted ${selectedIds.size} items`, 'info')
      setSelectedIds(new Set())
    } catch {
      addToast('Batch delete failed', 'error')
    } finally {
      setPendingBatchDelete(false)
    }
  }

  const handleAdd = async () => {
    if (!newContent.trim()) return
    try {
      await knowledgeController.add(newContent.trim(), newTopic, true)
      setNewContent('')
      setNewTopic('general')
      setShowAdd(false)
      addToast('Knowledge added', 'info')
      await fetchData()
    } catch {
      addToast('Failed to add knowledge', 'error')
    }
  }

  const handleSaveEdit = async (id: string) => {
    if (!editContent.trim()) return
    try {
      await knowledgeController.update(id, { content: editContent.trim(), topic: editTopic.trim() || 'general' })
      setItems(prev => prev.map(i => i.id === id ? { ...i, content: editContent.trim(), topic: editTopic.trim() || 'general' } : i))
      setEditingId(null)
      addToast('Updated', 'info')
    } catch {
      addToast('Update failed', 'error')
    }
  }

  const handleExport = () => {
    const data = items.map(i => ({ content: i.content, topic: i.topic, source: i.source, importance: i.importance }))
    downloadJson(data, `knowledge-export-${new Date().toISOString().slice(0, 10)}.json`)
    addToast(`Exported ${items.length} items`, 'success')
  }

  const handleImportFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setImporting(true)
    try {
      const text = await file.text()
      let facts: string[]
      if (file.name.endsWith('.json')) {
        const parsed = JSON.parse(text)
        facts = Array.isArray(parsed) ? parsed.map((i: any) => typeof i === 'string' ? i : i.content || JSON.stringify(i)).filter(Boolean) : []
      } else {
        facts = text.split('\n').map(l => l.trim()).filter(Boolean)
      }
      if (facts.length === 0) { addToast('No facts found in file', 'error'); return }
      await knowledgeController.bulkIngest(facts, 'imported', 'file-import')
      addToast(`Imported ${facts.length} facts`, 'success')
      await fetchData()
    } catch {
      addToast('Import failed', 'error')
    } finally {
      setImporting(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
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
    if (selectedIds.size === displayItems.length) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(displayItems.map(i => i.id)))
    }
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Knowledge" subtitle="Manage facts the AI remembers" />}
        right={
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline" className="h-7 text-xs" onClick={fetchData} disabled={loading}>
              <IconRefresh className={loading ? 'animate-spin h-3 w-3 mr-1' : 'h-3 w-3 mr-1'} />
              Refresh
            </Button>
            {items.length > 0 && (
              <Button size="sm" variant="outline" className="h-7 text-xs" onClick={handleExport}>
                Export
              </Button>
            )}
            <Button size="sm" variant="outline" className="h-7 text-xs" onClick={() => fileInputRef.current?.click()} disabled={importing}>
              {importing ? 'Importing...' : 'Import'}
            </Button>
            <input ref={fileInputRef} type="file" accept=".json,.txt" className="hidden" onChange={handleImportFile} />
            <Button size="sm" className="h-7 text-xs" onClick={() => setShowAdd(true)}>
              <IconPlus className="h-3 w-3 mr-1" />
              Add
            </Button>
          </div>
        }
      />

      <div className="space-y-4">
        {stats && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <Card>
              <CardContent className="p-3">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Total Facts</p>
                <p className="text-lg font-semibold mt-1">{stats.total_items}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-3">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Topics</p>
                <p className="text-lg font-semibold mt-1">{stats.topic_count}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-3">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Avg Importance</p>
                <p className="text-lg font-semibold mt-1">{stats.avg_importance.toFixed(1)}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-3">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Searchable</p>
                <p className="text-lg font-semibold mt-1">{stats.searchable ? 'Yes' : 'No'}</p>
              </CardContent>
            </Card>
          </div>
        )}

        {topics.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            <Chip
              label={`All (${items.length})`}
              onClick={() => setActiveTopic(null)}
              className={`text-[10px] px-2 py-0.5 rounded-full cursor-pointer transition-colors ${
                activeTopic === null
                  ? 'bg-primary/15 text-primary border-primary/30'
                  : 'bg-muted text-muted-foreground border-border/40 hover:bg-muted/80'
              }`}
            />
            {topics.map(t => (
              <Chip
                key={t.name}
                label={`${t.name} (${t.count})`}
                onClick={() => setActiveTopic(activeTopic === t.name ? null : t.name)}
                className={`text-[10px] px-2 py-0.5 rounded-full cursor-pointer transition-colors ${
                  activeTopic === t.name
                    ? 'bg-primary/15 text-primary border-primary/30'
                    : 'bg-muted text-muted-foreground border-border/40 hover:bg-muted/80'
                }`}
              />
            ))}
          </div>
        )}

        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <IconSearch className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground/60" />
            <Input
              placeholder="Search knowledge..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="h-8 text-xs pl-8"
            />
          </div>
          <select
            value={sortBy}
            onChange={e => setSortBy(e.target.value as typeof sortBy)}
            className="h-8 text-xs rounded-md border border-border/60 bg-background px-2 focus:outline-none focus:ring-1 focus:ring-primary/30"
            aria-label="Sort by"
          >
            <option value="date">Newest</option>
            <option value="importance">Importance</option>
            <option value="topic">Topic</option>
          </select>
          {selectedIds.size > 0 && (
            <Button
              size="sm"
              variant="destructive"
              className="h-8 text-xs"
              onClick={() => setPendingBatchDelete(true)}
            >
              <IconTrash className="h-3 w-3 mr-1" />
              Delete ({selectedIds.size})
            </Button>
          )}
        </div>

        {loading ? (
          <div className="space-y-2">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-16 rounded-lg" />
            ))}
          </div>
        ) : displayItems.length === 0 ? (
          <EmptyCard
            message={search ? 'No results found' : 'No knowledge stored. Add facts the AI should remember across conversations.'}
            action={null}
          />
        ) : (
          <>
            {displayItems.length > 1 && (
              <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer">
                <input
                  type="checkbox"
                  checked={selectedIds.size === displayItems.length && displayItems.length > 0}
                  onChange={toggleSelectAll}
                  className="rounded border-border"
                />
                Select all ({displayItems.length})
              </label>
            )}
            <div className="space-y-1.5">
              {displayItems.map(item => (
                <div
                  key={item.id}
                  className={`group relative p-3 rounded-lg border text-xs leading-relaxed transition-colors ${
                    selectedIds.has(item.id)
                      ? 'bg-primary/[0.06] border-primary/30'
                      : editingId === item.id
                        ? 'bg-primary/[0.04] border-primary/30'
                        : 'bg-card border-border/60 hover:bg-muted/30'
                  }`}
                >
                  <div className="flex items-start gap-2">
                    <input
                      type="checkbox"
                      checked={selectedIds.has(item.id)}
                      onChange={() => toggleSelect(item.id)}
                      className="mt-0.5 rounded border-border shrink-0"
                    />
                    <div className="flex-1 min-w-0">
                      {editingId === item.id ? (
                        <div className="space-y-2">
                          <textarea
                            className="w-full p-2 text-sm border border-input rounded-lg resize-none h-16 bg-background focus:outline-none focus:ring-1 focus:ring-primary/40"
                            value={editContent}
                            onChange={e => setEditContent(e.target.value)}
                            autoFocus
                            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSaveEdit(item.id) } if (e.key === 'Escape') setEditingId(null) }}
                          />
                          <div className="flex items-center gap-2">
                            <Input
                              value={editTopic}
                              onChange={e => setEditTopic(e.target.value)}
                              className="h-7 text-xs flex-1"
                              placeholder="Topic"
                              onKeyDown={e => { if (e.key === 'Enter') handleSaveEdit(item.id); if (e.key === 'Escape') setEditingId(null) }}
                            />
                            <button onClick={() => handleSaveEdit(item.id)} className="text-success hover:text-success/80 p-1" aria-label="Save edit">
                              <IconCheck className="h-3.5 w-3.5" />
                            </button>
                            <button onClick={() => setEditingId(null)} className="text-muted-foreground hover:text-foreground p-1" aria-label="Cancel edit">
                              <IconX className="h-3.5 w-3.5" />
                            </button>
                          </div>
                        </div>
                      ) : (
                        <>
                          <p className="text-sm text-foreground break-words">{item.content}</p>
                          <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
                            {item.topic && (
                              <span className="text-[9px] px-1.5 py-0.5 rounded font-medium bg-primary/10 text-primary">
                                {item.topic}
                              </span>
                            )}
                            {item.source && (
                              <span className="text-[9px] px-1.5 py-0.5 rounded font-medium bg-muted text-muted-foreground">
                                {item.source}
                              </span>
                            )}
                            {item.importance > 0 && (
                              <span className="text-[9px] text-muted-foreground/50">
                                importance: {item.importance.toFixed(1)}
                              </span>
                            )}
                            <span className="text-[9px] text-muted-foreground/50">
                              {new Date(item.timestamp * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                            </span>
                          </div>
                        </>
                      )}
                    </div>
                    {editingId !== item.id && (
                      <div className="flex items-center gap-0.5 shrink-0">
                        <button
                          onClick={() => { setEditingId(item.id); setEditContent(item.content); setEditTopic(item.topic || '') }}
                          className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-primary p-0.5 transition-opacity"
                          aria-label="Edit knowledge"
                        >
                          <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/></svg>
                        </button>
                        <button
                          onClick={() => setPendingDelete(item)}
                          className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive p-0.5 transition-opacity"
                          aria-label="Delete knowledge"
                        >
                          <IconTrash className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      <AlertDialog open={pendingDelete !== null} onOpenChange={() => setPendingDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete knowledge?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently remove this fact from the knowledge base.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDelete} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={pendingBatchDelete} onOpenChange={() => setPendingBatchDelete(false)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete {selectedIds.size} items?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently remove these facts from the knowledge base.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleBatchDelete} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
              Delete all
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={showAdd} onOpenChange={() => setShowAdd(false)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Add knowledge</AlertDialogTitle>
            <AlertDialogDescription>
              Add a fact the AI should remember across conversations.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <div className="space-y-3 py-2">
            <textarea
              className="w-full p-2.5 text-sm border border-input rounded-lg resize-none h-24 bg-background focus:outline-none focus:ring-1 focus:ring-primary/40"
              placeholder="Enter a fact, preference, or piece of context..."
              value={newContent}
              onChange={e => setNewContent(e.target.value)}
              autoFocus
            />
            <div className="flex items-center gap-2">
              <label className="text-xs text-muted-foreground shrink-0">Topic:</label>
              <Input
                value={newTopic}
                onChange={e => setNewTopic(e.target.value)}
                className="h-8 text-xs flex-1"
                placeholder="general"
              />
            </div>
          </div>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => { setNewContent(''); setNewTopic('general') }}>Cancel</AlertDialogCancel>
            <Button size="sm" disabled={!newContent.trim()} onClick={handleAdd}>Add</Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
