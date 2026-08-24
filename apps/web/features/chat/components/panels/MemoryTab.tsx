'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { cn, Switch, Button } from '@sloughgpt/strui'
import { IconRefresh, IconTrash, IconSearch, IconX, IconClock, IconEdit } from '@sloughgpt/strui'
import { memoryController, type MemoryItem, type MemoryStats } from '@/lib/memory-controller'
import { subscribeMemoryEvents } from '@/lib/memory-events'
import { formatRelativeTime } from '@/lib/format-bytes'
import { logger } from '@/lib/dev-log'

const MAX_VISIBLE = 8
const HIGHLIGHT_MS = 4000
const SEARCH_DEBOUNCE_MS = 300

/**
 * Compact memory panel for the chat tool sidebar. Surfaces what the AI
 * currently remembers (facts it stores automatically as you chat) with the
 * master Remember switch, debounced semantic search, per-item delete, and a
 * confirmed clear. The actual remembering happens in the chat loop.
 */
export function MemoryTab() {
  const [stats, setStats] = useState<MemoryStats | null>(null)
  const [items, setItems] = useState<MemoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [toggling, setToggling] = useState(false)
  const [pendingClear, setPendingClear] = useState(false)
  const [highlightedId, setHighlightedId] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [searchResults, setSearchResults] = useState<MemoryItem[] | null>(null)
  const [searched, setSearched] = useState(false)
  const [showAll, setShowAll] = useState(false)
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [showAdd, setShowAdd] = useState(false)
  const [newContent, setNewContent] = useState('')
  const [newTopic, setNewTopic] = useState('')
  const [adding, setAdding] = useState(false)
  const [addError, setAddError] = useState<string | null>(null)
  const [activeTopic, setActiveTopic] = useState<string | null>(null)
  const [sortOrder, setSortOrder] = useState<'newest' | 'oldest'>('newest')
  const [editingItem, setEditingItem] = useState<MemoryItem | null>(null)
  const [editContent, setEditContent] = useState('')
  const [editTopic, setEditTopic] = useState('')
  const [editImportance, setEditImportance] = useState(0.5)
  const [savingEdit, setSavingEdit] = useState(false)
  const [editError, setEditError] = useState<string | null>(null)
  const [consolidating, setConsolidating] = useState(false)
  const [consolidateMsg, setConsolidateMsg] = useState<string | null>(null)
  const pendingFactRef = useRef<string | null>(null)
  const highlightTimerRef = useRef<number | null>(null)
  const copyTimerRef = useRef<number | null>(null)
  const consolidateTimerRef = useRef<number | null>(null)
  const searchRef = useRef('')
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const handleSearch = useCallback(async (q: string) => {
    if (!q.trim()) {
      setSearchResults(null)
      setSearched(false)
      return
    }
    try {
      const result = await memoryController.search(q)
      setSearchResults(result.results || [])
      setSearched(true)
      setShowAll(false)
    } catch {
      setSearchResults([])
      setSearched(true)
    }
  }, [])

  useEffect(() => {
    searchRef.current = search
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => { void handleSearch(search) }, SEARCH_DEBOUNCE_MS)
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [search, handleSearch])

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const [statsResult, listResult] = await Promise.all([
        memoryController.stats().catch(() => null),
        memoryController.list().catch(() => ({ items: [], total: 0 })),
      ])
      setStats(statsResult)
      setItems(listResult.items || [])
      const pending = pendingFactRef.current
      if (pending) {
        pendingFactRef.current = null
        const match = (listResult.items || []).find(i => i.content === pending)
        if (match) {
          setHighlightedId(match.id)
          if (highlightTimerRef.current) window.clearTimeout(highlightTimerRef.current)
          highlightTimerRef.current = window.setTimeout(() => setHighlightedId(null), HIGHLIGHT_MS)
        }
      }
    } catch {
      // all fetches already fail-soft above; nothing left to surface
    } finally {
      setLoading(false)
    }
    if (searchRef.current.trim()) void handleSearch(searchRef.current)
  }, [handleSearch])

  useEffect(() => { fetchData() }, [fetchData])

  useEffect(() => {
    const unsubscribe = subscribeMemoryEvents((info) => {
      if (info.stored) {
        if (info.fact || (info.facts && info.facts.length > 0)) {
          pendingFactRef.current = info.facts?.[0] ?? info.fact ?? null
        }
        fetchData()
      }
    })
    return unsubscribe
  }, [fetchData])

  useEffect(() => {
    return () => {
      if (highlightTimerRef.current) window.clearTimeout(highlightTimerRef.current)
      if (copyTimerRef.current) window.clearTimeout(copyTimerRef.current)
      if (consolidateTimerRef.current) window.clearTimeout(consolidateTimerRef.current)
    }
  }, [])

  const handleCopy = useCallback(async (content: string, id: string) => {
    try {
      await navigator.clipboard.writeText(content)
      setCopiedId(id)
      if (copyTimerRef.current) window.clearTimeout(copyTimerRef.current)
      copyTimerRef.current = window.setTimeout(() => setCopiedId(null), 1500)
    } catch (err) {
      logger.debug('Could not memory copy', { exception: String(err) })
    }
  }, [])

  const handleAdd = useCallback(async () => {
    if (!newContent.trim()) return
    setAdding(true)
    setAddError(null)
    try {
      const result = await memoryController.store(newContent, newTopic.trim() || 'manual')
      if (result.stored) {
        pendingFactRef.current = newContent.trim()
        setNewContent('')
        setNewTopic('')
        setShowAdd(false)
        await fetchData()
      } else {
        setAddError('Already remembered (or memory is disabled)')
      }
    } catch {
      setAddError('Could not store fact')
    } finally {
      setAdding(false)
    }
  }, [newContent, newTopic, fetchData])

  const startEdit = useCallback((item: MemoryItem) => {
    setShowAdd(false)
    setEditingItem(item)
    setEditContent(item.content)
    setEditTopic(item.topic || '')
    setEditImportance(typeof item.importance === 'number' ? item.importance : 0.5)
    setEditError(null)
  }, [])

  const handleSaveEdit = useCallback(async () => {
    if (!editingItem || !editContent.trim()) return
    setSavingEdit(true)
    setEditError(null)
    try {
      const result = await memoryController.update(editingItem.id, editContent, editTopic, editImportance)
      if (result.updated > 0) {
        setEditingItem(null)
        const content = editContent.trim()
        const topic = editTopic.trim() || editingItem.topic
        const patch = (i: MemoryItem): MemoryItem => (i.id === editingItem.id ? { ...i, content, topic, importance: editImportance } : i)
        setItems(prev => prev.map(patch))
        setSearchResults(prev => (prev === null ? prev : prev.map(patch)))
        await fetchData()
      } else if (result.duplicate) {
        setEditError('That fact already exists in memory')
      } else {
        setEditError('Memory item not found')
      }
    } catch {
      setEditError('Could not update memory item')
    } finally {
      setSavingEdit(false)
    }
  }, [editingItem, editContent, editTopic, editImportance, fetchData])

  const handleConsolidate = useCallback(async () => {
    setConsolidating(true)
    setConsolidateMsg(null)
    try {
      const result = await memoryController.consolidate()
      setConsolidateMsg(
        result.removed > 0
          ? `Consolidated ${result.removed} duplicate fact(s), kept ${result.kept}`
          : 'No near-duplicate facts found'
      )
    } catch {
      setConsolidateMsg('Could not consolidate memory')
    } finally {
      setConsolidating(false)
      if (consolidateTimerRef.current) window.clearTimeout(consolidateTimerRef.current)
      consolidateTimerRef.current = window.setTimeout(() => setConsolidateMsg(null), 3500)
    }
    fetchData()
  }, [fetchData])

  const deleteItem = async (item: MemoryItem) => {
    setItems(prev => prev.filter(i => i.id !== item.id))
    setSearchResults(prev => (prev === null ? prev : prev.filter(i => i.id !== item.id)))
    try {
      await memoryController.delete(item.id)
    } catch (err) {
      logger.debug('Could not memory delete', { exception: String(err) })
      fetchData()
    }
  }

  const clearAll = async () => {
    setPendingClear(false)
    try {
      const result = await memoryController.clear()
      logger.debug('Memory cleared', { cleared: result.cleared })
    } catch (err) {
      logger.debug('Could not memory clear', { exception: String(err) })
    }
    fetchData()
  }

  const enabled = stats?.enabled ?? true

  const topics = useMemo(() => {
    const seen = new Set<string>()
    const list: string[] = []
    for (const i of items) {
      if (i.topic && !seen.has(i.topic)) {
        seen.add(i.topic)
        list.push(i.topic)
      }
    }
    return list.sort((a, b) => a.localeCompare(b))
  }, [items])

  const browseList = useMemo(() => {
    const base = [...items]
    return base.sort((a, b) => sortOrder === 'newest' ? b.timestamp - a.timestamp : a.timestamp - b.timestamp)
  }, [items, sortOrder])

  const topicFiltered = useMemo(() => {
    const base = searchResults !== null ? searchResults : browseList
    if (!activeTopic) return base
    return base.filter(i => i.topic === activeTopic)
  }, [browseList, searchResults, activeTopic])

  const displayed = showAll ? topicFiltered : topicFiltered.slice(0, MAX_VISIBLE)

  const toggleEnabled = async (next: boolean) => {
    setToggling(true)
    try {
      const result = await memoryController.setEnabled(next)
      setStats(prev => (prev ? { ...prev, enabled: result.enabled } : prev))
    } catch (err) {
      logger.debug('Could not memory toggle', { exception: String(err) })
    } finally {
      setToggling(false)
    }
    fetchData()
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground flex items-center gap-1.5">
            <span className={cn('inline-block h-1.5 w-1.5 rounded-full', enabled ? 'bg-success' : 'bg-muted-foreground/50')} />
            {enabled ? `${stats?.total_facts ?? items.length} fact${(stats?.total_facts ?? items.length) !== 1 ? 's' : ''}` : 'Memory off'}
          </span>
          {enabled && items.length > 0 && (
            <button
              onClick={handleConsolidate}
              disabled={consolidating}
              className="text-[10px] text-primary hover:underline disabled:opacity-40 disabled:no-underline transition-colors"
              title="Merge near-duplicate facts"
            >
              {consolidating ? 'Consolidating…' : 'Consolidate'}
            </button>
          )}
        </div>
        <div className="flex items-center gap-0.5">
          <div className="flex items-center gap-1 mr-1 pr-1.5 border-r border-border/60">
            <span className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">Remember</span>
            <Switch
              size="sm"
              checked={enabled}
              onCheckedChange={toggleEnabled}
              disabled={toggling}
              aria-label="Toggle automatic memory"
            />
          </div>
          <button
            onClick={fetchData}
            disabled={loading}
            className="h-6 w-6 flex items-center justify-center rounded text-muted-foreground hover:text-foreground hover:bg-muted/60 transition-colors disabled:opacity-40"
            aria-label="Refresh memory"
          >
            <IconRefresh className={cn('h-3 w-3', loading && 'animate-spin')} />
          </button>
          <button
            onClick={() => setPendingClear(true)}
            disabled={items.length === 0}
            className="h-6 w-6 flex items-center justify-center rounded text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors disabled:opacity-40"
            aria-label="Clear memory"
          >
            <IconTrash className="h-3 w-3" />
          </button>
        </div>
      </div>

      {consolidateMsg && <p className="text-[10px] text-muted-foreground">{consolidateMsg}</p>}

      {enabled && (
        <>
          <div className="flex items-center gap-1.5">
            <div className="relative flex-1">
              <IconSearch className="h-3 w-3 text-muted-foreground absolute left-2 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Search memory..."
                aria-label="Search memory"
                className="w-full h-7 pl-7 pr-6 rounded border border-border/40 bg-background text-xs focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50"
              />
              {search && (
                <button
                  onClick={() => setSearch('')}
                  className="absolute right-1.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                  aria-label="Clear search"
                >
                  <IconX className="h-3 w-3" />
                </button>
              )}
            </div>
            <Button
              size="sm"
              variant="outline"
              className="h-7 shrink-0 text-[10px] px-2"
              onClick={() => { setShowAdd(v => !v); setAddError(null); setEditingItem(null) }}
            >
              {showAdd ? 'Close' : '+ Store'}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              className="h-7 shrink-0 text-[10px] px-1.5 gap-1"
              onClick={() => setSortOrder(o => o === 'newest' ? 'oldest' : 'newest')}
              disabled={searchResults !== null}
              title={searchResults !== null ? 'Search results use relevance order' : undefined}
              aria-label="Toggle memory sort order"
            >
              <IconClock className="h-3 w-3" />
              {sortOrder === 'newest' ? 'Newest' : 'Oldest'}
            </Button>
          </div>

          {showAdd && (
            <div className="space-y-1.5 rounded border border-border/40 p-2">
              <textarea
                value={newContent}
                onChange={e => setNewContent(e.target.value)}
                placeholder="Type a fact the AI should remember..."
                aria-label="New memory fact"
                className="w-full h-14 resize-none rounded border border-border/40 bg-background p-2 text-xs focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50"
              />
              <div className="flex items-center gap-1.5">
                <input
                  value={newTopic}
                  onChange={e => setNewTopic(e.target.value)}
                  placeholder="topic"
                  aria-label="Memory fact topic"
                  className="flex-1 h-7 rounded border border-border/40 bg-background px-2 text-[10px] focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50"
                />
                <Button size="sm" className="h-6 text-[10px] px-2" disabled={!newContent.trim() || adding} onClick={handleAdd}>
                  {adding ? 'Saving…' : 'Save'}
                </Button>
              </div>
              {addError && <p className="text-[10px] text-destructive">{addError}</p>}
            </div>
          )}

          {topics.length > 0 && (
            <div className="flex items-center gap-1 overflow-x-auto pb-0.5 -mx-1 px-1" aria-label="Filter by topic">
              <button
                type="button"
                onClick={() => setActiveTopic(null)}
                className={`shrink-0 text-[10px] px-2 py-1 rounded-full font-medium transition-colors ${activeTopic === null ? 'bg-primary/15 text-primary' : 'bg-muted text-muted-foreground hover:bg-muted/70'}`}
              >
                All
              </button>
              {topics.map(topic => (
                <button
                  key={topic}
                  type="button"
                  onClick={() => setActiveTopic(activeTopic === topic ? null : topic)}
                  className={`shrink-0 text-[10px] px-2 py-1 rounded-full font-medium transition-colors ${activeTopic === topic ? 'bg-primary/15 text-primary' : 'bg-muted text-muted-foreground hover:bg-muted/70'}`}
                >
                  {topic}
                </button>
              ))}
            </div>
          )}

          {editingItem && (
            <div className="space-y-1.5 rounded border border-primary/40 p-2">
              <p className="text-[10px] font-medium text-muted-foreground flex items-center gap-1.5">
                <IconEdit className="h-3 w-3" />
                Edit memory fact
              </p>
              <textarea
                value={editContent}
                onChange={e => setEditContent(e.target.value)}
                aria-label="Edit memory fact text"
                className="w-full h-14 resize-none rounded border border-border/40 bg-background p-2 text-xs focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50"
              />
              <div className="flex items-center gap-1.5">
                <input
                  value={editTopic}
                  onChange={e => setEditTopic(e.target.value)}
                  placeholder={editingItem.topic || 'topic'}
                  aria-label="Edit memory fact topic"
                  className="flex-1 h-7 rounded border border-border/40 bg-background px-2 text-[10px] focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50"
                />
              </div>
              <div className="flex items-center gap-1.5">
                <span className="text-[10px] text-muted-foreground">Importance</span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.1}
                  value={editImportance}
                  onChange={e => setEditImportance(Number(e.target.value))}
                  aria-label="Edit memory fact importance"
                  className="flex-1 h-1 accent-primary"
                />
                <span className="text-[10px] text-muted-foreground font-mono w-8 text-right">{editImportance.toFixed(1)}</span>
              </div>
              <div className="flex items-center gap-1.5">
                <Button size="sm" className="h-6 text-[10px] px-2" disabled={!editContent.trim() || savingEdit} onClick={handleSaveEdit}>
                  {savingEdit ? 'Saving…' : 'Save'}
                </Button>
                <Button size="sm" variant="ghost" className="h-6 text-[10px] px-2" onClick={() => setEditingItem(null)}>
                  Cancel
                </Button>
              </div>
              {editError && <p className="text-[10px] text-destructive">{editError}</p>}
            </div>
          )}
        </>
      )}

      {loading ? (
        <p className="text-[10px] text-muted-foreground text-center py-3">Loading memory…</p>
      ) : !enabled ? (
        <p className="text-xs text-muted-foreground text-center py-4">
          Memory is off. Flip the Remember switch to start learning again.
        </p>
      ) : searched && searchResults !== null && searchResults.length === 0 ? (
        <div className="text-center py-4">
          <p className="text-xs text-muted-foreground">No memory matches that search.</p>
          <button
            onClick={() => setSearch('')}
            className="mt-1 text-[10px] text-primary hover:underline"
            aria-label="Clear search"
          >
            Clear search
          </button>
        </div>
      ) : topicFiltered.length === 0 ? (
        <p className="text-xs text-muted-foreground text-center py-4">
          {activeTopic
            ? `No memory in the "${activeTopic}" topic.`
            : 'Nothing remembered yet. The AI stores facts automatically as you chat.'}
        </p>
      ) : (
        <>
          <ul className="space-y-1 max-h-60 overflow-y-auto">
            {displayed.map(item => (
              <li key={item.id} className={cn(
                'group flex items-start justify-between gap-2 p-2 rounded bg-muted/30 border text-xs leading-relaxed transition-colors',
                item.id === highlightedId ? 'border-primary/60 bg-primary/10' : 'border-border/40',
              )}>
                <div className="min-w-0">
                  <div className="flex items-start justify-between gap-1">
                    <span
                      className="block cursor-pointer select-text hover:text-foreground/80 transition-colors"
                      title="Click to copy"
                      role="button"
                      tabIndex={0}
                      onClick={() => handleCopy(item.content, item.id)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault()
                          handleCopy(item.content, item.id)
                        }
                      }}
                    >
                      {item.content.length > 160 ? item.content.slice(0, 160) + '…' : item.content}
                    </span>
                    {copiedId === item.id && (
                      <span className="shrink-0 text-[9px] text-success font-medium pt-0.5">Copied</span>
                    )}
                  </div>
                  <div className="flex items-center gap-1.5 mt-1 flex-wrap">
                    {item.topic && (
                      <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground font-medium">{item.topic}</span>
                    )}
                    {item.source && (
                      <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground font-medium">{item.source}</span>
                    )}
                    {typeof item.importance === 'number' && (
                      <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground font-medium" title="Importance score">
                        importance {item.importance.toFixed(1)}
                      </span>
                    )}
                    {item.timestamp > 0 && (
                      <span
                        className="text-[9px] text-muted-foreground font-mono"
                        title={new Date(item.timestamp * 1000).toLocaleString()}
                      >
                        {formatRelativeTime(item.timestamp)}
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-0.5 shrink-0">
                  {typeof item.score === 'number' && searchResults !== null && (
                    <span className="text-[10px] text-muted-foreground font-mono shrink-0 mr-0.5">{item.score.toFixed(2)}</span>
                  )}
                  <button
                    onClick={() => startEdit(item)}
                    className="opacity-0 group-hover:opacity-100 focus-within:opacity-100 p-0.5 text-muted-foreground hover:text-primary transition-opacity"
                    aria-label="Edit memory item"
                  >
                    <IconEdit className="h-3 w-3" />
                  </button>
                  <button
                    onClick={() => deleteItem(item)}
                    className="opacity-0 group-hover:opacity-100 focus-within:opacity-100 p-0.5 text-muted-foreground hover:text-destructive transition-opacity"
                    aria-label="Delete memory item"
                  >
                    <IconTrash className="h-3 w-3" />
                  </button>
                </div>
              </li>
            ))}
          </ul>
          {searchResults === null && topicFiltered.length > MAX_VISIBLE && (
            <button
              onClick={() => setShowAll(v => !v)}
              className="block mx-auto mt-1.5 text-[10px] text-primary hover:underline"
            >
              {showAll ? 'Show fewer' : `Show all ${topicFiltered.length}`}
            </button>
          )}
        </>
      )}

      {pendingClear && (
        <div className="flex items-center gap-1 border-t border-border/30 pt-2">
          <span className="text-[10px] text-muted-foreground flex-1">Clear all stored memory?</span>
          <Button variant="destructive" size="sm" className="h-6 text-[10px] px-2" onClick={clearAll}>Clear</Button>
          <Button variant="outline" size="sm" className="h-6 text-[10px] px-2" onClick={() => setPendingClear(false)}>Cancel</Button>
        </div>
      )}

      <a
        href="/knowledge"
        className="block text-center text-[10px] text-muted-foreground hover:text-foreground pt-1 border-t border-border/30 transition-colors"
      >
        Manage memory →
      </a>
    </div>
  )
}
