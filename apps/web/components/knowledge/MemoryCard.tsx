'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from '@sloughgpt/strui'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import { Switch } from '@sloughgpt/strui'
import { FoldSection } from '@sloughgpt/strui'
import { EmptyCard } from '@sloughgpt/strui'
import {
  Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle,
} from '@sloughgpt/strui'
import {
  DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem,
} from '@sloughgpt/strui'
import { IconBrain, IconRefresh, IconTrash, IconPlus, IconSearch, IconX, IconFilter, IconFolder, IconSettings, IconDownload, IconUpload, IconChevronDown, IconClock, IconEdit } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { downloadJson, importFile } from '@/lib/download-utils'
import { todayDateString, formatRelativeTime } from '@/lib/format-bytes'
import {
  memoryController,
  type MemoryItem,
  type MemoryStats,
  type MemoryArchiveStats,
  type MemoryArchiveRecord,
} from '@/lib/memory-controller'

const SEARCH_DEBOUNCE_MS = 300

export interface MemoryImportEntry {
  content: string
  topic: string
}

export function parseMemoryImport(text: string, filename: string): MemoryImportEntry[] {
  const splitLines = (t: string) => t.split('\n').map(l => l.trim()).filter(Boolean)
  if (filename.toLowerCase().endsWith('.json')) {
    const parsed = JSON.parse(text)
    if (!Array.isArray(parsed)) return []
    return parsed.flatMap((entry: unknown) => {
      if (typeof entry === 'string') return entry.trim() ? [{ content: entry.trim(), topic: 'manual' }] : []
      if (entry && typeof entry === 'object') {
        const e = entry as Record<string, unknown>
        const content = typeof e.content === 'string' ? e.content.trim() : ''
        const topic = typeof e.topic === 'string' && e.topic.trim() ? e.topic.trim() : 'manual'
        return content ? [{ content, topic }] : []
      }
      return []
    })
  }
  if (filename.toLowerCase().endsWith('.csv')) {
    const lines = splitLines(text)
    if (lines.length === 0) return []
    const header = lines[0].toLowerCase()
    const contentIdx = header.split(',').findIndex(h => h.includes('content'))
    const topicIdx = header.split(',').findIndex(h => h.includes('topic'))
    return lines.slice(1).flatMap(line => {
      const cols = line.split(',').map(c => c.trim().replace(/^"|"$/g, ''))
      const content = contentIdx >= 0 ? cols[contentIdx] : cols[0]
      const topic = topicIdx >= 0 && cols[topicIdx] ? cols[topicIdx] : 'manual'
      return content ? [{ content, topic }] : []
    })
  }
  return splitLines(text).flatMap(line => {
    const match = line.match(/^(.*?)\s*\[([^\]]+)\]$/)
    if (match) return [{ content: match[1], topic: match[2] }]
    return [{ content: line, topic: 'manual' }]
  })
}

function archiveTypeLabel(taskType: string): string {
  if (taskType === 'memory.remember') return 'remember'
  if (taskType === 'memory.store') return 'store'
  if (taskType === 'memory.consolidate') return 'consolidate'
  return (taskType || 'task').replace(/^memory\./, '')
}

function archiveBadgeClass(taskType: string): string {
  if (taskType === 'memory.remember') return 'bg-primary/10 text-primary'
  if (taskType === 'memory.store') return 'bg-success/15 text-success'
  if (taskType === 'memory.consolidate') return 'bg-warning/15 text-warning'
  return 'bg-muted text-muted-foreground'
}

function archiveSummary(record: MemoryArchiveRecord): { text: string; detail: string } {
  if (record.task_type === 'memory.remember') {
    return { text: String(record.user_message ?? ''), detail: 'Learned from a conversation' }
  }
  if (record.task_type === 'memory.store') {
    return { text: String(record.content ?? ''), detail: record.topic ? `Topic: ${record.topic}` : 'Stored fact' }
  }
  if (record.task_type === 'memory.consolidate') {
    const removed = Number(record.removed ?? 0)
    const kept = Number(record.kept ?? 0)
    return { text: `Consolidated ${removed} duplicate(s), kept ${kept}`, detail: `Threshold ${Number(record.threshold ?? 0.8).toFixed(2)}` }
  }
  const { ts: _ts, task_id: _id, ...rest } = record as Record<string, unknown>
  const snippet = JSON.stringify(rest)
  return { text: snippet && snippet !== '{}' ? snippet : archiveTypeLabel(record.task_type), detail: 'Task record' }
}

/**
 * Self-contained card exposing the auto-memory layer on the Knowledge page.
 * Shows enabled state, stats, recent items, semantic search, manual store,
 * and a confirmed clear-all. Wired to `memoryController` (`/memory/*`).
 */
export function MemoryCard() {
  const addToast = useToastStore(s => s.addToast)
  const [stats, setStats] = useState<MemoryStats | null>(null)
  const [items, setItems] = useState<MemoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [searchResults, setSearchResults] = useState<MemoryItem[] | null>(null)
  const [showAdd, setShowAdd] = useState(false)
  const [newContent, setNewContent] = useState('')
  const [newTopic, setNewTopic] = useState('manual')
  const [pendingClear, setPendingClear] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<MemoryItem | null>(null)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [pendingBatchDelete, setPendingBatchDelete] = useState(false)
  const [editingItem, setEditingItem] = useState<MemoryItem | null>(null)
  const [editContent, setEditContent] = useState('')
  const [editTopic, setEditTopic] = useState('')
  const [editImportance, setEditImportance] = useState(0.5)
  const [savingEdit, setSavingEdit] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [adding, setAdding] = useState(false)
  const [searched, setSearched] = useState(false)
  const [archiveStats, setArchiveStats] = useState<MemoryArchiveStats | null>(null)
  const [consolidating, setConsolidating] = useState(false)
  const [pruning, setPruning] = useState(false)
  const [toggling, setToggling] = useState(false)
  const [archiveOpen, setArchiveOpen] = useState(false)
  const [archiveRecords, setArchiveRecords] = useState<MemoryArchiveRecord[]>([])
  const [archiveLoading, setArchiveLoading] = useState(false)
  const [importing, setImporting] = useState(false)
  const [importProgress, setImportProgress] = useState<{ current: number; total: number } | null>(null)
  const [expandedRecordId, setExpandedRecordId] = useState<string | null>(null)
  const [exportingArchive, setExportingArchive] = useState(false)
  const [retentionDays, setRetentionDays] = useState<number | null>(null)
  const [retentionLoading, setRetentionLoading] = useState(true)
  const [savingRetention, setSavingRetention] = useState(false)
  const [activeTopic, setActiveTopic] = useState<string | null>(null)
  const [showAllItems, setShowAllItems] = useState(false)
  const [sortOrder, setSortOrder] = useState<'newest' | 'oldest' | 'importance'>('newest')

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const [statsResult, listResult, archiveResult] = await Promise.all([
        memoryController.stats().catch((e) => { addToast('Could not load memory stats', 'error'); return null }),
        memoryController.list(),
        memoryController.archiveStats().catch((e) => { addToast('Could not load archive stats', 'error'); return null }),
      ])
      setStats(statsResult)
      setItems(listResult.items || [])
      setArchiveStats(archiveResult)
    } catch {
      addToast('Could not load memory', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => {
    let active = true
    const load = async () => {
      setLoading(true)
      try {
        const [statsResult, listResult, archiveResult] = await Promise.all([
          memoryController.stats().catch((e) => { addToast('Could not load memory stats', 'error'); return null }),
          memoryController.list(),
          memoryController.archiveStats().catch((e) => { addToast('Could not load archive stats', 'error'); return null }),
        ])
        if (active) {
          setStats(statsResult)
          setItems(listResult.items || [])
          setArchiveStats(archiveResult)
        }
      } catch {
        if (active) addToast('Could not load memory', 'error')
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    return () => { active = false }
  }, [addToast])

  const fetchConfig = useCallback(async () => {
    setRetentionLoading(true)
    try {
      const config = await memoryController.getConfig()
      setRetentionDays(config.archive_retention_days ?? 30)
    } catch {
      setRetentionDays(null)
    } finally {
      setRetentionLoading(false)
    }
  }, [])

  useEffect(() => {
    let active = true
    const load = async () => {
      setRetentionLoading(true)
      try {
        const config = await memoryController.getConfig()
        if (active) setRetentionDays(config.archive_retention_days ?? 30)
      } catch {
        if (active) setRetentionDays(null)
      } finally {
        if (active) setRetentionLoading(false)
      }
    }
    void load()
    return () => { active = false }
  }, [])

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
    } catch {
      addToast('Could not memory search', 'error')
    }
  }, [addToast])

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => { handleSearch(search) }, SEARCH_DEBOUNCE_MS)
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [search, handleSearch])

  const handleAdd = useCallback(async () => {
    if (!newContent.trim()) return
    setAdding(true)
    try {
      const result = await memoryController.store(newContent, newTopic.trim() || 'manual')
      if (result.stored) {
        addToast('Saved to memory', 'success')
        setNewContent('')
        setShowAdd(false)
        await fetchData()
      } else {
        addToast('Already remembered (or memory is disabled)', 'error')
      }
    } catch {
      addToast('Could not store memory', 'error')
    } finally {
      setAdding(false)
    }
  }, [newContent, newTopic, addToast, fetchData])

  const handleCopy = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      addToast('Memory fact copied', 'success')
    } catch {
      addToast('Could not copy fact', 'error')
    }
  }, [addToast])

  const handleClear = useCallback(async () => {
    setClearing(true)
    try {
      const result = await memoryController.clear()
      addToast(`Cleared ${result.cleared} memory items`, 'success')
      setPendingClear(false)
      setSearch('')
      setSearchResults(null)
      setSearched(false)
      await fetchData()
    } catch {
      addToast('Could not clear memory', 'error')
    } finally {
      setClearing(false)
    }
  }, [addToast, fetchData])

  const handleDeleteItem = useCallback(async () => {
    if (!pendingDelete) return
    setDeleting(true)
    try {
      const result = await memoryController.delete(pendingDelete.id)
      addToast(result.deleted > 0 ? 'Memory item deleted' : 'Memory item not found', result.deleted > 0 ? 'success' : 'error')
      setPendingDelete(null)
      if (searchResults !== null) {
        setSearchResults(searchResults.filter(i => i.id !== pendingDelete.id))
      }
      await fetchData()
    } catch {
      addToast('Could not delete memory item', 'error')
    } finally {
      setDeleting(false)
    }
  }, [pendingDelete, searchResults, addToast, fetchData])

  const toggleSelect = useCallback((id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const clearSelection = useCallback(() => {
    setSelectedIds(new Set())
  }, [])

  const handleExportSelected = useCallback(() => {
    const selected = items.filter(i => selectedIds.has(i.id))
    if (selected.length === 0) {
      addToast('Nothing selected to export', 'error')
      return
    }
    const data = selected.map(i => ({ content: i.content, topic: i.topic || 'manual', source: i.source || 'api' }))
    downloadJson(data, `memory-export-selected-${todayDateString()}.json`)
    addToast(`Exported ${selected.length} memory item(s)`, 'success')
  }, [items, selectedIds, addToast])

  const handleBatchDelete = useCallback(async () => {
    if (selectedIds.size === 0) return
    setDeleting(true)
    try {
      const ids = new Set(selectedIds)
      let deleted = 0
      for (const id of ids) {
        try {
          const result = await memoryController.delete(id)
          if (result.deleted > 0) deleted++
        } catch { /* continue deleting the rest */ }
      }
      addToast(deleted > 0 ? `Deleted ${deleted} memory item(s)` : 'Selected items not found', deleted > 0 ? 'success' : 'error')
      setPendingBatchDelete(false)
      setSelectedIds(new Set())
      if (searchResults !== null) {
        setSearchResults(searchResults.filter(i => !ids.has(i.id)))
      }
      await fetchData()
    } catch {
      addToast('Could not delete memory items', 'error')
    } finally {
      setDeleting(false)
    }
  }, [selectedIds, searchResults, addToast, fetchData])

  const startEdit = useCallback((itemToEdit: MemoryItem) => {
    setEditingItem(itemToEdit)
    setEditContent(itemToEdit.content)
    setEditTopic(itemToEdit.topic || '')
    setEditImportance(itemToEdit.importance != null ? itemToEdit.importance : 0.5)
  }, [])

  const handleSaveEdit = useCallback(async () => {
    if (!editingItem || !editContent.trim()) return
    setSavingEdit(true)
    try {
      const result = await memoryController.update(editingItem.id, editContent, editTopic, editImportance)
      if (result.updated > 0) {
        addToast('Memory item updated', 'success')
        setEditingItem(null)
        if (searchResults !== null) {
          setSearchResults(searchResults.map(i =>
            i.id === editingItem.id
              ? { ...i, content: editContent.trim(), topic: editTopic.trim() || i.topic, importance: editImportance }
              : i,
          ))
        }
        await fetchData()
      } else if (result.duplicate) {
        addToast('That fact already exists in memory', 'error')
      } else {
        addToast('Memory item not found', 'error')
      }
    } catch {
      addToast('Could not update memory item', 'error')
    } finally {
      setSavingEdit(false)
    }
  }, [editingItem, editContent, editTopic, editImportance, searchResults, addToast, fetchData])

  const handleConsolidate = useCallback(async () => {
    setConsolidating(true)
    try {
      const result = await memoryController.consolidate()
      if (result.removed > 0) {
        addToast(`Consolidated ${result.removed} duplicate fact(s), kept ${result.kept}`, 'success')
      } else {
        addToast('No near-duplicate facts found', 'info')
      }
      await fetchData()
    } catch {
      addToast('Could not consolidate memory', 'error')
    } finally {
      setConsolidating(false)
    }
  }, [addToast, fetchData])

  const handlePruneArchive = useCallback(async () => {
    setPruning(true)
    try {
      const result = await memoryController.archivePrune(retentionDays ?? undefined)
      addToast(result.pruned > 0 ? `Pruned ${result.pruned} archive record(s)` : 'Archive already within retention', 'success')
      await fetchData()
    } catch {
      addToast('Could not prune archive', 'error')
    } finally {
      setPruning(false)
    }
  }, [addToast, fetchData, retentionDays])

  const handleSaveRetention = useCallback(async () => {
    if (retentionDays == null || Number.isNaN(retentionDays)) {
      addToast('Enter a retention window in days', 'error')
      return
    }
    const days = Math.round(Math.max(0, Math.min(retentionDays, 3650)))
    setSavingRetention(true)
    try {
      const config = await memoryController.updateConfig({ archive_retention_days: days })
      setRetentionDays(config.archive_retention_days)
      addToast(`Archive retention set to ${config.archive_retention_days} day(s)`, 'success')
    } catch {
      addToast('Could not save retention', 'error')
    } finally {
      setSavingRetention(false)
    }
  }, [retentionDays, addToast])

  const loadArchive = useCallback(async () => {
    setArchiveLoading(true)
    try {
      const result = await memoryController.archive(20)
      setArchiveRecords(result.records || [])
    } catch {
      addToast('Could not load archive records', 'error')
    } finally {
      setArchiveLoading(false)
    }
  }, [addToast])

  const openArchive = useCallback(() => {
    setArchiveOpen(true)
    loadArchive()
  }, [loadArchive])

  const handleExportMemory = useCallback(async () => {
    try {
      const result = await memoryController.list(1000)
      const data = (result.items || []).map(i => ({ content: i.content, topic: i.topic || 'manual', source: i.source || 'api' }))
      downloadJson(data, `memory-export-${todayDateString()}.json`)
      addToast(`Exported ${data.length} memory item(s)`, 'success')
    } catch {
      addToast('Could not export memory', 'error')
    }
  }, [addToast])

  const handleImportMemory = useCallback(async () => {
    const file = await importFile('.json,.csv')
    if (!file) return
    setImporting(true)
    setImportProgress({ current: 0, total: 0 })
    try {
      const text = await file.text()
      const entries = parseMemoryImport(text, file.name)
      if (entries.length === 0) {
        addToast('No memory items found in file', 'error')
        return
      }
      setImportProgress({ current: 0, total: entries.length })
      let stored = 0
      for (let i = 0; i < entries.length; i++) {
        try {
          const result = await memoryController.store(entries[i].content, entries[i].topic)
          if (result.stored) stored++
        } catch { /* skip unimportable entry */ }
        setImportProgress({ current: i + 1, total: entries.length })
      }
      addToast(`Imported ${stored} of ${entries.length} memory item(s)`, 'success')
      await fetchData()
    } catch {
      addToast('Could not import memory', 'error')
    } finally {
      setImporting(false)
      setImportProgress(null)
    }
  }, [addToast, fetchData])

  const handleExportArchive = useCallback(async () => {
    setExportingArchive(true)
    try {
      const result = await memoryController.archive(1000)
      downloadJson(result.records || [], `memory-archive-${todayDateString()}.json`)
      addToast(`Exported ${result.records?.length ?? 0} archive record(s)`, 'success')
    } catch {
      addToast('Could not export archive', 'error')
    } finally {
      setExportingArchive(false)
    }
  }, [addToast])

  const handleToggleEnabled = useCallback(async (next: boolean) => {
    setToggling(true)
    try {
      const result = await memoryController.setEnabled(next)
      setStats(prev => (prev ? { ...prev, enabled: result.enabled } : prev))
      addToast(
        result.enabled ? 'Memory is on — the AI will keep learning from conversations' : 'Memory is off — the AI will stop storing new facts',
        'success',
      )
      setSearch('')
      setSearchResults(null)
      setSearched(false)
      await fetchData()
    } catch {
      addToast('Could not update memory setting', 'error')
    } finally {
      setToggling(false)
    }
  }, [addToast, fetchData])

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
  const visibleTopics = topics.slice(0, 8)
  const extraTopicCount = Math.max(0, topics.length - visibleTopics.length)

  const browseList = useMemo(() => {
    const base = activeTopic ? items.filter(i => i.topic === activeTopic) : items
    return [...base].sort((a, b) => {
      if (sortOrder === 'importance') return (b.importance ?? 0) - (a.importance ?? 0)
      return sortOrder === 'newest' ? b.timestamp - a.timestamp : a.timestamp - b.timestamp
    })
  }, [items, activeTopic, sortOrder])

  const filteredByTopic = useMemo(() => {
    const base = searchResults !== null ? searchResults : browseList
    if (!activeTopic) return base
    return base.filter(i => i.topic === activeTopic)
  }, [searchResults, browseList, activeTopic])

  const displayedItems = filteredByTopic
  const visibleList = showAllItems ? filteredByTopic : filteredByTopic.slice(0, 10)
  const itemCount = searchResults !== null ? searchResults.length : stats?.total_facts ?? items.length

  const toggleSelectAll = useCallback(() => {
    setSelectedIds(prev => {
      if (prev.size === displayedItems.length && displayedItems.length > 0) return new Set()
      return new Set(displayedItems.map(i => i.id))
    })
  }, [displayedItems])

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-base flex items-center gap-2">
          <IconBrain className="h-4 w-4 text-primary" />
          Memory
          {stats && (
            <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${stats.enabled ? 'bg-success/15 text-success' : 'bg-muted text-muted-foreground'}`}>
              {stats.enabled ? 'Active' : 'Off'}
            </span>
          )}
        </CardTitle>
        <div className="flex items-center gap-1.5">
          <div className="flex items-center gap-1.5 mr-1.5 pr-1.5 border-r border-border/60">
            <span className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">Remember</span>
            <Switch
              size="sm"
              checked={stats?.enabled ?? false}
              onCheckedChange={handleToggleEnabled}
              disabled={toggling || stats === null}
              aria-label="Toggle automatic memory"
            />
          </div>
          <Button size="sm" variant="outline" className="h-7 text-xs" onClick={fetchData} disabled={loading}>
            <IconRefresh className={loading ? 'animate-spin h-3 w-3 mr-1' : 'h-3 w-3 mr-1'} />
            Refresh
          </Button>
          <Button size="sm" variant="outline" className="h-7 text-xs text-destructive" onClick={() => setPendingClear(true)} disabled={clearing || itemCount === 0}>
            <IconTrash className="h-3 w-3 mr-1" />
            Clear
          </Button>
        </div>
      </CardHeader>

      <CardContent>
        <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium mb-3">
          Facts the AI remembers across conversations
        </p>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
          <div className="rounded-lg border border-border/60 p-3">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Facts</p>
            {loading ? <Skeleton className="h-6 w-12 mt-1" /> : <p className="text-xl font-semibold mt-1">{stats?.total_facts ?? 0}</p>}
          </div>
          <div className="rounded-lg border border-border/60 p-3">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Topics</p>
            {loading ? <Skeleton className="h-6 w-12 mt-1" /> : <p className="text-xl font-semibold mt-1">{stats?.topics ?? 0}</p>}
          </div>
          <div className="rounded-lg border border-border/60 p-3">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Visited URLs</p>
            {loading ? <Skeleton className="h-6 w-12 mt-1" /> : <p className="text-xl font-semibold mt-1">{stats?.visited_urls ?? 0}</p>}
          </div>
          <div className="rounded-lg border border-border/60 p-3">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Searchable</p>
            {loading ? <Skeleton className="h-6 w-12 mt-1" /> : (
              <p className="text-xl font-semibold mt-1">{stats?.enabled ? 'Yes' : 'No'}</p>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2 mb-3">
          <div className="relative flex-1">
            <IconSearch className="h-3.5 w-3.5 text-muted-foreground absolute left-2.5 top-1/2 -translate-y-1/2" />
            <Input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search memory..."
              className="pl-8 h-8 text-xs"
              aria-label="Search memory"
            />
            {search && (
              <button
                type="button"
                onClick={() => setSearch('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                aria-label="Clear search"
              >
                <IconX className="h-3 w-3" />
              </button>
            )}
          </div>
          <Button size="sm" variant="outline" className="h-8 text-xs" onClick={() => setShowAdd(v => !v)}>
            {showAdd ? <IconX className="h-3 w-3 mr-1" /> : <IconPlus className="h-3 w-3 mr-1" />}
            {showAdd ? 'Close' : 'Store fact'}
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                size="sm"
                variant="ghost"
                className="h-8 text-xs shrink-0"
                disabled={searchResults !== null}
                title={searchResults !== null ? 'Search results use relevance order' : undefined}
                aria-label="Toggle memory sort order"
              >
                <IconClock className="h-3 w-3 mr-1" />
                {sortOrder === 'importance' ? 'Importance' : sortOrder === 'newest' ? 'Newest' : 'Oldest'}
                <IconChevronDown className="h-3 w-3 ml-1" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => setSortOrder('newest')}>Newest</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortOrder('oldest')}>Oldest</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortOrder('importance')}>Importance</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        {showAdd && (
          <div className="rounded-lg border border-border/60 p-3 mb-3 space-y-2">
            <textarea
              className="w-full p-2 text-sm border rounded-lg resize-none h-16 bg-background focus:outline-none focus:ring-1 focus:ring-primary/40 border-input"
              placeholder="Type a fact the AI should remember..."
              value={newContent}
              onChange={e => setNewContent(e.target.value)}
              aria-label="New memory fact"
            />
            <div className="flex items-center gap-2">
              <label className="text-xs text-muted-foreground shrink-0">Topic:</label>
              <Input
                value={newTopic}
                onChange={e => setNewTopic(e.target.value)}
                className="h-8 text-xs flex-1"
                placeholder="manual"
              />
              <Button size="sm" className="h-8 text-xs" disabled={!newContent.trim() || adding} onClick={handleAdd}>
                {adding ? 'Saving…' : 'Save'}
              </Button>
            </div>
          </div>
        )}

        {editingItem && (
          <div className="rounded-lg border border-primary/40 p-3 mb-3 space-y-2">
            <p className="text-xs font-medium text-foreground flex items-center gap-1.5">
              <IconEdit className="h-3.5 w-3.5 text-muted-foreground" />
              Edit memory fact
            </p>
            <textarea
              className="w-full p-2 text-sm border rounded-lg resize-none h-16 bg-background focus:outline-none focus:ring-1 focus:ring-primary/40 border-input"
              placeholder="Fact text..."
              value={editContent}
              onChange={e => setEditContent(e.target.value)}
              aria-label="Edit memory fact text"
            />
            <div className="flex items-center gap-2">
              <label className="text-xs text-muted-foreground shrink-0">Topic:</label>
              <Input
                value={editTopic}
                onChange={e => setEditTopic(e.target.value)}
                className="h-8 text-xs flex-1"
                placeholder={editingItem.topic || 'manual'}
              />
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-muted-foreground shrink-0">Importance:</label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.1}
                value={editImportance}
                onChange={e => setEditImportance(Number(e.target.value))}
                className="flex-1 h-1 accent-primary"
                aria-label="Edit memory fact importance"
              />
              <span className="text-xs text-muted-foreground font-mono w-8 text-right shrink-0">{editImportance.toFixed(1)}</span>
            </div>
            <div className="flex items-center gap-2">
              <Button size="sm" className="h-8 text-xs" disabled={!editContent.trim() || savingEdit} onClick={handleSaveEdit}>
                {savingEdit ? 'Saving…' : 'Save'}
              </Button>
              <Button size="sm" variant="ghost" className="h-8 text-xs" onClick={() => setEditingItem(null)}>
                Cancel
              </Button>
            </div>
          </div>
        )}

        {topics.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 mb-3" aria-label="Filter by topic">
            <button
              type="button"
              onClick={() => setActiveTopic(null)}
              className={`text-[10px] px-2 py-1 rounded-full font-medium transition-colors ${activeTopic === null ? 'bg-primary/15 text-primary' : 'bg-muted text-muted-foreground hover:bg-muted/70'}`}
            >
              All
            </button>
            {visibleTopics.map(topic => (
              <button
                key={topic}
                type="button"
                onClick={() => setActiveTopic(activeTopic === topic ? null : topic)}
                className={`text-[10px] px-2 py-1 rounded-full font-medium transition-colors ${activeTopic === topic ? 'bg-primary/15 text-primary' : 'bg-muted text-muted-foreground hover:bg-muted/70'}`}
              >
                {topic}
              </button>
            ))}
            {extraTopicCount > 0 && (
              <span className="text-[10px] text-muted-foreground">+{extraTopicCount} more</span>
            )}
          </div>
        )}

        {loading ? (
          <div className="space-y-2">
            <Skeleton className="h-12 rounded-lg" />
            <Skeleton className="h-12 rounded-lg" />
            <Skeleton className="h-12 rounded-lg" />
          </div>
        ) : displayedItems.length === 0 ? (
          <div className="text-center py-6 text-sm text-muted-foreground">
            {activeTopic
              ? `No memory in the "${activeTopic}" topic.`
              : (searched ? 'No memory matches that search.' : 'Nothing remembered yet. The AI stores facts automatically as you chat.')}
            {searched && !activeTopic && (
              <div className="mt-2">
                <Button
                  size="sm"
                  variant="outline"
                  className="h-8 text-xs"
                  onClick={() => { setSearch(''); setSearchResults(null); setSearched(false) }}
                >
                  Clear search
                </Button>
              </div>
            )}
          </div>
        ) : (
          <div>
            {displayedItems.length > 1 && (
              <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer mb-1.5">
                <input
                  type="checkbox"
                  checked={selectedIds.size === displayedItems.length}
                  onChange={toggleSelectAll}
                  className="rounded border-border"
                  aria-label="Select all memory facts"
                />
                Select all ({displayedItems.length})
              </label>
            )}
            {selectedIds.size > 0 && (
              <div className="flex items-center gap-1.5 mb-2">
                <Button
                  size="sm"
                  variant="outline"
                  className="h-8 text-xs"
                  onClick={handleExportSelected}
                  aria-label={`Export ${selectedIds.size} selected memory items`}
                >
                  <IconDownload className="h-3 w-3 mr-1" />
                  Export ({selectedIds.size})
                </Button>
                <Button
                  size="sm"
                  variant="destructive"
                  className="h-8 text-xs"
                  onClick={() => setPendingBatchDelete(true)}
                  aria-label={`Delete ${selectedIds.size} selected memory items`}
                >
                  <IconTrash className="h-3 w-3 mr-1" />
                  Delete ({selectedIds.size})
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-8 text-xs ml-auto"
                  onClick={clearSelection}
                >
                  Cancel
                </Button>
              </div>
            )}
            <div className="space-y-1.5">
              {visibleList.map(item => (
                <div
                  key={item.id}
                  className={`group flex items-start justify-between gap-2 rounded-lg border px-3 py-2 transition-colors ${
                    selectedIds.has(item.id)
                      ? 'bg-primary/[0.06] border-primary/30'
                      : 'border-border/60 hover:bg-muted/40'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={selectedIds.has(item.id)}
                    onChange={() => toggleSelect(item.id)}
                    className="mt-1 rounded border-border shrink-0"
                    aria-label={`Select memory fact ${item.content}`}
                  />
                  <div className="min-w-0 flex-1">
                  <p
                    className="text-sm line-clamp-2 cursor-pointer select-text hover:text-foreground/80 transition-colors"
                    title="Copy to clipboard"
                    onClick={() => handleCopy(item.content)}
                  >
                    {item.content}
                  </p>
                  <div className="flex items-center gap-1.5 mt-1">
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
                {typeof item.score === 'number' && searchResults !== null && (
                  <span className="text-[10px] text-muted-foreground font-mono shrink-0">{item.score.toFixed(2)}</span>
                )}
                <button
                  type="button"
                  onClick={() => startEdit(item)}
                  className="h-7 w-7 shrink-0 flex items-center justify-center rounded text-muted-foreground opacity-60 lg:opacity-0 lg:group-hover:opacity-100 hover:text-primary hover:bg-primary/10 transition-all"
                  aria-label="Edit memory item"
                >
                  <IconEdit className="h-3.5 w-3.5" />
                </button>
                <button
                  type="button"
                  onClick={() => setPendingDelete(item)}
                  className="h-7 w-7 shrink-0 flex items-center justify-center rounded text-muted-foreground opacity-60 lg:opacity-0 lg:group-hover:opacity-100 hover:text-destructive hover:bg-destructive/10 transition-all"
                  aria-label="Delete memory item"
                >
                  <IconTrash className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
            </div>
          </div>
        )}

        {filteredByTopic.length > 10 && (
          <div className="flex items-center justify-between mt-3 pt-3 border-t border-border/60">
            <p className="text-[10px] text-muted-foreground">
              Showing {visibleList.length} of {filteredByTopic.length}
            </p>
            <Button
              size="sm"
              variant="ghost"
              className="h-7 text-xs"
              onClick={() => setShowAllItems(v => !v)}
            >
              {showAllItems ? 'Show fewer' : 'Show all'}
            </Button>
          </div>
        )}

        <FoldSection heading={
          <span className="flex items-center gap-2">
            <IconSettings className="h-4 w-4 text-muted-foreground" />
            Maintenance
          </span>
        } className="mt-4">
          <div className="space-y-3">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="text-sm font-medium text-foreground flex items-center gap-1.5">
                  <IconFilter className="h-3.5 w-3.5 text-muted-foreground" />
                  Consolidate duplicates
                </p>
                <p className="text-xs mt-0.5">Merge near-identical facts, keeping the longest copy.</p>
              </div>
              <Button
                size="sm"
                variant="outline"
                className="h-8 text-xs shrink-0"
                onClick={handleConsolidate}
                disabled={consolidating || itemCount === 0}
              >
                {consolidating ? 'Merging…' : 'Consolidate'}
              </Button>
            </div>
            <div className="flex items-start justify-between gap-3 border-t border-border pt-3">
              <div className="min-w-0">
                <p className="text-sm font-medium text-foreground flex items-center gap-1.5">
                  <IconFolder className="h-3.5 w-3.5 text-muted-foreground" />
                  Provenance archive
                </p>
                <p className="text-xs mt-0.5">
                  {loading ? 'Loading…' : (
                    <>
                      {archiveStats?.records ?? 0} record(s), {archiveStats?.bytes != null ? `${(archiveStats.bytes / 1024).toFixed(1)} KB` : '—'}
                    </>
                  )}
                </p>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <Button
                  size="sm"
                  variant="outline"
                  className="h-8 text-xs shrink-0"
                  onClick={openArchive}
                >
                  View records
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-8 text-xs shrink-0"
                  onClick={handlePruneArchive}
                  disabled={pruning || (archiveStats?.records ?? 0) === 0}
                >
                  {pruning ? 'Pruning…' : 'Prune old'}
                </Button>
              </div>
            </div>
            <div className="flex items-start justify-between gap-3 border-t border-border pt-3">
              <div className="min-w-0">
                <p className="text-sm font-medium text-foreground flex items-center gap-1.5">
                  <IconClock className="h-3.5 w-3.5 text-muted-foreground" />
                  Archive retention
                </p>
                <p className="text-xs mt-0.5">Pruning removes records older than this window (days).</p>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <Input
                  type="number"
                  min={0}
                  max={3650}
                  value={retentionDays ?? ''}
                  onChange={e => setRetentionDays(e.target.value === '' ? null : Number(e.target.value))}
                  placeholder={retentionLoading ? 'Loading…' : '30'}
                  className="h-8 w-20 text-xs"
                  aria-label="Archive retention days"
                />
                <Button
                  size="sm"
                  variant="outline"
                  className="h-8 text-xs shrink-0"
                  onClick={handleSaveRetention}
                  disabled={savingRetention || retentionLoading}
                >
                  {savingRetention ? 'Saving…' : 'Save'}
                </Button>
              </div>
            </div>
            <div className="flex items-start justify-between gap-3 border-t border-border pt-3">
              <div className="min-w-0">
                <p className="text-sm font-medium text-foreground flex items-center gap-1.5">
                  <IconDownload className="h-3.5 w-3.5 text-muted-foreground" />
                  Backup memory
                </p>
                <p className="text-xs mt-0.5">
                  {importProgress ? `Importing ${importProgress.current}/${importProgress.total}…` : 'Export all facts as JSON, or import from a JSON/CSV backup.'}
                </p>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <Button
                  size="sm"
                  variant="outline"
                  className="h-8 text-xs shrink-0"
                  onClick={handleExportMemory}
                >
                  Export
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-8 text-xs shrink-0"
                  onClick={handleImportMemory}
                  disabled={importing}
                >
                  {importing ? 'Importing…' : 'Import'}
                </Button>
              </div>
            </div>
          </div>
        </FoldSection>
      </CardContent>

      <AlertDialog open={pendingDelete !== null} onOpenChange={() => setPendingDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this memory?</AlertDialogTitle>
            <AlertDialogDescription>
              This permanently removes the item: “{pendingDelete?.content}”
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDeleteItem} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
              {deleting ? 'Deleting…' : 'Delete'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={pendingBatchDelete} onOpenChange={setPendingBatchDelete}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete {selectedIds.size} memory items?</AlertDialogTitle>
            <AlertDialogDescription>
              This permanently removes the selected memory items from the AI&apos;s memory.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleBatchDelete} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
              {deleting ? 'Deleting…' : 'Delete'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={pendingClear} onOpenChange={setPendingClear}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Clear all memory?</AlertDialogTitle>
            <AlertDialogDescription>
              This permanently removes every stored memory item the AI remembers.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleClear} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
              {clearing ? 'Clearing…' : 'Clear all'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <Dialog open={archiveOpen} onOpenChange={setArchiveOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="text-base flex items-center gap-2">
              <IconFolder className="h-4 w-4 text-primary" />
              Provenance archive
            </DialogTitle>
            <DialogDescription>
              {archiveStats
                ? `${archiveStats.records} record(s) — ${(archiveStats.bytes / 1024).toFixed(1)} KB`
                : 'Task-backed memory records'}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-1.5 max-h-[55vh] overflow-y-auto overscroll-contain pr-1">
            {archiveLoading && archiveRecords.length === 0 ? (
              <div className="space-y-2 py-2">
                <Skeleton className="h-12 rounded-lg" />
                <Skeleton className="h-12 rounded-lg" />
                <Skeleton className="h-12 rounded-lg" />
              </div>
            ) : archiveRecords.length === 0 ? (
              <EmptyCard
                message="No archive records yet"
                description="Task-backed memory records appear here as background tasks store facts."
                icon={<IconFolder className="h-5 w-5" />}
                action={null}
              />
            ) : (
              archiveRecords.map(record => {
                const recordId = record.task_id || `${record.task_type}-${record.ts}`
                const summary = archiveSummary(record)
                const expanded = expandedRecordId === recordId
                return (
                  <div key={recordId} className="rounded-lg border border-border/60">
                    <button
                      type="button"
                      onClick={() => setExpandedRecordId(expanded ? null : recordId)}
                      className="w-full text-left px-3 py-2 hover:bg-muted/40 transition-colors rounded-lg"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className={`text-[9px] px-1.5 py-0.5 rounded-full font-medium ${archiveBadgeClass(record.task_type)}`}>
                          {archiveTypeLabel(record.task_type)}
                        </span>
                        <div className="flex items-center gap-2">
                          {record.ts > 0 && (
                            <span className="text-[9px] text-muted-foreground font-mono">
                              {new Date(record.ts * 1000).toLocaleString()}
                            </span>
                          )}
                          <IconChevronDown className={`h-3.5 w-3.5 text-muted-foreground transition-transform ${expanded ? 'rotate-180' : ''}`} />
                        </div>
                      </div>
                      <p className="text-sm mt-1 line-clamp-2 break-words">{summary.text || '—'}</p>
                      {summary.detail && <p className="text-[10px] text-muted-foreground mt-0.5">{summary.detail}</p>}
                    </button>
                    {expanded && (
                      <pre className="text-[10px] font-mono leading-relaxed text-muted-foreground bg-muted/50 rounded-lg mx-3 mb-3 px-3 py-2 overflow-x-auto whitespace-pre-wrap break-words">
                        {JSON.stringify(record, null, 2)}
                      </pre>
                    )}
                  </div>
                )
              })
            )}
          </div>
          <DialogFooter className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Button size="sm" variant="outline" className="h-8 text-xs" onClick={handleExportArchive} disabled={exportingArchive || archiveRecords.length === 0}>
                <IconDownload className="h-3 w-3 mr-1" />
                {exportingArchive ? 'Exporting…' : 'Export'}
              </Button>
              <Button size="sm" variant="outline" className="h-8 text-xs" onClick={loadArchive} disabled={archiveLoading}>
                <IconRefresh className={archiveLoading ? 'animate-spin h-3 w-3 mr-1' : 'h-3 w-3 mr-1'} />
                Refresh
              </Button>
            </div>
            <Button size="sm" className="h-8 text-xs" onClick={() => setArchiveOpen(false)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  )
}
