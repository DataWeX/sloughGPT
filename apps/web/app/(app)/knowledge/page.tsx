'use client'
export const dynamic = 'force-dynamic'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { PageContainer } from '@/components/PageContainer'

import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from '@sloughgpt/strui'
import { Card, CardContent, Checkbox, EmptyCard, cn, Slider } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import { KnowledgeStatsSkeleton, KnowledgeCategoryChartSkeleton, KnowledgeAdapterSkeleton, KnowledgeRAGSkeleton, KnowledgeTopicsSkeleton } from '@/components/ui/PageSkeletons'
import { Chip } from '@sloughgpt/strui'
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem } from '@sloughgpt/strui'
import { IconRefresh, IconPlus, IconTrash, IconSearch, IconCheck, IconX, IconDownload, IconEdit, IconChevronDown, IconMapPin } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { knowledgeController, type KnowledgeItem, type KnowledgeStats, type TopicCount } from '@/lib/knowledge-controller'
import { getRAGStats, clearRAG, listRAGDocuments, syncKGToRAG, type RAGStats, type RAGDocument } from '@/lib/rag-controller'
import { KnowledgeCategoryChart } from '@/components/knowledge/KnowledgeCategoryChart'
import { MemoryCard } from '@/components/knowledge/MemoryCard'
import { SpacedReviewCard } from '@/components/knowledge/SpacedReviewCard'
import { KnowledgeIntelligenceCard } from '@/components/knowledge/KnowledgeIntelligenceCard'
import { MemorySettingsCard } from '@/components/knowledge/MemorySettingsCard'
import { LearnSection } from '@/components/learn/LearnSection'
import { downloadJson } from '@/lib/download-utils'
import { todayDateString, MS_PER_SECOND } from '@/lib/format-bytes'
import { knowledgeSchema } from '@/lib/validation-schemas'
import { logger } from '@/lib/dev-log'

const SEARCH_DEBOUNCE_MS = 300

export default function KnowledgePage() {
  const addToast = useToastStore(s => s.addToast)
  const [items, setItems] = useState<KnowledgeItem[]>([])
  const [stats, setStats] = useState<KnowledgeStats | null>(null)
  const [topics, setTopics] = useState<TopicCount[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [searchResults, setSearchResults] = useState<KnowledgeItem[] | null>(null)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [pendingDelete, setPendingDelete] = useState<KnowledgeItem | null>(null)
  const [pendingBatchDelete, setPendingBatchDelete] = useState(false)
  const [showBulkTopic, setShowBulkTopic] = useState(false)
  const [bulkTopic, setBulkTopic] = useState('')
  const [showAdd, setShowAdd] = useState(false)
  const [newContent, setNewContent] = useState('')
  const [newTopic, setNewTopic] = useState('general')
  const [activeTopic, setActiveTopic] = useState<string | null>(null)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editContent, setEditContent] = useState('')
  const [editTopic, setEditTopic] = useState('')
  const [addErrors, setAddErrors] = useState<{ content?: string; topic?: string }>({})
  const [editErrors, setEditErrors] = useState<{ content?: string; topic?: string }>({})
  const [ragStats, setRagStats] = useState<RAGStats | null>(null)
  const [ragDocs, setRagDocs] = useState<RAGDocument[]>([])
  const [ragClearing, setRagClearing] = useState(false)
  const [ragSyncing, setRagSyncing] = useState(false)
  const [showRagDocs, setShowRagDocs] = useState(false)
  const [editingImportanceId, setEditingImportanceId] = useState<string | null>(null)
  const [importanceValue, setImportanceValue] = useState(0)
  const [importing, setImporting] = useState(false)
  const [importProgress, setImportProgress] = useState<{ current: number; total: number } | null>(null)
  const [sortBy, setSortBy] = useState<'date' | 'importance' | 'topic'>('date')
  const [adapterStatus, setAdapterStatus] = useState<{ adapter_exists: boolean; fact_count: number; total_facts_available: number; trained_at?: number; post_training_loss?: number } | null>(null)
  const [adapterTraining, setAdapterTraining] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const suggestTopic = (content: string): string => {
    const lower = content.toLowerCase()
    if (/\b(prefer|like|love|hate|favorite|best|worst)\b/.test(lower)) return 'preferences'
    if (/\b(born|live|work|study|age|name|from)\b/.test(lower)) return 'personal'
    if (/\b(project|code|api|bug|fix|deploy|build)\b/.test(lower)) return 'technical'
    if (/\b(meeting|deadline|schedule|plan|goal)\b/.test(lower)) return 'planning'
    if (/\b(book|read|watch|listen|play|game|movie)\b/.test(lower)) return 'interests'
    if (/\b(cook|recipe|food|eat|drink|coffee|tea)\b/.test(lower)) return 'food'
    return 'general'
  }

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const [itemsResult, statsResult, topicsResult, adapterResult] = await Promise.all([
        knowledgeController.list(),
        knowledgeController.stats(),
        knowledgeController.topics(),
        knowledgeController.getAdapterStatus().catch((e) => { logger.warning('Could not adapter status', { exception: String(e) }); return null }),
      ])
      setItems(itemsResult)
      setStats(statsResult)
      setTopics(topicsResult.topics || [])
      if (adapterResult) setAdapterStatus(adapterResult)
    } catch {
      addToast('Could not load knowledge', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => { fetchData() }, [fetchData])

  const fetchRAGData = useCallback(async () => {
    try {
      const [s, docs] = await Promise.all([getRAGStats(), listRAGDocuments()])
      setRagStats(s)
      setRagDocs(docs.documents || [])
    } catch {
      addToast('Could not load deep memory data', 'error')
    }
  }, [])

  useEffect(() => { fetchRAGData() }, [fetchRAGData])

  const handleRAGClear = useCallback(async () => {
    setRagClearing(true)
    try {
      await clearRAG()
      setRagStats({ total_documents: 0, total_chunks: 0, index_size: 0 })
      setRagDocs([])
      addToast('Deep memory index cleared', 'success')
    } catch {
      addToast('Could not clear deep memory index', 'error')
    } finally {
      setRagClearing(false)
    }
  }, [addToast])

  const handleRAGSync = useCallback(async () => {
    setRagSyncing(true)
    try {
      const result = await syncKGToRAG()
      addToast(`Synced ${result.total_triples} knowledge connections to deep memory`, 'success')
      await fetchRAGData()
    } catch {
      addToast('Could not sync to deep memory', 'error')
    } finally {
      setRagSyncing(false)
    }
  }, [addToast, fetchRAGData])

  const handleTrainAdapter = useCallback(async () => {
    setAdapterTraining(true)
    try {
      const result = await knowledgeController.trainAdapter()
      setAdapterStatus(result.adapter_status)
      addToast(`Adapter trained on ${result.fact_count} facts in ${result.elapsed.toFixed(1)}s`, 'success')
    } catch {
      addToast('Could not adapter training', 'error')
    } finally {
      setAdapterTraining(false)
    }
  }, [addToast])

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
    try {
      const result = await knowledgeController.search(search)
      setSearchResults(result.results || [])
    } catch {
      addToast('Could not search', 'error')
    }
  }, [search, addToast])

  useEffect(() => {
    const timer = setTimeout(() => {
      if (search.trim()) handleSearch()
      else setSearchResults(null)
    }, SEARCH_DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [search, handleSearch])

  const displayItems = useMemo(() => (searchResults ?? (activeTopic
    ? items.filter(i => i.topic === activeTopic)
    : items)).slice().sort((a, b) => {
    if (sortBy === 'importance') return b.importance - a.importance
    if (sortBy === 'topic') return (a.topic || '').localeCompare(b.topic || '')
    return b.timestamp - a.timestamp
  }), [searchResults, activeTopic, items, sortBy])

  function highlightText(text: string, query: string): React.ReactNode {
    if (!query) return text
    const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    const regex = new RegExp(`(${escaped})`, 'gi')
    const parts = text.split(regex)
    return parts.map((part, i) =>
      regex.test(part) ? (
        <mark key={i} className="bg-warning/30 text-foreground rounded px-0.5">{part}</mark>
      ) : (
        part
      )
    )
  }

  const handleDelete = async () => {
    if (!pendingDelete) return
    const deleted = pendingDelete
    setItems(prev => prev.filter(i => i.id !== deleted.id))
    setStats(prev => prev ? { ...prev, total_items: prev.total_items - 1 } : prev)
    setPendingDelete(null)
    try {
      await knowledgeController.delete(deleted.id)
      addToast('Deleted', 'info', undefined, () => {
        setItems(prev => [deleted, ...prev])
        setStats(prev => prev ? { ...prev, total_items: prev.total_items + 1 } : prev)
        knowledgeController.add(deleted.content, deleted.topic, false).catch(() => {
          addToast('Could not restore item', 'error')
        })
      })
    } catch {
      setItems(prev => [deleted, ...prev])
      setStats(prev => prev ? { ...prev, total_items: prev.total_items + 1 } : prev)
      addToast('Could not delete', 'error')
    }
  }

  const handleBatchDelete = async () => {
    if (selectedIds.size === 0) return
    const deletedIds = new Set(selectedIds)
    const deletedItems = items.filter(i => deletedIds.has(i.id))
    setItems(prev => prev.filter(i => !deletedIds.has(i.id)))
    setStats(prev => prev ? { ...prev, total_items: prev.total_items - deletedIds.size } : prev)
    setSelectedIds(new Set())
    setPendingBatchDelete(false)
    try {
      await knowledgeController.batchDelete(Array.from(deletedIds))
      addToast(`Deleted ${deletedIds.size} items`, 'info', undefined, () => {
        setItems(prev => [...deletedItems, ...prev])
        setStats(prev => prev ? { ...prev, total_items: prev.total_items + deletedIds.size } : prev)
        knowledgeController.batchIngest(deletedItems.map(i => ({ content: i.content, tags: [i.topic] }))).catch(e => {
          logger.error('knowledge batch undo re-ingest failed', { count: deletedIds.size, exception: String(e) })
          addToast('Could not restore deleted items', 'error')
        })
      })
    } catch {
      setItems(prev => [...deletedItems, ...prev])
      setStats(prev => prev ? { ...prev, total_items: prev.total_items + deletedIds.size } : prev)
      addToast('Could not batch delete', 'error')
    }
  }

  const handleBulkTopicReassign = async () => {
    if (selectedIds.size === 0 || !bulkTopic.trim()) return
    const newTopic = bulkTopic.trim()
    const affectedIds = new Set(selectedIds)
    const oldTopics = new Map(items.filter(i => affectedIds.has(i.id)).map(i => [i.id, i.topic]))
    setItems(prev => prev.map(i => affectedIds.has(i.id) ? { ...i, topic: newTopic } : i))
    setSelectedIds(new Set())
    setShowBulkTopic(false)
    setBulkTopic('')
    try {
      await Promise.all(Array.from(affectedIds).map(id => knowledgeController.update(id, { topic: newTopic })))
      addToast(`Updated ${affectedIds.size} items to "${newTopic}"`, 'success')
    } catch {
      setItems(prev => prev.map(i => affectedIds.has(i.id) ? { ...i, topic: oldTopics.get(i.id) || 'general' } : i))
      addToast('Could not reassign topics', 'error')
    }
  }

  const handleAdd = async () => {
    const result = knowledgeSchema.safeParse({ content: newContent, topic: newTopic })
    if (!result.success) {
      const fieldErrors: { content?: string; topic?: string } = {}
      result.error.issues.forEach(issue => {
        const field = issue.path[0] as string
        if (field === 'content') fieldErrors.content = issue.message
        if (field === 'topic') fieldErrors.topic = issue.message
      })
      setAddErrors(fieldErrors)
      return
    }
    setAddErrors({})
    const tempId = `temp-${Date.now()}`
    const optimisticItem: KnowledgeItem = {
      id: tempId,
      content: newContent.trim(),
      topic: newTopic,
      source: 'manual',
      url: '',
      timestamp: Date.now(),
      importance: 0,
      score: 1,
    }
    setItems(prev => [optimisticItem, ...prev])
    setStats(prev => prev ? { ...prev, total_items: prev.total_items + 1 } : prev)
    setNewContent('')
    setNewTopic('general')
    setShowAdd(false)
    try {
      await knowledgeController.add(newContent.trim(), newTopic, true)
      addToast('Knowledge added', 'info')
      await fetchData()
    } catch {
      setItems(prev => prev.filter(i => i.id !== tempId))
      setStats(prev => prev ? { ...prev, total_items: prev.total_items - 1 } : prev)
      addToast('Could not add knowledge', 'error')
    }
  }

  const handleSaveEdit = async (id: string) => {
    const result = knowledgeSchema.safeParse({ content: editContent, topic: editTopic })
    if (!result.success) {
      const fieldErrors: { content?: string; topic?: string } = {}
      result.error.issues.forEach(issue => {
        const field = issue.path[0] as string
        if (field === 'content') fieldErrors.content = issue.message
        if (field === 'topic') fieldErrors.topic = issue.message
      })
      setEditErrors(fieldErrors)
      return
    }
    setEditErrors({})
    try {
      await knowledgeController.update(id, { content: editContent.trim(), topic: editTopic.trim() || 'general' })
      setItems(prev => prev.map(i => i.id === id ? { ...i, content: editContent.trim(), topic: editTopic.trim() || 'general' } : i))
      setEditingId(null)
      addToast('Updated', 'info')
    } catch {
      addToast('Could not update', 'error')
    }
  }

  const handleSaveImportance = async (id: string) => {
    try {
      await knowledgeController.update(id, { importance: importanceValue })
      setItems(prev => prev.map(i => i.id === id ? { ...i, importance: importanceValue } : i))
      setEditingImportanceId(null)
      addToast('Importance updated', 'info')
    } catch {
      addToast('Could not update', 'error')
    }
  }

  const handleExport = () => {
    const data = items.map(i => ({ content: i.content, topic: i.topic, source: i.source, importance: i.importance }))
    downloadJson(data, `knowledge-export-${todayDateString()}.json`)
    addToast(`Exported ${items.length} items`, 'success')
  }

  const handleExportSelected = () => {
    const selected = items.filter(i => selectedIds.has(i.id))
    const data = selected.map(i => ({ content: i.content, topic: i.topic, source: i.source, importance: i.importance }))
    downloadJson(data, `knowledge-export-selected-${todayDateString()}.json`)
    addToast(`Exported ${selected.length} selected items`, 'success')
  }

  const handleExportCSV = () => {
    const headers = ['content', 'topic', 'source', 'importance']
    const rows = items.map(i => [
      i.content, i.topic || '', i.source || '', i.importance.toString()
    ])
    const csv = [
      headers.join(','),
      ...rows.map(row => row.map(val =>
        val.includes(',') || val.includes('"') || val.includes('\n')
          ? `"${val.replace(/"/g, '""')}"` : val
      ).join(','))
    ].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = `knowledge-export-${todayDateString()}.csv`
    a.click(); URL.revokeObjectURL(url)
    addToast(`Exported ${items.length} items as CSV`, 'success')
  }

  const handleImportFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setImporting(true)
    setImportProgress({ current: 0, total: 0 })
    try {
      const text = await file.text()
      const splitLines = (t: string) => t.split('\n').map(l => l.trim()).filter(Boolean)
      let facts: string[]
      if (file.name.endsWith('.json')) {
        const parsed = JSON.parse(text)
        facts = Array.isArray(parsed) ? parsed.map((i: string | { content?: string }) => typeof i === 'string' ? i : i.content || JSON.stringify(i)).filter(Boolean) : []
      } else if (file.name.endsWith('.csv')) {
        const lines = splitLines(text)
        if (lines.length === 0) { addToast('No data found in CSV', 'error'); return }
        const header = lines[0].toLowerCase()
        const contentIdx = header.split(',').findIndex(h => h.includes('content'))
        const topicIdx = header.split(',').findIndex(h => h.includes('topic'))
        facts = lines.slice(1).map(line => {
          const cols = line.split(',').map(c => c.trim().replace(/^"|"$/g, ''))
          const content = contentIdx >= 0 ? cols[contentIdx] : cols[0]
          const topic = topicIdx >= 0 ? cols[topicIdx] : ''
          return content ? `${content}${topic ? ` [${topic}]` : ''}` : ''
        }).filter(Boolean)
      } else {
        facts = splitLines(text)
      }
      if (facts.length === 0) { addToast('No facts found in file', 'error'); return }
      setImportProgress({ current: 0, total: facts.length })
      const batchSize = 50
      for (let i = 0; i < facts.length; i += batchSize) {
        const batch = facts.slice(i, i + batchSize)
        await knowledgeController.bulkIngest(batch, 'imported', 'file-import')
        setImportProgress({ current: Math.min(i + batchSize, facts.length), total: facts.length })
      }
      addToast(`Imported ${facts.length} facts`, 'success')
      await fetchData()
    } catch {
      addToast('Could not import', 'error')
    } finally {
      setImporting(false)
      setImportProgress(null)
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

  const headerRight = useMemo(() => (
    <div className="flex items-center gap-2">
      <Button size="sm" variant="outline" className="h-7 text-xs" onClick={fetchData} disabled={loading}>
        <IconRefresh className={loading ? 'animate-spin h-3 w-3 mr-1' : 'h-3 w-3 mr-1'} />
        Refresh
      </Button>
      {items.length > 0 && (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button size="sm" variant="outline" className="h-7 text-xs">
              <IconDownload className="h-3 w-3 mr-1" />
              Export
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="min-w-[140px]">
            <DropdownMenuItem onClick={handleExport}>
              Export as JSON
            </DropdownMenuItem>
            <DropdownMenuItem onClick={handleExportCSV}>
              Export as CSV
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      )}
      <Button size="sm" variant="outline" className="h-7 text-xs" onClick={() => fileInputRef.current?.click()} disabled={importing}>
        {importing ? 'Importing...' : 'Import'}
      </Button>
      <input ref={fileInputRef} type="file" accept=".json,.txt,.csv" className="hidden" onChange={handleImportFile} aria-label="Import knowledge file" />
      {importProgress && importProgress.total > 0 && (
        <div className="flex items-center gap-2">
          <div className="w-24 h-1.5 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-primary transition-all duration-300"
              style={{ width: `${(importProgress.current / importProgress.total) * 100}%` }}
            />
          </div>
          <span className="text-xs text-muted-foreground font-mono">
            {importProgress.current}/{importProgress.total}
          </span>
        </div>
      )}
      <Button size="sm" className="h-7 text-xs" onClick={() => setShowAdd(true)}>
        <IconPlus className="h-3 w-3 mr-1" />
        Add
      </Button>
    </div>
  ), [loading, items.length, fetchData, handleExport, setShowAdd])

  const toolbar = useMemo(() => (
    <div className="flex items-center gap-2">
      <div className="relative flex-1">
        <IconSearch className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground/60" />
        <Input
          placeholder="Search knowledge..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="h-9 text-sm pl-9"
        />
      </div>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button size="sm" variant="outline" className="h-9 text-xs gap-1">
            {sortBy === 'date' ? 'Newest' : sortBy === 'importance' ? 'Importance' : 'Topic'}
            <IconChevronDown className="h-3 w-3" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onClick={() => setSortBy('date')}>Newest</DropdownMenuItem>
          <DropdownMenuItem onClick={() => setSortBy('importance')}>Importance</DropdownMenuItem>
          <DropdownMenuItem onClick={() => setSortBy('topic')}>Topic</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      {selectedIds.size > 0 && (
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            className="h-8 text-xs"
            onClick={handleExportSelected}
          >
            Export ({selectedIds.size})
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="h-8 text-xs"
            onClick={() => setShowBulkTopic(true)}
          >
            <IconMapPin className="h-3.5 w-3.5 mr-1" />
            Move to topic ({selectedIds.size})
          </Button>
          <Button
            size="sm"
            variant="destructive"
            className="h-8 text-xs"
            onClick={() => setPendingBatchDelete(true)}
          >
            <IconTrash className="h-3 w-3 mr-1" />
            Delete ({selectedIds.size})
          </Button>
        </div>
      )}
    </div>
  ), [search, sortBy, selectedIds.size, setPendingBatchDelete])

  return (
    <PageContainer
      title="Knowledge"
      subtitle="Manage facts the AI remembers"
      headerRight={headerRight}
      toolbar={toolbar}
    >
      {loading && !stats ? (
          <KnowledgeStatsSkeleton />
        ) : stats ? (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <Card>
              <CardContent className="p-4">
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">Things I remember</p>
                <p className="text-xl font-semibold mt-1">{stats.total_items}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">Topics</p>
                <p className="text-xl font-semibold mt-1">{stats.topic_count}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">How much I care</p>
                <p className="text-xl font-semibold mt-1">{stats.avg_importance >= 0.7 ? 'A lot' : stats.avg_importance >= 0.4 ? 'Somewhat' : 'A little'}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">Searchable</p>
                <p className="text-xl font-semibold mt-1">{stats.searchable ? 'Yes' : 'No'}</p>
              </CardContent>
            </Card>
          </div>
        ) : null}

        {items.length > 0 && <KnowledgeCategoryChart items={items} stats={stats} />}

        <SpacedReviewCard addToast={addToast} />

        {loading && !adapterStatus ? (
          <KnowledgeAdapterSkeleton />
        ) : adapterStatus ? (
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">Memory Training</p>
                  {adapterStatus.adapter_exists ? (
                    <span className="text-xs px-1.5 py-0.5 rounded-full bg-success/15 text-success font-medium">Ready</span>
                  ) : (
                    <span className="text-xs px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground font-medium">Not trained</span>
                  )}
                </div>
                <Button
                  size="sm"
                  variant={adapterStatus.adapter_exists ? 'outline' : 'default'}
                  className="h-7 text-xs px-2.5"
                  onClick={handleTrainAdapter}
                  disabled={adapterTraining || items.length === 0}
                >
                  {adapterTraining ? (
                    <span className="flex items-center gap-1">
                      <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent" />
                      Learning...
                    </span>
                  ) : adapterStatus.adapter_exists ? 'Retrain' : 'Train memory'}
                </Button>
              </div>
              <div className="flex gap-3 mt-2 text-xs text-muted-foreground">
                <span>{adapterStatus.fact_count} things learned</span>
                {adapterStatus.trained_at && (
                  <span>Last trained {new Date(adapterStatus.trained_at * MS_PER_SECOND).toLocaleDateString()}</span>
                )}
              </div>
            </CardContent>
          </Card>
        ) : null}

        {loading && !ragStats ? (
          <KnowledgeRAGSkeleton />
        ) : ragStats ? (
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">Deep Memory</p>
                  <span className="text-xs px-1.5 py-0.5 rounded-full bg-success/15 text-success font-medium">
                    {ragStats.total_chunks} pieces
                  </span>
                </div>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline" className="h-7 text-xs px-2.5" onClick={handleRAGSync} disabled={ragSyncing}>
                    {ragSyncing ? (
                      <span className="flex items-center gap-1">
                        <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent" />
                        Syncing...
                      </span>
                    ) : 'Sync'}
                  </Button>
                  <Button size="sm" variant="outline" className="h-7 text-xs px-2.5" onClick={() => setShowRagDocs(!showRagDocs)}>
                    {showRagDocs ? 'Hide' : 'Sources'}
                  </Button>
                  <Button size="sm" variant="outline" className="h-7 text-xs px-2.5 text-destructive hover:text-destructive" onClick={handleRAGClear} disabled={ragClearing}>
                    {ragClearing ? 'Clearing...' : 'Clear'}
                  </Button>
                </div>
              </div>
              <div className="flex gap-3 mt-2 text-xs text-muted-foreground">
                <span>{ragStats.total_documents} sources</span>
                <span>{ragStats.total_chunks} pieces indexed</span>
              </div>
              {showRagDocs && ragDocs.length > 0 && (
                <div className="mt-3 space-y-1 max-h-48 overflow-y-auto">
                  {ragDocs.map((doc, i) => (
                    <div key={i} className="flex items-center justify-between text-[11px] py-1 border-b border-border/40 last:border-0">
                      <div className="flex-1 min-w-0">
                        <span className="text-foreground truncate block">
                          {(doc.metadata?.source as string) || 'unknown'}
                        </span>
                        <span className="text-muted-foreground">
                          {doc.num_chunks} pieces
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
              {showRagDocs && ragDocs.length === 0 && (
                <p className="mt-3 text-[11px] text-muted-foreground">No sources yet.</p>
              )}
            </CardContent>
          </Card>
        ) : null}

        {loading && topics.length === 0 ? (
          <KnowledgeTopicsSkeleton />
        ) : topics.length > 0 ? (
          <Card>
            <CardContent className="p-3">
              <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium mb-2">What we talk about</p>
              <div className="space-y-1.5">
                {topics.slice(0, 8).map(t => {
                  const pct = stats ? Math.round((t.count / stats.total_items) * 100) : 0
                  return (
                    <div key={t.name} className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => setActiveTopic(activeTopic === t.name ? null : t.name)}
                        className={cn('text-[11px] w-24 text-left truncate transition-colors', activeTopic === t.name ? 'text-primary font-medium' : 'text-muted-foreground hover:text-foreground')}
                      >
                        {t.name}
                      </button>
                      <div className="flex-1 h-3 bg-muted/50 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-primary/40 rounded-full transition-all"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="text-xs text-muted-foreground w-8 text-right">{t.count}</span>
                    </div>
                  )
                })}
              </div>
            </CardContent>
          </Card>
        ) : null}

        {topics.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            <Chip
              label={`All (${items.length})`}
              onClick={() => setActiveTopic(null)}
              className={cn('text-xs px-2 py-0.5 rounded-full cursor-pointer transition-colors', activeTopic === null ? 'bg-primary/15 text-primary border-primary/30' : 'bg-muted text-muted-foreground border-border/40 hover:bg-muted/80')}
            />
            {topics.map(t => (
              <Chip
                key={t.name}
                label={`${t.name} (${t.count})`}
                onClick={() => setActiveTopic(activeTopic === t.name ? null : t.name)}
                className={cn('text-xs px-2 py-0.5 rounded-full cursor-pointer transition-colors', activeTopic === t.name ? 'bg-primary/15 text-primary border-primary/30' : 'bg-muted text-muted-foreground border-border/40 hover:bg-muted/80')}
              />
            ))}
          </div>
        )}

        {loading ? (
          <div className="space-y-2">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-16 rounded-lg" />
            ))}
          </div>
        ) : displayItems.length === 0 ? (
          <EmptyCard
            message={search ? 'No results found' : 'Nothing here yet'}
            description={search ? 'Try a different search term' : 'Add things you want your AI to remember about you. Click the + button to get started.'}
            icon={<IconSearch className="h-5 w-5" />}
            action={null}
          />
        ) : (
          <>
            {displayItems.length > 1 && (
              <label className="flex items-center gap-2 text-sm text-muted-foreground cursor-pointer">
                <Checkbox
                  checked={selectedIds.size === displayItems.length && displayItems.length > 0}
                  onCheckedChange={toggleSelectAll}
                  aria-label="Select all knowledge items"
                  className="rounded border-border"
                />
                Select all ({displayItems.length})
              </label>
            )}
            <div className="space-y-2 max-h-[60vh] overflow-y-auto overscroll-contain">
              {displayItems.map(item => (
                <div
                  key={item.id}
                  className={cn('group relative p-4 rounded-lg border text-sm leading-relaxed transition-colors', selectedIds.has(item.id) ? 'bg-primary/[0.06] border-primary/30' : editingId === item.id ? 'bg-primary/[0.04] border-primary/30' : 'bg-card border-border/60 hover:bg-muted/30')}
                >
                  <div className="flex items-start gap-2">
                    <Checkbox
                      checked={selectedIds.has(item.id)}
                      onCheckedChange={() => toggleSelect(item.id)}
                      aria-label={`Select knowledge item`}
                      className="mt-0.5 rounded border-border shrink-0"
                    />
                    <div className="flex-1 min-w-0">
                      {editingId === item.id ? (
                        <div className="space-y-2">
                          <div>
                            <textarea
                              className={cn('w-full p-2 text-sm border rounded-lg resize-none h-16 bg-background focus:outline-none focus:ring-1 focus:ring-primary/40', editErrors.content ? 'border-destructive ring-destructive/20' : 'border-input')}
                              value={editContent}
                              onChange={e => {
                                setEditContent(e.target.value)
                                if (editErrors.content) setEditErrors(prev => ({ ...prev, content: undefined }))
                              }}
                              autoFocus
                              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSaveEdit(item.id) } if (e.key === 'Escape') setEditingId(null) }}
                              aria-invalid={!!editErrors.content}
                              aria-describedby={editErrors.content ? 'edit-content-error' : undefined}
                              aria-label="Edit knowledge content"
                            />
                            {editErrors.content && (
                              <p id="edit-content-error" className="text-xs text-destructive mt-1" role="alert">{editErrors.content}</p>
                            )}
                          </div>
                          <div className="flex items-center gap-2">
                            <Input
                              value={editTopic}
                              onChange={e => setEditTopic(e.target.value)}
                              className="h-7 text-xs flex-1"
                              placeholder="Topic"
                              onKeyDown={e => { if (e.key === 'Enter') handleSaveEdit(item.id); if (e.key === 'Escape') setEditingId(null) }}
                            />
                            <button type="button" onClick={() => handleSaveEdit(item.id)} className="text-success hover:text-success/80 p-1" aria-label="Save edit">
                              <IconCheck className="h-3.5 w-3.5" />
                            </button>
                            <button type="button" onClick={() => setEditingId(null)} className="text-muted-foreground hover:text-foreground p-1" aria-label="Cancel edit">
                              <IconX className="h-3.5 w-3.5" />
                            </button>
                          </div>
                        </div>
                      ) : (
                        <>
                          <p className="text-sm text-foreground break-words">
                            {search ? highlightText(item.content, search) : item.content}
                          </p>
                          <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
                            {item.topic && (
                              <span className="text-xs px-1.5 py-0.5 rounded font-medium bg-primary/10 text-primary">
                                {item.topic}
                              </span>
                            )}
                            {item.source && (
                              <span className="text-xs px-1.5 py-0.5 rounded font-medium bg-muted text-muted-foreground">
                                {item.source}
                              </span>
                            )}
                            {editingImportanceId === item.id ? (
                              <div className="flex items-center gap-1.5 bg-muted/50 rounded px-1.5 py-0.5">
                                <Slider
                                  value={[importanceValue]}
                                  min={0}
                                  max={10}
                                  step={0.1}
                                  onValueChange={([v]) => setImportanceValue(v)}
                                  size="sm"
                                  className="w-16"
                                />
                                <Button
                                  size="icon-sm"
                                  variant="ghost"
                                  onClick={() => handleSaveImportance(item.id)}
                                  aria-label="Save importance"
                                >
                                  <IconCheck className="h-3 w-3" />
                                </Button>
                                <Button
                                  size="icon-sm"
                                  variant="ghost"
                                  onClick={() => setEditingImportanceId(null)}
                                  aria-label="Cancel editing"
                                >
                                  <IconX className="h-3 w-3" />
                                </Button>
                              </div>
                            ) : (
                              <button
                                type="button"
                                onClick={() => { setEditingImportanceId(item.id); setImportanceValue(item.importance) }}
                                className="text-xs text-muted-foreground/50 hover:text-muted-foreground transition-colors cursor-pointer"
                              >
                                importance: {item.importance.toFixed(1)}
                              </button>
                            )}
                            <span className="text-xs text-muted-foreground/50">
                              {new Date(item.timestamp * MS_PER_SECOND).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                            </span>
                          </div>
                        </>
                      )}
                    </div>
                    {editingId !== item.id && (
                      <div className="flex items-center gap-1 shrink-0">
                        <button
                          type="button"
                          onClick={() => { setEditingId(item.id); setEditContent(item.content); setEditTopic(item.topic || '') }}
                          className="opacity-0 group-hover:opacity-100 focus-within:opacity-100 text-muted-foreground hover:text-primary p-1 transition-opacity"
                          aria-label="Edit knowledge"
                        >
                          <IconEdit className="h-4 w-4" />
                        </button>
                        <button
                          type="button"
                          onClick={() => setPendingDelete(item)}
                          className="opacity-0 group-hover:opacity-100 focus-within:opacity-100 text-muted-foreground hover:text-destructive p-1 transition-opacity"
                          aria-label="Delete knowledge"
                        >
                          <IconTrash className="h-4 w-4" />
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        <MemoryCard />
        <MemorySettingsCard />

        <LearnSection />

        <KnowledgeIntelligenceCard />

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

        <AlertDialog open={showBulkTopic} onOpenChange={() => setShowBulkTopic(false)}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Move {selectedIds.size} items to topic</AlertDialogTitle>
              <AlertDialogDescription>
                Assign all selected knowledge items to a new topic.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <div className="py-2">
              <Input
                value={bulkTopic}
                onChange={e => setBulkTopic(e.target.value)}
                placeholder="Enter topic name..."
                autoFocus
                onKeyDown={e => { if (e.key === 'Enter' && bulkTopic.trim()) handleBulkTopicReassign() }}
              />
              <div className="flex flex-wrap gap-1 mt-2">
                {topics.slice(0, 6).map(t => (
                  <button
                    key={t.name}
                    type="button"
                    onClick={() => setBulkTopic(t.name)}
                    className="text-xs px-2 py-0.5 rounded-full border border-border/40 text-muted-foreground hover:bg-muted/50 transition-colors"
                  >
                    {t.name}
                  </button>
                ))}
              </div>
            </div>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <Button size="sm" disabled={!bulkTopic.trim()} onClick={handleBulkTopicReassign}>Move</Button>
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
              <div>
                <textarea
                  className={cn('w-full p-2.5 text-sm border rounded-lg resize-none h-24 bg-background focus:outline-none focus:ring-1 focus:ring-primary/40', addErrors.content ? 'border-destructive ring-destructive/20' : 'border-input')}
                  placeholder="Enter a fact, preference, or piece of context..."
                  value={newContent}
                  onChange={e => {
                    const val = e.target.value
                    setNewContent(val)
                    if (addErrors.content) setAddErrors(prev => ({ ...prev, content: undefined }))
                    if (newTopic === 'general' || newTopic === '') {
                      setNewTopic(suggestTopic(val))
                    }
                  }}
                  autoFocus
                  aria-invalid={!!addErrors.content}
                  aria-describedby={addErrors.content ? 'add-content-error' : undefined}
                  aria-label="New knowledge content"
                />
                {addErrors.content && (
                  <p id="add-content-error" className="text-xs text-destructive mt-1" role="alert">{addErrors.content}</p>
                )}
              </div>
              <div className="flex items-center gap-2">
                <label htmlFor="add-knowledge-topic" className="text-xs text-muted-foreground shrink-0">Topic:</label>
                <Input
                  id="add-knowledge-topic"
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
      </PageContainer>
  )
}
