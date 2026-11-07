'use client'

/**
 * Stateful core of the Knowledge page.
 *
 * Owns IO and state only — the knowledge/stats/topics/adapter/RAG fetches,
 * add/delete/batch mutations, inline edit and importance scoring, search
 * with its debounce, and the selection set. The panels own render.
 *
 * `suggestTopic` and `highlightText` moved out with the panels that use
 * them (format module / items list): both are pure and neither touches
 * state.
 */
import { notify } from '@/lib/notify'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useRouter } from '@/vite/next-compat/navigation'
import {
  knowledgeController,
  type KnowledgeItem,
  type KnowledgeStats,
  type TopicCount,
} from '@/lib/knowledge-controller'
import {
  getRAGStats,
  clearRAG,
  listRAGDocuments,
  syncKGToRAG,
  type RAGStats,
  type RAGDocument,
} from '@/lib/rag-controller'
import { downloadJson } from '@/lib/download-utils'
import { todayDateString } from '@/lib/format-bytes'
import { knowledgeSchema } from '@/lib/validation-schemas'
import { logger } from '@/lib/dev-log'
import { useRefreshShortcut } from '@/hooks/useRefreshShortcut'
import { SEARCH_DEBOUNCE_MS } from './knowledge-format'

export function useKnowledgeWorkbench() {
  const router = useRouter()
  const addToast = notify.toast
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
  const [importProgress, setImportProgress] = useState<{ current: number; total: number } | null>(
    null,
  )
  const [sortBy, setSortBy] = useState<'date' | 'importance' | 'topic'>('date')
  const [adapterStatus, setAdapterStatus] = useState<{
    adapter_exists: boolean
    fact_count: number
    total_facts_available: number
    trained_at?: number
    post_training_loss?: number
  } | null>(null)
  const [adapterTraining, setAdapterTraining] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const [itemsResult, statsResult, topicsResult, adapterResult] = await Promise.all([
        knowledgeController.list(),
        knowledgeController.stats(),
        knowledgeController.topics(),
        knowledgeController.getAdapterStatus().catch((e) => {
          logger.warning('Could not adapter status', { exception: String(e) })
          return null
        }),
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

  useRefreshShortcut(fetchData)

  useEffect(() => {
    fetchData()
  }, [fetchData])

  const fetchRAGData = useCallback(async () => {
    try {
      const [s, docs] = await Promise.all([getRAGStats(), listRAGDocuments()])
      setRagStats(s)
      setRagDocs(docs.documents || [])
    } catch {
      addToast('Could not load deep memory data', 'error')
    }
  }, [])

  useEffect(() => {
    fetchRAGData()
  }, [fetchRAGData])

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
      addToast(
        `Adapter trained on ${result.fact_count} facts in ${result.elapsed.toFixed(1)}s`,
        'success',
      )
    } catch {
      addToast('Could not adapter training', 'error')
    } finally {
      setAdapterTraining(false)
    }
  }, [addToast])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
      )
        return
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

  const displayItems = useMemo(
    () =>
      (searchResults ?? (activeTopic ? items.filter((i) => i.topic === activeTopic) : items))
        .slice()
        .sort((a, b) => {
          if (sortBy === 'importance') return b.importance - a.importance
          if (sortBy === 'topic') return (a.topic || '').localeCompare(b.topic || '')
          return b.timestamp - a.timestamp
        }),
    [searchResults, activeTopic, items, sortBy],
  )

  const handleDelete = async () => {
    if (!pendingDelete) return
    const deleted = pendingDelete
    setItems((prev) => prev.filter((i) => i.id !== deleted.id))
    setStats((prev) => (prev ? { ...prev, total_items: prev.total_items - 1 } : prev))
    setPendingDelete(null)
    try {
      await knowledgeController.delete(deleted.id)
      addToast('Deleted', 'info', undefined, () => {
        setItems((prev) => [deleted, ...prev])
        setStats((prev) => (prev ? { ...prev, total_items: prev.total_items + 1 } : prev))
        knowledgeController.add(deleted.content, deleted.topic, false).catch(() => {
          addToast('Could not restore item', 'error')
        })
      })
    } catch {
      setItems((prev) => [deleted, ...prev])
      setStats((prev) => (prev ? { ...prev, total_items: prev.total_items + 1 } : prev))
      addToast('Could not delete', 'error')
    }
  }

  const handleBatchDelete = async () => {
    if (selectedIds.size === 0) return
    const deletedIds = new Set(selectedIds)
    const deletedItems = items.filter((i) => deletedIds.has(i.id))
    setItems((prev) => prev.filter((i) => !deletedIds.has(i.id)))
    setStats((prev) => (prev ? { ...prev, total_items: prev.total_items - deletedIds.size } : prev))
    setSelectedIds(new Set())
    setPendingBatchDelete(false)
    try {
      await knowledgeController.batchDelete(Array.from(deletedIds))
      addToast(`Deleted ${deletedIds.size} items`, 'info', undefined, () => {
        setItems((prev) => [...deletedItems, ...prev])
        setStats((prev) =>
          prev ? { ...prev, total_items: prev.total_items + deletedIds.size } : prev,
        )
        knowledgeController
          .batchIngest(deletedItems.map((i) => ({ content: i.content, tags: [i.topic] })))
          .catch((e) => {
            logger.error('knowledge batch undo re-ingest failed', {
              count: deletedIds.size,
              exception: String(e),
            })
            addToast('Could not restore deleted items', 'error')
          })
      })
    } catch {
      setItems((prev) => [...deletedItems, ...prev])
      setStats((prev) =>
        prev ? { ...prev, total_items: prev.total_items + deletedIds.size } : prev,
      )
      addToast('Could not batch delete', 'error')
    }
  }

  const handleBulkTopicReassign = async () => {
    if (selectedIds.size === 0 || !bulkTopic.trim()) return
    const newTopic = bulkTopic.trim()
    const affectedIds = new Set(selectedIds)
    const oldTopics = new Map(
      items.filter((i) => affectedIds.has(i.id)).map((i) => [i.id, i.topic]),
    )
    setItems((prev) => prev.map((i) => (affectedIds.has(i.id) ? { ...i, topic: newTopic } : i)))
    setSelectedIds(new Set())
    setShowBulkTopic(false)
    setBulkTopic('')
    try {
      await Promise.all(
        Array.from(affectedIds).map((id) => knowledgeController.update(id, { topic: newTopic })),
      )
      addToast(`Updated ${affectedIds.size} items to "${newTopic}"`, 'success')
    } catch {
      setItems((prev) =>
        prev.map((i) =>
          affectedIds.has(i.id) ? { ...i, topic: oldTopics.get(i.id) || 'general' } : i,
        ),
      )
      addToast('Could not reassign topics', 'error')
    }
  }

  const handleAdd = async () => {
    const result = knowledgeSchema.safeParse({ content: newContent, topic: newTopic })
    if (!result.success) {
      const fieldErrors: { content?: string; topic?: string } = {}
      result.error.issues.forEach((issue) => {
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
    setItems((prev) => [optimisticItem, ...prev])
    setStats((prev) => (prev ? { ...prev, total_items: prev.total_items + 1 } : prev))
    setNewContent('')
    setNewTopic('general')
    setShowAdd(false)
    try {
      await knowledgeController.add(newContent.trim(), newTopic, true)
      addToast('Knowledge added', 'info')
      await fetchData()
    } catch {
      setItems((prev) => prev.filter((i) => i.id !== tempId))
      setStats((prev) => (prev ? { ...prev, total_items: prev.total_items - 1 } : prev))
      addToast('Could not add knowledge', 'error')
    }
  }

  const handleSaveEdit = async (id: string) => {
    const result = knowledgeSchema.safeParse({ content: editContent, topic: editTopic })
    if (!result.success) {
      const fieldErrors: { content?: string; topic?: string } = {}
      result.error.issues.forEach((issue) => {
        const field = issue.path[0] as string
        if (field === 'content') fieldErrors.content = issue.message
        if (field === 'topic') fieldErrors.topic = issue.message
      })
      setEditErrors(fieldErrors)
      return
    }
    setEditErrors({})
    try {
      await knowledgeController.update(id, {
        content: editContent.trim(),
        topic: editTopic.trim() || 'general',
      })
      setItems((prev) =>
        prev.map((i) =>
          i.id === id
            ? { ...i, content: editContent.trim(), topic: editTopic.trim() || 'general' }
            : i,
        ),
      )
      setEditingId(null)
      addToast('Updated', 'info')
    } catch {
      addToast('Could not update', 'error')
    }
  }

  const handleSaveImportance = async (id: string) => {
    try {
      await knowledgeController.update(id, { importance: importanceValue })
      setItems((prev) => prev.map((i) => (i.id === id ? { ...i, importance: importanceValue } : i)))
      setEditingImportanceId(null)
      addToast('Importance updated', 'info')
    } catch {
      addToast('Could not update', 'error')
    }
  }

  const handleExport = () => {
    const data = items.map((i) => ({
      content: i.content,
      topic: i.topic,
      source: i.source,
      importance: i.importance,
    }))
    downloadJson(data, `knowledge-export-${todayDateString()}.json`)
    addToast(`Exported ${items.length} items`, 'success')
  }

  const handleExportSelected = () => {
    const selected = items.filter((i) => selectedIds.has(i.id))
    const data = selected.map((i) => ({
      content: i.content,
      topic: i.topic,
      source: i.source,
      importance: i.importance,
    }))
    downloadJson(data, `knowledge-export-selected-${todayDateString()}.json`)
    addToast(`Exported ${selected.length} selected items`, 'success')
  }

  const handleExportCSV = () => {
    const headers = ['content', 'topic', 'source', 'importance']
    const rows = items.map((i) => [
      i.content,
      i.topic || '',
      i.source || '',
      i.importance.toString(),
    ])
    const csv = [
      headers.join(','),
      ...rows.map((row) =>
        row
          .map((val) =>
            val.includes(',') || val.includes('"') || val.includes('\n')
              ? `"${val.replace(/"/g, '""')}"`
              : val,
          )
          .join(','),
      ),
    ].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `knowledge-export-${todayDateString()}.csv`
    a.click()
    URL.revokeObjectURL(url)
    addToast(`Exported ${items.length} items as CSV`, 'success')
  }

  const handleImportFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setImporting(true)
    setImportProgress({ current: 0, total: 0 })
    try {
      const text = await file.text()
      const splitLines = (t: string) =>
        t
          .split('\n')
          .map((l) => l.trim())
          .filter(Boolean)
      let facts: string[]
      if (file.name.endsWith('.json')) {
        const parsed = JSON.parse(text)
        facts = Array.isArray(parsed)
          ? parsed
              .map((i: string | { content?: string }) =>
                typeof i === 'string' ? i : i.content || JSON.stringify(i),
              )
              .filter(Boolean)
          : []
      } else if (file.name.endsWith('.csv')) {
        const lines = splitLines(text)
        if (lines.length === 0) {
          addToast('No data found in CSV', 'error')
          return
        }
        const header = lines[0].toLowerCase()
        const contentIdx = header.split(',').findIndex((h) => h.includes('content'))
        const topicIdx = header.split(',').findIndex((h) => h.includes('topic'))
        facts = lines
          .slice(1)
          .map((line) => {
            const cols = line.split(',').map((c) => c.trim().replace(/^"|"$/g, ''))
            const content = contentIdx >= 0 ? cols[contentIdx] : cols[0]
            const topic = topicIdx >= 0 ? cols[topicIdx] : ''
            return content ? `${content}${topic ? ` [${topic}]` : ''}` : ''
          })
          .filter(Boolean)
      } else {
        facts = splitLines(text)
      }
      if (facts.length === 0) {
        addToast('No facts found in file', 'error')
        return
      }
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
    setSelectedIds((prev) => {
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
      setSelectedIds(new Set(displayItems.map((i) => i.id)))
    }
  }

  return {
    activeTopic,
    adapterStatus,
    adapterTraining,
    addErrors,
    addToast,
    bulkTopic,
    displayItems,
    editContent,
    editErrors,
    editTopic,
    editingId,
    editingImportanceId,
    fetchData,
    fetchRAGData,
    fileInputRef,
    handleAdd,
    handleBatchDelete,
    handleBulkTopicReassign,
    handleDelete,
    handleExport,
    handleExportCSV,
    handleExportSelected,
    handleImportFile,
    handleRAGClear,
    handleRAGSync,
    handleSaveEdit,
    handleSaveImportance,
    handleSearch,
    handleTrainAdapter,
    importProgress,
    importanceValue,
    importing,
    items,
    loading,
    newContent,
    newTopic,
    pendingBatchDelete,
    pendingDelete,
    ragClearing,
    ragDocs,
    ragStats,
    ragSyncing,
    router,
    search,
    searchResults,
    selectedIds,
    setActiveTopic,
    setAdapterStatus,
    setAdapterTraining,
    setAddErrors,
    setBulkTopic,
    setEditContent,
    setEditErrors,
    setEditTopic,
    setEditingId,
    setEditingImportanceId,
    setImportProgress,
    setImportanceValue,
    setImporting,
    setItems,
    setLoading,
    setNewContent,
    setNewTopic,
    setPendingBatchDelete,
    setPendingDelete,
    setRagClearing,
    setRagDocs,
    setRagStats,
    setRagSyncing,
    setSearch,
    setSearchResults,
    setSelectedIds,
    setShowAdd,
    setShowBulkTopic,
    setShowRagDocs,
    setSortBy,
    setStats,
    setTopics,
    showAdd,
    showBulkTopic,
    showRagDocs,
    sortBy,
    stats,
    toggleSelect,
    toggleSelectAll,
    topics,
  }
}

/** The contract the Knowledge panels render against. */
export type KnowledgeWorkbench = ReturnType<typeof useKnowledgeWorkbench>
