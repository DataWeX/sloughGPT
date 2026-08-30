'use client'

import { useCallback, useEffect, useMemo, useState, memo } from 'react'
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from '@sloughgpt/strui'
import { cn, Card, CardContent, CardHeader, CardTitle, Checkbox, Slider } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { Switch } from '@sloughgpt/strui'
import { IconBrain, IconRefresh, IconTrash, IconPlus, IconSearch, IconX, IconClock, IconEdit } from '@sloughgpt/strui'
import {
  DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { todayDateString } from '@/lib/format-bytes'
import { downloadJson } from '@/lib/download-utils'
import { memoryController, type MemoryItem } from '@/lib/memory-controller'
import { SectionErrorBoundary } from '@/components/SectionErrorBoundary'
import { parseMemoryImport } from '@/lib/memory-card-utils'
import { useMemoryData } from '@/hooks/useMemoryData'
import { MemoryStatsGrid } from './MemoryStatsGrid'
import { MemoryItemList } from './MemoryItemList'
import { MemoryMaintenancePanel } from './MemoryMaintenancePanel'
import { ArchiveDialog } from './ArchiveDialog'

/**
 * Self-contained card exposing the auto-memory layer on the Knowledge page.
 * Shows enabled state, stats, recent items, semantic search, manual store,
 * and a confirmed clear-all. Wired to `memoryController` (`/memory/*`).
 */
export const MemoryCard = memo(function MemoryCard() {
  const addToast = useToastStore(s => s.addToast)
  const {
    stats, items, archiveStats, loading, searched, searchResults,
    setSearch, search, fetchData, setSearchResults, setSearched,
  } = useMemoryData()

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
  const [toggling, setToggling] = useState(false)
  const [archiveOpen, setArchiveOpen] = useState(false)
  const [activeTopic, setActiveTopic] = useState<string | null>(null)
  const [showAllItems, setShowAllItems] = useState(false)
  const [sortOrder, setSortOrder] = useState<'newest' | 'oldest' | 'importance'>('newest')

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

  const itemCount = searchResults !== null ? searchResults.length : stats?.total_facts ?? items.length

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
  }, [addToast, fetchData, setSearch, setSearchResults, setSearched])

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
  }, [pendingDelete, searchResults, addToast, fetchData, setSearchResults])

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
  }, [selectedIds, searchResults, addToast, fetchData, setSearchResults])

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
  }, [editingItem, editContent, editTopic, editImportance, searchResults, addToast, fetchData, setSearchResults])

  const handleToggleEnabled = useCallback(async (next: boolean) => {
    setToggling(true)
    try {
      const result = await memoryController.setEnabled(next)
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
  }, [addToast, fetchData, setSearch, setSearchResults, setSearched])

  const toggleSelectAll = useCallback(() => {
    setSelectedIds(prev => {
      if (prev.size === filteredByTopic.length && filteredByTopic.length > 0) return new Set()
      return new Set(filteredByTopic.map(i => i.id))
    })
  }, [filteredByTopic])

  const openArchive = useCallback(() => setArchiveOpen(true), [])

  const clearSearch = useCallback(() => {
    setSearch('')
    setSearchResults(null)
    setSearched(false)
  }, [setSearch, setSearchResults, setSearched])

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-base flex items-center gap-2">
          <IconBrain className="h-4 w-4 text-primary" />
          Memory
          {stats && (
            <span className={cn('text-[10px] px-1.5 py-0.5 rounded-full font-medium', stats.enabled ? 'bg-success/15 text-success' : 'bg-muted text-muted-foreground')}>
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

        <MemoryStatsGrid stats={stats} loading={loading} />

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
                onClick={clearSearch}
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
          <SectionErrorBoundary sectionName="Memory editor">
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
              <Slider
                value={[editImportance]}
                min={0}
                max={1}
                step={0.1}
                showValue
                formatValue={(v) => v.toFixed(1)}
                onValueChange={([v]) => setEditImportance(v)}
                size="sm"
                className="flex-1"
                aria-label="Edit memory fact importance"
              />
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
          </SectionErrorBoundary>
        )}

        {topics.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 mb-3" aria-label="Filter by topic">
            <button
              type="button"
              onClick={() => setActiveTopic(null)}
              className={cn('text-[10px] px-2 py-1 rounded-full font-medium transition-colors', activeTopic === null ? 'bg-primary/15 text-primary' : 'bg-muted text-muted-foreground hover:bg-muted/70')}
            >
              All
            </button>
            {visibleTopics.map(topic => (
              <button
                key={topic}
                type="button"
                onClick={() => setActiveTopic(activeTopic === topic ? null : topic)}
                className={cn('text-[10px] px-2 py-1 rounded-full font-medium transition-colors', activeTopic === topic ? 'bg-primary/15 text-primary' : 'bg-muted text-muted-foreground hover:bg-muted/70')}
              >
                {topic}
              </button>
            ))}
            {extraTopicCount > 0 && (
              <span className="text-[10px] text-muted-foreground">+{extraTopicCount} more</span>
            )}
          </div>
        )}

        <MemoryItemList
          items={items}
          searchResults={searchResults}
          loading={loading}
          searched={searched}
          activeTopic={activeTopic}
          sortOrder={sortOrder}
          showAllItems={showAllItems}
          selectedIds={selectedIds}
          onToggleSelect={toggleSelect}
          onToggleSelectAll={toggleSelectAll}
          onClearSelection={clearSelection}
          onStartEdit={startEdit}
          onSetPendingDelete={setPendingDelete}
          onSetPendingBatchDelete={setPendingBatchDelete}
          onExportSelected={handleExportSelected}
          onCopy={handleCopy}
          onClearSearch={clearSearch}
          setShowAllItems={setShowAllItems}
          setSearch={setSearch}
          setSearchResults={setSearchResults}
          setSearched={setSearched}
        />

        <MemoryMaintenancePanel
          itemCount={itemCount}
          archiveStats={archiveStats}
          loading={loading}
          fetchData={fetchData}
          openArchive={openArchive}
        />
      </CardContent>

      <AlertDialog open={pendingDelete !== null} onOpenChange={() => setPendingDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this memory?</AlertDialogTitle>
            <AlertDialogDescription>
              This permanently removes the item: &ldquo;{pendingDelete?.content}&rdquo;
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

      <ArchiveDialog
        open={archiveOpen}
        onOpenChange={setArchiveOpen}
        archiveStats={archiveStats}
      />
    </Card>
  )
})
