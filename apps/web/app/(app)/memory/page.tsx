'use client'
export const dynamic = 'force-dynamic'

import { useState, useCallback, useEffect } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label, Progress } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { memoryController, type MemoryItem, type MemoryConfigResult, type MemoryArchiveStats } from '@/lib/memory-controller'

function importanceColor(i: number): string {
  if (i >= 0.8) return 'text-foreground font-medium'
  if (i >= 0.5) return 'text-muted-foreground'
  return 'text-muted-foreground/60'
}

export default function MemoryPage() {
  const addToast = useToastStore(s => s.addToast)
  const [items, setItems] = useState<MemoryItem[]>([])
  const [stats, setStats] = useState<{ enabled: boolean; total_facts: number; topics: number; visited_urls: number } | null>(null)
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<MemoryItem[] | null>(null)
  const [searching, setSearching] = useState(false)
  const [storeContent, setStoreContent] = useState('')
  const [storeTopic, setStoreTopic] = useState('manual')
  const [config, setConfig] = useState<MemoryConfigResult | null>(null)
  const [showStore, setShowStore] = useState(false)
  const [rememberContent, setRememberContent] = useState('')
  const [rememberTopic, setRememberTopic] = useState('')
  const [remembering, setRemembering] = useState(false)
  const [selectedItem, setSelectedItem] = useState<MemoryItem | null>(null)
  const [editContent, setEditContent] = useState('')
  const [editMode, setEditMode] = useState(false)
  const [archiveStats, setArchiveStats] = useState<MemoryArchiveStats | null>(null)
  const [archiving, setArchiving] = useState(false)

  const fetchAll = useCallback(async () => {
    setLoading(true)
    try {
      const [listResp, statsResp] = await Promise.all([
        memoryController.list(100),
        memoryController.stats(),
      ])
      setItems(listResp.items ?? [])
      setStats(statsResp)
    } catch {
      addToast('Could not load memory', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  const handleRemember = useCallback(async () => {
    if (!rememberContent.trim()) return
    setRemembering(true)
    try {
      await memoryController.remember(rememberContent, rememberTopic || rememberContent)
      addToast('Memory saved', 'success')
      setRememberContent('')
      setRememberTopic('')
      void fetchAll()
    } catch {
      addToast('Could not save memory', 'error')
    } finally {
      setRemembering(false)
    }
  }, [rememberContent, rememberTopic, addToast, fetchAll])

  const loadArchiveStats = useCallback(async () => {
    try {
      const resp = await memoryController.archiveStats()
      setArchiveStats(resp)
    } catch { /* silent */ }
  }, [])

  const handleArchive = useCallback(async () => {
    setArchiving(true)
    try {
      const resp = await memoryController.archive()
      addToast(`Archived: ${resp.total} items`, 'success')
      void loadArchiveStats()
      void fetchAll()
    } catch {
      addToast('Could not archive', 'error')
    } finally {
      setArchiving(false)
    }
  }, [addToast, loadArchiveStats, fetchAll])

  const handlePruneArchive = useCallback(async () => {
    try {
      const resp = await memoryController.archivePrune()
      addToast(`Pruned: ${resp.pruned} items`, 'success')
      void loadArchiveStats()
    } catch {
      addToast('Could not prune archive', 'error')
    }
  }, [addToast, loadArchiveStats])

  const handleToggleEnabled = useCallback(async () => {
    if (!stats) return
    try {
      await memoryController.setEnabled(!stats.enabled)
      setStats(prev => prev ? { ...prev, enabled: !prev.enabled } : prev)
      addToast(`Memory ${stats.enabled ? 'disabled' : 'enabled'}`, 'success')
    } catch {
      addToast('Could not toggle memory', 'error')
    }
  }, [stats, addToast])

  useEffect(() => { void fetchAll(); void loadArchiveStats() }, [fetchAll, loadArchiveStats])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'r' && !e.metaKey && !e.ctrlKey) { e.preventDefault(); void fetchAll() }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [fetchAll])

  const doSearch = useCallback(async () => {
    if (!searchQuery.trim()) { setSearchResults(null); return }
    setSearching(true)
    try {
      const resp = await memoryController.search(searchQuery, 20)
      setSearchResults(resp.results ?? [])
    } catch {
      addToast('Could not search', 'error')
    } finally {
      setSearching(false)
    }
  }, [searchQuery, addToast])

  const storeItem = useCallback(async () => {
    if (!storeContent.trim()) return
    try {
      await memoryController.store(storeContent, storeTopic)
      addToast('Stored', 'success')
      setStoreContent('')
      setShowStore(false)
      void fetchAll()
    } catch {
      addToast('Could not store', 'error')
    }
  }, [storeContent, storeTopic, addToast, fetchAll])

  const deleteItem = useCallback(async (id: string) => {
    try {
      await memoryController.delete(id)
      addToast('Deleted', 'success')
      setItems(prev => prev.filter(i => i.id !== id))
      if (selectedItem?.id === id) setSelectedItem(null)
    } catch {
      addToast('Could not delete', 'error')
    }
  }, [selectedItem, addToast])

  const saveEdit = useCallback(async () => {
    if (!selectedItem) return
    try {
      await memoryController.update(selectedItem.id, editContent)
      addToast('Updated', 'success')
      setEditMode(false)
      void fetchAll()
    } catch {
      addToast('Could not update', 'error')
    }
  }, [selectedItem, editContent, addToast, fetchAll])

  const consolidate = useCallback(async () => {
    try {
      const resp = await memoryController.consolidate()
      addToast(`Consolidated: ${resp.removed ?? 0} removed, ${resp.kept ?? 0} kept`, 'success')
      void fetchAll()
    } catch {
      addToast('Could not consolidate', 'error')
    }
  }, [addToast, fetchAll])

  const displayItems = searchResults ?? items

  return (
    <PageContainer
      title="Memory"
      subtitle="Conversation memory and knowledge retrieval"
      headerRight={
        <div className="flex items-center gap-2">
          <Button size="sm" variant="ghost" onClick={() => void fetchAll()}>Refresh</Button>
          <Button size="sm" variant="ghost" onClick={consolidate}>Consolidate</Button>
          <Button size="sm" variant="ghost" onClick={() => void handleArchive()} disabled={archiving}>{archiving ? 'Archiving...' : 'Archive'}</Button>
          <Button size="sm" variant={stats?.enabled ? 'outline' : 'ghost'} onClick={() => void handleToggleEnabled()}>
            {stats?.enabled ? 'Disable' : 'Enable'}
          </Button>
          <Button size="sm" onClick={() => setShowStore(!showStore)}>
            {showStore ? 'Cancel' : 'Store'}
          </Button>
        </div>
      }
    >
      {stats && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Card>
            <CardContent className="p-3">
              <p className="text-xs text-muted-foreground">Facts</p>
              <p className="text-lg font-medium">{stats.total_facts}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-3">
              <p className="text-xs text-muted-foreground">Topics</p>
              <p className="text-lg font-medium">{stats.topics}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-3">
              <p className="text-xs text-muted-foreground">URLs visited</p>
              <p className="text-lg font-medium">{stats.visited_urls}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-3">
              <p className="text-xs text-muted-foreground">Enabled</p>
              <p className="text-lg font-medium">{stats.enabled ? 'Yes' : 'No'}</p>
            </CardContent>
          </Card>
        </div>
      )}


      {archiveStats && (
        <Card>
          <CardContent className="p-3 flex items-center justify-between">
            <div className="flex gap-4">
              <div>
                <p className="text-xs text-muted-foreground">Archived</p>
                <p className="text-sm font-medium">{archiveStats.records} items</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Size</p>
                <p className="text-sm font-medium">{archiveStats.bytes} bytes</p>
              </div>
            </div>
            <Button size="sm" variant="ghost" className="text-destructive" onClick={() => void handlePruneArchive()}>Prune Old</Button>
          </CardContent>
        </Card>
      )}
      <div className="flex items-center gap-2">
        <Input
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') void doSearch() }}
          placeholder="Search memory..."
          className="h-8 text-xs"
        />
        <Button size="sm" variant="outline" onClick={() => { void doSearch() }} disabled={searching}>
          {searching ? 'Searching...' : 'Search'}
        </Button>
        {searchResults && (
          <Button size="sm" variant="ghost" onClick={() => { setSearchResults(null); setSearchQuery('') }}>
            Clear
          </Button>
        )}
      </div>

      <Card>
          <CardContent className="p-3">
            <div className="flex items-center gap-2">
              <Input
                value={rememberContent}
                onChange={e => setRememberContent(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') void handleRemember() }}
                placeholder="Quick remember: type a fact and press Enter"
                className="h-8 text-xs flex-1"
              />
              <Input
                value={rememberTopic}
                onChange={e => setRememberTopic(e.target.value)}
                placeholder="Topic (optional)"
                className="h-8 text-xs w-28"
              />
              <Button size="sm" onClick={() => void handleRemember()} disabled={remembering || !rememberContent.trim()}>
                {remembering ? 'Saving...' : 'Remember'}
              </Button>
            </div>
          </CardContent>
        </Card>

      {showStore && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Store memory</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-col gap-1">
              <Label htmlFor="memory-content" variant="uppercase">Content</Label>
              <textarea
                id="memory-content"
                value={storeContent}
                onChange={e => setStoreContent(e.target.value)}
                rows={3}
                aria-label="Memory content to store"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              />
            </div>
      <div className="flex items-center gap-2">
              <div className="flex flex-col gap-1">
                <Label htmlFor="memory-topic" variant="uppercase">Topic</Label>
                <Input id="memory-topic" value={storeTopic} onChange={e => setStoreTopic(e.target.value)}
                  className="h-8 text-xs w-40" />
              </div>
              <Button size="sm" className="mt-4" onClick={storeItem}>Store</Button>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">
              {searchResults ? `Results (${displayItems.length})` : `Memory (${displayItems.length})`}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <p className="text-xs text-muted-foreground">Loading...</p>
            ) : displayItems.length === 0 ? (
              <p className="text-xs text-muted-foreground">No memory items.</p>
            ) : (
              <div className="max-h-[500px] space-y-1 overflow-y-auto">
                {displayItems.map(item => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => { setSelectedItem(item); setEditContent(item.content); setEditMode(false) }}
                    className={`w-full rounded border p-2 text-left text-xs transition-colors ${
                      selectedItem?.id === item.id
                        ? 'border-primary bg-primary/5'
                        : 'border-border hover:bg-muted/30'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="truncate font-medium max-w-[70%]">{item.topic ?? 'untitled'}</span>
                      <span className={`text-[10px] ${importanceColor(item.importance ?? 0)}`}>
                        {((item.importance ?? 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                    <p className="mt-0.5 truncate text-muted-foreground">{item.content?.slice(0, 80)}</p>
                  </button>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">{selectedItem ? (selectedItem.topic ?? 'Detail') : 'Select item'}</CardTitle>
              {selectedItem && (
                <div className="flex items-center gap-1">
                  {editMode ? (
                    <>
                      <Button size="sm" onClick={saveEdit}>Save</Button>
                      <Button size="sm" variant="ghost" onClick={() => setEditMode(false)}>Cancel</Button>
                    </>
                  ) : (
                    <>
                      <Button size="sm" variant="ghost" onClick={() => setEditMode(true)}>Edit</Button>
                      <Button size="sm" variant="ghost" className="text-destructive" onClick={() => deleteItem(selectedItem.id)}>Delete</Button>
                    </>
                  )}
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {selectedItem ? (
              editMode ? (
                <textarea
                  value={editContent}
                  onChange={e => setEditContent(e.target.value)}
                  rows={12}
                  aria-label="Edit memory content"
                  className="w-full rounded-md border border-input bg-background p-3 text-sm"
                />
              ) : (
                <div className="space-y-3 text-sm">
                  <div className="flex gap-3 text-xs text-muted-foreground">
                    <span>Topic: {selectedItem.topic ?? '--'}</span>
                    <span>Importance: {((selectedItem.importance ?? 0) * 100).toFixed(0)}%</span>
                  </div>
                  {selectedItem.source && <p className="text-xs text-muted-foreground">Source: {selectedItem.source}</p>}
                  {selectedItem.timestamp && <p className="text-xs text-muted-foreground">Created: {new Date(selectedItem.timestamp).toLocaleString()}</p>}
                  <p className="whitespace-pre-wrap text-sm">{selectedItem.content}</p>
                </div>
              )
            ) : (
              <p className="text-xs text-muted-foreground">Click a memory item to view details.</p>
            )}
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  )
}
