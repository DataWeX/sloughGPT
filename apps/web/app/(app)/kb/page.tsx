'use client'

import { useState, useEffect, useCallback } from 'react'
import { ConfirmDialog } from '@/components/ConfirmDialog'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label, Textarea, cn } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { useToastStore } from '@/lib/toast-store'
import { kbController, type KnowledgeItem, type KnowledgeStats, type TopicItem } from '@/lib/kb-controller'
import { knowledgeController } from '@/lib/knowledge-controller'

type Tab = 'browse' | 'add' | 'search' | 'gaps'

export default function KbPage() {
  const addToast = useToastStore(s => s.addToast)
  const [tab, setTab] = useState<Tab>('browse')
  const [items, setItems] = useState<KnowledgeItem[]>([])
  const [stats, setStats] = useState<KnowledgeStats | null>(null)
  const [topics, setTopics] = useState<TopicItem[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null)
  const [page, setPage] = useState(0)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())

  const [newContent, setNewContent] = useState('')
  const [newTopic, setNewTopic] = useState('general')
  const [newImportance, setNewImportance] = useState(0.7)
  const [suggestResult, setSuggestResult] = useState<string | null>(null)

  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<KnowledgeItem[]>([])

  const [gapsResult, setGapsResult] = useState<{ gaps: string[]; suggestions: string[] } | null>(null)

  const [editingItem, setEditingItem] = useState<KnowledgeItem | null>(null)
  const [editContent, setEditContent] = useState('')
  const [editTopic, setEditTopic] = useState('')
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null)
  const [bulkConfirmOpen, setBulkConfirmOpen] = useState(false)

  const [relatedItems, setRelatedItems] = useState<KnowledgeItem[]>([])
  const [relatedLoading, setRelatedLoading] = useState<string | null>(null)

  const [urlInput, setUrlInput] = useState('')
  const [urlLoading, setUrlLoading] = useState(false)

  const [fileLoading, setFileLoading] = useState(false)

  const loadStats = useCallback(async () => {
    try {
      const [s, t] = await Promise.all([kbController.stats(), kbController.topics()])
      setStats(s)
      setTopics(t)
    } catch { /* silent */ }
  }, [])

  const loadItems = useCallback(async () => {
    setLoading(true)
    try {
      const result = await kbController.list(selectedTopic ?? undefined, 30, page * 30)
      setItems(result)
    } catch (e) {
      addToast(`Could not load knowledge: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setLoading(false)
    }
  }, [selectedTopic, page, addToast])

  useEffect(() => { void loadStats() }, [loadStats])
  useEffect(() => { if (tab === 'browse') void loadItems() }, [tab, loadItems])

  const handleAdd = async () => {
    if (!newContent.trim()) return
    setLoading(true)
    try {
      await kbController.add(newContent, newTopic, 'manual', newImportance)
      addToast('Entry added', 'success')
      setNewContent('')
      void loadStats()
    } catch (e) {
      addToast(`Could not add entry: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  const handleSuggest = async () => {
    if (!newContent.trim()) return
    try {
      const result = await kbController.suggestTopic(newContent)
      setSuggestResult(result.topic)
      setNewTopic(result.topic)
    } catch { /* silent */ }
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    setLoading(true)
    setSearchResults([])
    try {
      const results = await kbController.search(searchQuery, 20)
      setSearchResults(results)
    } catch (e) {
      addToast(`Search failed: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = (id: string) => {
    setPendingDeleteId(id)
    setConfirmOpen(true)
  }

  const confirmDelete = async () => {
    if (!pendingDeleteId) return
    try {
      await kbController.remove(pendingDeleteId)
      setItems(items.filter(i => i.id !== pendingDeleteId))
      addToast('Entry deleted', 'success')
      void loadStats()
    } catch (e) {
      addToast(`Could not delete: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setConfirmOpen(false)
      setPendingDeleteId(null)
    }
  }

  const handleBatchDelete = () => {
    if (selectedIds.size === 0) return
    setBulkConfirmOpen(true)
  }

  const confirmBatchDelete = async () => {
    try {
      await kbController.batchDelete(Array.from(selectedIds))
      addToast(`Deleted ${selectedIds.size} entries`, 'success')
      setSelectedIds(new Set())
      void loadItems()
      void loadStats()
    } catch (e) {
      addToast(`Could not batch delete: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setBulkConfirmOpen(false)
    }
  }

  const handleRelated = async (id: string) => {
    if (relatedLoading === id) { setRelatedItems([]); setRelatedLoading(null); return }
    setRelatedLoading(id)
    try {
      const result = await knowledgeController.related(id, 5)
      setRelatedItems(result.items)
    } catch {
      addToast('Could not load related items', 'error')
      setRelatedItems([])
    } finally {
      setUrlLoading(false)
    }
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setFileLoading(true)
    try {
      const result = await knowledgeController.ingestFile(file)
      addToast(`Uploaded ${result.filename} (${result.stored} facts, ${result.total_chunks} chunks)`, 'success')
      void loadItems()
      void loadStats()
    } catch (err) {
      addToast(`File upload failed: ${err instanceof Error ? err.message : 'Unknown error'}`, 'error')
    } finally {
      setFileLoading(false)
      e.target.value = ''
    }
  }

  const handleUpdate = async () => {
    if (!editingItem) return
    try {
      await kbController.update(editingItem.id, { content: editContent, topic: editTopic })
      addToast('Entry updated', 'success')
      setEditingItem(null)
      void loadItems()
    } catch (e) {
      addToast(`Could not update: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    }
  }

  const handleGaps = async () => {
    setLoading(true)
    setGapsResult(null)
    try {
      const result = await kbController.gaps()
      setGapsResult(result)
    } catch (e) {
      addToast(`Could not analyze gaps: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  const handleIngestUrl = async () => {
    if (!urlInput.trim()) return
    setUrlLoading(true)
    try {
      const result = await kbController.ingestUrl(urlInput)
      addToast('URL ingested', 'success')
      setUrlInput('')
      void loadItems()
      void loadStats()
    } catch (e) {
      addToast(`Could not ingest URL: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setUrlLoading(false)
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

  const tabs: { key: Tab; label: string }[] = [
    { key: 'browse', label: 'Browse' },
    { key: 'add', label: 'Add Entry' },
    { key: 'search', label: 'Search' },
    { key: 'gaps', label: 'Knowledge Gaps' },
  ]

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'r' && !e.metaKey && !e.ctrlKey) { e.preventDefault(); void loadStats() }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [loadStats])

  return (
    <PageContainer
      title="Knowledge Base"
      subtitle="Manage learned facts and knowledge"
      headerRight={
        <div className="flex items-center gap-2">
          {selectedIds.size > 0 && (
            <Button size="sm" variant="ghost" className="text-destructive" onClick={() => void handleBatchDelete()}>
              Delete {selectedIds.size}
            </Button>
          )}
          <Button size="sm" variant="ghost" onClick={() => { void loadStats(); void loadItems() }}>Refresh</Button>
        </div>
      }
    >
      <div className="space-y-4">
        {stats && (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {[
              { label: 'Total Entries', value: stats.total_items },
              { label: 'Topics', value: stats.topics.length },
              { label: 'Avg Importance', value: stats.avg_importance.toFixed(2) },
              { label: 'Sources', value: Object.keys(stats.sources).length },
            ].map(s => (
              <div key={s.label} className="rounded-md bg-muted/30 p-3 text-center">
                <div className="text-[10px] text-muted-foreground">{s.label}</div>
                <div className="text-lg font-mono font-medium">{s.value}</div>
              </div>
            ))}
          </div>
        )}

        <div className="flex gap-1 rounded-lg border border-border bg-muted/30 p-1">
          {tabs.map(t => (
            <button
              key={t.key}
              type="button"
              onClick={() => setTab(t.key)}
              className={cn('flex-1 rounded-md px-3 py-1.5 text-xs font-medium transition-colors', tab === t.key ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground')}
            >
              {t.label}
            </button>
          ))}
        </div>

        {tab === 'browse' && (
          <div className="space-y-3">
            <div className="flex flex-wrap gap-1">
              <button
                type="button"
                onClick={() => { setSelectedTopic(null); setPage(0) }}
                className={cn('rounded-full px-2.5 py-0.5 text-[10px] font-medium transition-colors', selectedTopic === null ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground hover:text-foreground')}
              >
                All
              </button>
              {topics.map(t => (
                <button
                  key={t.name}
                  type="button"
                  onClick={() => { setSelectedTopic(t.name); setPage(0) }}
                  className={cn('rounded-full px-2.5 py-0.5 text-[10px] font-medium transition-colors', selectedTopic === t.name ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground hover:text-foreground')}
                >
                  {t.name} ({t.count})
                </button>
              ))}
            </div>

            {loading ? (
              <div className="space-y-2">
                {Array.from({ length: 5 }).map((_, i) => (
                  <div key={i} className="h-16 animate-pulse rounded bg-muted/50" />
                ))}
              </div>
            ) : items.length === 0 ? (
              <Card>
                <CardContent className="py-8 text-center text-xs text-muted-foreground">No entries found.</CardContent>
              </Card>
            ) : (
              <div className="space-y-2">
                {items.map(item => (
                  <Card key={item.id}>
                    <CardContent className="py-3">
                      <div className="flex items-start gap-3">
                        <input
                          type="checkbox"
                          checked={selectedIds.has(item.id)}
                          onChange={() => toggleSelect(item.id)}
                          aria-label={`Select entry`}
                          className="mt-1 h-4 w-4 rounded border-border"
                        />
                        <div className="min-w-0 flex-1">
                          <p className="text-xs">{item.content.slice(0, 200)}{item.content.length > 200 ? '...' : ''}</p>
                          <div className="mt-1.5 flex gap-2">
                            <span className="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary">{item.topic}</span>
                            <span className="text-[10px] text-muted-foreground">{item.source}</span>
                            <span className="text-[10px] text-muted-foreground">{item.importance.toFixed(1)} importance</span>
                          </div>
                        </div>
                        <div className="flex gap-1">
                          <Button
                            size="sm"
                            variant="ghost"
                            className="h-6 px-1.5 text-[10px]"
                            onClick={() => void handleRelated(item.id)}
                          >
                            {relatedLoading === item.id ? '...' : 'Related'}
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="h-6 px-1.5 text-[10px]"
                            onClick={() => { setEditingItem(item); setEditContent(item.content); setEditTopic(item.topic) }}
                          >
                            Edit
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="h-6 px-1.5 text-[10px] text-destructive"
                            onClick={() => void handleDelete(item.id)}
                          >
                            Delete
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}

            {relatedItems.length > 0 && (
              <div className="rounded border border-border/50 bg-muted/20 p-3 space-y-2">
                <div className="flex items-center justify-between">
                  <p className="text-xs font-medium text-muted-foreground">Related items ({relatedItems.length})</p>
                  <Button size="sm" variant="ghost" className="text-[10px] h-5" onClick={() => { setRelatedItems([]); setRelatedLoading(null) }}>Clear</Button>
                </div>
                {relatedItems.map(ri => (
                  <div key={ri.id} className="text-xs space-y-0.5">
                    <p className="text-foreground/80">{ri.content.slice(0, 150)}{ri.content.length > 150 ? '...' : ''}</p>
                    <div className="flex gap-2">
                      <span className="text-[10px] text-primary">{ri.topic}</span>
                      <span className="text-[10px] text-muted-foreground">{ri.source}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}

            <div className="flex justify-center gap-2">
              <Button size="sm" variant="outline" disabled={page === 0} onClick={() => setPage(p => p - 1)}>Previous</Button>
              <span className="text-xs text-muted-foreground self-center">Page {page + 1}</span>
              <Button size="sm" variant="outline" disabled={items.length < 30} onClick={() => setPage(p => p + 1)}>Next</Button>
            </div>
          </div>
        )}

        {tab === 'add' && (
          <>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Add Knowledge Entry</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label className="text-xs">Content</Label>
                <Textarea value={newContent} onChange={e => setNewContent(e.target.value)} rows={4} className="text-xs" placeholder="Enter knowledge content..." />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <Label className="text-xs">Topic</Label>
                  <Input value={newTopic} onChange={e => setNewTopic(e.target.value)} className="h-8 text-xs" />
                  {suggestResult && <p className="text-[10px] text-muted-foreground">Suggested: {suggestResult}</p>}
                </div>
                <div className="space-y-2">
                  <Label htmlFor="importance-slider" className="text-xs">Importance ({newImportance.toFixed(1)})</Label>
                  <input id="importance-slider" type="range" min="0" max="1" step="0.1" value={newImportance} onChange={e => setNewImportance(Number(e.target.value))} className="w-full" />
                </div>
              </div>
              <div className="flex gap-2">
                <Button onClick={() => void handleAdd()} disabled={loading || !newContent.trim()} className="flex-1">
                  {loading ? 'Adding...' : 'Add Entry'}
                </Button>
                <Button variant="outline" onClick={() => void handleSuggest()} disabled={!newContent.trim()}>
                  Suggest Topic
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Import from URL</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-xs text-muted-foreground">Fetch content from a URL and add it to the knowledge base.</p>
              <div className="flex gap-2">
                <Input
                  value={urlInput}
                  onChange={e => setUrlInput(e.target.value)}
                  placeholder="https://example.com/article"
                  className="h-8 text-xs flex-1"
                  onKeyDown={e => e.key === 'Enter' && void handleIngestUrl()}
                />
                <Button size="sm" onClick={() => void handleIngestUrl()} disabled={urlLoading || !urlInput.trim()} className="shrink-0">
                  {urlLoading ? 'Importing...' : 'Import'}
                </Button>
               </div>
            </CardContent>
           </Card>

           <Card>
            <CardHeader>
              <CardTitle className="text-base">Upload File</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-xs text-muted-foreground">Upload a text file to add its contents to the knowledge base.</p>
              <div className="flex items-center gap-2">
                <label className="flex h-8 cursor-pointer items-center gap-2 rounded border border-border px-3 text-xs hover:bg-muted/30">
                  <input type="file" accept=".txt,.md,.json,.csv" className="hidden" onChange={e => void handleFileUpload(e)} disabled={fileLoading} />
                  {fileLoading ? 'Uploading...' : 'Choose file'}
                </label>
              </div>
            </CardContent>
           </Card>
          </>
        )}

        {tab === 'search' && (
          <div className="space-y-3">
            <div className="flex gap-2">
              <Input value={searchQuery} onChange={e => setSearchQuery(e.target.value)} placeholder="Search knowledge..." className="h-8 text-xs" onKeyDown={e => e.key === 'Enter' && void handleSearch()} />
              <Button onClick={() => void handleSearch()} disabled={loading} className="shrink-0">
                {loading ? 'Searching...' : 'Search'}
              </Button>
            </div>
            {searchResults.length > 0 && (
              <div className="space-y-2">
                {searchResults.map(item => (
                  <Card key={item.id}>
                    <CardContent className="py-3">
                      <div className="flex items-start gap-3">
                        <div className="min-w-0 flex-1">
                          <p className="text-xs">{item.content.slice(0, 300)}{item.content.length > 300 ? '...' : ''}</p>
                          <div className="mt-1.5 flex gap-2">
                            <span className="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary">{item.topic}</span>
                            <span className="text-[10px] text-muted-foreground">Score: {item.score.toFixed(3)}</span>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>
        )}

        {tab === 'gaps' && (
          <div className="space-y-3">
            <Button onClick={() => void handleGaps()} disabled={loading} className="w-full">
              {loading ? 'Analyzing...' : 'Analyze Knowledge Gaps'}
            </Button>
            {gapsResult && (
              <div className="grid gap-3 sm:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Identified Gaps</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {gapsResult.gaps.length === 0 ? (
                      <p className="text-xs text-muted-foreground">No gaps identified.</p>
                    ) : (
                      <ul className="space-y-1.5">
                        {gapsResult.gaps.map((g, i) => (
                          <li key={i} className="text-xs">{g}</li>
                        ))}
                      </ul>
                    )}
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Suggestions</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {gapsResult.suggestions.length === 0 ? (
                      <p className="text-xs text-muted-foreground">No suggestions.</p>
                    ) : (
                      <ul className="space-y-1.5">
                        {gapsResult.suggestions.map((s, i) => (
                          <li key={i} className="text-xs">{s}</li>
                        ))}
                      </ul>
                    )}
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        )}
      </div>

      {editingItem && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <Card className="w-full max-w-lg">
            <CardHeader>
              <CardTitle className="text-base">Edit Entry</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label className="text-xs">Content</Label>
                <Textarea value={editContent} onChange={e => setEditContent(e.target.value)} rows={4} className="text-xs" />
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Topic</Label>
                <Input value={editTopic} onChange={e => setEditTopic(e.target.value)} className="h-8 text-xs" />
              </div>
              <div className="flex justify-end gap-2">
                <Button variant="ghost" onClick={() => setEditingItem(null)}>Cancel</Button>
                <Button onClick={() => void handleUpdate()}>Save</Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      <ConfirmDialog
        open={confirmOpen}
        onOpenChange={setConfirmOpen}
        title="Delete knowledge entry"
        description="Delete this knowledge entry? This cannot be undone."
        confirmLabel="Delete"
        onConfirm={() => void confirmDelete()}
      />

      <ConfirmDialog
        open={bulkConfirmOpen}
        onOpenChange={setBulkConfirmOpen}
        title="Bulk delete entries"
        description={`Delete ${selectedIds.size} entries? This cannot be undone.`}
        confirmLabel="Delete all"
        onConfirm={() => void confirmBatchDelete()}
      />
    </PageContainer>
  )
}
