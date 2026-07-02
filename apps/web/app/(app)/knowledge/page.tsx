'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Switch } from '@/components/ui/switch'
import { Button } from '@/components/ui/button'
import { Badge, Chip, EmptyCard, KpiGrid, StatCard } from '@/components/ui'
import { SearchInput } from '@/components/ui/input'
import { IconSearch, IconPlus, IconTrash, IconDownload, IconUpload } from '@/components/icons/NavIcons'
import { IconRefresh } from '@/components/ui'
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@/components/ui/collapsible'
import { knowledgeController, type KnowledgeItem, type KnowledgeStats, type TopicCount, type AdapterStatus } from '@/lib/knowledge-controller'
import { useToastStore } from '@/lib/toast-store'

export default function KnowledgePage() {
  const [items, setItems] = useState<KnowledgeItem[]>([])
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(true)
  const [newContent, setNewContent] = useState('')
  const [newTopic, setNewTopic] = useState('')
  const [newUrl, setNewUrl] = useState('')
  const [adding, setAdding] = useState(false)
  const [urlIngesting, setUrlIngesting] = useState(false)
  const [fileUploading, setFileUploading] = useState(false)
  const [autoTag, setAutoTag] = useState(true)
  const [adapterStatus, setAdapterStatus] = useState<AdapterStatus | null>(null)
  const [trainingAdapter, setTrainingAdapter] = useState(false)
  const [suggestedTopic, setSuggestedTopic] = useState<string | null>(null)
  const suggestTimerRef = useRef<ReturnType<typeof setTimeout>>()
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [stats, setStats] = useState<KnowledgeStats | null>(null)
  const [topics, setTopics] = useState<TopicCount[]>([])
  const [activeTopic, setActiveTopic] = useState('')
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editContent, setEditContent] = useState('')
  const [editTopic, setEditTopic] = useState('')
  const [relatedItems, setRelatedItems] = useState<Record<string, KnowledgeItem[]>>({})
  const [loadingRelated, setLoadingRelated] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const txtFileInputRef = useRef<HTMLInputElement>(null)
  const addToast = useToastStore(s => s.addToast)

  const fetchItems = useCallback(async () => {
    setLoading(true)
    try {
      const data = search
        ? (await knowledgeController.search(search)).results
        : await knowledgeController.list(500, 0)
      const filtered = activeTopic ? data.filter(i => i.topic === activeTopic) : data
      setItems(filtered)
    } catch { addToast('Failed to load facts', 'error') }
    setLoading(false)
  }, [search, activeTopic, addToast])

  const fetchAdapterStatus = useCallback(async () => {
    try {
      const s = await knowledgeController.getAdapterStatus()
      setAdapterStatus(s)
    } catch { addToast('Failed to load status', 'error') }
  }, [addToast])

  const fetchStats = useCallback(async () => {
    try {
      const [s, t] = await Promise.all([knowledgeController.stats(), knowledgeController.topics()])
      setStats(s)
      setTopics(t.topics)
    } catch { addToast('Failed to load stats', 'error') }
  }, [addToast])

  useEffect(() => { fetchItems() }, [fetchItems])
  useEffect(() => { fetchStats() }, [fetchStats])
  useEffect(() => { fetchAdapterStatus() }, [fetchAdapterStatus])

  const handleAdd = async () => {
    if (!newContent.trim()) return
    setAdding(true)
    try {
      const res = await knowledgeController.add(newContent.trim(), newTopic || 'general', autoTag)
      setNewContent('')
      addToast(res.topic ? `Stored under "${res.topic}"` : 'Stored', 'success')
      await Promise.all([fetchItems(), fetchStats()])
    } catch (e) { addToast('Something went wrong adding this item', 'error') }
    setAdding(false)
  }

  const handleIngestUrl = async () => {
    if (!newUrl.trim()) return
    setUrlIngesting(true)
    try {
      const res = await knowledgeController.ingestUrl(newUrl)
      if (res.status === 'ok') {
        addToast(`Loaded ${res.new_facts} facts from "${res.title}"`, 'success')
      } else if (res.rejected) {
        addToast(`Couldn't load that URL`, 'info')
      } else {
        addToast(`Done`, 'info')
      }
      setNewUrl('')
      await Promise.all([fetchItems(), fetchStats()])
    } catch (e) { addToast('Loading failed', 'error') }
    setUrlIngesting(false)
  }

  const handleDelete = async (id: string) => {
    try {
      await knowledgeController.delete(id)
      setItems(prev => prev.filter(i => i.id !== id))
      setSelected(prev => { const n = new Set(prev); n.delete(id); return n })
      await fetchStats()
    } catch { addToast('Failed to delete item', 'error') }
  }

  const handleBatchDelete = async () => {
    const ids = Array.from(selected)
    try {
      const res = await knowledgeController.batchDelete(ids)
      addToast(`Deleted ${res.deleted} items`, 'success')
    } catch { addToast('Failed to delete items', 'error') }
    await Promise.all([fetchItems(), fetchStats()])
    setSelected(new Set())
  }

  const startEdit = async (item: KnowledgeItem) => {
    setEditingId(item.id)
    setEditContent(item.content)
    setEditTopic(item.topic)
    if (!relatedItems[item.id]) {
      setLoadingRelated(item.id)
      try {
        const res = await knowledgeController.related(item.id, 5)
        setRelatedItems(prev => ({ ...prev, [item.id]: res.items }))
      } catch { /* ignore — non-critical */ }
      setLoadingRelated(null)
    }
  }

  const cancelEdit = () => {
    setEditingId(null)
    setEditContent('')
    setEditTopic('')
  }

  const saveEdit = async (id: string) => {
    try {
      await knowledgeController.update(id, { content: editContent, topic: editTopic || undefined })
      addToast('Updated', 'success')
      setEditingId(null)
      await fetchItems()
    } catch (e) { addToast('Update failed', 'error') }
  }

  const toggleSelect = (id: string) => {
    setSelected(prev => {
      const n = new Set(prev)
      if (n.has(id)) n.delete(id); else n.add(id)
      return n
    })
  }

  const toggleSelectAll = () => {
    if (selected.size === items.length) {
      setSelected(new Set())
    } else {
      setSelected(new Set(items.map(i => i.id)))
    }
  }

  const exportJson = () => {
    const blob = new Blob([JSON.stringify(items, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'knowledge-export.json'; a.click()
    URL.revokeObjectURL(url)
  }

  const importJson = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const text = await file.text()
    const data = JSON.parse(text)
    const arr = Array.isArray(data) ? data : data.items ?? []
    let count = 0
    for (const item of arr) {
      try {
        await knowledgeController.add(item.content || item.text, item.topic || item.category || 'general')
        count++
      } catch { /* skip individual item */ }
    }
    addToast(`Imported ${count} items`, 'success')
    await Promise.all([fetchItems(), fetchStats()])
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setFileUploading(true)
    try {
      const res = await knowledgeController.ingestFile(file, 'imported')
      addToast(`Imported ${res.stored} facts from "${res.filename}" (${res.total_chunks} chunks)`, 'success')
      await Promise.all([fetchItems(), fetchStats()])
    } catch (err) {
      addToast(`File import failed: ${err instanceof Error ? err.message : 'unknown error'}`, 'error')
    }
    if (txtFileInputRef.current) txtFileInputRef.current.value = ''
    setFileUploading(false)
  }

  const handleTrainAdapter = async () => {
    setTrainingAdapter(true)
    try {
      const res = await knowledgeController.trainAdapter()
      addToast(`Trained on ${res.fact_count} facts in ${res.elapsed}s`, 'success')
      await fetchAdapterStatus()
    } catch (e) { addToast('Training failed', 'error') }
    setTrainingAdapter(false)
  }

  const handleRefresh = () => {
    void fetchItems()
    void fetchStats()
    void fetchAdapterStatus()
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Knowledge" />}
        right={<Button variant="outline" size="sm" onClick={handleRefresh}><IconRefresh className="w-3.5 h-3.5 mr-1" /> Refresh</Button>}
      />
      <div className="space-y-4">

        {/* Stats */}
        {stats && (
          <KpiGrid columns={4}>
            <StatCard label="Items" value={stats.total_items.toString()} />
            <StatCard label="Topics" value={stats.topic_count.toString()} />
            <StatCard label="Avg Importance" value={stats.avg_importance.toFixed(2)} />
            <StatCard label="Sources" value={Object.keys(stats.sources).length.toString()} />
          </KpiGrid>
        )}

        {/* Knowledge Adapter — bakes facts into model weights */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">AI Memory</CardTitle>
            <Button
              size="sm"
              onClick={handleTrainAdapter}
              disabled={trainingAdapter || (adapterStatus?.total_facts_available ?? 0) < 1}
            >
              {trainingAdapter ? 'Teaching…' : 'Teach the AI'}
            </Button>
          </CardHeader>
          <CardContent>
            {adapterStatus === null ? (
              <div className="h-8 animate-pulse bg-muted rounded" />
            ) : adapterStatus.adapter_exists ? (
              <div className="flex items-center gap-4 text-sm flex-wrap">
                <Badge variant="default" label="Trained" />
                <span className="text-muted-foreground">
                  {adapterStatus.fact_count} facts
                </span>
                {adapterStatus.trained_at && (
                  <span className="text-muted-foreground">
                    {new Date(adapterStatus.trained_at * 1000).toLocaleDateString()}
                  </span>
                )}
                {adapterStatus.post_training_loss !== undefined && (
                  <span className="text-muted-foreground">
                    loss {adapterStatus.post_training_loss.toFixed(4)}
                  </span>
                )}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                Not trained yet. Click &ldquo;Teach the AI&rdquo; to learn {stats?.total_items || 0} facts — no setup needed.
              </p>
            )}
          </CardContent>
        </Card>

        {/* Topic filter */}
        {topics.length > 0 && (
          <div className="flex items-center gap-2 flex-wrap">
            <Chip label="all" selected={!activeTopic} onClick={() => setActiveTopic('')} />
            {topics.map(t => (
              <Chip key={t.name} label={`${t.name} (${t.count})`} selected={activeTopic === t.name} onClick={() => setActiveTopic(t.name)} />
            ))}
          </div>
        )}

        {/* Semantic Search, Dedup, Gaps */}
        <KnowledgeOperationsCard />

        {/* Add new knowledge */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Add Knowledge</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
              <textarea
                className="w-full min-h-[100px] rounded-md border border-border bg-background px-3 py-2 text-sm resize-y focus:outline-none focus:ring-1 focus:ring-primary/30 transition-all"
                placeholder="Enter a fact to remember…"
                value={newContent}
                onChange={e => {
                  setNewContent(e.target.value)
                  if (autoTag && e.target.value.trim().length > 5) {
                    clearTimeout(suggestTimerRef.current)
                    suggestTimerRef.current = setTimeout(async () => {
                      try {
                        const res = await knowledgeController.suggestTopic(e.target.value.trim())
                        setSuggestedTopic(res.topic)
                      } catch { setSuggestedTopic(null) }
                    }, 300)
                  } else {
                    setSuggestedTopic(null)
                  }
                }}
                aria-label="Knowledge content"
              />
              {suggestedTopic && (
                <div className="text-xs text-muted-foreground flex items-center gap-1.5 animate-in fade-in slide-in-from-top-1">
                  <span className="opacity-70">Suggested topic:</span> <span className="font-medium text-primary">{suggestedTopic}</span>
                </div>
              )}
              <div className="flex items-center gap-3 flex-wrap">
                <input
                  type="text"
                  className="h-8 rounded-md border border-border bg-background px-2 text-xs w-40 focus:outline-none focus:ring-1 focus:ring-primary/30"
                  placeholder="Topic (optional)"
                  value={newTopic}
                  onChange={e => setNewTopic(e.target.value)}
                  aria-label="Topic"
                />
                <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer select-none hover:text-foreground transition-colors">
                  <Switch checked={autoTag} onCheckedChange={setAutoTag} />
                  Auto-tag
                </label>
                <Button onClick={handleAdd} disabled={adding || !newContent.trim()} size="sm" className="ml-auto">
                  {adding ? (
                    <span className="flex items-center gap-2">
                      <span className="h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent" />
                      Adding...
                    </span>
                  ) : (
                    <>
                      <IconPlus className="w-4 h-4 mr-1" />
                      Add
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>

        {/* Ingest URL */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Learn from a Web Page</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <input
                type="url"
                className="flex-1 h-8 rounded-md border border-border bg-background px-3 text-sm focus:outline-none focus:ring-1 focus:ring-primary/30 transition-all"
                placeholder="https://example.com/article"
                value={newUrl}
                onChange={e => setNewUrl(e.target.value)}
                aria-label="URL to ingest"
              />
              <Button onClick={handleIngestUrl} disabled={urlIngesting || !newUrl.trim()} size="sm" variant="outline" className="shrink-0">
                {urlIngesting ? (
                  <span className="flex items-center gap-2">
                    <span className="h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent" />
                    Learning...
                  </span>
                ) : 'Learn'}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Search & list */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Saved Facts</CardTitle>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={exportJson} disabled={items.length === 0}>
                  <IconDownload className="w-4 h-4 mr-1" /> Export
                </Button>
                <Button variant="outline" size="sm" onClick={() => fileInputRef.current?.click()}>
                  <IconUpload className="w-4 h-4 mr-1" /> Import
                </Button>
                <input ref={fileInputRef} type="file" accept=".json" className="hidden" onChange={importJson} aria-label="Import knowledge JSON file" />
                <Button variant="outline" size="sm" onClick={() => txtFileInputRef.current?.click()} disabled={fileUploading}>
                  <IconUpload className="w-4 h-4 mr-1" /> {fileUploading ? 'Uploading…' : 'Upload File'}
                </Button>
                <input ref={txtFileInputRef} type="file" accept=".txt,.md,.json" className="hidden" onChange={handleFileUpload} aria-label="Upload text or markdown file" />
                <span className="text-xs text-muted-foreground">{items.length} items{activeTopic ? ` in "${activeTopic}"` : ''}</span>
              </div>
            </div>
            <div className="flex items-center gap-2 mt-2">
              <SearchInput value={search} onChange={setSearch} placeholder="Search knowledge…" className="flex-1" />
              {selected.size > 0 && (
                <Button variant="destructive" size="sm" onClick={handleBatchDelete}>
                  <IconTrash className="w-4 h-4 mr-1" /> Delete {selected.size}
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="space-y-2">
                {[1,2,3].map(i => (
                  <div key={i} className="h-12 animate-pulse bg-muted rounded-md" />
                ))}
              </div>
            ) : items.length === 0 ? (
              <EmptyCard
                message={search ? 'Try a different search term' : activeTopic ? `No items in "${activeTopic}"` : 'Add your first fact above'}
              />
            ) : (
              <div className="space-y-2">
                <div className="flex items-center gap-2 pb-1 border-b border-border/30">
                  <input
                    type="checkbox"
                    checked={selected.size === items.length && items.length > 0}
                    onChange={toggleSelectAll}
                    className="rounded"
                    aria-label="Select all items"
                  />
                  <span className="text-xs text-muted-foreground">Select all</span>
                </div>
                {items.map(item => (
                  <div key={item.id} className="group flex items-start gap-3 rounded-lg border border-border/60 p-3 hover:bg-muted/40 transition-all duration-200">
                    <input
                      type="checkbox"
                      checked={selected.has(item.id)}
                      onChange={() => toggleSelect(item.id)}
                      className="mt-1 rounded border-border bg-background"
                      aria-label={`Select ${item.content.slice(0, 30)}`}
                    />
                    {editingId === item.id ? (
                      <div className="flex-1 min-w-0 space-y-3">
                        <textarea
                          className="w-full min-h-[60px] rounded-md border border-border bg-background px-2 py-1.5 text-sm resize-y focus:outline-none focus:ring-1 focus:ring-primary/30"
                          value={editContent}
                          onChange={e => setEditContent(e.target.value)}
                          aria-label="Edit knowledge content"
                        />
                        <div className="flex items-center gap-2">
                          <input
                            type="text"
                            className="h-7 rounded-md border border-border bg-background px-2 text-xs w-32 focus:outline-none focus:ring-1 focus:ring-primary/30"
                            value={editTopic}
                            onChange={e => setEditTopic(e.target.value)}
                            placeholder="Topic"
                            aria-label="Edit topic"
                          />
                          <Button size="sm" onClick={() => saveEdit(item.id)} className="h-7">Save</Button>
                          <Button size="sm" variant="outline" onClick={cancelEdit} className="h-7">Cancel</Button>
                        </div>
                        {/* Related items */}
                        {loadingRelated === item.id ? (
                          <div className="text-xs text-muted-foreground animate-pulse flex items-center gap-2">
                            <span className="h-1 w-1 rounded-full bg-current" /> Finding related items…
                          </div>
                        ) : relatedItems[item.id] && relatedItems[item.id].length > 0 ? (
                          <Collapsible className="text-xs group/related">
                            <CollapsibleTrigger className="cursor-pointer text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1">
                              Related ({relatedItems[item.id].length})
                            </CollapsibleTrigger>
                            <CollapsibleContent>
                              <div className="mt-2 space-y-1.5 pl-3 border-l-2 border-border/40">
                                {relatedItems[item.id].map(r => (
                                  <div key={r.id} className="text-muted-foreground line-clamp-1 hover:text-foreground transition-colors cursor-default">
                                    {r.content}
                                  </div>
                                ))}
                              </div>
                            </CollapsibleContent>
                          </Collapsible>
                        ) : null}
                      </div>
                    ) : (
                      <>
                        <div className="flex-1 min-w-0 cursor-pointer" onClick={() => startEdit(item)}>
                          <p className="text-sm leading-relaxed line-clamp-3">{item.content}</p>
                          <div className="flex items-center gap-2 mt-2">
                            <Badge variant="default" className="text-[10px] px-1.5 py-0 font-medium" label={item.topic} />
                            {item.source && (
                              <span className="text-[10px] text-muted-foreground/70 italic truncate max-w-[120px]">{item.source}</span>
                            )}
                            {item.url && (
                              <a href={item.url} target="_blank" rel="noreferrer" className="text-[10px] text-primary hover:underline truncate max-w-[160px]" onClick={e => e.stopPropagation()}>
                                {item.url.replace(/^https?:\/\//, '').split('/')[0]}
                              </a>
                            )}
                            <span className="text-[10px] text-muted-foreground/60 ml-auto font-mono">
                              {new Date(item.timestamp * 1000).toLocaleDateString()}
                            </span>
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="opacity-0 group-hover:opacity-100 shrink-0 h-8 w-8 transition-opacity"
                          onClick={() => handleDelete(item.id)}
                          aria-label={`Delete ${item.content.slice(0, 30)}`}
                        >
                          <IconTrash className="w-4 h-4 text-destructive" />
                        </Button>
                      </>
                    )}
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function KnowledgeOperationsCard() {
  const [searchQuery, setSearchQuery] = useState('')
  const [searchPath, setSearchPath] = useState('.')
  const [searchResults, setSearchResults] = useState<Array<{ path: string; line: number; snippet: string; score: number }>>([])
  const [searching, setSearching] = useState(false)

  const [dedupContent, setDedupContent] = useState('')
  const [dedupResult, setDedupResult] = useState<{ is_duplicate: boolean; best_match: string | null; score: number } | null>(null)
  const [checkingDup, setCheckingDup] = useState(false)

  const [gapData, setGapData] = useState<{ gaps: Array<{ topic: string; suggestion: string }>; total_facts: number } | null>(null)
  const [loadingGaps, setLoadingGaps] = useState(false)

  const addToast = useToastStore(s => s.addToast)

  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    setSearching(true)
    try {
      const res = await knowledgeController.searchFiles(searchQuery, searchPath)
      setSearchResults(res.results)
    } catch { addToast('Search failed', 'error') }
    setSearching(false)
  }

  const handleCheckDup = async () => {
    if (!dedupContent.trim()) return
    setCheckingDup(true)
    try {
      const res = await knowledgeController.checkDuplicate(dedupContent)
      setDedupResult(res)
    } catch { addToast('Duplicate check failed', 'error') }
    setCheckingDup(false)
  }

  const handleFindGaps = async () => {
    setLoadingGaps(true)
    try {
      const res = await knowledgeController.gaps()
      setGapData(res)
    } catch { addToast('Gap analysis failed', 'error') }
    setLoadingGaps(false)
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Semantic Tools</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* File search */}
        <div className="space-y-2">
          <label className="text-xs font-medium text-muted-foreground">Search codebase</label>
          <div className="flex items-center gap-2">
            <input
              className="flex-1 h-8 rounded-md border border-border bg-background px-3 text-sm focus:outline-none focus:ring-1 focus:ring-primary/30"
              placeholder="e.g. how does embedding work"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSearch()}
              aria-label="File search query"
            />
            <input
              className="h-8 rounded-md border border-border bg-background px-2 text-xs w-28 focus:outline-none focus:ring-1 focus:ring-primary/30"
              placeholder="path (.)"
              value={searchPath}
              onChange={e => setSearchPath(e.target.value)}
              aria-label="Search path"
            />
            <Button size="sm" onClick={handleSearch} disabled={searching || !searchQuery.trim()}>
              {searching ? 'Searching…' : 'Search'}
            </Button>
          </div>
          {searchResults.length > 0 && (
            <div className="space-y-1.5 mt-2 max-h-48 overflow-y-auto">
              {searchResults.map((r, i) => (
                <div key={i} className="text-xs rounded bg-muted/40 px-2 py-1.5 flex items-center gap-2">
                  <span className="font-mono text-muted-foreground shrink-0">{r.score.toFixed(3)}</span>
                  <span className="font-mono text-muted-foreground shrink-0">{r.path}:{r.line}</span>
                  <span className="truncate">{r.snippet.replace(/\n/g, ' ')}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="h-px bg-border/30" />

        {/* Duplicate check */}
        <div className="space-y-2">
          <label className="text-xs font-medium text-muted-foreground">Check for duplicates</label>
          <div className="flex items-center gap-2">
            <input
              className="flex-1 h-8 rounded-md border border-border bg-background px-3 text-sm focus:outline-none focus:ring-1 focus:ring-primary/30"
              placeholder="Paste content to check…"
              value={dedupContent}
              onChange={e => setDedupContent(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleCheckDup()}
              aria-label="Duplicate check content"
            />
            <Button size="sm" variant="outline" onClick={handleCheckDup} disabled={checkingDup || !dedupContent.trim()}>
              {checkingDup ? 'Checking…' : 'Check'}
            </Button>
          </div>
          {dedupResult && (
            <div className={`text-xs rounded px-2 py-1.5 ${dedupResult.is_duplicate ? 'bg-destructive/10 text-destructive' : 'bg-green-500/10 text-green-600'}`}>
              {dedupResult.is_duplicate
                ? `Duplicate (score: ${dedupResult.score.toFixed(3)}) — existing: "${dedupResult.best_match?.slice(0, 80)}…"`
                : `Unique (best match: ${dedupResult.score.toFixed(3)})`
              }
            </div>
          )}
        </div>

        <div className="h-px bg-border/30" />

        {/* Knowledge gaps */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-muted-foreground">Knowledge gaps</label>
            <Button size="sm" variant="ghost" onClick={handleFindGaps} disabled={loadingGaps}>
              {loadingGaps ? 'Analyzing…' : 'Analyze'}
            </Button>
          </div>
          {gapData && (
            <div className="space-y-1">
              {gapData.gaps.length === 0 ? (
                <p className="text-xs text-muted-foreground">No significant gaps — {gapData.total_facts} facts across all topics</p>
              ) : (
                gapData.gaps.map((g, i) => (
                  <div key={i} className="text-xs rounded bg-muted/40 px-2 py-1.5">
                    <span className="font-medium">{g.topic}:</span> <span className="text-muted-foreground">{g.suggestion}</span>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
