'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge, Chip, EmptyCard, KpiGrid, StatCard } from '@/components/ui'
import { SearchInput } from '@/components/ui/input'
import { IconSearch, IconPlus, IconTrash, IconDownload, IconUpload } from '@/components/icons/NavIcons'
import { IconRefresh } from '@/components/ui'
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
  const addToast = useToastStore(s => s.addToast)

  const fetchItems = useCallback(async () => {
    setLoading(true)
    try {
      const data = search
        ? (await knowledgeController.search(search)).results
        : await knowledgeController.list(500, 0)
      const filtered = activeTopic ? data.filter(i => i.topic === activeTopic) : data
      setItems(filtered)
    } catch { /* ignore */ }
    setLoading(false)
  }, [search, activeTopic])

  const fetchAdapterStatus = useCallback(async () => {
    try {
      const s = await knowledgeController.getAdapterStatus()
      setAdapterStatus(s)
    } catch { /* ignore */ }
  }, [])

  const fetchStats = useCallback(async () => {
    try {
      const [s, t] = await Promise.all([knowledgeController.stats(), knowledgeController.topics()])
      setStats(s)
      setTopics(t.topics)
    } catch { /* ignore */ }
  }, [])

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
    } catch (e) { addToast(e instanceof Error ? e.message : 'Failed to add', 'error') }
    setAdding(false)
  }

  const handleIngestUrl = async () => {
    if (!newUrl.trim()) return
    setUrlIngesting(true)
    try {
      const res = await knowledgeController.ingestUrl(newUrl)
      if (res.status === 'ok') {
        addToast(`Ingested ${res.new_facts} facts from "${res.title}"`, 'success')
      } else if (res.rejected) {
        addToast(`URL rejected: ${res.reason || 'filtered'}`, 'info')
      } else {
        addToast(`Status: ${res.status}`, 'info')
      }
      setNewUrl('')
      await Promise.all([fetchItems(), fetchStats()])
    } catch (e) { addToast('Ingestion failed', 'error') }
    setUrlIngesting(false)
  }

  const handleDelete = async (id: string) => {
    try {
      await knowledgeController.delete(id)
      setItems(prev => prev.filter(i => i.id !== id))
      setSelected(prev => { const n = new Set(prev); n.delete(id); return n })
      await fetchStats()
    } catch { /* ignore */ }
  }

  const handleBatchDelete = async () => {
    const ids = Array.from(selected)
    try {
      const res = await knowledgeController.batchDelete(ids)
      addToast(`Deleted ${res.deleted} items`, 'success')
    } catch { /* ignore */ }
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
      } catch { /* ignore */ }
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
      } catch { /* ignore */ }
    }
    addToast(`Imported ${count} items`, 'success')
    await Promise.all([fetchItems(), fetchStats()])
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const handleTrainAdapter = async () => {
    setTrainingAdapter(true)
    try {
      const res = await knowledgeController.trainAdapter()
      addToast(`Adapter trained on ${res.fact_count} facts in ${res.elapsed}s`, 'success')
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
        right={<Button variant="secondary" size="sm" onClick={handleRefresh}><IconRefresh className="w-3.5 h-3.5 mr-1" /> Refresh</Button>}
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
            <CardTitle className="text-base">Model Weights</CardTitle>
            <Button
              size="sm"
              onClick={handleTrainAdapter}
              disabled={trainingAdapter || (adapterStatus?.total_facts_available ?? 0) < 1}
            >
              {trainingAdapter ? 'Training…' : 'Train on Knowledge'}
            </Button>
          </CardHeader>
          <CardContent>
            {adapterStatus === null ? (
              <div className="h-8 animate-pulse bg-muted rounded" />
            ) : adapterStatus.adapter_exists ? (
              <div className="flex items-center gap-4 text-sm flex-wrap">
                <Badge variant="default" label="Trained" />
                <span className="text-muted-foreground">
                  {adapterStatus.fact_count} facts · rank {adapterStatus.lora_rank}
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
                No adapter yet. Click &ldquo;Train on Knowledge&rdquo; to bake {stats?.total_items || 0} facts into model weights — no prompt tuning needed.
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

        {/* Add new knowledge */}
        <Card>
          <CardHeader><CardTitle className="text-base">Add Knowledge</CardTitle></CardHeader>
          <CardContent className="space-y-3">
              <textarea
                className="w-full min-h-[80px] rounded-md border border-border bg-background px-3 py-2 text-sm resize-y"
                placeholder="Enter knowledge content…"
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
              />
              {suggestedTopic && (
                <div className="text-xs text-muted-foreground flex items-center gap-1">
                  Suggested topic: <span className="font-medium text-primary">{suggestedTopic}</span>
                </div>
              )}
            <div className="flex items-center gap-3 flex-wrap">
              <input
                type="text"
                className="h-8 rounded-md border border-border bg-background px-2 text-xs w-40"
                placeholder="Topic (optional)"
                value={newTopic}
                onChange={e => setNewTopic(e.target.value)}
              />
              <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer">
                <input type="checkbox" checked={autoTag} onChange={e => setAutoTag(e.target.checked)} className="rounded" />
                Auto-tag
              </label>
              <Button onClick={handleAdd} disabled={adding || !newContent.trim()} size="sm">
                <IconPlus className="w-4 h-4 mr-1" />
                {adding ? 'Adding…' : 'Add'}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Ingest URL */}
        <Card>
          <CardHeader><CardTitle className="text-base">Ingest from URL</CardTitle></CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <input
                type="url"
                className="flex-1 h-8 rounded-md border border-border bg-background px-2 text-sm"
                placeholder="https://example.com/article"
                value={newUrl}
                onChange={e => setNewUrl(e.target.value)}
              />
              <Button onClick={handleIngestUrl} disabled={urlIngesting || !newUrl.trim()} size="sm" variant="outline">
                {urlIngesting ? 'Ingesting…' : 'Ingest'}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Search & list */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Knowledge Base</CardTitle>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={exportJson} disabled={items.length === 0}>
                  <IconDownload className="w-4 h-4 mr-1" /> Export
                </Button>
                <Button variant="outline" size="sm" onClick={() => fileInputRef.current?.click()}>
                  <IconUpload className="w-4 h-4 mr-1" /> Import
                </Button>
                <input ref={fileInputRef} type="file" accept=".json" className="hidden" onChange={importJson} />
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
                message={search ? 'Try a different search term' : activeTopic ? `No items in "${activeTopic}"` : 'Add your first knowledge item above'}
              />
            ) : (
              <div className="space-y-2">
                <div className="flex items-center gap-2 pb-1 border-b border-border/30">
                  <input
                    type="checkbox"
                    checked={selected.size === items.length && items.length > 0}
                    onChange={toggleSelectAll}
                    className="rounded"
                  />
                  <span className="text-xs text-muted-foreground">Select all</span>
                </div>
                {items.map(item => (
                  <div key={item.id} className="group flex items-start gap-3 rounded-md border border-border/50 p-3 hover:bg-muted/30 transition-colors">
                    <input
                      type="checkbox"
                      checked={selected.has(item.id)}
                      onChange={() => toggleSelect(item.id)}
                      className="mt-1 rounded"
                    />
                    {editingId === item.id ? (
                      <div className="flex-1 min-w-0 space-y-2">
                        <textarea
                          className="w-full min-h-[60px] rounded border border-border bg-background px-2 py-1 text-sm resize-y"
                          value={editContent}
                          onChange={e => setEditContent(e.target.value)}
                        />
                        <div className="flex items-center gap-2">
                          <input
                            type="text"
                            className="h-7 rounded border border-border bg-background px-2 text-xs w-32"
                            value={editTopic}
                            onChange={e => setEditTopic(e.target.value)}
                            placeholder="Topic"
                          />
                          <Button size="sm" onClick={() => saveEdit(item.id)}>Save</Button>
                          <Button size="sm" variant="outline" onClick={cancelEdit}>Cancel</Button>
                        </div>
                        {/* Related items */}
                        {loadingRelated === item.id ? (
                          <div className="text-xs text-muted-foreground animate-pulse">Finding related items…</div>
                        ) : relatedItems[item.id] && relatedItems[item.id].length > 0 ? (
                          <details className="text-xs">
                            <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                              Related ({relatedItems[item.id].length})
                            </summary>
                            <div className="mt-1 space-y-1 pl-2 border-l-2 border-border/30">
                              {relatedItems[item.id].map(r => (
                                <div key={r.id} className="text-muted-foreground line-clamp-1">{r.content}</div>
                              ))}
                            </div>
                          </details>
                        ) : null}
                      </div>
                    ) : (
                      <>
                        <div className="flex-1 min-w-0 cursor-pointer" onClick={() => startEdit(item)}>
                          <p className="text-sm line-clamp-2">{item.content}</p>
                          <div className="flex items-center gap-2 mt-1">
                            <Badge variant="default" className="text-[10px]" label={item.topic} />
                            {item.source && (
                              <span className="text-[10px] text-muted-foreground">{item.source}</span>
                            )}
                            {item.url && (
                              <a href={item.url} target="_blank" rel="noreferrer" className="text-[10px] text-primary hover:underline truncate max-w-[160px]" onClick={e => e.stopPropagation()}>
                                {item.url.replace(/^https?:\/\//, '').split('/')[0]}
                              </a>
                            )}
                            <span className="text-[10px] text-muted-foreground ml-auto">
                              {new Date(item.timestamp * 1000).toLocaleDateString()}
                            </span>
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="opacity-0 group-hover:opacity-100 shrink-0"
                          onClick={() => handleDelete(item.id)}
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
