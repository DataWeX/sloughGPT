'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge, Chip, EmptyCard } from '@/components/ui'
import { SearchInput } from '@/components/ui/input'
import { IconSearch, IconPlus, IconTrash, IconDownload, IconUpload } from '@/components/icons/NavIcons'
import { knowledgeController, type KnowledgeItem } from '@/lib/knowledge-controller'

export default function KnowledgePage() {
  const [items, setItems] = useState<KnowledgeItem[]>([])
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(true)
  const [newContent, setNewContent] = useState('')
  const [newTopic, setNewTopic] = useState('general')
  const [adding, setAdding] = useState(false)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const fileInputRef = useRef<HTMLInputElement>(null)

  const fetchItems = useCallback(async () => {
    setLoading(true)
    try {
      const data = search
        ? (await knowledgeController.search(search)).results
        : await knowledgeController.list()
      setItems(data)
    } catch { /* ignore */ }
    setLoading(false)
  }, [search])

  useEffect(() => { fetchItems() }, [fetchItems])

  const handleAdd = async () => {
    if (!newContent.trim()) return
    setAdding(true)
    try {
      await knowledgeController.add(newContent.trim(), newTopic)
      setNewContent('')
      await fetchItems()
    } catch { /* ignore */ }
    setAdding(false)
  }

  const handleDelete = async (id: string) => {
    try {
      await knowledgeController.delete(id)
      setItems(prev => prev.filter(i => i.id !== id))
      setSelected(prev => { const n = new Set(prev); n.delete(id); return n })
    } catch { /* ignore */ }
  }

  const handleBatchDelete = async () => {
    for (const id of selected) {
      try { await knowledgeController.delete(id) } catch { /* ignore */ }
    }
    await fetchItems()
    setSelected(new Set())
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
    for (const item of arr) {
      try {
        await knowledgeController.add(item.content || item.text, item.topic || item.category || 'general')
      } catch { /* ignore */ }
    }
    await fetchItems()
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const topics = ['general', 'code', 'docs', 'reference', 'persona']

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="Knowledge" />} />
      <div className="space-y-4">
        {/* Add new knowledge */}
        <Card>
          <CardHeader><CardTitle className="text-base">Add Knowledge</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <textarea
              className="w-full min-h-[80px] rounded-md border border-border bg-background px-3 py-2 text-sm resize-y"
              placeholder="Enter knowledge content…"
              value={newContent}
              onChange={e => setNewContent(e.target.value)}
            />
            <div className="flex flex-wrap gap-2">
              {topics.map(topic => (
                <Chip key={topic} label={topic} selected={newTopic === topic} onClick={() => setNewTopic(topic)} />
              ))}
            </div>
            <div className="flex items-center gap-2">
              <Button onClick={handleAdd} disabled={adding || !newContent.trim()} size="sm">
                <IconPlus className="w-4 h-4 mr-1" />
                {adding ? 'Adding…' : 'Add'}
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
                <span className="text-xs text-muted-foreground">{items.length} items</span>
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
                message={search ? 'Try a different search term' : 'Add your first knowledge item above'}
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
                    <div className="flex-1 min-w-0">
                      <p className="text-sm line-clamp-2">{item.content}</p>
                      <div className="flex items-center gap-2 mt-1">
                        <Badge variant="default" className="text-[10px]" label={item.topic} />
                        {item.source && (
                          <span className="text-[10px] text-muted-foreground">source: {item.source}</span>
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
