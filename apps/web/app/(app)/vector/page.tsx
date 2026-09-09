'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { vectorController, type VectorSearchResult } from '@/lib/vector-controller'
import { useToastStore } from '@/lib/toast-store'
import { useRefreshShortcut } from '@/hooks/useRefreshShortcut'

export default function VectorPage() {
  const addToast = useToastStore(s => s.addToast)
  const [provider, setProvider] = useState('in_memory')
  const [count, setCount] = useState(0)
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<VectorSearchResult[]>([])
  const [searchTime, setSearchTime] = useState<number | null>(null)
  const [searching, setSearching] = useState(false)
  const [upsertText, setUpsertText] = useState('')
  const [upserting, setUpserting] = useState(false)
  const [initializing, setInitializing] = useState(false)

  const fetchStats = useCallback(async () => {
    try {
      setLoading(true)
      const stats = await vectorController.getStats()
      setProvider(stats.provider)
      setCount(stats.count)
    } catch {
      addToast('Could not load vector store data', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useRefreshShortcut(fetchStats)

  useEffect(() => { fetchStats() }, [fetchStats])

  const handleInit = async (prov: string) => {
    setInitializing(true)
    try {
      const result = await vectorController.init(prov)
      setProvider(result.provider)
      addToast(result.note || `Initialized ${result.provider} vector store`, 'success')
      await fetchStats()
    } catch {
      addToast('Could not initialize vector store', 'error')
    } finally {
      setInitializing(false)
    }
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    setSearching(true)
    try {
      const result = await vectorController.search(searchQuery)
      setSearchResults(result.results)
      setSearchTime(result.elapsed_ms)
    } catch {
      addToast('Could not search vector store', 'error')
    } finally {
      setSearching(false)
    }
  }

  const handleUpsert = async () => {
    const texts = upsertText.split('\n').filter(t => t.trim())
    if (texts.length === 0) return
    setUpserting(true)
    try {
      const result = await vectorController.upsert(texts)
      addToast(`Added ${result.count} entries in ${result.elapsed_ms}ms`, 'success')
      setUpsertText('')
      await fetchStats()
    } catch {
      addToast('Could not add entries', 'error')
    } finally {
      setUpserting(false)
    }
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'r' && !e.metaKey && !e.ctrlKey) { e.preventDefault(); void fetchStats() }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [fetchStats])

  return (
    <PageContainer
      title="Vector Store"
      subtitle="Manage embeddings and similarity search"
      headerRight={
        <Button size="sm" variant="outline" className="h-7 text-[11px]" onClick={fetchStats} disabled={loading}>
          <IconRefresh className="h-3 w-3 mr-1" />
          Refresh
        </Button>
      }
    >
      <KpiGrid>
        <StatCard label="Provider" value={provider === 'in_memory' ? 'In Memory' : provider} />
        <StatCard label="Vectors" value={loading ? <Skeleton className="h-5 w-10 inline-block" /> : String(count)} />
        <StatCard label="Dimension" value="384" />
        <StatCard label="Status" value={count > 0 ? 'Active' : 'Empty'} />
      </KpiGrid>

      <Card>
        <CardHeader className="pb-2 pt-2.5 px-2.5">
          <CardTitle className="text-[11px] font-medium">Initialize</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 px-2.5 pb-2.5">
          <p className="text-[10px] text-muted-foreground/60">Choose a vector store backend. ChromaDB persists to disk; in-memory is faster but does not survive restarts.</p>
          <div className="flex gap-1">
            <Button size="sm" variant={provider === 'in_memory' ? 'default' : 'outline'} className="h-7 text-[11px]" onClick={() => handleInit('in_memory')} disabled={initializing}>
              In Memory
            </Button>
            <Button size="sm" variant={provider === 'chromadb' ? 'default' : 'outline'} className="h-7 text-[11px]" onClick={() => handleInit('chromadb')} disabled={initializing}>
              ChromaDB
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2 pt-2.5 px-2.5">
          <CardTitle className="text-[11px] font-medium">Add Entries</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 px-2.5 pb-2.5">
          <p className="text-[10px] text-muted-foreground/60">Add text entries to the vector store. Each line becomes one entry.</p>
          <textarea
            value={upsertText}
            onChange={e => setUpsertText(e.target.value)}
            placeholder={"Enter text entries, one per line:\nSloughGPT is an AI framework\nIt learns from conversations\nMemory persists across sessions"}
            className="w-full h-20 text-[11px] font-mono rounded-lg border border-border/40 bg-background px-2.5 py-2 resize-none focus:outline-none focus:ring-1 focus:ring-ring"
            aria-label="Vector store entries"
          />
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-muted-foreground/60 font-mono tabular-nums">{upsertText.split('\n').filter(t => t.trim()).length} entries</span>
            <Button size="sm" className="h-7 text-[11px]" onClick={handleUpsert} disabled={upserting || !upsertText.trim()}>
              {upserting ? 'Adding...' : 'Add entries'}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2 pt-2.5 px-2.5">
          <CardTitle className="text-[11px] font-medium">Similarity Search</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 px-2.5 pb-2.5">
          <div className="flex gap-1.5">
            <Input
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="Search for similar text..."
              className="h-7 text-[11px] flex-1"
              onKeyDown={e => e.key === 'Enter' && handleSearch()}
            />
            <Button size="sm" className="h-7 text-[11px]" onClick={handleSearch} disabled={searching || !searchQuery.trim()}>
              {searching ? 'Searching...' : 'Search'}
            </Button>
          </div>

          {searchResults.length > 0 && (
            <div className="space-y-1">
              <p className="text-[10px] text-muted-foreground/60">{searchResults.length} results in {searchTime?.toFixed(1)}ms</p>
              {searchResults.map((r, i) => (
                <div key={r.id || i} className="border border-border/40 rounded-lg p-2.5 space-y-0.5">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px]">{r.text}</span>
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded-full bg-primary/10 text-primary tabular-nums">
                      {(r.score * 100).toFixed(1)}%
                    </span>
                  </div>
                  {r.id && <p className="text-[10px] text-muted-foreground/60 font-mono">{r.id}</p>}
                </div>
              ))}
            </div>
          )}

          {searchResults.length === 0 && !searching && searchQuery && (
            <p className="text-[10px] text-muted-foreground/60 text-center py-3">No results found</p>
          )}
        </CardContent>
      </Card>
    </PageContainer>
  )
}
