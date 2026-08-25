'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { vectorController, type VectorSearchResult } from '@/lib/vector-controller'
import { useToastStore } from '@/lib/toast-store'

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
        <Button size="sm" variant="outline" onClick={fetchStats} disabled={loading}>
          <IconRefresh className="h-3.5 w-3.5 mr-1" />
          Refresh
        </Button>
      }
    >
      <KpiGrid>
        <StatCard label="Provider" value={provider === 'in_memory' ? 'In Memory' : provider} />
        <StatCard label="Vectors" value={loading ? '...' : String(count)} />
        <StatCard label="Dimension" value="384" />
        <StatCard label="Status" value={count > 0 ? 'Active' : 'Empty'} />
      </KpiGrid>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Initialize</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-sm text-muted-foreground">Choose a vector store backend. ChromaDB persists to disk; in-memory is faster but does not survive restarts.</p>
          <div className="flex gap-2">
            <Button size="sm" variant={provider === 'in_memory' ? 'default' : 'outline'} onClick={() => handleInit('in_memory')} disabled={initializing}>
              In Memory
            </Button>
            <Button size="sm" variant={provider === 'chromadb' ? 'default' : 'outline'} onClick={() => handleInit('chromadb')} disabled={initializing}>
              ChromaDB
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Add Entries</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-sm text-muted-foreground">Add text entries to the vector store. Each line becomes one entry.</p>
          <textarea
            value={upsertText}
            onChange={e => setUpsertText(e.target.value)}
            placeholder={"Enter text entries, one per line:\nSloughGPT is an AI framework\nIt learns from conversations\nMemory persists across sessions"}
            className="w-full h-24 text-sm font-mono rounded-md border bg-background px-3 py-2 resize-none focus:outline-none focus:ring-1 focus:ring-ring"
            aria-label="Vector store entries"
          />
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">{upsertText.split('\n').filter(t => t.trim()).length} entries</span>
            <Button size="sm" onClick={handleUpsert} disabled={upserting || !upsertText.trim()}>
              {upserting ? 'Adding...' : 'Add entries'}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Similarity Search</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex gap-2">
            <Input
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="Search for similar text..."
              onKeyDown={e => e.key === 'Enter' && handleSearch()}
            />
            <Button size="sm" onClick={handleSearch} disabled={searching || !searchQuery.trim()}>
              {searching ? 'Searching...' : 'Search'}
            </Button>
          </div>

          {searchResults.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground">{searchResults.length} results in {searchTime?.toFixed(1)}ms</p>
              {searchResults.map((r, i) => (
                <div key={r.id || i} className="border rounded-md p-3 space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">{r.text}</span>
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-primary/10 text-primary tabular-nums">
                      {(r.score * 100).toFixed(1)}%
                    </span>
                  </div>
                  {r.id && <p className="text-[10px] text-muted-foreground font-mono">{r.id}</p>}
                </div>
              ))}
            </div>
          )}

          {searchResults.length === 0 && !searching && searchQuery && (
            <p className="text-xs text-muted-foreground text-center py-4">No results found</p>
          )}
        </CardContent>
      </Card>
    </PageContainer>
  )
}
