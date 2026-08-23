'use client'
export const dynamic = 'force-dynamic'

import { useState, useCallback, useEffect } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { apiGet, apiPost } from '@/lib/http-client'

interface VectorStats {
  total_vectors: number
  dimensions: number
  provider: string
  index_type?: string
}

interface SearchResult {
  id: string
  score: number
  content?: string
  metadata?: Record<string, unknown>
}

export default function VectorPage() {
  const addToast = useToastStore(s => s.addToast)
  const [stats, setStats] = useState<VectorStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [initProvider, setInitProvider] = useState('local')
  const [initDimensions, setInitDimensions] = useState(384)
  const [initializing, setInitializing] = useState(false)

  const [upsertId, setUpsertId] = useState('')
  const [upsertContent, setUpsertContent] = useState('')
  const [upserting, setUpserting] = useState(false)

  const [searchQuery, setSearchQuery] = useState('')
  const [searchLimit, setSearchLimit] = useState(5)
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [searching, setSearching] = useState(false)

  const [ingestStatus, setIngestStatus] = useState<Record<string, unknown> | null>(null)

  const fetchStats = useCallback(async () => {
    setLoading(true)
    try {
      const data = await apiGet<VectorStats>('/vector/stats')
      setStats(data)
    } catch {
      addToast('Failed to load vector stats', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  const fetchIngestStatus = useCallback(async () => {
    try {
      const data = await apiGet<Record<string, unknown>>('/vector/ingest/status')
      setIngestStatus(data)
    } catch { /* ignore */ }
  }, [])

  useEffect(() => {
    void fetchStats()
    void fetchIngestStatus()
  }, [fetchStats, fetchIngestStatus])

  const initStore = useCallback(async () => {
    setInitializing(true)
    try {
      await apiPost('/vector/init', { provider: initProvider, dimensions: initDimensions })
      addToast('Vector store initialized', 'success')
      void fetchStats()
    } catch {
      addToast('Failed to initialize', 'error')
    } finally {
      setInitializing(false)
    }
  }, [initProvider, initDimensions, addToast, fetchStats])

  const upsert = useCallback(async () => {
    if (!upsertId.trim() || !upsertContent.trim()) {
      addToast('ID and content required', 'error')
      return
    }
    setUpserting(true)
    try {
      await apiPost('/vector/upsert', { id: upsertId, content: upsertContent })
      addToast('Vector upserted', 'success')
      setUpsertId('')
      setUpsertContent('')
      void fetchStats()
    } catch {
      addToast('Failed to upsert', 'error')
    } finally {
      setUpserting(false)
    }
  }, [upsertId, upsertContent, addToast, fetchStats])

  const search = useCallback(async () => {
    if (!searchQuery.trim()) return
    setSearching(true)
    try {
      const data = await apiPost<{ results: SearchResult[] }>('/vector/search', {
        query: searchQuery, limit: searchLimit,
      })
      setSearchResults(data.results ?? [])
    } catch {
      addToast('Search failed', 'error')
    } finally {
      setSearching(false)
    }
  }, [searchQuery, searchLimit, addToast])

  return (
    <PageContainer
      title="Vector store"
      subtitle="Embedding storage and semantic search"
      headerRight={
        <div className="flex items-center gap-2">
          <Button size="sm" variant="ghost" onClick={() => { void fetchStats(); void fetchIngestStatus() }}>Refresh</Button>
        </div>
      }
    >
      {stats && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Card>
            <CardContent className="p-3">
              <p className="text-xs text-muted-foreground">Vectors</p>
              <p className="text-lg font-medium">{stats.total_vectors}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-3">
              <p className="text-xs text-muted-foreground">Dimensions</p>
              <p className="text-lg font-medium">{stats.dimensions}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-3">
              <p className="text-xs text-muted-foreground">Provider</p>
              <p className="text-lg font-medium">{stats.provider}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-3">
              <p className="text-xs text-muted-foreground">Index</p>
              <p className="text-lg font-medium">{stats.index_type ?? '--'}</p>
            </CardContent>
          </Card>
        </div>
      )}

      {!loading && !stats && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Initialize vector store</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-xs text-muted-foreground">No vector store configured. Initialize one to get started.</p>
            <div className="grid grid-cols-2 gap-3">
              <div className="flex flex-col gap-1">
                <Label variant="uppercase">Provider</Label>
                <select value={initProvider} onChange={e => setInitProvider(e.target.value)}
                  className="h-8 rounded-md border border-input bg-background px-2 text-sm">
                  <option value="local">Local (numpy)</option>
                  <option value="faiss">FAISS</option>
                  <option value="qdrant">Qdrant</option>
                </select>
              </div>
              <div className="flex flex-col gap-1">
                <Label variant="uppercase">Dimensions</Label>
                <Input type="number" value={initDimensions} onChange={e => setInitDimensions(Number(e.target.value))}
                  className="h-8 text-xs font-mono" />
              </div>
            </div>
            <Button size="sm" onClick={initStore} disabled={initializing}>
              {initializing ? 'Initializing...' : 'Initialize'}
            </Button>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Upsert vector</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-col gap-1">
              <Label variant="uppercase">ID</Label>
              <Input value={upsertId} onChange={e => setUpsertId(e.target.value)}
                placeholder="vec-001" className="h-8 text-xs font-mono" />
            </div>
            <div className="flex flex-col gap-1">
              <Label variant="uppercase">Content</Label>
              <textarea value={upsertContent} onChange={e => setUpsertContent(e.target.value)}
                rows={4} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                placeholder="Text content to embed and store..." />
            </div>
            <Button size="sm" onClick={upsert} disabled={upserting}>
              {upserting ? 'Upserting...' : 'Upsert'}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Semantic search</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex gap-2">
              <Input value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') void search() }}
                placeholder="Search query..." className="h-8 text-xs flex-1" />
              <Input type="number" value={searchLimit} onChange={e => setSearchLimit(Number(e.target.value))}
                min={1} max={50} className="h-8 text-xs w-16" />
              <Button size="sm" onClick={() => void search()} disabled={searching}>
                {searching ? '...' : 'Search'}
              </Button>
            </div>
            {searchResults.length > 0 && (
              <div className="max-h-[300px] space-y-1 overflow-y-auto">
                {searchResults.map(r => (
                  <div key={r.id} className="rounded border p-2 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="truncate font-mono font-medium">{r.id}</span>
                      <span className="text-muted-foreground">score: {r.score.toFixed(4)}</span>
                    </div>
                    {r.content && <p className="mt-1 truncate text-muted-foreground">{r.content.slice(0, 100)}</p>}
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {ingestStatus && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Ingest status</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="max-h-[200px] overflow-y-auto rounded bg-muted/30 p-3 text-xs">
              {JSON.stringify(ingestStatus, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}
    </PageContainer>
  )
}
