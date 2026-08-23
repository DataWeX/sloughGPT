'use client'
export const dynamic = 'force-dynamic'

import { useState, useCallback, useEffect } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { apiGet, apiPost, apiDelete } from '@/lib/http-client'

interface Pipeline {
  id: string
  name: string
  source_type: string
  store_type: string
  records_count?: number
  last_run?: string
}

interface CollectionStats {
  pipelines: number
  sources: number
  stores: number
  filters: number
}

export default function CollectionsPage() {
  const addToast = useToastStore(s => s.addToast)
  const [pipelines, setPipelines] = useState<Pipeline[]>([])
  const [stats, setStats] = useState<CollectionStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [showCreate, setShowCreate] = useState(false)
  const [newName, setNewName] = useState('')
  const [newSourceType, setNewSourceType] = useState('file')
  const [newStoreType, setNewStoreType] = useState('memory')
  const [creating, setCreating] = useState(false)
  const [runningId, setRunningId] = useState<string | null>(null)

  const fetchPipelines = useCallback(async () => {
    try {
      const data = await apiGet<{ pipelines: Pipeline[]; counts: CollectionStats }>('/collections')
      setPipelines(data.pipelines ?? [])
      setStats(data.counts ?? null)
    } catch {
      addToast('Failed to load collections', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => {
    void fetchPipelines()
  }, [fetchPipelines])

  const createPipeline = useCallback(async () => {
    if (!newName.trim()) {
      addToast('Name is required', 'error')
      return
    }
    setCreating(true)
    try {
      await apiPost('/collections/create', {
        name: newName,
        source_type: newSourceType,
        store_type: newStoreType,
      })
      addToast(`Created pipeline: ${newName}`, 'success')
      setNewName('')
      setShowCreate(false)
      void fetchPipelines()
    } catch {
      addToast('Failed to create pipeline', 'error')
    } finally {
      setCreating(false)
    }
  }, [newName, newSourceType, newStoreType, addToast, fetchPipelines])

  const runPipeline = useCallback(async (pipelineId: string) => {
    setRunningId(pipelineId)
    try {
      await apiPost(`/collections/run?name=${encodeURIComponent(pipelineId)}`)
      addToast('Pipeline executed', 'success')
      void fetchPipelines()
    } catch {
      addToast('Failed to run pipeline', 'error')
    } finally {
      setRunningId(null)
    }
  }, [addToast, fetchPipelines])

  const deletePipeline = useCallback(async (pipelineId: string) => {
    try {
      await apiDelete(`/collections/${pipelineId}`)
      addToast('Pipeline deleted', 'success')
      void fetchPipelines()
    } catch {
      addToast('Failed to delete pipeline', 'error')
    }
  }, [addToast, fetchPipelines])

  const sourceTypes = ['file', 'url', 'rss', 'api', 'sse', 'watch', 'generator']
  const storeTypes = ['memory', 'file', 'callback', 'chained', 'stats']

  return (
    <PageContainer
      title="Collections"
      subtitle="Data collection pipelines"
      headerRight={
        <div className="flex items-center gap-2">
          <Button size="sm" variant="ghost" onClick={() => void fetchPipelines()}>Refresh</Button>
          <Button size="sm" onClick={() => setShowCreate(!showCreate)}>
            {showCreate ? 'Cancel' : 'New pipeline'}
          </Button>
        </div>
      }
    >
      {stats && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Card>
            <CardContent className="p-3">
              <p className="text-xs text-muted-foreground">Pipelines</p>
              <p className="text-lg font-medium">{stats.pipelines}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-3">
              <p className="text-xs text-muted-foreground">Sources</p>
              <p className="text-lg font-medium">{stats.sources}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-3">
              <p className="text-xs text-muted-foreground">Stores</p>
              <p className="text-lg font-medium">{stats.stores}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-3">
              <p className="text-xs text-muted-foreground">Filters</p>
              <p className="text-lg font-medium">{stats.filters}</p>
            </CardContent>
          </Card>
        </div>
      )}

      {showCreate && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Create pipeline</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-col gap-1">
              <Label htmlFor="pipe-name" variant="uppercase">Name</Label>
              <Input id="pipe-name" value={newName} onChange={e => setNewName(e.target.value)}
                placeholder="my-pipeline" className="h-8 text-sm" />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="flex flex-col gap-1">
                <Label htmlFor="pipe-source" variant="uppercase">Source</Label>
                <select id="pipe-source" value={newSourceType} onChange={e => setNewSourceType(e.target.value)}
                  className="h-8 rounded-md border border-input bg-background px-2 text-sm">
                  {sourceTypes.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="pipe-store" variant="uppercase">Store</Label>
                <select id="pipe-store" value={newStoreType} onChange={e => setNewStoreType(e.target.value)}
                  className="h-8 rounded-md border border-input bg-background px-2 text-sm">
                  {storeTypes.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
            </div>
            <Button size="sm" onClick={createPipeline} disabled={creating || !newName.trim()}>
              {creating ? 'Creating...' : 'Create'}
            </Button>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Pipelines ({pipelines.length})</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <p className="text-xs text-muted-foreground">Loading...</p>
          ) : pipelines.length === 0 ? (
            <p className="text-xs text-muted-foreground">No pipelines configured. Create one above.</p>
          ) : (
            <div className="space-y-2">
              {pipelines.map(p => (
                <div key={p.id} className="flex items-center justify-between rounded border p-3 text-sm">
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-medium">{p.name}</p>
                    <div className="flex gap-3 text-xs text-muted-foreground">
                      <span>Source: {p.source_type}</span>
                      <span>Store: {p.store_type}</span>
                      {p.records_count != null && <span>{p.records_count} records</span>}
                      {p.last_run && <span>Last: {new Date(p.last_run).toLocaleDateString()}</span>}
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button size="sm" variant="ghost" onClick={() => runPipeline(p.id)} disabled={runningId === p.id}>
                      {runningId === p.id ? 'Running...' : 'Run'}
                    </Button>
                    <Button size="sm" variant="ghost" className="text-destructive" onClick={() => deletePipeline(p.id)}>
                      Delete
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </PageContainer>
  )
}
