'use client'
import { notify } from '@/lib/notify'

import { useState, useCallback, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label } from '@sloughgpt/strui'
import { apiGet, apiPost, apiDelete } from '@/lib/http-client'
// Module scope: `addToast` must keep a stable identity — it sits in
// useCallback/useEffect dependency arrays, and a per-render arrow would
// re-fire those effects on every render.
const addToast = (message: string, type?: string) =>
  notify.toast(message, { type: type as 'success' | 'error' | 'info' })

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

export default function CollectionsContent() {
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
      notify.toast('Could not load collections', { type: 'error' })
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void fetchPipelines()
  }, [fetchPipelines])

  const createPipeline = useCallback(async () => {
    if (!newName.trim()) {
      notify.toast('Name is required', { type: 'error' })
      return
    }
    setCreating(true)
    try {
      await apiPost('/collections/create', {
        name: newName,
        source_type: newSourceType,
        store_type: newStoreType,
      })
      notify.toast(`Created pipeline: ${newName}`, { type: 'success' })
      setNewName('')
      setShowCreate(false)
      void fetchPipelines()
    } catch {
      notify.toast('Could not create pipeline', { type: 'error' })
    } finally {
      setCreating(false)
    }
  }, [newName, newSourceType, newStoreType, fetchPipelines])

  const runPipeline = useCallback(
    async (pipelineId: string) => {
      setRunningId(pipelineId)
      try {
        await apiPost(`/collections/run?name=${encodeURIComponent(pipelineId)}`)
        notify.toast('Pipeline executed', { type: 'success' })
        void fetchPipelines()
      } catch {
        notify.toast('Could not run pipeline', { type: 'error' })
      } finally {
        setRunningId(null)
      }
    },
    [fetchPipelines],
  )

  const deletePipeline = useCallback(
    async (pipelineId: string) => {
      try {
        await apiDelete(`/collections/${pipelineId}`)
        notify.toast('Pipeline deleted', { type: 'success' })
        void fetchPipelines()
      } catch {
        notify.toast('Could not delete pipeline', { type: 'error' })
      }
    },
    [fetchPipelines],
  )

  const sourceTypes = ['file', 'url', 'rss', 'api', 'sse', 'watch', 'generator']
  const storeTypes = ['memory', 'file', 'callback', 'chained', 'stats']

  return (
    <>
      {stats && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Card>
            <CardContent className="p-3">
              <p className="text-xs text-muted-foreground">Pipelines</p>
              <p className="text-base font-medium">{stats.pipelines}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-3">
              <p className="text-xs text-muted-foreground">Sources</p>
              <p className="text-base font-medium">{stats.sources}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-3">
              <p className="text-xs text-muted-foreground">Stores</p>
              <p className="text-base font-medium">{stats.stores}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-3">
              <p className="text-xs text-muted-foreground">Filters</p>
              <p className="text-base font-medium">{stats.filters}</p>
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
              <Label htmlFor="pipe-name" variant="uppercase">
                Name
              </Label>
              <Input
                id="pipe-name"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="my-pipeline"
                className="h-8 text-sm"
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="flex flex-col gap-1">
                <Label htmlFor="pipe-source" variant="uppercase">
                  Source
                </Label>
                <select
                  id="pipe-source"
                  value={newSourceType}
                  onChange={(e) => setNewSourceType(e.target.value)}
                  className="h-8 rounded-md border border-input bg-background px-2 text-sm"
                >
                  {sourceTypes.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="pipe-store" variant="uppercase">
                  Store
                </Label>
                <select
                  id="pipe-store"
                  value={newStoreType}
                  onChange={(e) => setNewStoreType(e.target.value)}
                  className="h-8 rounded-md border border-input bg-background px-2 text-sm"
                >
                  {storeTypes.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
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
          <div className="flex items-center justify-between">
            <CardTitle className="text-base">Pipelines ({pipelines.length})</CardTitle>
            <div className="flex items-center gap-2">
              <Button size="sm" variant="ghost" onClick={() => void fetchPipelines()}>
                Refresh
              </Button>
              <Button
                size="sm"
                onClick={() => setShowCreate(!showCreate)}
                aria-pressed={showCreate}
              >
                {showCreate ? 'Cancel' : 'New pipeline'}
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="space-y-2">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-16 animate-pulse bg-muted/50 rounded" />
              ))}
            </div>
          ) : pipelines.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              No pipelines configured. Create one above.
            </p>
          ) : (
            <div className="space-y-2">
              {pipelines.map((p) => (
                <div
                  key={p.id}
                  className="flex items-center justify-between rounded border p-3 text-sm"
                >
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
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => runPipeline(p.id)}
                      disabled={runningId === p.id}
                    >
                      {runningId === p.id ? 'Running...' : 'Run'}
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-destructive"
                      onClick={() => deletePipeline(p.id)}
                    >
                      Delete
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </>
  )
}
