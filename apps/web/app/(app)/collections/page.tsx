'use client'
export const dynamic = 'force-dynamic'

import { useState, useCallback, useEffect } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label, Spinner } from '@sloughgpt/strui'
import { CollectionStatsCard } from '@/components/collections/CollectionStatsCard'
import { CollectionPipelineCard } from '@/components/collections/CollectionPipelineCard'
import { CollectionCreateCard } from '@/components/collections/CollectionCreateCard'
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
      addToast('Could not load collections', 'error')
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
      addToast('Could not create pipeline', 'error')
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
      addToast('Could not run pipeline', 'error')
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
      addToast('Could not delete pipeline', 'error')
    }
  }, [addToast, fetchPipelines])

  const sourceTypes = ['file', 'url', 'rss', 'api', 'sse', 'watch', 'generator']
  const storeTypes = ['memory', 'file', 'callback', 'chained', 'stats']

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'r' && !e.metaKey && !e.ctrlKey) { e.preventDefault(); void fetchPipelines() }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [fetchPipelines])

  return (
    <PageContainer
      title="Collections"
      subtitle="Data collection pipelines"
      headerRight={
        <div className="flex items-center gap-1">
          <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => void fetchPipelines()}>Refresh</Button>
          <Button size="sm" className="h-6 text-[10px]" onClick={() => setShowCreate(!showCreate)} aria-pressed={showCreate}>
            {showCreate ? 'Cancel' : 'New pipeline'}
          </Button>
        </div>
      }
    >
      <CollectionStatsCard stats={stats} />

      <CollectionCreateCard onCreate={async (name, source, store) => {
        await apiPost('/collections/create', { name, source_type: source, store_type: store })
        addToast(`Created pipeline: ${name}`, 'success')
        void fetchPipelines()
      }} />

      <CollectionPipelineCard
        pipelines={pipelines}
        runningId={runningId}
        onRun={runPipeline}
        onDelete={deletePipeline}
      />
    </PageContainer>
  )
}
