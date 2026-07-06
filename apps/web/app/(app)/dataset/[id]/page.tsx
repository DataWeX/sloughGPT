'use client'
export const dynamic = 'force-dynamic'

import { useCallback, useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import { StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import { IconTrash, IconDownload, IconEdit, IconCheck, IconX, IconRefresh } from '@sloughgpt/strui'
import { datasetController, type Dataset, type DatasetStats } from '@/lib/dataset-controller'
import { DatasetPreview } from '@/components/DatasetPreview'
import { useToastStore } from '@/lib/toast-store'

function formatBytes(bytes: number): string {
  if (!bytes) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function DatasetDetailPage() {
  const params = useParams()
  const router = useRouter()
  const addToast = useToastStore(s => s.addToast)
  const datasetId = decodeURIComponent((params.id as string) || '')

  const [dataset, setDataset] = useState<Dataset | null>(null)
  const [loading, setLoading] = useState(true)
  const [renaming, setRenaming] = useState(false)
  const [renameText, setRenameText] = useState('')
  const [stats, setStats] = useState<DatasetStats | null>(null)

  const fetchDataset = useCallback(async () => {
    setLoading(true)
    try {
      const d = await datasetController.get(datasetId)
      setDataset(d)
    } catch {
      addToast('Failed to load dataset', 'error')
    } finally {
      setLoading(false)
    }
  }, [datasetId, addToast])

  const fetchStats = useCallback(async () => {
    if (!datasetId) return
    try {
      const s = await datasetController.getStats(datasetId)
      setStats(s)
    } catch {
      // stats are optional — silently ignore
    }
  }, [datasetId])

  useEffect(() => {
    if (!datasetId) { router.push('/datasets'); return }
    fetchDataset()
  }, [datasetId, fetchDataset, router])

  useEffect(() => {
    if (dataset) {
      fetchStats()
    }
  }, [dataset, fetchStats])

  const startRename = () => {
    setRenameText(dataset?.name || '')
    setRenaming(true)
  }

  const commitRename = async () => {
    if (!renameText.trim() || !dataset) return
    try {
      await datasetController.update(dataset.id, { name: renameText.trim() })
      setDataset(prev => prev ? { ...prev, name: renameText.trim() } : prev)
      addToast('Renamed', 'success')
      setRenaming(false)
    } catch {
      addToast('Rename failed', 'error')
    }
  }

  const handleDelete = async () => {
    if (!dataset) return
    if (!confirm(`Delete dataset "${dataset.name}"?`)) return
    try {
      await datasetController.delete(dataset.id)
      addToast(`Deleted "${dataset.name}"`, 'info')
      router.push('/datasets')
    } catch {
      addToast('Delete failed', 'error')
    }
  }

  const handleExport = async () => {
    if (!dataset) return
    try {
      const blob = await datasetController.export(dataset.id)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url; a.download = `${dataset.name || dataset.id}.jsonl`; a.click()
      URL.revokeObjectURL(url)
      addToast('Exported', 'success')
    } catch {
      addToast('Export failed', 'error')
    }
  }

  if (!datasetId) return null

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="sm" onClick={() => router.push('/datasets')} className="h-7 px-1.5 text-xs text-muted-foreground hover:text-foreground">
              ← Datasets
            </Button>
            <AppRouteHeaderLead title={loading ? '...' : dataset?.name || datasetId} />
          </div>
        }
        right={
          <Button variant="secondary" size="sm" onClick={fetchDataset} disabled={loading}>
            <IconRefresh className={loading ? 'animate-spin h-3.5 w-3.5 mr-1' : 'h-3.5 w-3.5 mr-1'} />
            Refresh
          </Button>
        }
      />

      <div className="space-y-4">
        {loading ? (
          <div className="space-y-3">
            <Skeleton className="h-28 rounded-lg" />
            <Skeleton className="h-64 rounded-lg" />
          </div>
        ) : !dataset ? (
          <Card>
            <CardContent className="py-8 text-center text-muted-foreground text-sm">
              Dataset not found
            </CardContent>
          </Card>
        ) : (
          <>
            {/* Metadata card */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <CardTitle className="text-base">Details</CardTitle>
                    {dataset.type && <Badge variant="secondary" className="text-xs">{dataset.type}</Badge>}
                  </div>
                  <div className="flex items-center gap-2">
                    <Button size="sm" variant="outline" className="h-7 text-xs" onClick={handleExport}>
                      <IconDownload className="h-3 w-3 mr-1" /> Export
                    </Button>
                    <Button size="sm" variant="outline" className="h-7 text-xs text-destructive hover:text-destructive" onClick={handleDelete}>
                      <IconTrash className="h-3 w-3 mr-1" /> Delete
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {renaming ? (
                  <div className="flex items-center gap-2 mb-3">
                    <Input
                      value={renameText}
                      onChange={e => setRenameText(e.target.value)}
                      className="h-8 text-sm max-w-xs"
                      autoFocus
                      onKeyDown={e => { if (e.key === 'Enter') commitRename(); if (e.key === 'Escape') setRenaming(false) }}
                    />
                    <Button variant="ghost" size="icon-sm" className="h-7 w-7" onClick={commitRename} aria-label="Confirm rename">
                      <IconCheck className="h-4 w-4 text-success" />
                    </Button>
                    <Button variant="ghost" size="icon-sm" className="h-7 w-7" onClick={() => setRenaming(false)} aria-label="Cancel rename">
                      <IconX className="h-4 w-4" />
                    </Button>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-sm font-medium">{dataset.name}</span>
                    <Button variant="ghost" size="icon-sm" className="h-6 w-6" onClick={startRename} aria-label="Rename dataset">
                      <IconEdit className="h-3 w-3" />
                    </Button>
                  </div>
                )}

                <KpiGrid columns={4}>
                  <StatCard label="ID" value={dataset.id} />
                  <StatCard label="Source" value={dataset.source || 'local'} />
                  <StatCard label="Size" value={formatBytes(dataset.size)} />
                  {dataset.samples != null && <StatCard label="Samples" value={dataset.samples.toLocaleString()} />}
                </KpiGrid>

                {dataset.tags && dataset.tags.length > 0 && (
                  <div className="flex items-center gap-1.5 mt-3">
                    <span className="text-xs text-muted-foreground">Tags:</span>
                    {dataset.tags.map(t => (
                      <Badge key={t} variant="default" size="sm">{t}</Badge>
                    ))}
                  </div>
                )}

                {dataset.created_at && (
                  <p className="text-xs text-muted-foreground mt-2">
                    Created: {new Date(dataset.created_at).toLocaleString()}
                  </p>
                )}

                <div className="flex items-center gap-2 mt-3">
                  <Button size="sm" onClick={() => router.push(`/training?dataset=${dataset.id}&method=distill`)}>
                    Train on this dataset
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => router.push(`/training?dataset=${dataset.id}&method=finetune`)}>
                    Fine-tune
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Stats card */}
            {stats && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Stats</CardTitle>
                </CardHeader>
                <CardContent>
                  <KpiGrid columns={4}>
                    <StatCard label="Format" value={stats.format || '—'} />
                    <StatCard label="Rows (lines)" value={stats.lines?.toLocaleString() || '—'} />
                    <StatCard label="Avg length" value={stats.avg_length ? `${stats.avg_length.toFixed(0)} chars` : '—'} />
                    <StatCard label="Total chars" value={stats.chars?.toLocaleString() || '—'} />
                  </KpiGrid>
                  {stats.suggested_method && stats.suggested_method !== 'unknown' && (
                    <div className="flex items-center gap-2 mt-2">
                      <span className="text-xs text-muted-foreground">Recommended method:</span>
                      <Badge variant="secondary" size="sm">{stats.suggested_method}</Badge>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Preview card */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Preview</CardTitle>
              </CardHeader>
              <CardContent>
                <DatasetPreview datasetId={dataset.id} onUseForTraining={() => router.push(`/training?dataset=${dataset.id}&method=distill`)} />
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  )
}
