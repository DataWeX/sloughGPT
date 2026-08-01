'use client'
export const dynamic = 'force-dynamic'

import { useCallback, useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import { StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import { IconTrash, IconDownload, IconEdit, IconCheck, IconX, IconRefresh, IconClock } from '@sloughgpt/strui'
import { datasetController, type Dataset, type DatasetStats } from '@/lib/dataset-controller'
import { DatasetPreview } from '@/components/DatasetPreview'
import { formatBytes } from '@/lib/format-bytes'
import { downloadBlob } from '@/lib/download-utils'
import { useToastStore } from '@/lib/toast-store'

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
  const [showDelete, setShowDelete] = useState(false)
  const [versions, setVersions] = useState<string[]>([])
  const [versionsLoading, setVersionsLoading] = useState(false)
  const [snapshotting, setSnapshotting] = useState(false)
  const [restoreTarget, setRestoreTarget] = useState<string | null>(null)

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

  const fetchVersions = useCallback(async () => {
    if (!datasetId) return
    setVersionsLoading(true)
    try {
      const res = await datasetController.listVersions(datasetId)
      setVersions(res.versions || [])
    } catch {
      setVersions([])
    } finally {
      setVersionsLoading(false)
    }
  }, [datasetId])

  const handleCreateVersion = async () => {
    if (!dataset) return
    setSnapshotting(true)
    try {
      await datasetController.createVersion(dataset.id)
      addToast('Snapshot created', 'success')
      fetchVersions()
    } catch {
      addToast('Snapshot failed', 'error')
    } finally {
      setSnapshotting(false)
    }
  }

  const handleRestoreVersion = async () => {
    if (!dataset || !restoreTarget) return
    try {
      const res = await datasetController.restoreVersion(dataset.id, restoreTarget)
      addToast(res.message || 'Version restored', 'success')
      fetchVersions()
    } catch {
      addToast('Restore failed', 'error')
    } finally {
      setRestoreTarget(null)
    }
  }

  useEffect(() => {
    if (!datasetId) { router.push('/datasets'); return }
    fetchDataset()
  }, [datasetId, fetchDataset, router])

  useEffect(() => {
    if (dataset) {
      fetchStats()
      fetchVersions()
    }
  }, [dataset, fetchStats, fetchVersions])

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
    try {
      await datasetController.delete(dataset.id)
      addToast(`Deleted "${dataset.name}"`, 'info')
      router.push('/datasets')
    } catch {
      addToast('Delete failed', 'error')
    } finally {
      setShowDelete(false)
    }
  }

  const handleExport = async () => {
    if (!dataset) return
    try {
      const blob = await datasetController.export(dataset.id)
      downloadBlob(blob, `${dataset.name || dataset.id}.jsonl`)
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
                    {dataset.type && <Badge variant={"secondary" as const} className="text-xs">{dataset.type}</Badge>}
                  </div>
                  <div className="flex items-center gap-2">
                    <Button size="sm" variant="outline" className="h-7 text-xs" onClick={handleExport}>
                      <IconDownload className="h-3 w-3 mr-1" /> Export
                    </Button>
                    <Button size="sm" variant="outline" className="h-7 text-xs text-destructive hover:text-destructive" onClick={() => setShowDelete(true)}>
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
                      onChange={(e: React.ChangeEvent<HTMLInputElement>) => setRenameText(e.target.value)}
                      className="h-8 text-sm max-w-xs"
                      autoFocus
                      onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => { if (e.key === 'Enter') commitRename(); if (e.key === 'Escape') setRenaming(false) }}
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
                      <Badge key={t} variant={"default" as const} size="sm">{t}</Badge>
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
                      <Badge variant={"secondary" as const} size="sm">{stats.suggested_method}</Badge>
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

            {/* Versions card */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">Versions</CardTitle>
                  <Button size="sm" className="h-7 text-xs" onClick={handleCreateVersion} disabled={snapshotting}>
                    <IconClock className="h-3 w-3 mr-1" />
                    {snapshotting ? 'Snapshotting…' : 'Create snapshot'}
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {versionsLoading ? (
                  <Skeleton className="h-16 rounded-lg" />
                ) : versions.length === 0 ? (
                  <div className="text-center py-6 text-sm text-muted-foreground">
                    No snapshots yet. Create one to freeze the current files, then restore them later.
                  </div>
                ) : (
                  <ul className="space-y-2">
                    {versions.map(v => (
                      <li key={v} className="flex items-center justify-between rounded-md border border-border/60 px-3 py-2">
                        <div className="flex items-center gap-2 text-sm">
                          <IconClock className="h-3.5 w-3.5 text-muted-foreground" />
                          <span className="font-mono text-xs">{v}</span>
                          <span className="text-xs text-muted-foreground">
                            {new Date(`${v.slice(0,4)}-${v.slice(4,6)}-${v.slice(6,8)}T${v.slice(8,10)}:${v.slice(10,12)}:${v.slice(12,14)}Z`).toLocaleString()}
                          </span>
                        </div>
                        <Button size="sm" variant="outline" className="h-7 text-xs" onClick={() => setRestoreTarget(v)}>
                          Restore
                        </Button>
                      </li>
                    ))}
                  </ul>
                )}
              </CardContent>
            </Card>
          </>
        )}
      </div>

      <AlertDialog open={restoreTarget !== null} onOpenChange={(open: boolean) => { if (!open) setRestoreTarget(null) }}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Restore version</AlertDialogTitle>
            <AlertDialogDescription>
              Restore dataset &ldquo;{dataset?.name}&rdquo; to version {restoreTarget ? `"${restoreTarget}"` : ''}? This overwrites the current files.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleRestoreVersion}>Restore</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={showDelete} onOpenChange={setShowDelete}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete dataset</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete &ldquo;{dataset?.name}&rdquo;? This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDelete} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
