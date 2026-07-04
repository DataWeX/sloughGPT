'use client'

import { useCallback, useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Skeleton } from '@/components/ui/display'
import { IconRefresh, IconPlus, IconTrash } from '@/components/ui'
import { useToastStore } from '@/lib/toast-store'
import { datasetController, type Dataset } from '@/lib/dataset-controller'

function formatBytes(bytes: number): string {
  if (!bytes) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatDate(dateStr: string): string {
  try {
    const d = new Date(dateStr)
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
  } catch {
    return dateStr
  }
}

export default function DatasetsPage() {
  const router = useRouter()
  const addToast = useToastStore(s => s.addToast)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')

  const fetchDatasets = useCallback(async () => {
    setLoading(true)
    try {
      const list = await datasetController.list()
      setDatasets(list)
    } catch {
      addToast('Failed to load datasets', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => {
    fetchDatasets()
  }, [fetchDatasets])

  const handleDelete = async (ds: Dataset) => {
    if (!confirm(`Delete dataset "${ds.name}"?`)) return
    try {
      await datasetController.delete(ds.id)
      setDatasets(prev => prev.filter(d => d.id !== ds.id))
      addToast(`Deleted "${ds.name}"`, 'info')
    } catch {
      addToast('Delete failed', 'error')
    }
  }

  const filtered = datasets.filter(ds =>
    !search || ds.name.toLowerCase().includes(search.toLowerCase()) || ds.id.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Datasets" />}
        right={
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline" className="h-7 text-xs" onClick={fetchDatasets} disabled={loading}>
              <IconRefresh className={loading ? 'animate-spin h-3 w-3 mr-1' : 'h-3 w-3 mr-1'} />
              Refresh
            </Button>
            <Button size="sm" className="h-7 text-xs" onClick={() => router.push('/training')}>
              <IconPlus className="h-3 w-3 mr-1" />
              Import
            </Button>
          </div>
        }
      />

      <div className="space-y-4">
        {datasets.length > 0 && (
          <Input
            placeholder="Search datasets..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="h-8 text-sm max-w-xs"
          />
        )}

        {loading ? (
          <div className="grid gap-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-20 rounded-lg" />
            ))}
          </div>
        ) : filtered.length === 0 ? (
          <Card>
            <CardContent className="py-10 text-center">
              <p className="text-sm text-muted-foreground mb-1">{datasets.length === 0 ? 'No datasets yet.' : 'No datasets match your search.'}</p>
              <p className="text-xs text-muted-foreground mb-3">Import a dataset to get started with training.</p>
              <Button size="sm" onClick={() => router.push('/training')}>Import Dataset</Button>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-2">
            {filtered.map(ds => (
              <Card key={ds.id} className="cursor-pointer hover:bg-accent/40 transition-colors" onClick={() => router.push(`/dataset/${encodeURIComponent(ds.id)}`)}>
                <CardContent className="flex items-center justify-between py-3 px-4">
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium truncate">{ds.name}</p>
                    <div className="flex items-center gap-3 text-xs text-muted-foreground mt-0.5">
                      <span>{ds.source || 'local'}</span>
                      <span>{formatBytes(ds.size)}</span>
                      {ds.samples != null && <span>{ds.samples.toLocaleString()} samples</span>}
                      {ds.created_at && <span>{formatDate(ds.created_at)}</span>}
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 w-7 ml-2 shrink-0 text-muted-foreground hover:text-destructive"
                    onClick={e => { e.stopPropagation(); handleDelete(ds) }}
                    aria-label={`Delete ${ds.name}`}
                  >
                    <IconTrash className="h-3.5 w-3.5" />
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
