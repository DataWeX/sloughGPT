'use client'
export const dynamic = 'force-dynamic'

import { useCallback, useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from '@sloughgpt/strui'
import { Card, CardContent, CardHeader, CardTitle, EmptyCard } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import { IconRefresh, IconPlus, IconTrash } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { datasetController, type Dataset } from '@/lib/dataset-controller'
import { formatBytes } from '@/lib/format-bytes'
import { formatDate } from '@/lib/conversations-utils'

export default function DatasetsPage() {
  const router = useRouter()
  const addToast = useToastStore(s => s.addToast)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [pendingDelete, setPendingDelete] = useState<Dataset | null>(null)

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

  const handleDelete = async () => {
    if (!pendingDelete) return
    try {
      await datasetController.delete(pendingDelete.id)
      setDatasets(prev => prev.filter(d => d.id !== pendingDelete.id))
      addToast(`Deleted "${pendingDelete.name}"`, 'info')
    } catch {
      addToast('Delete failed', 'error')
    } finally {
      setPendingDelete(null)
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
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearch(e.target.value)}
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
          <EmptyCard
            message={datasets.length === 0 ? 'No datasets yet.' : 'No datasets match your search.'}
            action={<Button size="sm" onClick={() => router.push('/training')}>Import Dataset</Button>}
          />
        ) : (
          <div className="grid gap-2">
            {filtered.map(ds => (
              <Card key={ds.id} className="cursor-pointer hover:bg-accent/40 transition-colors" onClick={() => router.push(`/dataset/${encodeURIComponent(ds.id)}`)}>
                <CardContent className="flex items-center justify-between py-3 px-4">
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium truncate">{ds.name}</p>
                    <div className="flex items-center gap-2 mt-1 flex-wrap">
                      {ds.source && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
                          {ds.source}
                        </span>
                      )}
                      {ds.type && ds.type !== 'text' && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">
                          {ds.type}
                        </span>
                      )}
                      {ds.vlm_metadata && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-accent/15 text-accent font-medium">
                          VLM · {ds.vlm_metadata.image_count} images
                        </span>
                      )}
                      {ds.tags && ds.tags.length > 0 && ds.tags.slice(0, 3).map(tag => (
                        <span key={tag} className="text-[10px] px-1.5 py-0.5 rounded bg-secondary text-secondary-foreground font-medium">
                          {tag}
                        </span>
                      ))}
                    </div>
                    <div className="flex items-center gap-3 text-xs text-muted-foreground mt-1.5">
                      <span>{formatBytes(ds.size)}</span>
                      {ds.samples != null && <span>{ds.samples.toLocaleString()} samples</span>}
                      {ds.created_at && <span>{formatDate(ds.created_at)}</span>}
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 w-7 ml-2 shrink-0 text-muted-foreground hover:text-destructive"
                    onClick={(e: React.MouseEvent<HTMLButtonElement>) => { e.stopPropagation(); setPendingDelete(ds) }}
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

      <AlertDialog open={!!pendingDelete} onOpenChange={(open) => { if (!open) setPendingDelete(null) }}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete dataset</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete &ldquo;{pendingDelete?.name}&rdquo;? This cannot be undone.
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
