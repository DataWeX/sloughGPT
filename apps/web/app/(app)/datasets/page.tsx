'use client'
export const dynamic = 'force-dynamic'

import { useCallback, useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { PageContainer } from '@/components/PageContainer'
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from '@sloughgpt/strui'
import { Card, CardContent, CardHeader, CardTitle, EmptyCard, cn } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import { IconRefresh, IconPlus, IconTrash, IconChevronDown, IconDownload, IconPlay } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { datasetController, type Dataset, type DatasetPreview as PreviewData } from '@/lib/dataset-controller'
import { formatBytes } from '@/lib/format-bytes'
import { formatDate } from '@/lib/conversations-utils'
import { DatasetImportModal } from '@/components/DatasetImportModal'

export default function DatasetsPage() {
  const router = useRouter()
  const addToast = useToastStore(s => s.addToast)
  const [datasets, setDatasets] = useState<Dataset[]>([])

  const [loading, setLoading] = useState(true)
  const [fetchError, setFetchError] = useState(false)
  const [search, setSearch] = useState('')
  const [pendingDelete, setPendingDelete] = useState<Dataset | null>(null)
  const [importOpen, setImportOpen] = useState(false)
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [previewData, setPreviewData] = useState<PreviewData | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewSearch, setPreviewSearch] = useState('')
  const [compareIds, setCompareIds] = useState<Set<string>>(new Set())
  const [compareData, setCompareData] = useState<Array<{ id: string; name: string; preview: PreviewData | null }>>([])
  const [sortBy, setSortBy] = useState<'date' | 'size' | 'name'>('date')
  const [versionCounts, setVersionCounts] = useState<Record<string, number>>({})
  const [searching, setSearching] = useState(false)

  const fetchDatasets = useCallback(async () => {
    setLoading(true)
    setFetchError(false)
    try {
      const list = await datasetController.list()
      setDatasets(list)
    } catch {
      setFetchError(true)
      addToast('Could not load datasets', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => {
    fetchDatasets()
  }, [fetchDatasets])

  useEffect(() => {
    if (datasets.length === 0) return
    const fetchVersions = async () => {
      const counts: Record<string, number> = {}
      await Promise.all(
        datasets.map(async ds => {
          try {
            const res = await datasetController.listVersions(ds.id)
            counts[ds.id] = res.count || 0
          } catch {
            counts[ds.id] = 0
          }
        })
      )
      setVersionCounts(counts)
    }
    fetchVersions()
  }, [datasets])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return
      if (e.key === 'n' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault()
        setImportOpen(true)
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
        e.preventDefault()
        document.querySelector<HTMLInputElement>('[placeholder="Search datasets..."]')?.focus()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  useEffect(() => {
    if (!search) {
      fetchDatasets()
      return
    }
    setSearching(true)
    const timer = setTimeout(async () => {
      try {
        const results = await datasetController.search(search)
        setDatasets(results)
      } catch {
        // fallback to client-side filter
      } finally {
        setSearching(false)
      }
    }, 300)
    return () => clearTimeout(timer)
  }, [search])

  const handleDelete = async () => {
    if (!pendingDelete) return
    const deleted = pendingDelete
    setDatasets(prev => prev.filter(d => d.id !== deleted.id))
    setPendingDelete(null)
    try {
      await datasetController.delete(deleted.id)
      addToast(`Deleted "${deleted.name}"`, 'info', undefined, () => {
        setDatasets(prev => [deleted, ...prev])
      })
    } catch {
      setDatasets(prev => [deleted, ...prev])
      addToast('Could not delete', 'error')
    }
  }

  const handleExport = async (ds: Dataset, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      const blob = await datasetController.export(ds.id, 'jsonl')
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${ds.name || ds.id}.jsonl`
      a.click()
      URL.revokeObjectURL(url)
      addToast(`Exported "${ds.name}"`, 'success')
    } catch {
      addToast('Could not export', 'error')
    }
  }

  const filtered = datasets
    .filter(ds => !search || ds.name.toLowerCase().includes(search.toLowerCase()) || ds.id.toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => {
      if (sortBy === 'date') return (b.created_at || '').localeCompare(a.created_at || '')
      if (sortBy === 'size') return (b.size || 0) - (a.size || 0)
      return a.name.localeCompare(b.name)
    })

  const handlePreview = async (ds: Dataset, e: React.MouseEvent) => {
    e.stopPropagation()
    if (expandedId === ds.id) {
      setExpandedId(null)
      setPreviewData(null)
      setPreviewSearch('')
      return
    }
    setExpandedId(ds.id)
    setPreviewSearch('')
    setPreviewLoading(true)
    try {
      const preview = await datasetController.preview(ds.id, 5)
      setPreviewData(preview)
    } catch {
      addToast('Could not load preview', 'error')
    } finally {
      setPreviewLoading(false)
    }
  }

  const toggleCompare = (id: string) => {
    setCompareIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const loadCompare = async () => {
    if (compareIds.size < 2) return
    const results = await Promise.all(
      Array.from(compareIds).map(async id => {
        const ds = datasets.find(d => d.id === id)
        try {
          const preview = await datasetController.preview(id, 3)
          return { id, name: ds?.name || id, preview }
        } catch {
          return { id, name: ds?.name || id, preview: null }
        }
      })
    )
    setCompareData(results)
  }

  const headerRight = (
    <div className="flex items-center gap-2">
      <Button size="sm" variant="outline" className="h-8 text-xs" onClick={fetchDatasets} disabled={loading}>
        <IconRefresh className={loading ? 'animate-spin h-3.5 w-3.5 mr-1' : 'h-3.5 w-3.5 mr-1'} />
        Refresh
      </Button>
      <Button size="sm" className="h-8 text-xs" onClick={() => setImportOpen(true)}>
        <IconPlus className="h-3.5 w-3.5 mr-1" />
        Import
      </Button>
    </div>
  )

  const toolbar = datasets.length > 0 ? (
    <div className="flex items-center gap-3">
      <Input
        placeholder="Search datasets..."
        value={search}
        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearch(e.target.value)}
        className="h-9 text-sm max-w-xs"
      />
      {searching && (
        <span className="text-xs text-muted-foreground flex items-center gap-1.5" role="status" aria-live="polite">
          <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent" />
          Searching…
        </span>
      )}
      <div className="flex items-center gap-1 ml-auto">
        {(['date', 'size', 'name'] as const).map(s => (
          <button
            key={s}
            type="button"
            aria-pressed={sortBy === s}
            aria-label={`Sort by ${s}`}
            onClick={() => setSortBy(s)}
            className={cn('text-xs px-2 py-1 rounded border transition-colors', sortBy === s ? 'bg-primary/15 text-primary border-primary/30' : 'border-border/40 text-muted-foreground hover:bg-muted/80')}
          >
            {s}
          </button>
        ))}
      </div>
      {datasets.length > 1 && (
        <div className="flex-1 max-w-xs">
          <div className="text-xs text-muted-foreground mb-1">Size comparison</div>
          <div className="flex items-end gap-1 h-8">
            {datasets.slice(0, 8).map(ds => {
              const maxSize = Math.max(...datasets.map(d => d.size || 1))
              const height = Math.max(((ds.size || 0) / maxSize) * 100, 4)
              return (
                <div
                  key={ds.id}
                  className="flex-1 bg-primary/30 rounded-t transition-all hover:bg-primary/50"
                  style={{ height: `${height}%` }}
                  title={`${ds.name}: ${formatBytes(ds.size)}`}
                />
              )
            })}
          </div>
        </div>
      )}
    </div>
  ) : undefined

  return (
    <PageContainer
      title="Datasets"
      headerRight={headerRight}
      toolbar={toolbar}
    >

        {compareIds.size >= 2 && (
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline" className="h-8 text-xs" onClick={loadCompare}>
              Compare {compareIds.size} datasets
            </Button>
            <Button size="sm" variant="ghost" className="h-8 text-xs" onClick={() => { setCompareIds(new Set()); setCompareData([]) }}>
              Clear
            </Button>
          </div>
        )}

        {compareData.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Dataset comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${compareData.length}, 1fr)` }}>
                {compareData.map(d => (
                  <div key={d.id} className="space-y-2">
                    <p className="text-sm font-medium truncate">{d.name}</p>
                    {d.preview ? (
                      <>
                        <div className="text-[10px] text-muted-foreground">
                          {d.preview.total_samples} samples · {d.preview.total_chars.toLocaleString()} chars
                        </div>
                        <div className="space-y-1">
                          {d.preview.samples.slice(0, 2).map((s, i) => (
                            <pre key={i} className="text-[10px] bg-muted/30 rounded p-1.5 overflow-x-auto max-h-16 font-mono whitespace-pre-wrap">
                              {s.content.slice(0, 150)}{s.content.length > 150 ? '…' : ''}
                            </pre>
                          ))}
                        </div>
                      </>
                    ) : (
                      <p className="text-[10px] text-muted-foreground">No preview</p>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {loading ? (
          <div className="grid gap-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-20 rounded-lg" />
            ))}
          </div>
        ) : fetchError && datasets.length === 0 ? (
          <Card>
            <CardContent className="py-8 text-center">
              <p className="text-sm text-destructive mb-3">Could not load datasets</p>
              <Button size="sm" variant="outline" onClick={fetchDatasets}>
                <IconRefresh className="h-3.5 w-3.5 mr-1.5" />
                Retry
              </Button>
            </CardContent>
          </Card>
        ) : filtered.length === 0 ? (
          <EmptyCard
            message={datasets.length === 0 ? 'No datasets yet' : 'No datasets match your search'}
            description={datasets.length === 0 ? 'Import a dataset to start training models.' : 'Try a different search term.'}
            icon={<IconPlus className="h-5 w-5" />}
            action={<Button size="sm" onClick={() => setImportOpen(true)}>Import Dataset</Button>}
          />
        ) : (
          <div className="grid gap-2 max-h-[60vh] overflow-y-auto overscroll-contain">
            {filtered.map(ds => (
              <div key={ds.id}>
              <Card
                className={cn('group transition-colors', expandedId === ds.id ? 'border-primary/40 bg-primary/[0.08]' : 'hover:bg-accent/40')}
                onClick={() => router.push(`/dataset/${encodeURIComponent(ds.id)}`)}
              >
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
                        {ds.size > 100 * 1024 * 1024 && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-warning/15 text-warning font-medium">
                            Large dataset
                          </span>
                        )}
                        {ds.samples != null && <span>{ds.samples.toLocaleString()} samples</span>}
                        {versionCounts[ds.id] != null && versionCounts[ds.id] > 0 && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">
                            {versionCounts[ds.id]} version{versionCounts[ds.id] !== 1 ? 's' : ''}
                          </span>
                        )}
                        {ds.created_at && <span>{formatDate(ds.created_at)}</span>}
                      </div>
                    </div>
                    <div className="flex items-center gap-1 ml-2 shrink-0" onClick={(e) => e.stopPropagation()}>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 w-8 text-muted-foreground hover:text-primary"
                        onClick={() => router.push(`/training?dataset=${encodeURIComponent(ds.id)}`)}
                        aria-label={`Train with ${ds.name}`}
                      >
                        <IconPlay className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 w-8 text-muted-foreground hover:text-primary"
                        onClick={(e) => handleExport(ds, e)}
                        aria-label={`Export ${ds.name}`}
                      >
                        <IconDownload className="h-4 w-4" />
                      </Button>
                      <button
                        type="button"
                        className={cn('h-6 w-6 rounded border transition-colors flex items-center justify-center', compareIds.has(ds.id) ? 'bg-primary border-primary text-primary-foreground' : 'border-border hover:border-primary/50')}
                        onClick={() => toggleCompare(ds.id)}
                        aria-label={`Select ${ds.name} for comparison`}
                      >
                        {compareIds.has(ds.id) && <span className="text-xs font-bold">✓</span>}
                      </button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 w-8 text-muted-foreground hover:text-primary"
                        onClick={(e) => handlePreview(ds, e)}
                        aria-label={expandedId === ds.id ? `Hide preview for ${ds.name}` : `Preview ${ds.name}`}
                      >
                        <IconChevronDown className={cn('h-4 w-4 transition-transform', expandedId === ds.id && 'rotate-180')} />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 w-8 text-muted-foreground hover:text-destructive"
                        onClick={() => setPendingDelete(ds)}
                        aria-label={`Delete ${ds.name}`}
                      >
                        <IconTrash className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
                {expandedId === ds.id && (
                  <div className="mt-1 rounded-lg border border-border/40 bg-muted/20 px-4 py-3 text-sm">
                    {previewLoading ? (
                      <Skeleton className="h-16 rounded" />
                    ) : previewData && previewData.samples.length > 0 ? (
                      <div className="space-y-2">
                        <div className="flex items-center gap-3 text-xs text-muted-foreground">
                          <span>{previewData.total_samples.toLocaleString()} samples</span>
                          <span>{previewData.total_chars.toLocaleString()} chars</span>
                        </div>
                        {previewData.samples.length > 3 && (
                          <input
                            type="text"
                            value={previewSearch}
                            onChange={e => setPreviewSearch(e.target.value)}
                            placeholder="Filter samples..."
                            aria-label="Filter samples"
                            className="h-7 w-full max-w-[200px] rounded-md border border-border/60 bg-background px-2 text-[10px] placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-primary/30"
                          />
                        )}
                        {previewData.samples
                          .filter(s => !previewSearch || s.content.toLowerCase().includes(previewSearch.toLowerCase()))
                          .slice(0, 10)
                          .map((sample, i) => (
                          <pre key={i} className="text-xs bg-card rounded p-2 overflow-x-auto max-h-24 overflow-y-auto font-mono whitespace-pre-wrap">
                            {sample.content.slice(0, 300)}{sample.content.length > 300 ? '…' : ''}
                          </pre>
                        ))}
                      </div>
                    ) : (
                      <p className="text-xs text-muted-foreground">No preview available</p>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
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

      <DatasetImportModal open={importOpen} onOpenChange={setImportOpen} onImportComplete={fetchDatasets} />
    </PageContainer>
  )
}
