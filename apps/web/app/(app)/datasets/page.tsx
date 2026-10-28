'use client'
export const dynamic = 'force-dynamic'

import { useCallback, useEffect, useState } from 'react'
import { useRouter } from '@/vite/next-compat/navigation'
import { PageContainer } from '@/components/PageContainer'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@sloughgpt/strui'
import { Card, CardContent, EmptyCard, cn, Spinner } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import { DatasetListSkeleton } from '@/components/ui/PageSkeletons'
import { IconRefresh, IconPlus, IconTrash, IconDownload, IconPlay } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import {
  datasetController,
  type Dataset,
  type DatasetPreview as PreviewData,
} from '@/lib/dataset-controller'
import { formatBytes } from '@/lib/format-bytes'
import { formatDate } from '@/lib/conversations-utils'
import { DatasetImportDialog } from '@/components/DatasetImportDialog'
import { DatasetDropZone } from '@/components/DatasetDropZone'
import { useDatasetList, useDeleteDataset } from '@/lib/cache'

export default function DatasetsPage() {
  const router = useRouter()
  const addToast = useToastStore((s) => s.addToast)
  const { data: allDatasets = [], isLoading: loading, refetch: refetchDatasets } = useDatasetList()
  const deleteDataset = useDeleteDataset()
  const [search, setSearch] = useState('')
  const [pendingDelete, setPendingDelete] = useState<Dataset | null>(null)
  const [importOpen, setImportOpen] = useState(false)
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [previewData, setPreviewData] = useState<PreviewData | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [fetchError, setFetchError] = useState(false)

  const fetchDatasets = useCallback(async () => {
    setFetchError(false)
    try {
      await refetchDatasets()
    } catch {
      setFetchError(true)
      addToast('Could not load files', 'error')
    }
  }, [refetchDatasets, addToast])

  const filtered = allDatasets.filter(
    (ds) =>
      !search ||
      ds.name.toLowerCase().includes(search.toLowerCase()) ||
      ds.id.toLowerCase().includes(search.toLowerCase()),
  )

  const handleDelete = async () => {
    if (!pendingDelete) return
    const deleted = pendingDelete
    setPendingDelete(null)
    try {
      await deleteDataset.mutateAsync(deleted.id)
      addToast(`Deleted "${deleted.name}"`, 'info')
    } catch {
      addToast('Could not delete', 'error')
    }
  }

  const handlePreview = async (ds: Dataset) => {
    if (expandedId === ds.id) {
      setExpandedId(null)
      setPreviewData(null)
      return
    }
    setExpandedId(ds.id)
    setPreviewLoading(true)
    try {
      const preview = await datasetController.preview(ds.id, 3)
      setPreviewData(preview)
    } catch {
      addToast('Could not load preview', 'error')
    } finally {
      setPreviewLoading(false)
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

  return (
    <PageContainer
      title="My Files"
      subtitle="Drop a file to chat with it — PDF, Word, text"
      headerRight={
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            className="h-7 text-xs"
            onClick={fetchDatasets}
            disabled={loading}
          >
            <Spinner className="h-3 w-3 mr-1" />
            Refresh
          </Button>
          <Button size="sm" className="h-7 text-xs" onClick={() => setImportOpen(true)}>
            <IconPlus className="h-3.5 w-3.5 mr-1" />
            Add file
          </Button>
        </div>
      }
      toolbar={
        filtered.length > 0 ? (
          <Input
            placeholder="Search files..."
            value={search}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearch(e.target.value)}
            className="h-7 text-xs max-w-xs"
          />
        ) : undefined
      }
    >
      <DatasetDropZone onUploadComplete={fetchDatasets} className="mb-4" />

      {loading ? (
        <DatasetListSkeleton />
      ) : fetchError && filtered.length === 0 ? (
        <Card>
          <CardContent className="text-center py-6">
            <p className="text-sm text-destructive mb-3">Could not load files</p>
            <Button size="sm" variant="outline" onClick={fetchDatasets}>
              <IconRefresh className="h-3.5 w-3.5 mr-1" />
              Retry
            </Button>
          </CardContent>
        </Card>
      ) : filtered.length === 0 ? (
        <EmptyCard
          message={allDatasets.length === 0 ? 'No files yet' : 'No files match your search'}
          description={
            allDatasets.length === 0
              ? 'Drop a PDF or text file above to start chatting with it.'
              : 'Try a different search term.'
          }
          icon={<IconPlus className="h-5 w-5" />}
          action={
            <Button size="sm" className="h-7 text-xs" onClick={() => setImportOpen(true)}>
              Add file
            </Button>
          }
        />
      ) : (
        <div className="grid gap-2 max-h-[60vh] overflow-y-auto overscroll-contain">
          {filtered.map((ds) => (
            <div key={ds.id}>
              <Card
                className={cn(
                  'group transition-colors hover:bg-accent/40',
                  expandedId === ds.id && 'border-primary/40 bg-primary/[0.06]',
                )}
                onClick={() => router.push(`/dataset/${encodeURIComponent(ds.id)}`)}
              >
                <CardContent className="flex items-center justify-between py-2.5 px-3">
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium truncate">{ds.name}</p>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground mt-1">
                      <span>{formatBytes(ds.size)}</span>
                      {ds.samples != null && <span>{ds.samples.toLocaleString()} samples</span>}
                      {ds.created_at && <span>{formatDate(ds.created_at)}</span>}
                    </div>
                    {ds.tags && ds.tags.length > 0 && (
                      <div className="flex gap-1 mt-1 flex-wrap">
                        {ds.tags.slice(0, 3).map((t) => (
                          <span
                            key={t}
                            className="text-[10px] px-1.5 py-0.5 rounded bg-secondary text-secondary-foreground"
                          >
                            {t}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  <div
                    className="flex items-center gap-1 ml-2 shrink-0"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6"
                      onClick={() => router.push(`/training?dataset=${encodeURIComponent(ds.id)}`)}
                      aria-label={`Train with ${ds.name}`}
                    >
                      <IconPlay className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6"
                      onClick={(e) => handleExport(ds, e)}
                      aria-label={`Export ${ds.name}`}
                    >
                      <IconDownload className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 text-muted-foreground hover:text-primary"
                      onClick={() => handlePreview(ds)}
                      aria-label={expandedId === ds.id ? `Hide preview` : `Preview ${ds.name}`}
                    >
                      {expandedId === ds.id ? '−' : '+'}
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 text-muted-foreground hover:text-destructive"
                      onClick={() => setPendingDelete(ds)}
                      aria-label={`Delete ${ds.name}`}
                    >
                      <IconTrash className="h-4 w-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
              {expandedId === ds.id && (
                <div className="mt-1 rounded-lg border border-border/40 bg-muted/20 px-3 py-3 text-sm">
                  {previewLoading ? (
                    <Skeleton className="h-12 bg-muted/20 rounded" />
                  ) : previewData && previewData.samples.length > 0 ? (
                    <div className="space-y-2">
                      <div className="flex gap-3 text-xs text-muted-foreground">
                        <span>{previewData.total_samples.toLocaleString()} samples</span>
                        <span>{previewData.total_chars.toLocaleString()} chars</span>
                      </div>
                      {previewData.samples.slice(0, 3).map((s, i) => (
                        <pre
                          key={i}
                          className="text-xs bg-card rounded p-2 overflow-x-auto max-h-20 overflow-y-auto font-mono whitespace-pre-wrap"
                        >
                          {s.content.slice(0, 300)}
                          {s.content.length > 300 ? '…' : ''}
                        </pre>
                      ))}
                      <div className="flex gap-2 pt-1">
                        <Button
                          size="sm"
                          className="h-6 text-xs"
                          onClick={() =>
                            router.push(`/knowledge?file=${encodeURIComponent(ds.id)}`)
                          }
                        >
                          Chat about this file
                        </Button>
                        <span className="text-xs text-muted-foreground self-center">
                          Try: Summarize · What are key points? · Explain in simple terms
                        </span>
                      </div>
                    </div>
                  ) : (
                    <p className="text-xs text-muted-foreground">
                      No preview — try opening the file.
                    </p>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <AlertDialog
        open={!!pendingDelete}
        onOpenChange={(open) => {
          if (!open) setPendingDelete(null)
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete file</AlertDialogTitle>
            <AlertDialogDescription>
              Delete “{pendingDelete?.name}”? Cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      <DatasetImportDialog
        open={importOpen}
        onOpenChange={setImportOpen}
        onImportComplete={fetchDatasets}
      />
    </PageContainer>
  )
}
