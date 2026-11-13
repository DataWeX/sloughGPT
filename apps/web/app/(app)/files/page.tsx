'use client'

import { notify } from '@/lib/notify'
import { useState, useEffect, useRef, useCallback } from 'react'
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  Button,
  Checkbox,
  Input,
  Progress,
  cn,
} from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import {
  filesController,
  type FileEntry,
  type FileDetail,
  type ImportStatus,
} from '@/lib/files-controller'
import { FileStatsCard } from '@/components/files/FileStatsCard'
import { ImagesPanel } from '@/components/images/ImagesPanel'
import { useRefreshShortcut } from '@/hooks/useRefreshShortcut'

export default function FilesPage() {
  const [files, setFiles] = useState<FileEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploadMsg, setUploadMsg] = useState<string | null>(null)
  const [ingesting, setIngesting] = useState<string | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [batchDeleting, setBatchDeleting] = useState(false)
  const [previewFile, setPreviewFile] = useState<FileDetail | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [importOpen, setImportOpen] = useState(false)
  const [importUrl, setImportUrl] = useState('')
  const [importing, setImporting] = useState(false)
  const [importJob, setImportJob] = useState<ImportStatus | null>(null)
  const [importTick, setImportTick] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const addToast = notify.toast

  const fetchFiles = useCallback(async () => {
    try {
      setLoadError(null)
      setFiles(await filesController.list())
    } catch {
      setLoadError('Could not load files. Please try again.')
    } finally {
      setLoading(false)
    }
  }, [])

  useRefreshShortcut(fetchFiles)

  useEffect(() => {
    fetchFiles()
  }, [fetchFiles])

  // Poll the import job every 500 ms while it runs; the percentage it carries
  // is decoded/decoded (disk byte-space) — wire savings are a separate receipt.
  useEffect(() => {
    if (!importJob || importJob.state !== 'running') return
    const t = setTimeout(() => {
      filesController
        .importStatus(importJob.job_id)
        .then((status) => {
          setImportJob(status)
          if (status.state === 'done') {
            addToast(`Imported ${status.filename}`, 'success')
            void fetchFiles()
          } else if (status.state === 'error') {
            addToast(status.error ?? 'Import failed', 'error')
          }
        })
        .catch(() => {
          /* transient poll failure — keep polling on the next tick */
        })
        .finally(() => setImportTick((n) => n + 1))
    }, 500)
    return () => clearTimeout(t)
  }, [importJob, importTick, addToast, fetchFiles])

  const handleStartImport = async () => {
    const url = importUrl.trim()
    if (!url) return
    setImporting(true)
    try {
      const { job_id } = await filesController.startImport(url)
      setImportJob({
        job_id,
        state: 'running',
        filename: url.split('/').pop() || url,
        done: 0,
        total: 0,
        pct: 0,
      })
      setImportTick((n) => n + 1)
    } catch (err) {
      addToast(err instanceof Error ? err.message : 'Could not start import', 'error')
    } finally {
      setImporting(false)
    }
  }

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    setUploadMsg(null)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const data = await filesController.upload(formData)
      setUploadMsg(`Uploaded ${data.filename ?? file.name}`)
      await fetchFiles()
    } catch (err) {
      addToast(err instanceof Error ? err.message : 'Could not upload', 'error')
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleDelete = async (id: string) => {
    try {
      await filesController.delete(id)
      await fetchFiles()
    } catch {
      addToast('Could not delete file', 'error')
    }
  }

  const handleIngest = async (id: string) => {
    setIngesting(id)
    try {
      await filesController.ingest(id)
      await fetchFiles()
    } catch {
      addToast('Could not index file', 'error')
    } finally {
      setIngesting(null)
    }
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      await fetchFiles()
      return
    }
    try {
      setFiles(await filesController.search(searchQuery))
    } catch {
      addToast('Could not search files', 'error')
    }
  }

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const toggleSelectAll = () => {
    if (selected.size === filtered.length) {
      setSelected(new Set())
    } else {
      setSelected(new Set(filtered.map((f) => f.id)))
    }
  }

  const handleBatchDelete = async () => {
    if (selected.size === 0) return
    setBatchDeleting(true)
    try {
      await filesController.deleteBatch(Array.from(selected))
      setSelected(new Set())
      await fetchFiles()
      addToast(`Deleted ${selected.size} files`, 'success')
    } catch {
      addToast('Could not batch delete', 'error')
    } finally {
      setBatchDeleting(false)
    }
  }

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const handlePreview = async (file: FileEntry) => {
    if (previewFile?.id === file.id) {
      setPreviewFile(null)
      return
    }
    setPreviewLoading(true)
    try {
      const detail = await filesController.getDetail(file.id)
      setPreviewFile(detail)
    } catch {
      addToast('Could not load file preview', 'error')
    } finally {
      setPreviewLoading(false)
    }
  }

  const filtered = searchQuery.trim()
    ? files.filter(
        (f) =>
          f.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
          f.content_type.toLowerCase().includes(searchQuery.toLowerCase()),
      )
    : files

  if (loading) {
    return (
      <PageContainer
        title="Files"
        subtitle="Manage uploaded files"
        loadingContent={
          <div className="space-y-4">
            <Card>
              <CardContent>
                <div className="h-32 animate-pulse bg-muted/50 rounded" />
              </CardContent>
            </Card>
          </div>
        }
      >
        <></>
      </PageContainer>
    )
  }

  return (
    <PageContainer
      title="Files"
      subtitle={`${files.length} files`}
      error={loadError}
      onRetry={fetchFiles}
    >
      <FileStatsCard files={files} />

      <Card>
        <CardHeader className="flex flex-row items-center justify-between pb-2 pt-2.5 px-2.5">
          <CardTitle className="text-[11px] font-medium">Files</CardTitle>
          <div className="flex gap-1">
            <Button
              size="sm"
              variant="ghost"
              className="h-6 text-[10px]"
              onClick={fetchFiles}
              aria-label="Refresh files"
            >
              <IconRefresh className="h-3 w-3" />
            </Button>
            <Button
              size="sm"
              variant="ghost"
              className="h-6 text-[10px]"
              onClick={() => setImportOpen((o) => !o)}
              aria-label="Toggle import from URL"
              aria-expanded={importOpen}
            >
              Import URL
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              aria-label="Upload file"
              className="hidden"
              onChange={handleUpload}
              accept=".txt,.md,.json,.jsonl,.csv,.pdf,.py,.js,.ts,.html,.css"
            />
            <Button
              size="sm"
              className="h-6 text-[10px]"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
            >
              {uploading ? 'Uploading...' : 'Upload'}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-2 px-2.5 pb-2.5">
          {uploadMsg && (
            <div className="rounded-lg bg-primary/5 border border-primary/20 px-2.5 py-1.5 text-[11px] text-primary">
              {uploadMsg}
            </div>
          )}
          {importOpen && (
            <div className="flex gap-1.5" data-testid="import-row">
              <Input
                value={importUrl}
                onChange={(e) => setImportUrl(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && void handleStartImport()}
                placeholder="https://example.com/file.txt"
                aria-label="Import URL"
                className="h-7 text-[11px] flex-1"
              />
              <Button
                size="sm"
                className="h-7 text-[11px]"
                onClick={() => void handleStartImport()}
                disabled={importing || !importUrl.trim() || importJob?.state === 'running'}
              >
                {importing ? 'Starting...' : 'Start import'}
              </Button>
            </div>
          )}
          {importJob && (
            <ImportJobCard
              job={importJob}
              formatSize={formatSize}
              onDismiss={() => setImportJob(null)}
            />
          )}
          <div className="flex gap-1.5">
            <Input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Search files..."
              className="h-7 text-[11px] flex-1"
            />
            <Button size="sm" variant="outline" className="h-7 text-[11px]" onClick={handleSearch}>
              Search
            </Button>
          </div>
          {filtered.length === 0 ? (
            <p className="text-[10px] text-muted-foreground/60">
              No files uploaded yet. Click Upload to add one.
            </p>
          ) : (
            <>
              {selected.size > 0 && (
                <div className="flex items-center gap-1.5 rounded-lg bg-destructive/5 border border-destructive/20 px-2.5 py-1.5">
                  <span className="text-[11px] text-destructive font-medium">
                    {selected.size} selected
                  </span>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="text-destructive h-6 text-[10px] ml-auto"
                    onClick={handleBatchDelete}
                    disabled={batchDeleting}
                  >
                    {batchDeleting ? 'Deleting...' : 'Delete Selected'}
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 text-[10px]"
                    onClick={() => setSelected(new Set())}
                  >
                    Clear
                  </Button>
                </div>
              )}
              <div className="space-y-0.5 max-h-96 overflow-y-auto">
                <label className="flex items-center gap-1.5 px-2.5 py-1 text-[10px] text-muted-foreground/60 cursor-pointer hover:bg-muted/20 rounded-lg">
                  <Checkbox
                    checked={selected.size === filtered.length && filtered.length > 0}
                    onCheckedChange={toggleSelectAll}
                    aria-label="Select all files"
                    className="h-3.5 w-3.5 rounded border-border"
                  />
                  Select all ({filtered.length})
                </label>
                {filtered.map((f) => (
                  <div
                    key={f.id}
                    className={cn(
                      'flex items-center justify-between rounded-lg border px-2.5 py-2 text-[11px] group hover:bg-muted/20 transition-colors',
                      selected.has(f.id) ? 'border-primary/40 bg-primary/5' : 'border-border/40',
                    )}
                  >
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <Checkbox
                        checked={selected.has(f.id)}
                        onCheckedChange={() => toggleSelect(f.id)}
                        aria-label={`Select file ${f.filename}`}
                        className="h-3.5 w-3.5 rounded border-border shrink-0"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5">
                          <span className="font-medium truncate">{f.filename}</span>
                          {f.ingested && (
                            <span className="text-[9px] bg-success/10 text-success px-1.5 py-0.5 rounded-full">
                              indexed
                            </span>
                          )}
                        </div>
                        <div className="text-[10px] text-muted-foreground/60 mt-0.5 font-mono tabular-nums">
                          {formatSize(f.size)} · {f.content_type ?? 'unknown'} ·{' '}
                          {f.uploaded_at ? new Date(f.uploaded_at).toLocaleDateString() : '—'}
                          {f.chunk_count != null && ` · ${f.chunk_count} chunks`}
                        </div>
                      </div>
                    </div>
                    <div className="flex gap-0.5 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-6 text-[10px]"
                        onClick={() => void handlePreview(f)}
                        disabled={previewLoading && previewFile?.id !== f.id}
                      >
                        {previewFile?.id === f.id ? 'Close' : 'Preview'}
                      </Button>
                      {!f.ingested && (
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-6 text-[10px]"
                          onClick={() => handleIngest(f.id)}
                          disabled={ingesting === f.id}
                        >
                          {ingesting === f.id ? 'Indexing...' : 'Index'}
                        </Button>
                      )}
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-6 text-[10px] text-destructive"
                        onClick={() => handleDelete(f.id)}
                      >
                        Delete
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {previewFile && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-[11px] font-medium truncate">
              {previewFile.filename}
            </CardTitle>
            <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground/60 shrink-0">
              {previewFile.chars != null && (
                <span className="font-mono tabular-nums">
                  {previewFile.chars.toLocaleString()} chars
                </span>
              )}
              {previewFile.pages != null && (
                <span className="font-mono tabular-nums">{previewFile.pages} pages</span>
              )}
              <Button
                size="sm"
                variant="ghost"
                className="h-6 text-[10px]"
                onClick={() => setPreviewFile(null)}
              >
                Close
              </Button>
            </div>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            {previewLoading ? (
              <div className="h-40 animate-pulse bg-muted/20 rounded-lg" />
            ) : previewFile.text ? (
              <pre className="max-h-96 overflow-y-auto whitespace-pre-wrap break-all rounded-lg bg-muted/20 p-2.5 font-mono text-[11px] leading-relaxed">
                {previewFile.text}
              </pre>
            ) : (
              <p className="text-[10px] text-muted-foreground/60 text-center py-3">
                No text content extracted
              </p>
            )}
          </CardContent>
        </Card>
      )}

      {/* Create Images journey — absorbed from the retired /images route
            (route-map row: /files <- /files + /images, card 7839a7c6). */}
      <ImagesPanel />
    </PageContainer>
  )
}

/**
 * Progress receipt for an import job.
 *
 * The bar lives in disk byte-space (decoded vs decoded) — wire bytes are
 * reported only in the final savings line, never mixed into the percentage.
 */
function ImportJobCard({
  job,
  formatSize,
  onDismiss,
}: {
  job: ImportStatus
  formatSize: (bytes: number) => string
  onDismiss: () => void
}) {
  const value = job.state === 'done' ? 100 : job.total > 0 && job.total > job.done ? job.pct : null
  const indeterminate = job.state === 'running' && value === null
  const savedPct =
    job.wire_bytes != null && job.saved_bytes != null
      ? Math.round((100 * job.saved_bytes) / Math.max(1, job.wire_bytes + job.saved_bytes))
      : null

  return (
    <div
      className="rounded-lg border border-border/40 px-2.5 py-2 space-y-1.5"
      data-testid="import-job"
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-medium truncate">
          {job.state === 'running' && `Importing ${job.filename}…`}
          {job.state === 'done' && `Imported ${job.filename}`}
          {job.state === 'error' && `Import failed: ${job.error ?? 'unknown error'}`}
        </span>
        <Button size="sm" variant="ghost" className="h-5 text-[10px] shrink-0" onClick={onDismiss}>
          Dismiss
        </Button>
      </div>
      <Progress
        value={value}
        indeterminate={indeterminate}
        max={100}
        size="xs"
        variant={job.state === 'done' ? 'success' : job.state === 'error' ? 'error' : 'default'}
        showValue
      />
      {job.state === 'done' && job.wire_bytes != null && job.saved_bytes != null && (
        <p className="text-[10px] text-muted-foreground/70" data-testid="import-savings">
          Saved {formatSize(job.saved_bytes)} on the wire ({savedPct}%) — transferred{' '}
          {formatSize(job.wire_bytes)}, file is {formatSize(job.wire_bytes + job.saved_bytes)} ·{' '}
          {job.content_encoding ?? 'identity'}
        </p>
      )}
    </div>
  )
}
