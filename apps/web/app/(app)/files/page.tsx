'use client'

import { useState, useEffect, useRef } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { filesController, type FileEntry } from '@/lib/files-controller'
import { FileStatsCard } from '@/components/files/FileStatsCard'
import { useToastStore } from '@/lib/toast-store'

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
  const fileInputRef = useRef<HTMLInputElement>(null)
  const addToast = useToastStore(s => s.addToast)

  const fetchFiles = async () => {
    try {
      setLoadError(null)
      setFiles(await filesController.list())
    } catch {
      setLoadError('Could not load files. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchFiles() }, [])

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
    if (!searchQuery.trim()) { await fetchFiles(); return }
    try {
      setFiles(await filesController.search(searchQuery))
    } catch {
      addToast('Could not search files', 'error')
    }
  }

  const toggleSelect = (id: string) => {
    setSelected(prev => {
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
      setSelected(new Set(filtered.map(f => f.id)))
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

  const filtered = searchQuery.trim()
    ? files.filter(f => f.filename.toLowerCase().includes(searchQuery.toLowerCase()) || f.content_type.toLowerCase().includes(searchQuery.toLowerCase()))
    : files

  if (loading) {
    return (
      <PageContainer
        title="Files"
        subtitle="Manage uploaded files"
        loadingContent={
          <div className="space-y-4">
            <Card><CardContent><div className="h-32 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
          </div>
        }
      ><></>
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
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Files</CardTitle>
            <div className="flex gap-2">
              <Button size="sm" variant="ghost" onClick={fetchFiles} aria-label="Refresh files">
                <IconRefresh className="h-4 w-4" />
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                aria-label="Upload file"
                className="hidden"
                onChange={handleUpload}
                accept=".txt,.md,.json,.jsonl,.csv,.pdf,.py,.js,.ts,.html,.css"
              />
              <Button size="sm" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
                {uploading ? 'Uploading...' : 'Upload'}
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {uploadMsg && (
              <div className="rounded-md bg-primary/10 border border-primary/20 px-3 py-2 text-sm text-primary">{uploadMsg}</div>
            )}
            <div className="flex gap-2">
              <Input
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSearch()}
                placeholder="Search files..."
              />
              <Button size="sm" variant="outline" onClick={handleSearch}>Search</Button>
            </div>
            {filtered.length === 0 ? (
              <p className="text-sm text-muted-foreground">No files uploaded yet. Click Upload to add one.</p>
            ) : (
              <>
                {selected.size > 0 && (
                  <div className="flex items-center gap-2 rounded-md bg-destructive/5 border border-destructive/20 px-3 py-2">
                    <span className="text-sm text-destructive font-medium">{selected.size} selected</span>
                    <Button size="sm" variant="ghost" className="text-destructive h-8 text-xs ml-auto" onClick={handleBatchDelete} disabled={batchDeleting}>
                      {batchDeleting ? 'Deleting...' : 'Delete Selected'}
                    </Button>
                    <Button size="sm" variant="ghost" className="h-8 text-xs" onClick={() => setSelected(new Set())}>
                      Clear
                    </Button>
                  </div>
                )}
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  <label className="flex items-center gap-2 px-3 py-1 text-xs text-muted-foreground cursor-pointer hover:bg-muted/30 rounded">
                    <input
                      type="checkbox"
                      checked={selected.size === filtered.length && filtered.length > 0}
                      onChange={toggleSelectAll}
                      aria-label="Select all files"
                      className="h-4 w-4 rounded border-border"
                    />
                    Select all ({filtered.length})
                  </label>
                  {filtered.map(f => (
                    <div key={f.id} className={`flex items-center justify-between rounded-md border px-3 py-2 text-sm group hover:bg-muted/50 transition-colors ${selected.has(f.id) ? 'border-primary/40 bg-primary/5' : 'border-border/60'}`}>
                      <div className="flex items-center gap-3 flex-1 min-w-0">
                        <input
                          type="checkbox"
                          checked={selected.has(f.id)}
                          onChange={() => toggleSelect(f.id)}
                          aria-label={`Select file ${f.filename}`}
                          className="h-4 w-4 rounded border-border shrink-0"
                        />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="font-medium truncate">{f.filename}</span>
                            {f.ingested && <span className="text-xs bg-success/10 text-success px-1 rounded">indexed</span>}
                          </div>
                          <div className="text-xs text-muted-foreground mt-0.5">
                            {formatSize(f.size)} · {f.content_type ?? 'unknown'} · {f.uploaded_at ? new Date(f.uploaded_at).toLocaleDateString() : '—'}
                            {f.chunk_count != null && ` · ${f.chunk_count} chunks`}
                          </div>
                        </div>
                      </div>
                      <div className="flex gap-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                        {!f.ingested && (
                          <Button size="sm" variant="ghost" onClick={() => handleIngest(f.id)} disabled={ingesting === f.id}>
                            {ingesting === f.id ? 'Indexing...' : 'Index'}
                          </Button>
                        )}
                        <Button size="sm" variant="ghost" className="text-destructive" onClick={() => handleDelete(f.id)}>
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
    </PageContainer>
  )
}
