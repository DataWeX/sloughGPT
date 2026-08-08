'use client'

import { useState, useEffect, useRef } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { filesController, type FileEntry } from '@/lib/files-controller'
import { useToastStore } from '@/lib/toast-store'

export default function FilesPage() {
  const [files, setFiles] = useState<FileEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploadMsg, setUploadMsg] = useState<string | null>(null)
  const [ingesting, setIngesting] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const addToast = useToastStore(s => s.addToast)

  const fetchFiles = async () => {
    try {
      setFiles(await filesController.list())
    } catch {
      addToast('Failed to load files', 'error')
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
      addToast(err instanceof Error ? err.message : 'Upload failed', 'error')
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
      addToast('Failed to delete file', 'error')
    }
  }

  const handleIngest = async (id: string) => {
    setIngesting(id)
    try {
      await filesController.ingest(id)
      await fetchFiles()
    } catch {
      addToast('Failed to index file', 'error')
    } finally {
      setIngesting(null)
    }
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) { await fetchFiles(); return }
    try {
      setFiles(await filesController.search(searchQuery))
    } catch {
      addToast('Failed to search files', 'error')
    }
  }

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const filtered = searchQuery.trim()
    ? files
    : files

  if (loading) {
    return (
      <div className="sl-page mx-auto max-w-4xl">
        <AppRouteHeader left={<AppRouteHeaderLead title="Files" subtitle="Manage uploaded files" />} />
        <div className="space-y-4">
          <Card><CardContent><div className="h-32 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
        </div>
      </div>
    )
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="Files" subtitle={`${files.length} files`} />} />
      <div className="space-y-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Files</CardTitle>
            <div className="flex gap-2">
              <Button size="sm" variant="ghost" onClick={fetchFiles}>
                <IconRefresh className="h-3.5 w-3.5" />
              </Button>
              <input
                ref={fileInputRef}
                type="file"
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
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {filtered.map(f => (
                  <div key={f.id} className="flex items-center justify-between rounded-md border border-border/60 px-3 py-2 text-sm group hover:bg-muted/50 transition-colors">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-medium truncate">{f.filename}</span>
                        {f.ingested && <span className="text-[10px] bg-success/10 text-success px-1 rounded">indexed</span>}
                      </div>
                      <div className="text-xs text-muted-foreground mt-0.5">
                        {formatSize(f.size)} · {f.content_type ?? 'unknown'} · {f.uploaded_at ? new Date(f.uploaded_at).toLocaleDateString() : '—'}
                        {f.chunk_count != null && ` · ${f.chunk_count} chunks`}
                      </div>
                    </div>
                    <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
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
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
