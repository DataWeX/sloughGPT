'use client'
export const dynamic = 'force-dynamic'

import { useCallback, useEffect, useRef, useState } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import { IconSearch, IconPlus, IconTrash, IconDownload, IconUpload, IconX, IconCheck } from '@sloughgpt/strui'
import { filesController, type FileItem } from '@/lib/controllers'
import { useToastStore } from '@/lib/toast-store'

export default function FilesPage() {
  const addToast = useToastStore(s => s.addToast)
  const [files, setFiles] = useState<FileItem[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [uploading, setUploading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [ingesting, setIngesting] = useState<Set<string>>(new Set())
  const fileInputRef = useRef<HTMLInputElement>(null)
  const dragCounterRef = useRef(0)

  const fetchFiles = useCallback(async () => {
    try {
      const res = searchQuery
        ? await filesController.search(searchQuery)
        : await filesController.list()
      setFiles(res.files)
      setTotal(res.total)
    } catch {
      addToast('Failed to load files', 'error')
    } finally {
      setLoading(false)
    }
  }, [searchQuery, addToast])

  useEffect(() => { fetchFiles() }, [fetchFiles])

  const doUpload = async (file: File) => {
    setUploading(true)
    try {
      const res = await filesController.upload(file)
      addToast(`Uploaded ${res.filename} (${filesController.formatSize(res.size_bytes)})`, 'success')
      await fetchFiles()
    } catch {
      addToast('Upload failed', 'error')
    } finally {
      setUploading(false)
    }
  }

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    await doUpload(file)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const handleDragEnter = (e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); dragCounterRef.current++; setDragOver(true) }
  const handleDragLeave = (e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); dragCounterRef.current--; if (dragCounterRef.current <= 0) { dragCounterRef.current = 0; setDragOver(false) } }
  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); e.stopPropagation() }
  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation()
    setDragOver(false); dragCounterRef.current = 0
    const file = e.dataTransfer.files?.[0]
    if (!file) return
    await doUpload(file)
  }

  const handleDelete = async (id: string) => {
    try {
      await filesController.delete(id)
      addToast('File deleted', 'success')
      setSelected(prev => { const n = new Set(prev); n.delete(id); return n })
      await fetchFiles()
    } catch {
      addToast('Failed to delete file', 'error')
    }
  }

  const handleBatchDelete = async () => {
    for (const id of selected) await handleDelete(id)
    setSelected(new Set())
  }

  const handleIngest = async (id: string) => {
    setIngesting(prev => new Set(prev).add(id))
    try {
      const res = await filesController.ingest(id)
      addToast(`Ingested ${res.filename}: ${res.facts_stored} facts stored`, 'success')
    } catch {
      addToast('Failed to ingest file', 'error')
    } finally {
      setIngesting(prev => { const n = new Set(prev); n.delete(id); return n })
    }
  }

  const toggleSelect = (id: string) => {
    setSelected(prev => {
      const n = new Set(prev)
      if (n.has(id)) n.delete(id)
      else n.add(id)
      return n
    })
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="Files" />} />

      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Upload & Search</CardTitle>
          </CardHeader>
          <CardContent
            className="space-y-3 relative"
            data-testid="drop-zone"
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            {dragOver && (
              <div className="absolute inset-0 z-10 flex items-center justify-center rounded-md border-2 border-dashed border-primary bg-primary/5 backdrop-blur-[1px]">
                <div className="text-center">
                  <IconUpload className="h-8 w-8 mx-auto mb-2 text-primary" />
                  <p className="text-sm font-medium text-primary">Drop file to upload</p>
                  <p className="text-xs text-muted-foreground mt-1">PDF, DOCX, TXT, code, and more</p>
                </div>
              </div>
            )}
            <div className="flex items-center gap-3">
              <div className="relative flex-1">
                <IconSearch className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  className="pl-9 text-sm"
                  placeholder="Search files..."
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                />
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,.txt,.md,.csv,.json,.py,.js,.ts,.html,.css,.yaml,.yml,.xml,.log"
                className="hidden"
                onChange={handleUpload}
              />
              <Button size="sm" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
                <IconUpload className="h-4 w-4 mr-1" />
                {uploading ? 'Uploading...' : 'Upload'}
              </Button>
            </div>
            {selected.size > 0 && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <span>{selected.size} selected</span>
                <Button size="sm" variant="outline" onClick={() => setSelected(new Set())}>
                  <IconX className="h-3 w-3 mr-1" /> Clear
                </Button>
                <Button size="sm" variant="outline" onClick={handleBatchDelete}>
                  <IconTrash className="h-3 w-3 mr-1" /> Delete all
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">
              {searchQuery ? `Search: "${searchQuery}"` : `All Files`}
              <span className="ml-2 text-xs text-muted-foreground font-normal">({total})</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="space-y-2">
                {[1,2,3].map(i => (
                  <div key={i} className="h-12 animate-pulse bg-muted rounded" />
                ))}
              </div>
            ) : files.length === 0 ? (
              <div className="py-8 text-center text-sm text-muted-foreground space-y-3">
                {searchQuery
                  ? 'No files match your search.'
                  : <>
                      <p>No files uploaded yet.</p>
                      <Button size="sm" variant="outline" onClick={() => fileInputRef.current?.click()}>
                        <IconUpload className="h-4 w-4 mr-1" />
                        Upload a file
                      </Button>
                    </>
                }
              </div>
            ) : (
              <div className="divide-y divide-border/50">
                {files.map(file => (
                  <div
                    key={file.id}
                    className={`flex items-center gap-3 py-2.5 px-1 rounded transition-colors ${
                      selected.has(file.id) ? 'bg-primary/5' : ''
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={selected.has(file.id)}
                      onChange={() => toggleSelect(file.id)}
                      className="shrink-0"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium truncate">{file.filename}</span>
                        <Badge label={file.extension || '?'} variant="outline" />
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {filesController.formatSize(file.size_bytes)} &middot; {filesController.formatDate(file.uploaded_at)}
                      </div>
                    </div>
                    <div className="flex items-center gap-1 shrink-0">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => handleIngest(file.id)}
                        disabled={ingesting.has(file.id)}
                        title="Ingest into knowledge base"
                      >
                        {ingesting.has(file.id) ? (
                          <span className="h-3 w-3 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                        ) : (
                          <IconDownload className="h-3.5 w-3.5" />
                        )}
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => handleDelete(file.id)}
                        title="Delete file"
                      >
                        <IconTrash className="h-3.5 w-3.5 text-destructive" />
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
