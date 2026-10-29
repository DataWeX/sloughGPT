'use client'
export const dynamic = 'force-dynamic'

import { useState, useCallback, useEffect } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label, AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, cn } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { apiGet, apiPut, apiDelete, apiPatch } from '@/lib/http-client'

const COLLECTIONS = [
  'sessions', 'pendingMessages', 'knowledge', 'bookmarks',
  'prompts', 'drafts', 'kv', 'errors',
] as const

type CollectionName = typeof COLLECTIONS[number]

interface DocEntry {
  _id: string
  [key: string]: unknown
}

interface CollectionMeta {
  name: string
  count: number
}

export default function DocstorePage() {
  const addToast = useToastStore(s => s.addToast)
  const [selected, setSelected] = useState<CollectionName>('sessions')
  const [docs, setDocs] = useState<DocEntry[]>([])
  const [loading, setLoading] = useState(false)
  const [collectionMeta, setCollectionMeta] = useState<Record<string, number>>({})
  const [selectedDoc, setSelectedDoc] = useState<DocEntry | null>(null)
  const [editMode, setEditMode] = useState(false)
  const [editContent, setEditContent] = useState('')
  const [newDocId, setNewDocId] = useState('')
  const [pendingClear, setPendingClear] = useState(false)
  const [newDocContent, setNewDocContent] = useState('{}')
  const [showCreate, setShowCreate] = useState(false)
  const [page, setPage] = useState(1)
  const [searchQuery, setSearchQuery] = useState('')

  const fetchDocs = useCallback(async (collection: CollectionName) => {
    setLoading(true)
    try {
      const params: Record<string, string> = { sort: '_id', direction: '-1' }
      const data = await apiGet<{ documents: DocEntry[]; total: number }>(`/docstore/${collection}`, params)
      setDocs(data.documents ?? [])
      setCollectionMeta(prev => ({ ...prev, [collection]: data.total ?? data.documents?.length ?? 0 }))
    } catch {
      addToast(`Could not load ${collection}`, 'error')
      setDocs([])
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => {
    void fetchDocs(selected)
    setSelectedDoc(null)
    setEditMode(false)
    setPage(1)
    setSearchQuery('')
  }, [selected, fetchDocs])

  const deleteDoc = useCallback(async (docId: string) => {
    try {
      await apiDelete(`/docstore/${selected}/${docId}`)
      addToast('Document deleted', 'success')
      setDocs(prev => prev.filter(d => d._id !== docId))
      if (selectedDoc?._id === docId) setSelectedDoc(null)
    } catch {
      addToast('Could not delete document', 'error')
    }
  }, [selected, selectedDoc, addToast])

  const createDoc = useCallback(async () => {
    if (!newDocId.trim()) {
      addToast('Document ID is required', 'error')
      return
    }
    try {
      let parsed: Record<string, unknown>
      try {
        parsed = JSON.parse(newDocContent)
      } catch {
        addToast('Invalid JSON', 'error')
        return
      }
      await apiPut(`/docstore/${selected}/${newDocId}`, parsed)
      addToast('Document created', 'success')
      setNewDocId('')
      setNewDocContent('{}')
      setShowCreate(false)
      void fetchDocs(selected)
    } catch {
      addToast('Could not create document', 'error')
    }
  }, [selected, newDocId, newDocContent, addToast, fetchDocs])

  const saveDoc = useCallback(async () => {
    if (!selectedDoc) return
    try {
      let parsed: Record<string, unknown>
      try {
        parsed = JSON.parse(editContent)
      } catch {
        addToast('Invalid JSON', 'error')
        return
      }
      await apiPatch(`/docstore/${selected}/${selectedDoc._id}`, parsed)
      addToast('Document saved', 'success')
      setEditMode(false)
      void fetchDocs(selected)
    } catch {
      addToast('Could not save document', 'error')
    }
  }, [selected, selectedDoc, editContent, addToast, fetchDocs])

  const clearCollection = useCallback(async () => {
    setPendingClear(true)
    try {
      await apiDelete(`/docstore/${selected}`)
      addToast(`Cleared ${selected}`, 'success')
      setDocs([])
      setSelectedDoc(null)
      setCollectionMeta(prev => ({ ...prev, [selected]: 0 }))
    } catch {
      addToast('Could not clear collection', 'error')
    }
  }, [selected, addToast])

  const filtered = searchQuery
    ? docs.filter(d => JSON.stringify(d).toLowerCase().includes(searchQuery.toLowerCase()))
    : docs

  const pageSize = 50
  const totalPages = Math.ceil(filtered.length / pageSize)
  const pageDocs = filtered.slice((page - 1) * pageSize, page * pageSize)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'r' && !e.metaKey && !e.ctrlKey) { e.preventDefault(); void fetchDocs(selected) }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [fetchDocs, selected])

  return (
    <PageContainer
      title="Document store"
      subtitle="Browse and manage stored documents"
      headerRight={
        <div className="flex items-center gap-1">
          <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => void fetchDocs(selected)}>Refresh</Button>
          <Button size="sm" variant="ghost" className="h-6 text-[10px] text-destructive" onClick={clearCollection}>Clear</Button>
        </div>
      }
    >
      <div className="grid grid-cols-4 gap-1 sm:grid-cols-8">
        {COLLECTIONS.map(c => (
          <button
            key={c}
            type="button"
            onClick={() => setSelected(c)}
            className={cn('rounded-lg border px-2 py-1.5 text-[10px] text-left transition-colors', selected === c ? 'border-primary/40 bg-primary/5 text-primary font-medium' : 'border-border/40 hover:bg-muted/20')}
          >
            <span className="block truncate">{c}</span>
            {collectionMeta[c] != null && (
              <span className="text-[9px] text-muted-foreground/60 font-mono tabular-nums">{collectionMeta[c]}</span>
            )}
          </button>
        ))}
      </div>

      <div className="flex items-center gap-1.5">
        <Input
          value={searchQuery}
          onChange={e => { setSearchQuery(e.target.value); setPage(1) }}
          placeholder={`Search ${selected}...`}
          className="h-7 text-[11px] flex-1"
        />
        <Button size="sm" variant="outline" className="h-7 text-[11px]" onClick={() => setShowCreate(!showCreate)} aria-pressed={showCreate}>
          {showCreate ? 'Cancel' : 'New doc'}
        </Button>
      </div>

      {showCreate && (
        <Card>
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-[11px] font-medium">Create document in {selected}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 px-2.5 pb-2.5">
            <div className="flex flex-col gap-0.5">
              <Label htmlFor="doc-id" className="text-[10px] text-muted-foreground/60 uppercase tracking-wider">Document ID</Label>
              <Input id="doc-id" value={newDocId} onChange={e => setNewDocId(e.target.value)}
                placeholder="my-doc-id" className="h-7 text-[11px] font-mono" />
            </div>
            <div className="flex flex-col gap-0.5">
              <Label htmlFor="doc-content" className="text-[10px] text-muted-foreground/60 uppercase tracking-wider">Content (JSON)</Label>
              <textarea id="doc-content" value={newDocContent} onChange={e => setNewDocContent(e.target.value)}
                rows={6} className="w-full rounded-lg border border-border/40 bg-background px-2.5 py-2 font-mono text-[11px]" />
            </div>
            <Button size="sm" className="h-7 text-[11px]" onClick={createDoc}>Create</Button>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-3 lg:grid-cols-[1fr_1fr]">
        <Card>
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-[11px] font-medium">
              {selected} ({filtered.length} docs)
              {totalPages > 1 && <span className="text-[10px] text-muted-foreground/60 font-normal ml-1.5">Page {page}/{totalPages}</span>}
            </CardTitle>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            {loading ? (
              <p className="text-[10px] text-muted-foreground/60">Loading...</p>
            ) : pageDocs.length === 0 ? (
              <p className="text-[10px] text-muted-foreground/60">No documents.</p>
            ) : (
              <div className="max-h-[400px] space-y-0.5 overflow-y-auto">
                {pageDocs.map(d => (
                  <button
                    key={d._id}
                    type="button"
                    onClick={() => { setSelectedDoc(d); setEditMode(false); setEditContent(JSON.stringify(d, null, 2)) }}
                    className={cn('w-full rounded-lg border p-2 text-left text-[11px] transition-colors font-mono', selectedDoc?._id === d._id ? 'border-primary/40 bg-primary/5' : 'border-border/40 hover:bg-muted/20')}
                  >
                    <span className="block truncate">{d._id}</span>
                  </button>
                ))}
              </div>
            )}
            {totalPages > 1 && (
              <div className="mt-1.5 flex items-center gap-1.5">
                <Button size="sm" variant="ghost" className="h-6 text-[10px]" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>Prev</Button>
                <span className="text-[10px] text-muted-foreground/60 font-mono tabular-nums">{page}/{totalPages}</span>
                <Button size="sm" variant="ghost" className="h-6 text-[10px]" disabled={page >= totalPages} onClick={() => setPage(p => p + 1)}>Next</Button>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <div className="flex items-center justify-between">
              <CardTitle className="text-[11px] font-medium">
                {selectedDoc ? selectedDoc._id : 'Select a document'}
              </CardTitle>
              {selectedDoc && (
                <div className="flex items-center gap-0.5">
                  {editMode ? (
                    <>
                      <Button size="sm" className="h-6 text-[10px]" onClick={saveDoc}>Save</Button>
                      <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => setEditMode(false)}>Cancel</Button>
                    </>
                  ) : (
                    <>
                      <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => setEditMode(true)}>Edit</Button>
                      <Button size="sm" variant="ghost" className="h-6 text-[10px] text-destructive" onClick={() => deleteDoc(selectedDoc._id)}>Delete</Button>
                    </>
                  )}
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            {selectedDoc ? (
              editMode ? (
                <textarea
                  aria-label="Edit document content"
                  value={editContent}
                  onChange={e => setEditContent(e.target.value)}
                  rows={20}
                  className="w-full rounded-lg border border-border/40 bg-background p-2.5 font-mono text-[11px]"
                />
              ) : (
                <pre className="max-h-[400px] overflow-y-auto rounded-lg bg-muted/20 p-2.5 text-[11px] font-mono">
                  {JSON.stringify(selectedDoc, null, 2)}
                </pre>
              )
            ) : (
              <p className="text-[10px] text-muted-foreground/60">Click a document to view details.</p>
            )}
          </CardContent>
        </Card>
      </div>

      <AlertDialog open={pendingClear} onOpenChange={setPendingClear}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete all documents?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete all documents in &quot;{selected}&quot;. This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={async () => { setPendingClear(false); try { await apiDelete(`/docstore/${selected}`); addToast(`Cleared ${selected}`, 'success'); setDocs([]); setSelectedDoc(null); setCollectionMeta(prev => ({ ...prev, [selected]: 0 })) } catch { addToast('Could not clear collection', 'error') } }} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
              Delete all
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </PageContainer>
  )
}
