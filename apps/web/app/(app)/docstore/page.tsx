'use client'
export const dynamic = 'force-dynamic'

import { useState, useCallback, useEffect } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label, AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from '@sloughgpt/strui'
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
        <div className="flex items-center gap-2">
          <Button size="sm" variant="ghost" onClick={() => void fetchDocs(selected)}>Refresh</Button>
          <Button size="sm" variant="ghost" className="text-destructive" onClick={clearCollection}>Clear</Button>
        </div>
      }
    >
      <div className="grid grid-cols-4 gap-1 sm:grid-cols-8">
        {COLLECTIONS.map(c => (
          <button
            key={c}
            type="button"
            onClick={() => setSelected(c)}
            className={`rounded border px-2 py-1.5 text-xs text-left transition-colors ${
              selected === c
                ? 'border-primary bg-primary/10 text-primary font-medium'
                : 'border-border hover:bg-muted/50'
            }`}
          >
            <span className="block truncate">{c}</span>
            {collectionMeta[c] != null && (
              <span className="text-[10px] text-muted-foreground">{collectionMeta[c]}</span>
            )}
          </button>
        ))}
      </div>

      <div className="flex items-center gap-2">
        <Input
          value={searchQuery}
          onChange={e => { setSearchQuery(e.target.value); setPage(1) }}
          placeholder={`Search ${selected}...`}
          className="h-8 text-xs"
        />
        <Button size="sm" variant="outline" onClick={() => setShowCreate(!showCreate)}>
          {showCreate ? 'Cancel' : 'New doc'}
        </Button>
      </div>

      {showCreate && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Create document in {selected}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-col gap-1">
              <Label htmlFor="doc-id" variant="uppercase">Document ID</Label>
              <Input id="doc-id" value={newDocId} onChange={e => setNewDocId(e.target.value)}
                placeholder="my-doc-id" className="h-8 text-sm font-mono" />
            </div>
            <div className="flex flex-col gap-1">
              <Label htmlFor="doc-content" variant="uppercase">Content (JSON)</Label>
              <textarea id="doc-content" value={newDocContent} onChange={e => setNewDocContent(e.target.value)}
                rows={6} className="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-xs" />
            </div>
            <Button size="sm" onClick={createDoc}>Create</Button>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">
              {selected} ({filtered.length} docs)
              {totalPages > 1 && <span className="text-xs text-muted-foreground font-normal ml-2">Page {page}/{totalPages}</span>}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <p className="text-xs text-muted-foreground">Loading...</p>
            ) : pageDocs.length === 0 ? (
              <p className="text-xs text-muted-foreground">No documents.</p>
            ) : (
              <div className="max-h-[400px] space-y-1 overflow-y-auto">
                {pageDocs.map(d => (
                  <button
                    key={d._id}
                    type="button"
                    onClick={() => { setSelectedDoc(d); setEditMode(false); setEditContent(JSON.stringify(d, null, 2)) }}
                    className={`w-full rounded border p-2 text-left text-xs transition-colors ${
                      selectedDoc?._id === d._id
                        ? 'border-primary bg-primary/5'
                        : 'border-border hover:bg-muted/30'
                    }`}
                  >
                    <span className="block truncate font-mono">{d._id}</span>
                  </button>
                ))}
              </div>
            )}
            {totalPages > 1 && (
              <div className="mt-2 flex items-center gap-2">
                <Button size="sm" variant="ghost" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>Prev</Button>
                <span className="text-xs text-muted-foreground">{page}/{totalPages}</span>
                <Button size="sm" variant="ghost" disabled={page >= totalPages} onClick={() => setPage(p => p + 1)}>Next</Button>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">
                {selectedDoc ? selectedDoc._id : 'Select a document'}
              </CardTitle>
              {selectedDoc && (
                <div className="flex items-center gap-1">
                  {editMode ? (
                    <>
                      <Button size="sm" onClick={saveDoc}>Save</Button>
                      <Button size="sm" variant="ghost" onClick={() => setEditMode(false)}>Cancel</Button>
                    </>
                  ) : (
                    <>
                      <Button size="sm" variant="ghost" onClick={() => setEditMode(true)}>Edit</Button>
                      <Button size="sm" variant="ghost" className="text-destructive" onClick={() => deleteDoc(selectedDoc._id)}>Delete</Button>
                    </>
                  )}
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {selectedDoc ? (
              editMode ? (
                <textarea
                  aria-label="Edit document content"
                  value={editContent}
                  onChange={e => setEditContent(e.target.value)}
                  rows={20}
                  className="w-full rounded-md border border-input bg-background p-3 font-mono text-xs"
                />
              ) : (
                <pre className="max-h-[400px] overflow-y-auto rounded bg-muted/30 p-3 text-xs">
                  {JSON.stringify(selectedDoc, null, 2)}
                </pre>
              )
            ) : (
              <p className="text-xs text-muted-foreground">Click a document to view details.</p>
            )}
          </CardContent>
        </Card>
      </div>

      <AlertDialog open={pendingClear} onOpenChange={setPendingClear}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete all documents?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete all documents in "{selected}". This cannot be undone.
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
