'use client'

import { useState, useEffect, useCallback, useMemo, useRef, memo } from 'react'
import {
  Card, CardContent, Button, Badge, Input, Textarea,
  Dialog, DialogContent, DialogHeader, DialogTitle,
  cn,
} from '@sloughgpt/strui'
import {
  IconRefresh, IconPlus, IconSearch, IconEdit, IconTrash,
  IconCheck, IconX, IconClock, IconGrid, IconDocument,
} from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'

// ── Types ──────────────────────────────────────────────────────────────

interface KanbanCard {
  id: string
  title: string
  description: string
  column: string
  priority: 'low' | 'medium' | 'high' | 'critical'
  tags: string[]
  created_at: string
  updated_at: string
  due_date: string
  assignee: string
  notes: { id: string; text: string; author: string }[]
}

interface KanbanColumn {
  name: string
  wip_limit: number
  order: number
}

interface KanbanBoard {
  name: string
  columns: KanbanColumn[]
  cards: KanbanCard[]
}

interface Note {
  id: string
  title: string
  created_at: string
  updated_at: string
  tags: string[]
  status: string
  body: string
  sprint: string
  gh: string
}

type Tab = 'board' | 'notes'

// ── Constants ──────────────────────────────────────────────────────────

const COLUMN_LABELS: Record<string, string> = {
  todo: 'To Do',
  in_progress: 'In Progress',
  review: 'Review',
  done: 'Done',
}

const COLUMN_COLORS: Record<string, string> = {
  todo: 'bg-secondary text-secondary-foreground',
  in_progress: 'bg-primary/15 text-primary',
  review: 'bg-accent/20 text-accent-foreground',
  done: 'bg-success/15 text-success',
}

const PRIORITY_COLORS: Record<string, string> = {
  low: 'bg-secondary text-secondary-foreground',
  medium: 'bg-warning/15 text-warning',
  high: 'bg-accent/20 text-accent',
  critical: 'bg-destructive/15 text-destructive',
}

const STATUS_OPTIONS = ['open', 'wip', 'done', 'blocked', 'review', 'todo']

// ── Main Page ──────────────────────────────────────────────────────────

export default function PlannerPage() {
  const [tab, setTab] = useState<Tab>('board')
  const [board, setBoard] = useState<KanbanBoard | null>(null)
  const [notes, setNotes] = useState<Note[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filterTag, setFilterTag] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [syncing, setSyncing] = useState(false)
  const [showNewNote, setShowNewNote] = useState(false)
  const [editingNote, setEditingNote] = useState<Note | null>(null)
  const [draggedCard, setDraggedCard] = useState<string | null>(null)

  const fetchAll = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [boardRes, notesRes] = await Promise.all([
        fetch('/api/planner/board'),
        fetch('/api/planner/notes'),
      ])
      if (boardRes.ok) {
        const bd = await boardRes.json()
        setBoard(bd.board)
      }
      if (notesRes.ok) {
        const nd = await notesRes.json()
        setNotes(nd.notes || [])
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load planner')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchAll() }, [fetchAll])

  const handleSync = useCallback(async () => {
    setSyncing(true)
    try {
      const res = await fetch('/api/planner/sync', { method: 'POST' })
      if (res.ok) {
        await fetchAll()
      }
    } finally {
      setSyncing(false)
    }
  }, [fetchAll])

  const handleMoveCard = useCallback(async (cardId: string, column: string) => {
    try {
      const res = await fetch('/api/planner/board', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: cardId, column }),
      })
      if (res.ok && board) {
        setBoard(prev => {
          if (!prev) return prev
          return {
            ...prev,
            cards: prev.cards.map(c => c.id === cardId ? { ...c, column } : c),
          }
        })
      }
    } catch {}
  }, [board])

  const handleCreateNote = useCallback(async (data: { title: string; tags: string[]; status: string; body: string }) => {
    const res = await fetch('/api/planner/notes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (res.ok) {
      const { note } = await res.json()
      setNotes(prev => [note, ...prev])
      setShowNewNote(false)
    }
  }, [])

  const handleUpdateNote = useCallback(async (id: string, data: Partial<Note>) => {
    const res = await fetch(`/api/planner/notes/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (res.ok) {
      const { note } = await res.json()
      setNotes(prev => prev.map(n => n.id === id ? note : n))
      setEditingNote(null)
    }
  }, [])

  const handleDeleteNote = useCallback(async (id: string) => {
    const res = await fetch(`/api/planner/notes/${id}`, { method: 'DELETE' })
    if (res.ok) {
      setNotes(prev => prev.filter(n => n.id !== id))
    }
  }, [])

  // ── Drag handlers ──────────────────────────────────────────────────

  const onDragStart = useCallback((e: React.DragEvent, cardId: string) => {
    setDraggedCard(cardId)
    e.dataTransfer.effectAllowed = 'move'
  }, [])

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop = useCallback((e: React.DragEvent, column: string) => {
    e.preventDefault()
    if (draggedCard) {
      handleMoveCard(draggedCard, column)
      setDraggedCard(null)
    }
  }, [draggedCard, handleMoveCard])

  // ── Derived data ───────────────────────────────────────────────────

  const allTags = useMemo(() => {
    const tags = new Set<string>()
    for (const card of board?.cards || []) {
      for (const t of card.tags) tags.add(t)
    }
    for (const note of notes) {
      for (const t of note.tags) tags.add(t)
    }
    return Array.from(tags).sort()
  }, [board, notes])

  const filteredCards = useMemo(() => {
    let cards = board?.cards || []
    if (filterTag) cards = cards.filter(c => c.tags.includes(filterTag))
    if (searchQuery) {
      const q = searchQuery.toLowerCase()
      cards = cards.filter(c => c.title.toLowerCase().includes(q) || c.description.toLowerCase().includes(q))
    }
    return cards
  }, [board, filterTag, searchQuery])

  const filteredNotes = useMemo(() => {
    let list = notes
    if (filterTag) list = list.filter(n => n.tags.includes(filterTag))
    if (searchQuery) {
      const q = searchQuery.toLowerCase()
      list = list.filter(n => n.title.toLowerCase().includes(q) || n.body.toLowerCase().includes(q))
    }
    return list
  }, [notes, filterTag, searchQuery])

  const cardsByColumn = useMemo(() => {
    const map: Record<string, KanbanCard[]> = {}
    for (const card of filteredCards) {
      if (!map[card.column]) map[card.column] = []
      map[card.column].push(card)
    }
    return map
  }, [filteredCards])

  const sortedColumns = useMemo(() => {
    return [...(board?.columns || [])].sort((a, b) => a.order - b.order)
  }, [board])

  const stats = useMemo(() => ({
    cards: board?.cards.length || 0,
    notes: notes.length,
    byColumn: (() => {
      const m: Record<string, number> = {}
      for (const c of board?.cards || []) m[c.column] = (m[c.column] || 0) + 1
      return m
    })(),
  }), [board, notes])

  return (
    <PageContainer
      title="Planner"
      subtitle="Board, notes, and sync"
      maxWidth="max-w-7xl"
      headerRight={
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={handleSync} disabled={syncing}>
            <IconRefresh className={cn('h-4 w-4', syncing && 'animate-spin')} />
            Sync
          </Button>
          <Button variant="ghost" size="sm" onClick={fetchAll} disabled={loading}>
            <IconRefresh className={cn('h-4 w-4', loading && 'animate-spin')} />
          </Button>
        </div>
      }
      toolbar={
        <div className="flex items-center gap-3 flex-wrap">
          {/* Tab switcher */}
          <div className="flex items-center rounded-lg border border-border p-0.5">
            <button
              onClick={() => setTab('board')}
              className={cn(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
                tab === 'board' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
              )}
            >
              <IconGrid className="h-3.5 w-3.5" />
              Board
              <span className="text-xs opacity-70">{stats.cards}</span>
            </button>
            <button
              onClick={() => setTab('notes')}
              className={cn(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
                tab === 'notes' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
              )}
            >
              <IconDocument className="h-3.5 w-3.5" />
              Notes
              <span className="text-xs opacity-70">{stats.notes}</span>
            </button>
          </div>

          {/* Search */}
          <div className="relative flex-1 min-w-[200px] max-w-md">
            <IconSearch className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search cards and notes..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="pl-9 h-9"
            />
          </div>

          {/* Tag filter */}
          {allTags.length > 0 && (
            <div className="flex items-center gap-1 flex-wrap">
              <Button
                variant={filterTag === null ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setFilterTag(null)}
              >
                All
              </Button>
              {allTags.slice(0, 8).map(tag => (
                <Badge
                  key={tag}
                  variant={filterTag === tag ? 'default' : 'secondary'}
                  className="cursor-pointer"
                  onClick={() => setFilterTag(filterTag === tag ? null : tag)}
                >
                  {tag}
                </Badge>
              ))}
              {allTags.length > 8 && (
                <span className="text-xs text-muted-foreground">+{allTags.length - 8}</span>
              )}
            </div>
          )}

          {tab === 'notes' && (
            <Button size="sm" onClick={() => setShowNewNote(true)}>
              <IconPlus className="h-4 w-4 mr-1" />
              New Note
            </Button>
          )}
        </div>
      }
      loading={loading}
      error={error}
      onRetry={fetchAll}
    >
      {tab === 'board' ? (
        <BoardView
          columns={sortedColumns}
          cardsByColumn={cardsByColumn}
          draggedCard={draggedCard}
          onDragStart={onDragStart}
          onDragOver={onDragOver}
          onDrop={onDrop}
        />
      ) : (
        <NotesView
          notes={filteredNotes}
          onEdit={setEditingNote}
          onDelete={handleDeleteNote}
        />
      )}

      {/* New note dialog */}
      {showNewNote && (
        <NoteDialog
          onClose={() => setShowNewNote(false)}
          onSave={handleCreateNote}
        />
      )}

      {/* Edit note dialog */}
      {editingNote && (
        <NoteDialog
          note={editingNote}
          onClose={() => setEditingNote(null)}
          onSave={data => handleUpdateNote(editingNote.id, data)}
        />
      )}
    </PageContainer>
  )
}

// ── Board View ─────────────────────────────────────────────────────────

const BoardView = memo(function BoardView({
  columns,
  cardsByColumn,
  draggedCard,
  onDragStart,
  onDragOver,
  onDrop,
}: {
  columns: KanbanColumn[]
  cardsByColumn: Record<string, KanbanCard[]>
  draggedCard: string | null
  onDragStart: (e: React.DragEvent, cardId: string) => void
  onDragOver: (e: React.DragEvent) => void
  onDrop: (e: React.DragEvent, column: string) => void
}) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {columns.map(col => {
        const cards = cardsByColumn[col.name] || []
        const atWip = col.wip_limit > 0 && cards.length >= col.wip_limit
        return (
          <div
            key={col.name}
            className="space-y-3"
            onDragOver={onDragOver}
            onDrop={e => onDrop(e, col.name)}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-medium text-foreground">
                  {COLUMN_LABELS[col.name] || col.name}
                </h3>
                <span className={cn(
                  'inline-flex items-center justify-center h-5 min-w-5 px-1.5 rounded-full text-xs font-medium',
                  COLUMN_COLORS[col.name] || 'bg-muted text-muted-foreground'
                )}>
                  {cards.length}
                </span>
                {col.wip_limit > 0 && (
                  <span className="text-xs text-muted-foreground">/ {col.wip_limit}</span>
                )}
              </div>
              {atWip && <span className="text-xs text-warning font-medium">WIP limit</span>}
            </div>

            <div className="space-y-2 min-h-[4rem]">
              {cards.length === 0 && (
                <div className="text-xs text-muted-foreground italic p-3 text-center border border-dashed border-border rounded-lg">
                  Drop cards here
                </div>
              )}
              {cards.map(card => (
                <Card
                  key={card.id}
                  className={cn(
                    'transition-shadow hover:shadow-md cursor-grab active:cursor-grabbing',
                    draggedCard === card.id && 'opacity-50'
                  )}
                  draggable
                  onDragStart={e => onDragStart(e, card.id)}
                >
                  <CardContent className="p-3 space-y-2">
                    <p className="text-sm font-medium text-foreground leading-snug">{card.title}</p>
                    {card.description && (
                      <p className="text-xs text-muted-foreground line-clamp-2">{card.description}</p>
                    )}
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className={cn(
                        'inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium uppercase tracking-wider',
                        PRIORITY_COLORS[card.priority] || 'bg-muted text-muted-foreground'
                      )}>
                        {card.priority}
                      </span>
                      {card.tags.slice(0, 3).map(tag => (
                        <span key={tag} className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] bg-muted text-muted-foreground">
                          {tag}
                        </span>
                      ))}
                      {card.tags.length > 3 && (
                        <span className="text-[10px] text-muted-foreground">+{card.tags.length - 3}</span>
                      )}
                    </div>
                    {card.due_date && (
                      <p className="text-[10px] text-muted-foreground flex items-center gap-1">
                        <IconClock className="h-3 w-3" />
                        {new Date(card.due_date).toLocaleDateString()}
                      </p>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
})

// ── Notes View ─────────────────────────────────────────────────────────

const NotesView = memo(function NotesView({
  notes,
  onEdit,
  onDelete,
}: {
  notes: Note[]
  onEdit: (note: Note) => void
  onDelete: (id: string) => void
}) {
  if (notes.length === 0) {
    return (
      <div className="flex min-h-[20vh] items-center justify-center">
        <div className="text-center space-y-2">
          <IconDocument className="h-8 w-8 mx-auto text-muted-foreground" />
          <p className="text-sm text-muted-foreground">No notes yet</p>
          <p className="text-xs text-muted-foreground">Create a note to get started</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {notes.map(note => (
        <Card key={note.id} className="transition-shadow hover:shadow-md">
          <CardContent className="p-4">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0 flex-1 space-y-1">
                <p className="text-sm font-medium text-foreground leading-snug">{note.title}</p>
                {note.body && (
                  <p className="text-xs text-muted-foreground line-clamp-2">{note.body}</p>
                )}
                <div className="flex items-center gap-2 flex-wrap">
                  <Badge variant={note.status === 'done' ? 'default' : 'secondary'} className="text-[10px]">
                    {note.status}
                  </Badge>
                  {note.tags.map(tag => (
                    <span key={tag} className="text-[10px] text-muted-foreground">#{tag}</span>
                  ))}
                  <span className="text-[10px] text-muted-foreground">
                    {note.created_at ? new Date(note.created_at).toLocaleDateString() : ''}
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-1 shrink-0">
                <Button variant="ghost" size="sm" onClick={() => onEdit(note)}>
                  <IconEdit className="h-3.5 w-3.5" />
                </Button>
                <Button variant="ghost" size="sm" onClick={() => onDelete(note.id)}>
                  <IconTrash className="h-3.5 w-3.5" />
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
})

// ── Note Dialog ────────────────────────────────────────────────────────

const NoteDialog = memo(function NoteDialog({
  note,
  onClose,
  onSave,
}: {
  note?: Note
  onClose: () => void
  onSave: (data: { title: string; tags: string[]; status: string; body: string }) => void
}) {
  const [title, setTitle] = useState(note?.title || '')
  const [tags, setTags] = useState(note?.tags.join(', ') || '')
  const [status, setStatus] = useState(note?.status || 'open')
  const [body, setBody] = useState(note?.body || '')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!title.trim()) return
    onSave({
      title: title.trim(),
      tags: tags.split(',').map(t => t.trim()).filter(Boolean),
      status,
      body,
    })
  }

  return (
    <Dialog open onOpenChange={onClose}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>{note ? 'Edit Note' : 'New Note'}</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Title</label>
            <Input
              value={title}
              onChange={e => setTitle(e.target.value)}
              placeholder="Note title"
              autoFocus
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Tags</label>
              <Input
                value={tags}
                onChange={e => setTags(e.target.value)}
                placeholder="comma-separated"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Status</label>
              <select
                value={status}
                onChange={e => setStatus(e.target.value)}
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 text-sm"
              >
                {STATUS_OPTIONS.map(s => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Body</label>
            <Textarea
              value={body}
              onChange={e => setBody(e.target.value)}
              placeholder="Note content..."
              rows={6}
            />
          </div>
          <div className="flex justify-end gap-2">
            <Button type="button" variant="ghost" onClick={onClose}>Cancel</Button>
            <Button type="submit" disabled={!title.trim()}>
              <IconCheck className="h-4 w-4 mr-1" />
              {note ? 'Save' : 'Create'}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
})
