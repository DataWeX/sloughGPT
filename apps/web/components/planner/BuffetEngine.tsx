'use client'

import { useState, useCallback, useMemo } from 'react'
import { cn, Button, IconPlus, IconSearch, IconRefresh } from '@sloughgpt/strui'
import type { Card, DragState, SyncState } from './types'
import { useOonBoard } from '@/lib/useOonBoard'
import { oon } from '@/lib/oon'
import { Scene } from './Scene'
import { CardEditor } from './CardEditor'

const DEFAULT_COLUMNS = [
  { name: 'todo', label: 'To Do', wip_limit: 5, order: 0 },
  { name: 'in_progress', label: 'In Progress', wip_limit: 3, order: 1 },
  { name: 'review', label: 'Review', wip_limit: 2, order: 2 },
  { name: 'done', label: 'Done', wip_limit: 0, order: 3 },
]

export function BuffetEngine() {
  const {
    board, tags, loading, error,
    refresh, optimisticMove, optimisticAdd, optimisticDelete,
  } = useOonBoard(500)

  const [input, setInput] = useState<{ drag: DragState | null; selected: Card | null }>({
    drag: null,
    selected: null,
  })
  const [sync, setSync] = useState<SyncState>({ status: 'idle', lastSync: null, error: null })
  const [searchQuery, setSearchQuery] = useState('')
  const [filterTag, setFilterTag] = useState<string | null>(null)
  const [showCreate, setShowCreate] = useState(false)

  const handleMove = useCallback(
    async (cardId: string, toColumn: string) => {
      optimisticMove(cardId, toColumn)
      try {
        await oon.move(cardId, toColumn)
      } catch {
        refresh()
      }
    },
    [optimisticMove, refresh],
  )

  const handleCreate = useCallback(
    async (data: { title: string; column: string; priority: string; description: string }) => {
      try {
        const { card } = await oon.create(data)
        optimisticAdd(card)
        setShowCreate(false)
      } catch (err) {
        refresh()
      }
    },
    [oon, optimisticAdd, refresh],
  )

  const handleUpdate = useCallback(
    async (cardId: string, data: Partial<Card>) => {
      optimisticMove(cardId, data.column || '')
      setInput((prev) => ({ ...prev, selected: null }))
      try {
        await oon.update(cardId, data)
      } catch {
        refresh()
      }
    },
    [oon, optimisticMove, refresh],
  )

  const handleDelete = useCallback(
    async (cardId: string) => {
      optimisticDelete(cardId)
      setInput((prev) => ({ ...prev, selected: null }))
      try {
        await oon.delete(cardId)
      } catch {
        refresh()
      }
    },
    [oon, optimisticDelete, refresh],
  )

  const handleSync = useCallback(async () => {
    setSync({ status: 'syncing', lastSync: null, error: null })
    try {
      await oon.sync()
      setSync({ status: 'idle', lastSync: new Date().toISOString(), error: null })
      refresh()
    } catch (err) {
      setSync({ status: 'error', lastSync: null, error: err instanceof Error ? err.message : 'Sync failed' })
    }
  }, [oon, refresh])

  const handleDragStart = useCallback((e: React.DragEvent, cardId: string, column: string) => {
    e.dataTransfer.setData('text/plain', cardId)
    e.dataTransfer.effectAllowed = 'move'
    setInput((prev) => ({ ...prev, drag: { cardId, fromColumn: column } }))
  }, [])

  const handleDragEnd = useCallback(() => {
    setInput((prev) => ({ ...prev, drag: null }))
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.dataTransfer.dropEffect = 'move'
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent, targetColumn: string) => {
      const cardId = e.dataTransfer.getData('text/plain')
      if (cardId && (!input.drag || input.drag.fromColumn !== targetColumn)) {
        handleMove(cardId, targetColumn)
      }
      setInput((prev) => ({ ...prev, drag: null }))
    },
    [input.drag, handleMove],
  )

  const filteredCards = useMemo(() => {
    let cards = board?.cards || []
    if (filterTag) {
      cards = cards.filter((c) => c.tags.includes(filterTag))
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase()
      cards = cards.filter(
        (c) =>
          c.title.toLowerCase().includes(q) ||
          c.description.toLowerCase().includes(q) ||
          c.tags.some((t) => t.toLowerCase().includes(q)),
      )
    }
    return cards
  }, [board, filterTag, searchQuery])

  const uniqueTags = useMemo(() => tags.map((t) => t.name).sort(), [tags])

  if (loading && !board) {
    return (
      <div className="relative min-h-screen bg-[hsl(40,30%,95%)] overflow-hidden">
        <div
          className={cn(
            'absolute inset-0 pointer-events-none opacity-20',
            '[background-image:',
              'radial-gradient(circle,hsl(40,20%,75%)_1.5px,transparent_1.5px)',
            ']',
            '[background-size:12px_12px]',
            '[background-position:0_0]',
          )}
          aria-hidden="true"
        />
        <div className="relative sl-page mx-auto max-w-7xl">
          <div className="flex items-center justify-center min-h-[40vh]">
            <div className="text-sm text-[hsl(40,20%,45%)]">Loading board...</div>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="relative min-h-screen bg-[hsl(40,30%,95%)] overflow-hidden">
        <div
          className={cn(
            'absolute inset-0 pointer-events-none opacity-20',
            '[background-image:',
              'radial-gradient(circle,hsl(40,20%,75%)_1.5px,transparent_1.5px)',
            ']',
            '[background-size:12px_12px]',
            '[background-position:0_0]',
          )}
          aria-hidden="true"
        />
        <div className="relative sl-page mx-auto max-w-7xl">
          <div className="flex flex-col items-center justify-center min-h-[40vh] gap-3">
            <p className="text-sm text-[hsl(0,50%,50%)]">{error}</p>
            <Button size="sm" variant="outline" onClick={refresh}>
              <IconRefresh className="h-4 w-4 mr-1.5" />
              Retry
            </Button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="relative min-h-screen bg-[hsl(40,30%,95%)] overflow-hidden">
      {/* Halftone dot pattern overlay */}
      <div
        className={cn(
          'absolute inset-0 pointer-events-none opacity-20',
          '[background-image:',
            'radial-gradient(circle,hsl(40,20%,75%)_1.5px,transparent_1.5px)',
          ']',
          '[background-size:12px_12px]',
          '[background-position:0_0]',
        )}
        aria-hidden="true"
      />

      <div className="relative sl-page mx-auto max-w-7xl space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-[hsl(40,20%,20%)]">Planner</h1>
            <p className="text-sm text-[hsl(40,20%,45%)]">Board</p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={refresh} disabled={loading}>
              <IconRefresh className={cn('h-4 w-4', loading && 'animate-spin')} />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleSync}
              disabled={sync.status === 'syncing'}
              title="Sync notes to board"
            >
              <IconRefresh className={cn('h-4 w-4', sync.status === 'syncing' && 'animate-spin')} />
            </Button>
            <Button size="sm" onClick={() => setShowCreate(true)} aria-label="Create card">
              <IconPlus className="h-4 w-4 mr-1" />
              New card
            </Button>
          </div>
        </div>

        {/* Toolbar */}
        <div className="flex items-center gap-3">
          <div className="relative flex-1 max-w-xs">
            <IconSearch className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-[hsl(40,20%,50%)]" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search cards..."
              aria-label="Search cards"
              className="w-full rounded-lg border border-[hsl(40,20%,80%)] bg-[hsl(40,25%,97%)] pl-8 pr-3 py-1.5 text-sm text-[hsl(40,20%,20%)] placeholder:text-[hsl(40,20%,55%)] focus:outline-none focus:ring-2 focus:ring-[hsl(270,50%,60%)]"
            />
          </div>
        {uniqueTags.length > 0 && (
          <div className="flex items-center gap-1 flex-wrap">
            <Button
              variant={filterTag === null ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setFilterTag(null)}
            >
              All
            </Button>
            {uniqueTags.slice(0, 6).map((tag) => (
              <Button
                key={tag}
                variant={filterTag === tag ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setFilterTag(filterTag === tag ? null : tag)}
              >
                {tag}
              </Button>
            ))}
            {uniqueTags.length > 6 && (
              <span className="text-xs text-[hsl(40,20%,50%)]">+{uniqueTags.length - 6}</span>
            )}
          </div>
        )}
      </div>

      {/* Scene */}
      <Scene
        board={board || { columns: DEFAULT_COLUMNS, cards: [] }}
        filteredCards={filteredCards}
        onDragStart={handleDragStart}
        onDragEnd={handleDragEnd}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onCardClick={(card) => setInput((prev) => ({ ...prev, selected: card }))}
        draggingId={input.drag?.cardId ?? null}
      />

      {/* Card Editor */}
      <CardEditor
        card={input.selected}
        onClose={() => setInput((prev) => ({ ...prev, selected: null }))}
        onUpdate={handleUpdate}
        onDelete={handleDelete}
      />

      {/* Create Card Dialog */}
      {showCreate && (
        <CreateCardDialog
          onClose={() => setShowCreate(false)}
          onCreate={handleCreate}
        />
      )}
      </div>
    </div>
  )
}

function CreateCardDialog({
  onClose,
  onCreate,
}: {
  onClose: () => void
  onCreate: (data: { title: string; column: string; priority: string; description: string }) => void
}) {
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [column, setColumn] = useState('todo')
  const [priority, setPriority] = useState('medium')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (title.trim()) {
      onCreate({ title: title.trim(), column, priority, description: description.trim() })
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-[hsl(40,25%,97%)] rounded-lg border border-[hsl(40,20%,80%)] shadow-[6px_6px_0_0_hsl(40,20%,78%)] p-6 w-full max-w-md">
        <h2 className="text-lg font-semibold mb-4 text-[hsl(40,20%,20%)]">New Card</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="text-sm font-medium text-[hsl(40,20%,30%)]">Title</label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="w-full mt-1 rounded-lg border border-[hsl(40,20%,80%)] bg-[hsl(40,30%,95%)] px-3 py-2 text-sm text-[hsl(40,20%,20%)]"
              autoFocus
            />
          </div>
          <div>
            <label className="text-sm font-medium text-[hsl(40,20%,30%)]">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full mt-1 rounded-lg border border-[hsl(40,20%,80%)] bg-[hsl(40,30%,95%)] px-3 py-2 text-sm text-[hsl(40,20%,20%)]"
              rows={3}
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium text-[hsl(40,20%,30%)]">Column</label>
              <select
                value={column}
                onChange={(e) => setColumn(e.target.value)}
                className="w-full mt-1 rounded-lg border border-[hsl(40,20%,80%)] bg-[hsl(40,30%,95%)] px-3 py-2 text-sm text-[hsl(40,20%,20%)]"
              >
                <option value="todo">To Do</option>
                <option value="in_progress">In Progress</option>
                <option value="review">Review</option>
                <option value="done">Done</option>
              </select>
            </div>
            <div>
              <label className="text-sm font-medium text-[hsl(40,20%,30%)]">Priority</label>
              <select
                value={priority}
                onChange={(e) => setPriority(e.target.value)}
                className="w-full mt-1 rounded-lg border border-[hsl(40,20%,80%)] bg-[hsl(40,30%,95%)] px-3 py-2 text-sm text-[hsl(40,20%,20%)]"
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
                <option value="critical">Critical</option>
              </select>
            </div>
          </div>
          <div className="flex justify-end gap-2">
            <Button type="button" variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={!title.trim()}>
              Create
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}
