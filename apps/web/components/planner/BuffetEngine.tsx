'use client'

import { useState, useCallback, useMemo, useRef } from 'react'
import { cn, Button, IconPlus, IconSearch, Spinner } from '@sloughgpt/strui'
import type { Card } from './types'
import { usePlannerBoard } from '@/lib/usePlannerBoard'
import { Table } from './Table'
import { CardEditor } from './CardEditor'

export function BuffetEngine() {
  const {
    board,
    columns,
    tags,
    loading,
    error,
    refresh,
    moveCard,
    createCard,
    updateCard,
    deleteCard,
    undo,
    canUndo,
    isOverWip,
    canDrop,
    syncNotes,
    drag,
    setDrag,
  } = usePlannerBoard(500)

  const [selected, setSelected] = useState<Card | null>(null)
  const [focusCol, setFocusCol] = useState(0)
  const [focusCard, setFocusCard] = useState(0)
  const [searchQuery, setSearchQuery] = useState('')
  const [filterTag, setFilterTag] = useState<string | null>(null)
  const [showCreate, setShowCreate] = useState(false)
  const [syncing, setSyncing] = useState(false)
  const [light, setLight] = useState({ x: 50, y: 30 })
  const boardRef = useRef<HTMLDivElement>(null)

  const sortedColumns = useMemo(() => [...columns].sort((a, b) => a.order - b.order), [columns])

  const filteredCards = useMemo(() => {
    let cards = board?.cards || []
    if (filterTag) cards = cards.filter((c) => c.tags.includes(filterTag))
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

  const cardsByColumn = useMemo(() => {
    const map: Record<string, Card[]> = {}
    for (const col of sortedColumns) {
      map[col.name] = filteredCards
        .filter((c) => c.column === col.name)
        .sort((a, b) => {
          const order: Record<string, number> = { critical: 0, high: 1, medium: 2, low: 3 }
          const pa = order[a.priority] ?? 4
          const pb = order[b.priority] ?? 4
          if (pa !== pb) return pa - pb
          return new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
        })
    }
    return map
  }, [filteredCards, sortedColumns])

  const uniqueTags = useMemo(() => tags.map((t) => t.name).sort(), [tags])

  const handleLight = useCallback((e: React.MouseEvent) => {
    if (!boardRef.current) return
    const r = boardRef.current.getBoundingClientRect()
    setLight({
      x: ((e.clientX - r.left) / r.width) * 100,
      y: ((e.clientY - r.top) / r.height) * 100,
    })
  }, [])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (selected || showCreate) return
      const col = sortedColumns[focusCol]
      if (!col) return
      const colCards = cardsByColumn[col.name] || []
      switch (e.key) {
        case 'ArrowLeft':
          e.preventDefault()
          setFocusCol((c) => Math.max(0, c - 1))
          setFocusCard(0)
          break
        case 'ArrowRight':
          e.preventDefault()
          setFocusCol((c) => Math.min(sortedColumns.length - 1, c + 1))
          setFocusCard(0)
          break
        case 'ArrowUp':
          e.preventDefault()
          setFocusCard((c) => Math.max(0, c - 1))
          break
        case 'ArrowDown':
          e.preventDefault()
          setFocusCard((c) => Math.min(colCards.length - 1, c + 1))
          break
        case 'Enter':
        case ' ':
          if (colCards[focusCard]) {
            e.preventDefault()
            setSelected(colCards[focusCard])
          }
          break
        case 'z':
          if (e.metaKey || e.ctrlKey) {
            e.preventDefault()
            undo()
          }
          break
      }
    },
    [sortedColumns, focusCol, focusCard, cardsByColumn, selected, showCreate, undo],
  )

  const handleDragStart = useCallback(
    (e: React.DragEvent, cardId: string, column: string) => {
      e.dataTransfer.setData('text/plain', cardId)
      e.dataTransfer.effectAllowed = 'move'
      setDrag({ cardId, fromColumn: column })
    },
    [setDrag],
  )

  const handleDragEnd = useCallback(() => setDrag(null), [setDrag])

  const handleDragOver = useCallback(
    (e: React.DragEvent, column: string) => {
      e.preventDefault()
      e.dataTransfer.dropEffect = canDrop(column) ? 'move' : 'none'
    },
    [canDrop],
  )

  const handleDrop = useCallback(
    (e: React.DragEvent, targetColumn: string) => {
      e.preventDefault()
      const cardId = e.dataTransfer.getData('text/plain')
      if (cardId && (!drag || drag.fromColumn !== targetColumn)) moveCard(cardId, targetColumn)
      setDrag(null)
    },
    [drag, moveCard, setDrag],
  )

  const handleUpdate = useCallback(
    (cardId: string, data: Partial<Card>) => {
      setSelected(null)
      updateCard(cardId, data)
    },
    [updateCard],
  )

  const handleDelete = useCallback(
    (cardId: string) => {
      setSelected(null)
      deleteCard(cardId)
    },
    [deleteCard],
  )

  const handleCreate = useCallback(
    async (data: { title: string; column: string; priority: string; description: string }) => {
      try {
        await createCard(data)
        setShowCreate(false)
      } catch {
        refresh()
      }
    },
    [createCard, refresh],
  )

  const handleSync = useCallback(async () => {
    setSyncing(true)
    try {
      await syncNotes()
    } catch {
      /* noop */
    } finally {
      setSyncing(false)
    }
  }, [syncNotes])

  if (loading && !board) {
    return (
      <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
        <div className="text-sm text-zinc-500 font-mono animate-pulse">loading board...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-zinc-950 flex flex-col items-center justify-center gap-3">
        <p className="text-sm text-red-400 font-mono">{error}</p>
        <Button size="sm" variant="outline" onClick={refresh}>
          Retry
        </Button>
      </div>
    )
  }

  return (
    <div
      ref={boardRef}
      className="relative h-screen flex flex-col overflow-hidden select-none bg-zinc-950"
      onMouseMove={handleLight}
      onKeyDown={handleKeyDown}
      tabIndex={-1}
      role="application"
      aria-label="Kanban board"
      style={{
        backgroundImage: `radial-gradient(ellipse 900px 500px at ${light.x}% ${light.y}%, rgba(255,245,220,0.035), transparent 65%)`,
      }}
    >
      {/* Vignette */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(ellipse at 50% 40%, transparent 30%, rgba(0,0,0,0.45) 100%)',
        }}
        aria-hidden
      />

      {/* Header — fixed top bar */}
      <header
        className="relative shrink-0 px-5 py-3"
        style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-baseline gap-3">
            <h1 className="text-lg font-light tracking-tight text-zinc-200">Planner</h1>
            <p className="text-[11px] font-mono text-zinc-600">
              {board?.cards.length ?? 0} cards · {uniqueTags.length} tags
            </p>
          </div>
          <div className="flex items-center gap-1.5">
            <Button
              variant="ghost"
              size="sm"
              onClick={undo}
              disabled={!canUndo}
              title="Undo (Ctrl+Z)"
              aria-label="Undo"
              className="text-zinc-500 hover:text-zinc-300"
            >
              <svg
                className="h-4 w-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M3 10h10a5 5 0 015 5v2m0 0l-3-3m3 3l3-3"
                />
              </svg>
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={refresh}
              disabled={loading}
              aria-label="Refresh"
              className="text-zinc-500 hover:text-zinc-300"
            >
              <Spinner className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleSync}
              disabled={syncing}
              title="Sync notes to board"
              aria-label="Sync notes"
              className="text-zinc-500 hover:text-zinc-300"
            >
              {syncing ? (
                <Spinner className="h-4 w-4" />
              ) : (
                <svg
                  className="h-4 w-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                  />
                </svg>
              )}
            </Button>
            <button
              onClick={() => setShowCreate(true)}
              className={cn(
                'px-3 py-1.5 rounded-md text-sm font-medium',
                'bg-zinc-800 text-zinc-300 border border-zinc-700',
                'hover:bg-zinc-700 hover:text-zinc-100 hover:border-zinc-600',
                'active:scale-[0.97] transition-all duration-150',
                'shadow-[0_1px_2px_rgba(0,0,0,0.3)]',
              )}
            >
              <IconPlus className="h-3.5 w-3.5 mr-1 inline" />
              New
            </button>
          </div>
        </div>

        {/* Toolbar row */}
        <div className="flex items-center gap-3 mt-3">
          <div className="relative flex-1 max-w-xs">
            <IconSearch className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-600" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search..."
              aria-label="Search cards"
              className="w-full rounded-md bg-zinc-900 border border-zinc-800 pl-8 pr-3 py-1.5 text-sm text-zinc-300 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-zinc-600 focus:border-zinc-600 transition-colors"
            />
          </div>
          {uniqueTags.length > 0 && (
            <div className="flex items-center gap-1 flex-wrap">
              {uniqueTags.slice(0, 6).map((tag) => (
                <button
                  key={tag}
                  onClick={() => setFilterTag(filterTag === tag ? null : tag)}
                  className={cn(
                    'px-2 py-0.5 rounded text-[11px] font-mono transition-all duration-150',
                    filterTag === tag
                      ? 'bg-zinc-800 text-zinc-300 border border-zinc-700'
                      : 'text-zinc-600 hover:text-zinc-400 border border-transparent',
                  )}
                >
                  {tag}
                </button>
              ))}
            </div>
          )}
        </div>
      </header>

      {/* Board — fills remaining height */}
      <main className="relative flex-1 min-h-0 px-5 py-4 overflow-hidden">
        <Table
          columns={sortedColumns}
          cardsByColumn={cardsByColumn}
          onDragStart={handleDragStart}
          onDragEnd={handleDragEnd}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onCardClick={setSelected}
          draggingId={drag?.cardId ?? null}
          focusCol={focusCol}
          focusCard={focusCard}
          canDrop={canDrop}
          isOverWip={isOverWip}
        />
      </main>

      {/* Status bar */}
      <footer
        className="relative shrink-0 px-5 py-1.5 flex items-center justify-between text-[10px] font-mono text-zinc-700"
        style={{ borderTop: '1px solid rgba(255,255,255,0.04)' }}
      >
        <div className="flex items-center gap-4">
          <span>←→ columns</span>
          <span>↑↓ cards</span>
          <span>↵ open</span>
          <span>ctrl+z undo</span>
        </div>
        <div className="flex items-center gap-4">
          {filterTag && <span className="text-zinc-500">filter: {filterTag}</span>}
          {searchQuery && <span className="text-zinc-500">search: &quot;{searchQuery}&quot;</span>}
          <span>
            {filteredCards.length} / {board?.cards.length ?? 0}
          </span>
        </div>
      </footer>

      <CardEditor
        card={selected}
        onClose={() => setSelected(null)}
        onUpdate={handleUpdate}
        onDelete={handleDelete}
      />
      {showCreate && (
        <CreateCardDialog onClose={() => setShowCreate(false)} onCreate={handleCreate} />
      )}
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
    if (title.trim())
      onCreate({ title: title.trim(), column, priority, description: description.trim() })
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 w-full max-w-md shadow-2xl">
        <h2 className="text-base font-medium mb-4 text-zinc-200">New Card</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Title"
            autoFocus
            className="w-full rounded-md bg-zinc-800 border border-zinc-700 px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-zinc-600"
          />
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Description"
            rows={3}
            className="w-full rounded-md bg-zinc-800 border border-zinc-700 px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-zinc-600"
          />
          <div className="grid grid-cols-2 gap-4">
            <select
              value={column}
              onChange={(e) => setColumn(e.target.value)}
              className="rounded-md bg-zinc-800 border border-zinc-700 px-3 py-2 text-sm text-zinc-200"
            >
              <option value="todo">To Do</option>
              <option value="wip">In Progress</option>
              <option value="review">Review</option>
              <option value="done">Done</option>
            </select>
            <select
              value={priority}
              onChange={(e) => setPriority(e.target.value)}
              className="rounded-md bg-zinc-800 border border-zinc-700 px-3 py-2 text-sm text-zinc-200"
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
              <option value="critical">Critical</option>
            </select>
          </div>
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-3 py-1.5 rounded-md text-sm text-zinc-500 hover:text-zinc-300 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!title.trim()}
              className="px-3 py-1.5 rounded-md text-sm bg-zinc-800 text-zinc-300 border border-zinc-700 hover:bg-zinc-700 disabled:opacity-40 transition-colors"
            >
              Create
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
