'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import type { Card, Board, TagCount, DragState } from '@/components/planner/types'
import * as planner from '@/lib/planner-client'

interface BoardState {
  board: Board | null
  tags: TagCount[]
  loading: boolean
  error: string | null
}

interface UndoEntry {
  type: 'move' | 'delete' | 'update'
  cardId: string
  prev?: Partial<Card>
  card?: Card
}

const DEFAULT_COLUMNS = [
  { name: 'todo', label: 'To Do', wip_limit: 5, order: 0 },
  { name: 'wip', label: 'In Progress', wip_limit: 3, order: 1 },
  { name: 'review', label: 'Review', wip_limit: 2, order: 2 },
  { name: 'done', label: 'Done', wip_limit: 0, order: 3 },
]

const PRIORITY_ORDER: Record<string, number> = { critical: 0, high: 1, medium: 2, low: 3 }

export function usePlannerBoard(pollMs = 500) {
  const [state, setState] = useState<BoardState>({
    board: null,
    tags: [],
    loading: true,
    error: null,
  })
  const [drag, setDrag] = useState<DragState | null>(null)
  const [undoStack, setUndoStack] = useState<UndoEntry[]>([])
  const hashRef = useRef<string>('')
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const activeRef = useRef(true)

  const fetchBoard = useCallback(async (silent = false) => {
    try {
      if (!silent) setState((s) => ({ ...s, loading: true, error: null }))
      const [boardRes, tagsRes] = await Promise.all([planner.fetchBoard(), planner.fetchTags()])
      const newHash = JSON.stringify(
        boardRes.board.cards.map((c: Card) => [c.id, c.column, c.updated_at, c.root_hash]),
      )
      if (newHash !== hashRef.current) {
        hashRef.current = newHash
        setState({ board: boardRes.board, tags: tagsRes.tags, loading: false, error: null })
      } else if (!silent) {
        setState((s) => ({ ...s, loading: false }))
      }
    } catch (err) {
      setState((s) => ({
        ...s,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to load board',
      }))
    }
  }, [])

  const refresh = useCallback(async () => {
    hashRef.current = ''
    await fetchBoard(false)
  }, [fetchBoard])

  useEffect(() => {
    activeRef.current = true
    fetchBoard(false)
    return () => {
      activeRef.current = false
    }
  }, [fetchBoard])

  useEffect(() => {
    if (pollMs <= 0) return
    timerRef.current = setInterval(() => {
      if (activeRef.current) fetchBoard(true)
    }, pollMs)
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [pollMs, fetchBoard])

  // ── Optimistic primitives ──────────────────────────────────────────

  const patchCard = useCallback((cardId: string, patch: Partial<Card>) => {
    setState((s) => {
      if (!s.board) return s
      const cards = s.board.cards.map((c) => (c.id === cardId ? { ...c, ...patch } : c))
      return { ...s, board: { ...s.board, cards } }
    })
  }, [])

  const pushUndo = useCallback((entry: UndoEntry) => {
    setUndoStack((prev) => [...prev.slice(-19), entry])
  }, [])

  // ── Actions ────────────────────────────────────────────────────────

  const moveCard = useCallback(
    async (cardId: string, toColumn: string) => {
      const card = state.board?.cards.find((c) => c.id === cardId)
      if (!card || card.column === toColumn) return

      // WIP enforcement
      if (toColumn !== 'todo' && toColumn !== 'done') {
        const col = state.board?.columns.find((c) => c.name === toColumn)
        const count = state.board?.cards.filter((c) => c.column === toColumn).length ?? 0
        if (col && col.wip_limit > 0 && count >= col.wip_limit) return
      }

      pushUndo({ type: 'move', cardId, prev: { column: card.column } })
      patchCard(cardId, { column: toColumn, updated_at: new Date().toISOString() })
      try {
        await planner.moveCard({ card_id: cardId, column: toColumn })
      } catch {
        refresh()
      }
    },
    [state.board, patchCard, pushUndo, refresh],
  )

  const createCard = useCallback(
    async (data: { title: string; column: string; priority: string; description: string }) => {
      try {
        const { card } = await planner.createCard(data)
        setState((s) => {
          if (!s.board) return s
          return { ...s, board: { ...s.board, cards: [...s.board.cards, card] } }
        })
        return card
      } catch (err) {
        refresh()
        throw err
      }
    },
    [refresh],
  )

  const updateCard = useCallback(
    async (cardId: string, data: Partial<Card>) => {
      const card = state.board?.cards.find((c) => c.id === cardId)
      if (!card) return

      pushUndo({ type: 'update', cardId, prev: card })
      patchCard(cardId, { ...data, updated_at: new Date().toISOString() })
      try {
        await planner.updateCard(cardId, data)
      } catch {
        refresh()
      }
    },
    [state.board, patchCard, pushUndo, refresh],
  )

  const deleteCard = useCallback(
    async (cardId: string) => {
      const card = state.board?.cards.find((c) => c.id === cardId)
      if (!card) return

      pushUndo({ type: 'delete', cardId, card })
      setState((s) => {
        if (!s.board) return s
        return { ...s, board: { ...s.board, cards: s.board.cards.filter((c) => c.id !== cardId) } }
      })
      try {
        await planner.deleteCard(cardId)
      } catch {
        refresh()
      }
    },
    [state.board, pushUndo, refresh],
  )

  const undo = useCallback(async () => {
    const entry = undoStack[undoStack.length - 1]
    if (!entry) return

    setUndoStack((prev) => prev.slice(0, -1))

    if (entry.type === 'move' && entry.prev?.column) {
      patchCard(entry.cardId, entry.prev)
      await planner.moveCard({ card_id: entry.cardId, column: entry.prev.column })
    } else if (entry.type === 'delete' && entry.card) {
      setState((s) => {
        if (!s.board) return s
        return { ...s, board: { ...s.board, cards: [...s.board.cards, entry.card!] } }
      })
      await planner.createCard({
        title: entry.card.title,
        description: entry.card.description,
        column: entry.card.column,
        priority: entry.card.priority,
        tags: entry.card.tags,
      })
    } else if (entry.type === 'update' && entry.prev) {
      patchCard(entry.cardId, entry.prev)
      await planner.updateCard(entry.cardId, entry.prev)
    }
  }, [undoStack, patchCard])

  // ── Sorting ────────────────────────────────────────────────────────

  const getSortedCards = useCallback(
    (column: string): Card[] => {
      const cards = state.board?.cards.filter((c) => c.column === column) ?? []
      return [...cards].sort((a, b) => {
        const pa = PRIORITY_ORDER[a.priority] ?? 4
        const pb = PRIORITY_ORDER[b.priority] ?? 4
        if (pa !== pb) return pa - pb
        return new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
      })
    },
    [state.board],
  )

  const isOverWip = useCallback(
    (column: string): boolean => {
      const col = state.board?.columns.find((c) => c.name === column)
      if (!col || col.wip_limit <= 0) return false
      const count = state.board?.cards.filter((c) => c.column === column).length ?? 0
      return count > col.wip_limit
    },
    [state.board],
  )

  const canDrop = useCallback(
    (column: string): boolean => {
      const col = state.board?.columns.find((c) => c.name === column)
      if (!col || col.wip_limit <= 0) return true
      const count = state.board?.cards.filter((c) => c.column === column).length ?? 0
      return count < col.wip_limit
    },
    [state.board],
  )

  const syncNotes = useCallback(async (): Promise<{
    added: number
    updated: number
    total: number
  }> => {
    const result = await planner.syncNotes()
    await refresh()
    return result
  }, [refresh])

  return {
    ...state,
    columns: state.board?.columns ?? DEFAULT_COLUMNS,
    drag,
    setDrag,
    canUndo: undoStack.length > 0,
    refresh,
    moveCard,
    createCard,
    updateCard,
    deleteCard,
    undo,
    syncNotes,
    getSortedCards,
    isOverWip,
    canDrop,
  }
}
