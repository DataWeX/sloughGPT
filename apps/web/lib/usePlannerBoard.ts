'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import type { Card, Board, TagCount } from '@/components/planner/types'
import * as planner from '@/lib/planner-client'

interface BoardState {
  board: Board | null
  tags: TagCount[]
  loading: boolean
  error: string | null
}

export function usePlannerBoard(pollMs = 500) {
  const [state, setState] = useState<BoardState>({
    board: null,
    tags: [],
    loading: true,
    error: null,
  })
  const hashRef = useRef<string>('')
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const activeRef = useRef(true)

  const fetchBoard = useCallback(async (silent = false) => {
    try {
      if (!silent) setState((s) => ({ ...s, loading: true, error: null }))
      const [boardRes, tagsRes] = await Promise.all([planner.fetchBoard(), planner.fetchTags()])
      const newHash = JSON.stringify(
        boardRes.board.cards.map((c: Card) => [c.id, c.column, c.updated_at]),
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

  const forceRefresh = useCallback(async () => {
    hashRef.current = ''
    await fetchBoard(false)
  }, [fetchBoard])

  // Initial load
  useEffect(() => {
    activeRef.current = true
    fetchBoard(false)
    return () => {
      activeRef.current = false
    }
  }, [fetchBoard])

  // Poll
  useEffect(() => {
    if (pollMs <= 0) return
    timerRef.current = setInterval(() => {
      if (activeRef.current) fetchBoard(true)
    }, pollMs)
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [pollMs, fetchBoard])

  // Optimistic update — update card locally before server confirms
  const optimisticUpdate = useCallback((cardId: string, patch: Partial<Card>) => {
    setState((s) => {
      if (!s.board) return s
      const cards = s.board.cards.map((c) => (c.id === cardId ? { ...c, ...patch } : c))
      return { ...s, board: { ...s.board, cards } }
    })
  }, [])

  // Optimistic move — move card column locally
  const optimisticMove = useCallback(
    (cardId: string, column: string) => {
      optimisticUpdate(cardId, { column, updated_at: new Date().toISOString() })
    },
    [optimisticUpdate],
  )

  // Optimistic add — prepend card locally
  const optimisticAdd = useCallback((card: Card) => {
    setState((s) => {
      if (!s.board) return s
      return { ...s, board: { ...s.board, cards: [...s.board.cards, card] } }
    })
  }, [])

  // Optimistic delete — remove card locally
  const optimisticDelete = useCallback((cardId: string) => {
    setState((s) => {
      if (!s.board) return s
      return { ...s, board: { ...s.board, cards: s.board.cards.filter((c) => c.id !== cardId) } }
    })
  }, [])

  return {
    ...state,
    refresh: forceRefresh,
    optimisticUpdate,
    optimisticMove,
    optimisticAdd,
    optimisticDelete,
  }
}
