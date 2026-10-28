/**
 * Browser-side fetch wrappers for the /api/planner/ routes.
 *
 * Reads/writes .kanban/board.jsonl and .dev-notes/store/notes.journal.jsonl
 * via Next.js API routes.
 */

import { apiGet, apiPost, apiPut, apiDelete } from './http-client'
import type { Card, Board, Note, TagCount, Stats, HashTree } from '@/components/planner/types'

// ── Board ───────────────────────────────────────────────────────────────

export async function fetchBoard(): Promise<{ board: Board }> {
  return apiGet<{ board: Board }>('/api/planner/board')
}

export async function moveCard(payload: { card_id: string; column: string }): Promise<void> {
  await apiPost('/api/planner/board/move', payload)
}

export async function createCard(payload: {
  title: string
  description?: string
  column?: string
  priority?: string
  tags?: string[]
  due_date?: string
  assignee?: string
  sprint?: string
  gh?: string
}): Promise<{ card: Card }> {
  return apiPost<{ card: Card }>('/api/planner/board/cards', payload)
}

export async function updateCard(
  id: string,
  payload: Partial<
    Pick<
      Card,
      | 'title'
      | 'description'
      | 'priority'
      | 'tags'
      | 'due_date'
      | 'assignee'
      | 'column'
      | 'sprint'
      | 'gh'
    >
  >,
): Promise<{ card: Card }> {
  return apiPut<{ card: Card }>(`/api/planner/board/cards/${id}`, payload)
}

export async function deleteCard(id: string): Promise<void> {
  await apiDelete(`/api/planner/board/cards/${id}`)
}

// ── Tags & Stats ────────────────────────────────────────────────────────

export async function fetchTags(): Promise<{ tags: TagCount[] }> {
  return apiGet<{ tags: TagCount[] }>('/api/planner/tags')
}

export async function fetchStats(): Promise<{ stats: Stats }> {
  return apiGet<{ stats: Stats }>('/api/planner/stats')
}

// ── Notes ───────────────────────────────────────────────────────────────

export async function fetchNotes(): Promise<{ notes: Note[] }> {
  return apiGet<{ notes: Note[] }>('/api/planner/notes')
}

export async function createNote(payload: {
  title: string
  body?: string
  status?: string
  tags?: string[]
  sprint?: string
  gh?: string
}): Promise<{ note: Note }> {
  return apiPost<{ note: Note }>('/api/planner/notes', payload)
}

export async function updateNote(
  id: string,
  payload: Partial<Pick<Note, 'title' | 'body' | 'status' | 'tags' | 'sprint' | 'gh'>>,
): Promise<{ note: Note }> {
  return apiPut<{ note: Note }>(`/api/planner/notes/${id}`, payload)
}

export async function deleteNote(id: string): Promise<void> {
  await apiDelete(`/api/planner/notes/${id}`)
}

// ── Sync ────────────────────────────────────────────────────────────────

export async function syncNotes(): Promise<{ added: number; updated: number; total: number }> {
  return apiPost<{ added: number; updated: number; total: number }>('/api/planner/sync', {})
}

// ── Hash tree ───────────────────────────────────────────────────────────

export async function fetchHashTree(cardId: string): Promise<HashTree | null> {
  try {
    return await apiGet<HashTree>('/api/planner/hashtree', { cardId })
  } catch {
    return null
  }
}

export async function fetchHashTrees(): Promise<{ trees: HashTree[]; count: number }> {
  return apiGet<{ trees: HashTree[]; count: number }>('/api/planner/hashtree')
}

// ── Card notes (writes card.notes + hash tree in one save) ──────────────

export async function addCardNote(
  cardId: string,
  text: string,
  author?: string,
): Promise<{ card: Card }> {
  return apiPost<{ card: Card }>(`/api/planner/board/cards/${cardId}/notes`, {
    action: 'add',
    text,
    author,
  })
}

export async function updateCardNote(
  cardId: string,
  noteId: string,
  text: string,
): Promise<{ card: Card }> {
  return apiPost<{ card: Card }>(`/api/planner/board/cards/${cardId}/notes`, {
    action: 'update',
    note_id: noteId,
    text,
  })
}

export async function deleteCardNote(cardId: string, noteId: string): Promise<{ card: Card }> {
  return apiPost<{ card: Card }>(`/api/planner/board/cards/${cardId}/notes`, {
    action: 'delete',
    note_id: noteId,
  })
}

// ── Transition (engine-validated) ───────────────────────────────────────

export interface TransitionResult {
  ok: boolean
  card_id: string
  from_state: string
  to_state: string
  error: string
  wip_count: number
  wip_limit: number
  event: {
    type: string
    card_id: string
    from_state: string
    to_state: string
    actor: string
    timestamp: string
    metadata: Record<string, unknown>
  } | null
}

export async function transitionCard(payload: {
  card_id: string
  to_state: string
  actor?: string
  force?: boolean
}): Promise<TransitionResult> {
  const res = await fetch('/api/planner/transition', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return res.json()
}
