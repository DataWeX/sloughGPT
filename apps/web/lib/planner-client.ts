/**
 * Browser-side fetch wrappers — delegates to oon.ts.
 *
 * Kept for backward compatibility with existing imports.
 * New code should import { oon } from '@/lib/oon' directly.
 */

import { oon } from './oon'
import type { Card, Board, Note, TagCount, Stats } from '@/components/planner/types'

export async function fetchBoard(): Promise<{ board: Board }> {
  return oon.board() as Promise<{ board: Board }>
}

export async function moveCard(payload: { card_id: string; column: string }): Promise<void> {
  await oon.move(payload.card_id, payload.column)
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
  return oon.create(payload) as Promise<{ card: Card }>
}

export async function updateCard(
  id: string,
  payload: Partial<Pick<Card, 'title' | 'description' | 'priority' | 'tags' | 'due_date' | 'assignee' | 'column' | 'sprint' | 'gh'>>,
): Promise<{ card: Card }> {
  return oon.update(id, payload) as Promise<{ card: Card }>
}

export async function deleteCard(id: string): Promise<void> {
  await oon.delete(id)
}

export async function fetchTags(): Promise<{ tags: TagCount[] }> {
  return oon.tags() as Promise<{ tags: TagCount[] }>
}

export async function fetchNotes(): Promise<{ notes: Note[] }> {
  return oon.list() as Promise<{ notes: Note[] }>
}

export async function createNote(payload: {
  title: string
  body?: string
  status?: string
  tags?: string[]
  sprint?: string
  gh?: string
}): Promise<{ note: Note }> {
  return oon.createNote(payload) as Promise<{ note: Note }>
}

export async function updateNote(
  id: string,
  payload: Partial<Pick<Note, 'title' | 'body' | 'status' | 'tags' | 'sprint' | 'gh'>>,
): Promise<{ note: Note }> {
  return oon.updateNote(id, payload) as Promise<{ note: Note }>
}

export async function deleteNote(id: string): Promise<void> {
  await oon.deleteNote(id)
}

export async function fetchStats(): Promise<{ stats: Stats }> {
  const { stats } = await oon.stats()
  return { stats: stats as unknown as Stats }
}

export async function syncNotes(): Promise<{ added: number; updated: number; total: number }> {
  const { added, moved, total } = await oon.sync()
  return { added, updated: moved, total }
}
