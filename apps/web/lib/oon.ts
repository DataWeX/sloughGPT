/**
 * OON API client — single endpoint, action-based.
 *
 * Usage:
 *   import { oon } from '@/lib/oon'
 *   const { board } = await oon.board()
 *   const { card } = await oon.create({ title: 'Task', column: 'todo' })
 *   await oon.move(card.id, 'done')
 *   await oon.sync()
 */

import { apiGet, apiPost, apiDelete } from './http-client'

const ENDPOINT = '/api/oon/card'

async function call<T = any>(action: string, data: Record<string, any> = {}): Promise<T> {
  return apiPost<T>(ENDPOINT, { action, ...data })
}

// ── Types ──────────────────────────────────────────────────────────────

export interface OonCard {
  id: string
  title: string
  description: string
  column: string
  priority: 'low' | 'medium' | 'high' | 'critical'
  tags: string[]
  due_date: string
  assignee: string
  sprint: string
  gh: string
  notes: { id: string; text: string; author: string; created_at: string }[]
  created_at: string
  updated_at: string
  root_hash: string
}

export interface OonBoard {
  name: string
  columns: { name: string; label: string; wip_limit: number; order: number }[]
  cards: OonCard[]
}

export interface OonNote {
  id: string
  title: string
  body: string
  status: string
  tags: string[]
  sprint: string
  gh: string
  created_at: string
  updated_at: string
}

export interface OonTag {
  name: string
  count: number
}

export interface OonStats {
  total_cards: number
  byColumn: Record<string, number>
  columns: number
  total_notes: number
}

export interface OonHashTree {
  root: {
    root: string
    card_id: string
    slot_id: string
    tray: string
    position: number
    placed_at: string
    created_at: string
  }
  notes: {
    note_id: string
    hash_value: string
    root_ref: string
    version: number
    created_at: string
    updated_at: string
  }[]
  history: {
    root_ref: string
    old_hash: string
    new_hash: string
    change_type: string
    note_id: string
    timestamp: string
  }[]
}

// ── API ────────────────────────────────────────────────────────────────

export const oon = {
  /** Get the full board */
  board: () => call<{ board: OonBoard }>('board'),

  /** Get all tags with counts */
  tags: () => call<{ tags: OonTag[] }>('tags'),

  /** Get board stats */
  stats: () => call<{ stats: OonStats }>('stats'),

  /** List all notes */
  list: () => call<{ notes: OonNote[] }>('list'),

  /** Create a new card */
  create: (data: {
    title: string
    description?: string
    column?: string
    priority?: string
    tags?: string[]
    due_date?: string
    assignee?: string
    sprint?: string
    gh?: string
  }) => call<{ card: OonCard }>('create', data),

  /** Update a card */
  update: (id: string, data: Partial<OonCard>) =>
    call<{ card: OonCard }>('update', { id, ...data }),

  /** Delete a card */
  delete: (id: string) => call<{ deleted: boolean }>('delete', { id }),

  /** Move a card to a column */
  move: (id: string, column: string) =>
    call<{ moved: boolean }>('move', { id, column }),

  /** Sync notes to board */
  sync: () => call<{ added: number; moved: number; total: number }>('sync'),

  /** Create a note */
  createNote: (data: { title: string; body?: string; status?: string; tags?: string[]; sprint?: string; gh?: string; card_id?: string }) =>
    call<{ note: OonNote }>('create_note', data),

  /** Update a note */
  updateNote: (id: string, data: Partial<OonNote> & { card_id?: string }) =>
    call<{ note: OonNote }>('update_note', { id, ...data }),

  /** Delete a note */
  deleteNote: (id: string, card_id?: string) =>
    call<{ deleted: boolean }>('delete_note', { id, card_id }),

  /** Get hash tree for a card */
  hashTree: (cardId: string) =>
    apiGet<OonHashTree | null>(`/api/oon/hashtree?cardId=${cardId}`),

  /** Get all hash trees */
  hashTrees: () =>
    apiGet<{ trees: OonHashTree[]; count: number }>('/api/oon/hashtree'),

  /** Create hash tree for a card */
  createHashTree: (cardId: string, cardContent: string, tray: string, position: number = 0) =>
    apiPost<OonHashTree>('/api/oon/hashtree', { cardId, cardContent, tray, position }),

  /** Delete hash tree */
  deleteHashTree: (cardId: string) =>
    apiDelete(`/api/oon/hashtree?cardId=${cardId}`),
}
