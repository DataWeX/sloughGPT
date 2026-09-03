'use client'

export interface Card {
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
  notes: CardNote[]
  created_at: string
  updated_at: string
  root_hash: string
}

export interface CardNote {
  id: string
  text: string
  author: string
  created_at: string
}

export interface Column {
  name: string
  label: string
  wip_limit: number
  order: number
}

export interface Board {
  columns: Column[]
  cards: Card[]
}

export interface Note {
  id: string
  title: string
  created_at: string
  updated_at: string
  tags: string[]
  status: string
  sprint: string
  gh: string
  body: string
}

export interface TagCount {
  name: string
  count: number
}

export interface Stats {
  total_cards: number
  by_column: Record<string, number>
  by_priority: Record<string, number>
  total_notes: number
}

export interface DragState {
  cardId: string
  fromColumn: string
}

export interface InputState {
  drag: DragState | null
  hover: string | null
  selected: Card | null
}

export interface SyncState {
  status: 'idle' | 'syncing' | 'error'
  lastSync: string | null
  error: string | null
}

export interface Scene {
  board: Board
  notes: Note[]
}

export const PRIORITY_CLASSES: Record<string, string> = {
  low: 'bg-success/15 text-success',
  medium: 'bg-warning/15 text-warning',
  high: 'bg-accent/20 text-accent',
  critical: 'bg-destructive/15 text-destructive',
}

export interface HashTree {
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
  commits: {
    commit_hash: string
    parent_hash: string
    root_ref: string
    changes: any[]
    color: string
    pixel: string
    timestamp: string
  }[]
}

export const COLUMN_LABELS: Record<string, string> = {
  todo: 'To Do',
  in_progress: 'In Progress',
  review: 'Review',
  done: 'Done',
}
