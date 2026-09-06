/**
 * Shared helpers for reading/writing planner data.
 *
 * Board: <repo>/.kanban/board.jsonl
 * Notes: <repo>/.dev-notes/store/notes.journal.jsonl
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs'
import { join } from 'path'

// ── Paths ──────────────────────────────────────────────────────────────

function findRepoRoot(): string {
  let dir = process.cwd()
  while (dir !== '/') {
    if (existsSync(join(dir, '.kanban', 'board.jsonl')) && !dir.includes('.next')) return dir
    dir = join(dir, '..')
  }
  dir = process.cwd()
  while (dir !== '/') {
    if (existsSync(join(dir, '.kanban', 'board.json')) && !dir.includes('.next')) return dir
    dir = join(dir, '..')
  }
  return process.cwd()
}

const REPO_ROOT = findRepoRoot()

export function boardPath(): string {
  return join(REPO_ROOT, '.kanban', 'board.jsonl')
}

export function notesPath(): string {
  return join(REPO_ROOT, '.dev-notes', 'store', 'notes.journal.jsonl')
}

// ── Types ──────────────────────────────────────────────────────────────

export interface BoardColumn {
  name: string
  wip_limit: number
  order: number
}

export interface BoardCard {
  id: string
  title: string
  description: string
  column: string
  priority: string
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

export interface Board {
  name: string
  columns: BoardColumn[]
  cards: BoardCard[]
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

// ── Read / Write Board ────────────────────────────────────────────────

export function readBoard(): Board {
  const bp = boardPath()
  if (!existsSync(bp)) {
    return {
      name: 'Main',
      columns: [
        { name: 'todo', wip_limit: 5, order: 0 },
        { name: 'in_progress', wip_limit: 3, order: 1 },
        { name: 'review', wip_limit: 2, order: 2 },
        { name: 'done', wip_limit: 0, order: 3 },
      ],
      cards: [],
    }
  }
  const raw = readFileSync(bp, 'utf-8')
  const lines = raw.split('\n').filter(Boolean)
  const board: Board = {
    name: 'Main',
    columns: [
      { name: 'todo', wip_limit: 5, order: 0 },
      { name: 'in_progress', wip_limit: 3, order: 1 },
      { name: 'review', wip_limit: 2, order: 2 },
      { name: 'done', wip_limit: 0, order: 3 },
    ],
    cards: [],
  }
  for (const line of lines) {
    try {
      const obj = JSON.parse(line)
      if (obj.schema === 'planner/1' && obj.columns) {
        board.name = obj.name || 'Main'
        board.columns = obj.columns
      } else if (obj.id && obj.title) {
        board.cards.push(obj as BoardCard)
      }
    } catch {
      // skip malformed lines
    }
  }
  return board
}

export function writeBoard(board: Board): void {
  const bp = boardPath()
  const dir = join(REPO_ROOT, '.kanban')
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })

  const lines: string[] = []
  lines.push(JSON.stringify({ schema: 'planner/1', name: board.name, columns: board.columns }))
  for (const card of board.cards) {
    lines.push(JSON.stringify(card))
  }
  writeFileSync(bp, lines.join('\n') + '\n', 'utf-8')
}

// ── Read / Write Notes ────────────────────────────────────────────────

export function readNotes(): Note[] {
  const np = notesPath()
  if (!existsSync(np)) return []
  const raw = readFileSync(np, 'utf-8')
  const lines = raw.split('\n').filter(Boolean)
  const notes: Note[] = []
  for (const line of lines) {
    try {
      const obj = JSON.parse(line)
      if (obj.id && obj.title) {
        notes.push(obj as Note)
      }
    } catch {
      // skip malformed lines
    }
  }
  return notes
}

export function writeNotes(notes: Note[]): void {
  const np = notesPath()
  const dir = join(REPO_ROOT, '.dev-notes', 'store')
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })

  const lines = notes.map((note) => JSON.stringify(note))
  writeFileSync(np, lines.join('\n') + '\n', 'utf-8')
}

// ── Card Operations ────────────────────────────────────────────────────

export function moveCard(cardId: string, column: string): boolean {
  const board = readBoard()
  const card = board.cards.find(c => c.id === cardId)
  if (!card) return false
  card.column = column
  card.updated_at = new Date().toISOString()
  writeBoard(board)
  return true
}

export function createCard(data: {
  title: string
  description?: string
  priority?: string
  tags?: string[]
  due_date?: string
  assignee?: string
  sprint?: string
  gh?: string
  column?: string
}): BoardCard {
  const board = readBoard()
  const now = new Date().toISOString()
  const card: BoardCard = {
    id: `card-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    title: data.title,
    description: data.description || '',
    column: data.column || 'todo',
    priority: data.priority || 'medium',
    tags: data.tags || [],
    due_date: data.due_date || '',
    assignee: data.assignee || '',
    sprint: data.sprint || '',
    gh: data.gh || '',
    notes: [],
    created_at: now,
    updated_at: now,
    root_hash: '',
  }
  board.cards.push(card)
  writeBoard(board)
  return card
}

export function updateCard(
  id: string,
  data: Partial<Pick<BoardCard, 'title' | 'description' | 'priority' | 'tags' | 'due_date' | 'assignee' | 'column' | 'sprint' | 'gh'>>,
): BoardCard | null {
  const board = readBoard()
  const card = board.cards.find(c => c.id === id)
  if (!card) return null
  Object.assign(card, data, { updated_at: new Date().toISOString() })
  writeBoard(board)
  return card
}

export function deleteCard(id: string): boolean {
  const board = readBoard()
  const idx = board.cards.findIndex(c => c.id === id)
  if (idx === -1) return false
  board.cards.splice(idx, 1)
  writeBoard(board)
  return true
}

// ── Note Operations ────────────────────────────────────────────────────

export function createNote(data: {
  title: string
  body?: string
  status?: string
  tags?: string[]
  sprint?: string
  gh?: string
}): Note {
  const notes = readNotes()
  const now = new Date().toISOString()
  const note: Note = {
    id: `${now.replace(/[-:T]/g, '').slice(0, 15)}_${data.title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')}`,
    title: data.title,
    body: data.body || '',
    status: data.status || 'open',
    tags: data.tags || [],
    sprint: data.sprint || '',
    gh: data.gh || '',
    created_at: now,
    updated_at: now,
  }
  notes.push(note)
  writeNotes(notes)
  return note
}

export function updateNote(
  id: string,
  data: Partial<Pick<Note, 'title' | 'body' | 'status' | 'tags' | 'sprint' | 'gh'>>,
): Note | null {
  const notes = readNotes()
  const note = notes.find(n => n.id === id)
  if (!note) return null
  Object.assign(note, data, { updated_at: new Date().toISOString() })
  writeNotes(notes)
  return note
}

export function deleteNote(id: string): boolean {
  const notes = readNotes()
  const idx = notes.findIndex(n => n.id === id)
  if (idx === -1) return false
  notes.splice(idx, 1)
  writeNotes(notes)
  return true
}

// ── Tags & Stats ───────────────────────────────────────────────────────

export function getAllTags(): { name: string; count: number }[] {
  const board = readBoard()
  const tagMap = new Map<string, number>()
  for (const card of board.cards) {
    for (const tag of (card.tags || [])) {
      tagMap.set(tag, (tagMap.get(tag) || 0) + 1)
    }
  }
  return Array.from(tagMap.entries())
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)
}

export function getStats() {
  const board = readBoard()
  const notes = readNotes()
  const byColumn: Record<string, number> = {}
  for (const card of board.cards) {
    byColumn[card.column] = (byColumn[card.column] || 0) + 1
  }
  return {
    total_cards: board.cards.length,
    byColumn,
    columns: board.columns.length,
    total_notes: notes.length,
  }
}

// ── Hash Tree Helpers ──────────────────────────────────────────────────

import { createHash } from 'crypto'

function sha256(data: string): string {
  return createHash('sha256').update(data, 'utf-8').digest('hex')
}

export interface HashTreeData {
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

function hashtreesDir(): string {
  return join(REPO_ROOT, '.planner', 'hashtrees')
}

function hashtreesFile(): string {
  return join(hashtreesDir(), 'trees.jsonl')
}

export function readHashTrees(): Map<string, HashTreeData> {
  const dir = hashtreesDir()
  const file = hashtreesFile()
  if (!existsSync(file)) return new Map()
  const raw = readFileSync(file, 'utf-8')
  const trees = new Map<string, HashTreeData>()
  for (const line of raw.split('\n').filter(Boolean)) {
    try {
      const tree = JSON.parse(line) as HashTreeData
      if (tree.root?.card_id) {
        trees.set(tree.root.card_id, tree)
      }
    } catch {}
  }
  return trees
}

export function writeHashTrees(trees: Map<string, HashTreeData>): void {
  const dir = hashtreesDir()
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  const lines = Array.from(trees.values()).map(t => JSON.stringify(t))
  writeFileSync(hashtreesFile(), lines.join('\n') + '\n', 'utf-8')
}

export function getHashTree(cardId: string): HashTreeData | null {
  return readHashTrees().get(cardId) || null
}

export function createHashTree(cardId: string, cardContent: string, tray: string, position: number): HashTreeData {
  const now = new Date().toISOString()
  const rootHash = sha256(`${cardContent}:${tray}:${position}:${now}`)
  const slotId = sha256(`${cardId}:${tray}:${position}`)
  return {
    root: {
      root: rootHash,
      card_id: cardId,
      slot_id: slotId,
      tray,
      position,
      placed_at: now,
      created_at: now,
    },
    notes: [],
    history: [],
  }
}

export function addNoteToTree(tree: HashTreeData, noteId: string, noteContent: string): void {
  const now = new Date().toISOString()
  const existing = tree.notes.find(n => n.note_id === noteId)
  if (existing) {
    const oldHash = existing.hash_value
    existing.hash_value = sha256(`${tree.root.root}:${noteContent}`)
    existing.version += 1
    existing.updated_at = now
    tree.history.push({
      root_ref: tree.root.root,
      old_hash: oldHash,
      new_hash: existing.hash_value,
      change_type: 'note_edit',
      note_id: noteId,
      timestamp: now,
    })
  } else {
    const hashValue = sha256(`${tree.root.root}:${noteContent}`)
    tree.notes.push({
      note_id: noteId,
      hash_value: hashValue,
      root_ref: tree.root.root,
      version: 1,
      created_at: now,
      updated_at: now,
    })
    tree.history.push({
      root_ref: tree.root.root,
      old_hash: '',
      new_hash: hashValue,
      change_type: 'note_add',
      note_id: noteId,
      timestamp: now,
    })
  }
}

export function removeNoteFromTree(tree: HashTreeData, noteId: string): boolean {
  const now = new Date().toISOString()
  const idx = tree.notes.findIndex(n => n.note_id === noteId)
  if (idx === -1) return false
  const [removed] = tree.notes.splice(idx, 1)
  tree.history.push({
    root_ref: tree.root.root,
    old_hash: removed.hash_value,
    new_hash: '',
    change_type: 'note_delete',
    note_id: noteId,
    timestamp: now,
  })
  return true
}

const COMMIT_COLORS = [
  '#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#1abc9c',
  '#3498db', '#9b59b6', '#e91e63', '#00bcd4', '#8bc34a',
]

export function rehashTree(tree: HashTreeData): void {
  const now = new Date().toISOString()
  const parentHash = tree.history.length > 0
    ? tree.history[tree.history.length - 1].new_hash
    : tree.root.root

  // New hash = parent + all note hashes
  const noteHashes = tree.notes.map(n => n.hash_value).sort().join(':')
  const newHash = sha256(`${parentHash}:${noteHashes}:${now}`)

  // Append to history — the chain
  tree.history.push({
    root_ref: tree.root.root,
    old_hash: parentHash,
    new_hash: newHash,
    change_type: 'rehash',
    note_id: '',
    timestamp: now,
  })
}
