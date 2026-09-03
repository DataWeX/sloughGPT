import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import os from 'os'
import { join } from 'path'
import { mkdirSync, writeFileSync, readFileSync, existsSync } from 'fs'

const tmpDir = join(os.tmpdir(), `planner-test-${Date.now()}`)
const tmpKanban = join(tmpDir, '.kanban')
const tmpNotesDir = join(tmpDir, '.dev-notes', 'store')
const tmpBoard = join(tmpKanban, 'board.jsonl')
const tmpNotes = join(tmpNotesDir, 'notes.journal.jsonl')

const originalCwd = process.cwd

beforeEach(() => {
  vi.resetModules()
  mkdirSync(tmpKanban, { recursive: true })
  mkdirSync(tmpNotesDir, { recursive: true })
  process.cwd = () => tmpDir
})

afterEach(() => {
  process.cwd = originalCwd
  try { require('fs').unlinkSync(tmpBoard) } catch {}
  try { require('fs').unlinkSync(tmpNotes) } catch {}
  try { require('fs').rmdirSync(tmpNotesDir) } catch {}
  try { require('fs').rmdirSync(join(tmpDir, '.dev-notes')) } catch {}
  try { require('fs').rmdirSync(tmpKanban) } catch {}
  try { require('fs').rmdirSync(tmpDir) } catch {}
})

const sampleBoard = {
  name: 'Test Board',
  columns: [
    { name: 'todo', wip_limit: 5, order: 0 },
    { name: 'in_progress', wip_limit: 3, order: 1 },
    { name: 'done', wip_limit: 0, order: 2 },
  ],
  cards: [
    {
      id: 'card-1', title: 'Task 1', description: 'Desc 1', column: 'todo',
      priority: 'high', tags: ['frontend', 'bug'], due_date: '', assignee: '',
      sprint: 'sprint-1', gh: '', notes: [],
      created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
    },
    {
      id: 'card-2', title: 'Task 2', description: '', column: 'in_progress',
      priority: 'low', tags: ['backend'], due_date: '', assignee: '',
      sprint: '', gh: '', notes: [],
      created_at: '2026-01-02T00:00:00Z', updated_at: '2026-01-02T00:00:00Z',
    },
  ],
}

function writeSampleBoard() {
  const lines = [
    JSON.stringify({ schema: 'planner/1', name: sampleBoard.name, columns: sampleBoard.columns }),
    ...sampleBoard.cards.map(c => JSON.stringify(c)),
  ]
  writeFileSync(tmpBoard, lines.join('\n') + '\n', 'utf-8')
}

let readBoard: typeof import('./helpers').readBoard
let writeBoard: typeof import('./helpers').writeBoard
let createCard: typeof import('./helpers').createCard
let updateCard: typeof import('./helpers').updateCard
let deleteCard: typeof import('./helpers').deleteCard
let moveCard: typeof import('./helpers').moveCard
let getAllTags: typeof import('./helpers').getAllTags
let getStats: typeof import('./helpers').getStats
let readNotes: typeof import('./helpers').readNotes
let createNote: typeof import('./helpers').createNote
let updateNote: typeof import('./helpers').updateNote
let deleteNote: typeof import('./helpers').deleteNote

beforeEach(async () => {
  vi.resetModules()
  mkdirSync(tmpKanban, { recursive: true })
  mkdirSync(tmpNotesDir, { recursive: true })
  process.cwd = () => tmpDir
  const mod = await import('./helpers')
  readBoard = mod.readBoard
  writeBoard = mod.writeBoard
  createCard = mod.createCard
  updateCard = mod.updateCard
  deleteCard = mod.deleteCard
  moveCard = mod.moveCard
  getAllTags = mod.getAllTags
  getStats = mod.getStats
  readNotes = mod.readNotes
  createNote = mod.createNote
  updateNote = mod.updateNote
  deleteNote = mod.deleteNote
})

describe('readBoard / writeBoard', () => {
  it('returns default board when file does not exist', () => {
    const board = readBoard()
    expect(board.name).toBe('Main')
    expect(board.columns.length).toBe(4)
    expect(board.cards).toEqual([])
  })

  it('reads a board from JSONL file', () => {
    writeSampleBoard()
    const board = readBoard()
    expect(board.name).toBe('Test Board')
    expect(board.cards.length).toBe(2)
    expect(board.cards[0].title).toBe('Task 1')
  })

  it('skips malformed lines', () => {
    writeFileSync(tmpBoard, '{"bad json\n{"id":"c1","title":"OK"}\n', 'utf-8')
    const board = readBoard()
    expect(board.cards.length).toBe(1)
    expect(board.cards[0].title).toBe('OK')
  })
})

describe('createCard', () => {
  beforeEach(() => writeSampleBoard())

  it('creates a card with defaults', () => {
    const card = createCard({ title: 'New Task' })
    expect(card.title).toBe('New Task')
    expect(card.column).toBe('todo')
    expect(card.priority).toBe('medium')
    expect(card.id).toMatch(/^card-/)
    expect(card.notes).toEqual([])
  })

  it('creates a card in a specific column', () => {
    const card = createCard({ title: 'In Progress', column: 'in_progress' })
    expect(card.column).toBe('in_progress')
  })

  it('persists the card to disk', () => {
    createCard({ title: 'Persisted' })
    const board = readBoard()
    expect(board.cards.find(c => c.title === 'Persisted')).toBeTruthy()
  })

  it('includes sprint and gh fields', () => {
    const card = createCard({ title: 'Sprint Task', sprint: 'sprint-5', gh: '#123' })
    expect(card.sprint).toBe('sprint-5')
    expect(card.gh).toBe('#123')
  })
})

describe('updateCard', () => {
  beforeEach(() => writeSampleBoard())

  it('updates card fields', () => {
    const updated = updateCard('card-1', { title: 'Updated Task', priority: 'low' })
    expect(updated).not.toBeNull()
    expect(updated!.title).toBe('Updated Task')
    expect(updated!.priority).toBe('low')
  })

  it('returns null for nonexistent card', () => {
    expect(updateCard('nonexistent', { title: 'X' })).toBeNull()
  })

  it('persists changes', () => {
    updateCard('card-1', { title: 'Persisted Update' })
    const board = readBoard()
    expect(board.cards.find(c => c.id === 'card-1')!.title).toBe('Persisted Update')
  })
})

describe('deleteCard', () => {
  beforeEach(() => writeSampleBoard())

  it('deletes an existing card', () => {
    expect(deleteCard('card-1')).toBe(true)
    const board = readBoard()
    expect(board.cards.find(c => c.id === 'card-1')).toBeUndefined()
  })

  it('returns false for nonexistent card', () => {
    expect(deleteCard('nonexistent')).toBe(false)
  })
})

describe('moveCard', () => {
  beforeEach(() => writeSampleBoard())

  it('moves a card to a new column', () => {
    expect(moveCard('card-1', 'done')).toBe(true)
    const board = readBoard()
    expect(board.cards.find(c => c.id === 'card-1')!.column).toBe('done')
  })

  it('returns false for nonexistent card', () => {
    expect(moveCard('nonexistent', 'done')).toBe(false)
  })
})

describe('getAllTags', () => {
  beforeEach(() => writeSampleBoard())

  it('returns tags sorted by count', () => {
    const tags = getAllTags()
    expect(tags.length).toBe(3)
    expect(tags[0].name).toBe('frontend')
    expect(tags[0].count).toBe(1)
  })
})

describe('getStats', () => {
  beforeEach(() => writeSampleBoard())

  it('returns board and note stats', () => {
    const stats = getStats()
    expect(stats.total_cards).toBe(2)
    expect(stats.columns).toBe(3)
    expect(stats.byColumn.todo).toBe(1)
    expect(stats.byColumn.in_progress).toBe(1)
  })
})

describe('readNotes / writeNotes', () => {
  it('returns empty array when file does not exist', () => {
    const notes = readNotes()
    expect(notes).toEqual([])
  })

  it('reads notes from JSONL file', () => {
    const note = {
      id: 'n1', title: 'Test Note', body: 'body', status: 'open',
      tags: ['tag1'], sprint: '', gh: '',
      created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
    }
    writeFileSync(tmpNotes, JSON.stringify(note) + '\n', 'utf-8')
    const notes = readNotes()
    expect(notes.length).toBe(1)
    expect(notes[0].title).toBe('Test Note')
  })
})

describe('createNote', () => {
  it('creates a note with defaults', () => {
    const note = createNote({ title: 'New Note' })
    expect(note.title).toBe('New Note')
    expect(note.status).toBe('open')
    expect(note.id).toBeTruthy()
  })

  it('persists the note', () => {
    createNote({ title: 'Persisted Note' })
    const notes = readNotes()
    expect(notes.find(n => n.title === 'Persisted Note')).toBeTruthy()
  })
})

describe('updateNote', () => {
  it('updates an existing note', () => {
    const note = createNote({ title: 'Original' })
    const updated = updateNote(note.id, { title: 'Updated', status: 'done' })
    expect(updated).not.toBeNull()
    expect(updated!.title).toBe('Updated')
    expect(updated!.status).toBe('done')
  })

  it('returns null for nonexistent note', () => {
    expect(updateNote('nonexistent', { title: 'X' })).toBeNull()
  })
})

describe('deleteNote', () => {
  it('deletes an existing note', () => {
    const note = createNote({ title: 'To Delete' })
    expect(deleteNote(note.id)).toBe(true)
    expect(readNotes().find(n => n.id === note.id)).toBeUndefined()
  })

  it('returns false for nonexistent note', () => {
    expect(deleteNote('nonexistent')).toBe(false)
  })
})
