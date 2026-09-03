import { describe, it, expect, vi, beforeEach } from 'vitest'

const mockOon = vi.hoisted(() => ({
  board: vi.fn(),
  move: vi.fn(),
  create: vi.fn(),
  update: vi.fn(),
  delete: vi.fn(),
  tags: vi.fn(),
  list: vi.fn(),
  createNote: vi.fn(),
  updateNote: vi.fn(),
  deleteNote: vi.fn(),
  stats: vi.fn(),
  sync: vi.fn(),
}))

vi.mock('./oon', () => ({
  oon: mockOon,
}))

import {
  fetchBoard,
  moveCard,
  createCard,
  updateCard,
  deleteCard,
  fetchTags,
  fetchNotes,
  createNote,
  updateNote,
  deleteNote,
  fetchStats,
  syncNotes,
} from './planner-client'

beforeEach(() => {
  vi.clearAllMocks()
})

describe('fetchBoard', () => {
  it('delegates to oon.board()', async () => {
    const board = { columns: [], cards: [] }
    mockOon.board.mockResolvedValue({ board })
    const result = await fetchBoard()
    expect(result).toEqual({ board })
    expect(mockOon.board).toHaveBeenCalledOnce()
  })

  it('throws on failure', async () => {
    mockOon.board.mockRejectedValue(new Error('oon.board failed'))
    await expect(fetchBoard()).rejects.toThrow('oon.board failed')
  })
})

describe('moveCard', () => {
  it('delegates to oon.move()', async () => {
    mockOon.move.mockResolvedValue(undefined)
    await moveCard({ card_id: 'c1', column: 'done' })
    expect(mockOon.move).toHaveBeenCalledWith('c1', 'done')
  })

  it('throws on failure', async () => {
    mockOon.move.mockRejectedValue(new Error('oon.move failed'))
    await expect(moveCard({ card_id: 'c1', column: 'done' })).rejects.toThrow('oon.move failed')
  })
})

describe('createCard', () => {
  it('delegates to oon.create()', async () => {
    const card = { id: 'c1', title: 'Test' }
    mockOon.create.mockResolvedValue({ card })
    const result = await createCard({ title: 'Test', column: 'todo', priority: 'high' })
    expect(result).toEqual({ card })
    expect(mockOon.create).toHaveBeenCalledWith({ title: 'Test', column: 'todo', priority: 'high' })
  })

  it('throws on failure', async () => {
    mockOon.create.mockRejectedValue(new Error('oon.create failed'))
    await expect(createCard({ title: 'Test' })).rejects.toThrow('oon.create failed')
  })
})

describe('updateCard', () => {
  it('delegates to oon.update()', async () => {
    const card = { id: 'c1', title: 'Updated' }
    mockOon.update.mockResolvedValue({ card })
    const result = await updateCard('c1', { title: 'Updated' })
    expect(result).toEqual({ card })
    expect(mockOon.update).toHaveBeenCalledWith('c1', { title: 'Updated' })
  })

  it('throws on failure', async () => {
    mockOon.update.mockRejectedValue(new Error('oon.update failed'))
    await expect(updateCard('c1', { title: 'X' })).rejects.toThrow('oon.update failed')
  })
})

describe('deleteCard', () => {
  it('delegates to oon.delete()', async () => {
    mockOon.delete.mockResolvedValue(undefined)
    await deleteCard('c1')
    expect(mockOon.delete).toHaveBeenCalledWith('c1')
  })

  it('throws on failure', async () => {
    mockOon.delete.mockRejectedValue(new Error('oon.delete failed'))
    await expect(deleteCard('c1')).rejects.toThrow('oon.delete failed')
  })
})

describe('fetchTags', () => {
  it('delegates to oon.tags()', async () => {
    const tags = [{ name: 'bug', count: 5 }]
    mockOon.tags.mockResolvedValue({ tags })
    const result = await fetchTags()
    expect(result).toEqual({ tags })
    expect(mockOon.tags).toHaveBeenCalledOnce()
  })

  it('throws on failure', async () => {
    mockOon.tags.mockRejectedValue(new Error('oon.tags failed'))
    await expect(fetchTags()).rejects.toThrow('oon.tags failed')
  })
})

describe('fetchNotes', () => {
  it('delegates to oon.list()', async () => {
    const notes = [{ id: 'n1', title: 'Note 1' }]
    mockOon.list.mockResolvedValue({ notes })
    const result = await fetchNotes()
    expect(result).toEqual({ notes })
    expect(mockOon.list).toHaveBeenCalledOnce()
  })

  it('throws on failure', async () => {
    mockOon.list.mockRejectedValue(new Error('oon.list failed'))
    await expect(fetchNotes()).rejects.toThrow('oon.list failed')
  })
})

describe('createNote', () => {
  it('delegates to oon.createNote()', async () => {
    const note = { id: 'n1', title: 'Test' }
    mockOon.createNote.mockResolvedValue({ note })
    const result = await createNote({ title: 'Test', body: 'Content' })
    expect(result).toEqual({ note })
    expect(mockOon.createNote).toHaveBeenCalledWith({ title: 'Test', body: 'Content' })
  })

  it('throws on failure', async () => {
    mockOon.createNote.mockRejectedValue(new Error('oon.createNote failed'))
    await expect(createNote({ title: 'X' })).rejects.toThrow('oon.createNote failed')
  })
})

describe('updateNote', () => {
  it('delegates to oon.updateNote()', async () => {
    const note = { id: 'n1', title: 'Updated' }
    mockOon.updateNote.mockResolvedValue({ note })
    const result = await updateNote('n1', { title: 'Updated' })
    expect(result).toEqual({ note })
    expect(mockOon.updateNote).toHaveBeenCalledWith('n1', { title: 'Updated' })
  })

  it('throws on failure', async () => {
    mockOon.updateNote.mockRejectedValue(new Error('oon.updateNote failed'))
    await expect(updateNote('n1', { title: 'X' })).rejects.toThrow('oon.updateNote failed')
  })
})

describe('deleteNote', () => {
  it('delegates to oon.deleteNote()', async () => {
    mockOon.deleteNote.mockResolvedValue(undefined)
    await deleteNote('n1')
    expect(mockOon.deleteNote).toHaveBeenCalledWith('n1')
  })

  it('throws on failure', async () => {
    mockOon.deleteNote.mockRejectedValue(new Error('oon.deleteNote failed'))
    await expect(deleteNote('n1')).rejects.toThrow('oon.deleteNote failed')
  })
})

describe('fetchStats', () => {
  it('delegates to oon.stats()', async () => {
    const stats = { total_cards: 10, byColumn: { todo: 5 }, columns: 4, total_notes: 3 }
    mockOon.stats.mockResolvedValue({ stats })
    const result = await fetchStats()
    expect(result).toEqual({ stats })
    expect(mockOon.stats).toHaveBeenCalledOnce()
  })

  it('throws on failure', async () => {
    mockOon.stats.mockRejectedValue(new Error('oon.stats failed'))
    await expect(fetchStats()).rejects.toThrow('oon.stats failed')
  })
})

describe('syncNotes', () => {
  it('delegates to oon.sync()', async () => {
    mockOon.sync.mockResolvedValue({ added: 2, moved: 0, total: 10 })
    const result = await syncNotes()
    expect(result).toEqual({ added: 2, updated: 0, total: 10 })
    expect(mockOon.sync).toHaveBeenCalledOnce()
  })

  it('throws on failure', async () => {
    mockOon.sync.mockRejectedValue(new Error('oon.sync failed'))
    await expect(syncNotes()).rejects.toThrow('oon.sync failed')
  })
})
