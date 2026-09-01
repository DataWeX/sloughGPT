import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
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

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

function jsonResponse(data: unknown, ok = true, status = 200) {
  return {
    ok,
    status,
    json: () => Promise.resolve(data),
  }
}

beforeEach(() => {
  mockFetch.mockReset()
})

// ── fetchBoard ─────────────────────────────────────────────────────

describe('fetchBoard', () => {
  it('fetches board from /api/planner/board', async () => {
    const board = { columns: [], cards: [] }
    mockFetch.mockResolvedValue(jsonResponse({ board }))
    const result = await fetchBoard()
    expect(result).toEqual({ board })
    expect(mockFetch).toHaveBeenCalledWith('/api/planner/board')
  })

  it('throws on non-ok response', async () => {
    mockFetch.mockResolvedValue(jsonResponse(null, false, 500))
    await expect(fetchBoard()).rejects.toThrow('Failed to fetch board')
  })
})

// ── moveCard ───────────────────────────────────────────────────────

describe('moveCard', () => {
  it('POSTs to /api/planner/board/move', async () => {
    mockFetch.mockResolvedValue(jsonResponse({}))
    await moveCard({ card_id: 'c1', column: 'done' })
    expect(mockFetch).toHaveBeenCalledWith('/api/planner/board/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ card_id: 'c1', column: 'done' }),
    })
  })

  it('throws on failure', async () => {
    mockFetch.mockResolvedValue(jsonResponse(null, false, 400))
    await expect(moveCard({ card_id: 'c1', column: 'done' })).rejects.toThrow('Failed to move card')
  })
})

// ── createCard ─────────────────────────────────────────────────────

describe('createCard', () => {
  it('POSTs card data', async () => {
    const card = { id: 'c1', title: 'Test' }
    mockFetch.mockResolvedValue(jsonResponse({ card }))
    const result = await createCard({ title: 'Test', column: 'todo', priority: 'high' })
    expect(result).toEqual({ card })
    expect(mockFetch).toHaveBeenCalledWith('/api/planner/board/cards', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: 'Test', column: 'todo', priority: 'high' }),
    })
  })

  it('throws on failure', async () => {
    mockFetch.mockResolvedValue(jsonResponse(null, false, 400))
    await expect(createCard({ title: 'Test' })).rejects.toThrow('Failed to create card')
  })
})

// ── updateCard ─────────────────────────────────────────────────────

describe('updateCard', () => {
  it('PUTs updated fields', async () => {
    const card = { id: 'c1', title: 'Updated' }
    mockFetch.mockResolvedValue(jsonResponse({ card }))
    const result = await updateCard('c1', { title: 'Updated' })
    expect(result).toEqual({ card })
    expect(mockFetch).toHaveBeenCalledWith('/api/planner/board/cards/c1', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: 'Updated' }),
    })
  })

  it('throws on failure', async () => {
    mockFetch.mockResolvedValue(jsonResponse(null, false, 404))
    await expect(updateCard('c1', { title: 'X' })).rejects.toThrow('Failed to update card')
  })
})

// ── deleteCard ─────────────────────────────────────────────────────

describe('deleteCard', () => {
  it('DELETEs card by id', async () => {
    mockFetch.mockResolvedValue(jsonResponse({}))
    await deleteCard('c1')
    expect(mockFetch).toHaveBeenCalledWith('/api/planner/board/cards/c1', { method: 'DELETE' })
  })

  it('throws on failure', async () => {
    mockFetch.mockResolvedValue(jsonResponse(null, false, 404))
    await expect(deleteCard('c1')).rejects.toThrow('Failed to delete card')
  })
})

// ── fetchTags ──────────────────────────────────────────────────────

describe('fetchTags', () => {
  it('fetches tags', async () => {
    const tags = [{ tag: 'bug', count: 5 }]
    mockFetch.mockResolvedValue(jsonResponse({ tags }))
    const result = await fetchTags()
    expect(result).toEqual({ tags })
    expect(mockFetch).toHaveBeenCalledWith('/api/planner/tags')
  })

  it('throws on failure', async () => {
    mockFetch.mockResolvedValue(jsonResponse(null, false, 500))
    await expect(fetchTags()).rejects.toThrow('Failed to fetch tags')
  })
})

// ── fetchNotes ─────────────────────────────────────────────────────

describe('fetchNotes', () => {
  it('fetches notes', async () => {
    const notes = [{ id: 'n1', title: 'Note 1' }]
    mockFetch.mockResolvedValue(jsonResponse({ notes }))
    const result = await fetchNotes()
    expect(result).toEqual({ notes })
    expect(mockFetch).toHaveBeenCalledWith('/api/planner/notes')
  })

  it('throws on failure', async () => {
    mockFetch.mockResolvedValue(jsonResponse(null, false, 500))
    await expect(fetchNotes()).rejects.toThrow('Failed to fetch notes')
  })
})

// ── createNote ─────────────────────────────────────────────────────

describe('createNote', () => {
  it('POSTs note data', async () => {
    const note = { id: 'n1', title: 'Test' }
    mockFetch.mockResolvedValue(jsonResponse({ note }))
    const result = await createNote({ title: 'Test', body: 'Content' })
    expect(result).toEqual({ note })
  })

  it('throws on failure', async () => {
    mockFetch.mockResolvedValue(jsonResponse(null, false, 400))
    await expect(createNote({ title: 'X' })).rejects.toThrow('Failed to create note')
  })
})

// ── updateNote ─────────────────────────────────────────────────────

describe('updateNote', () => {
  it('PUTs updated fields', async () => {
    const note = { id: 'n1', title: 'Updated' }
    mockFetch.mockResolvedValue(jsonResponse({ note }))
    const result = await updateNote('n1', { title: 'Updated' })
    expect(result).toEqual({ note })
    expect(mockFetch).toHaveBeenCalledWith('/api/planner/notes/n1', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: 'Updated' }),
    })
  })

  it('throws on failure', async () => {
    mockFetch.mockResolvedValue(jsonResponse(null, false, 404))
    await expect(updateNote('n1', { title: 'X' })).rejects.toThrow('Failed to update note')
  })
})

// ── deleteNote ─────────────────────────────────────────────────────

describe('deleteNote', () => {
  it('DELETEs note by id', async () => {
    mockFetch.mockResolvedValue(jsonResponse({}))
    await deleteNote('n1')
    expect(mockFetch).toHaveBeenCalledWith('/api/planner/notes/n1', { method: 'DELETE' })
  })

  it('throws on failure', async () => {
    mockFetch.mockResolvedValue(jsonResponse(null, false, 404))
    await expect(deleteNote('n1')).rejects.toThrow('Failed to delete note')
  })
})

// ── fetchStats ─────────────────────────────────────────────────────

describe('fetchStats', () => {
  it('fetches stats', async () => {
    const stats = { total: 10, by_column: { todo: 5 } }
    mockFetch.mockResolvedValue(jsonResponse({ stats }))
    const result = await fetchStats()
    expect(result).toEqual({ stats })
  })

  it('throws on failure', async () => {
    mockFetch.mockResolvedValue(jsonResponse(null, false, 500))
    await expect(fetchStats()).rejects.toThrow('Failed to fetch stats')
  })
})

// ── syncNotes ──────────────────────────────────────────────────────

describe('syncNotes', () => {
  it('POSTs to /api/planner/sync', async () => {
    const result_data = { added: 2, updated: 1, total: 10 }
    mockFetch.mockResolvedValue(jsonResponse(result_data))
    const result = await syncNotes()
    expect(result).toEqual(result_data)
    expect(mockFetch).toHaveBeenCalledWith('/api/planner/sync', { method: 'POST' })
  })

  it('throws on failure', async () => {
    mockFetch.mockResolvedValue(jsonResponse(null, false, 500))
    await expect(syncNotes()).rejects.toThrow('Failed to sync notes')
  })
})
