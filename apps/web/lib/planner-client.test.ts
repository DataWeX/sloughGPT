import { describe, it, expect, vi, beforeEach } from 'vitest'

const mockApi = vi.hoisted(() => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
  apiPut: vi.fn(),
  apiDelete: vi.fn(),
}))

vi.mock('./http-client', () => mockApi)

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
  it('calls GET /api/planner/board', async () => {
    const board = { columns: [], cards: [] }
    mockApi.apiGet.mockResolvedValue({ board })
    const result = await fetchBoard()
    expect(result).toEqual({ board })
    expect(mockApi.apiGet).toHaveBeenCalledWith('/api/planner/board')
  })

  it('throws on failure', async () => {
    mockApi.apiGet.mockRejectedValue(new Error('fetch failed'))
    await expect(fetchBoard()).rejects.toThrow('fetch failed')
  })
})

describe('moveCard', () => {
  it('calls POST /api/planner/board/move', async () => {
    mockApi.apiPost.mockResolvedValue(undefined)
    await moveCard({ card_id: 'c1', column: 'done' })
    expect(mockApi.apiPost).toHaveBeenCalledWith('/api/planner/board/move', {
      card_id: 'c1',
      column: 'done',
    })
  })

  it('throws on failure', async () => {
    mockApi.apiPost.mockRejectedValue(new Error('move failed'))
    await expect(moveCard({ card_id: 'c1', column: 'done' })).rejects.toThrow('move failed')
  })
})

describe('createCard', () => {
  it('calls POST /api/planner/board/cards', async () => {
    const card = { id: 'c1', title: 'Test' }
    mockApi.apiPost.mockResolvedValue({ card })
    const result = await createCard({ title: 'Test', column: 'todo', priority: 'high' })
    expect(result).toEqual({ card })
    expect(mockApi.apiPost).toHaveBeenCalledWith('/api/planner/board/cards', {
      title: 'Test',
      column: 'todo',
      priority: 'high',
    })
  })

  it('throws on failure', async () => {
    mockApi.apiPost.mockRejectedValue(new Error('create failed'))
    await expect(createCard({ title: 'Test' })).rejects.toThrow('create failed')
  })
})

describe('updateCard', () => {
  it('calls PUT /api/planner/board/cards/:id', async () => {
    const card = { id: 'c1', title: 'Updated' }
    mockApi.apiPut.mockResolvedValue({ card })
    const result = await updateCard('c1', { title: 'Updated' })
    expect(result).toEqual({ card })
    expect(mockApi.apiPut).toHaveBeenCalledWith('/api/planner/board/cards/c1', { title: 'Updated' })
  })

  it('throws on failure', async () => {
    mockApi.apiPut.mockRejectedValue(new Error('update failed'))
    await expect(updateCard('c1', { title: 'X' })).rejects.toThrow('update failed')
  })
})

describe('deleteCard', () => {
  it('calls DELETE /api/planner/board/cards/:id', async () => {
    mockApi.apiDelete.mockResolvedValue(undefined)
    await deleteCard('c1')
    expect(mockApi.apiDelete).toHaveBeenCalledWith('/api/planner/board/cards/c1')
  })

  it('throws on failure', async () => {
    mockApi.apiDelete.mockRejectedValue(new Error('delete failed'))
    await expect(deleteCard('c1')).rejects.toThrow('delete failed')
  })
})

describe('fetchTags', () => {
  it('calls GET /api/planner/tags', async () => {
    const tags = [{ name: 'bug', count: 5 }]
    mockApi.apiGet.mockResolvedValue({ tags })
    const result = await fetchTags()
    expect(result).toEqual({ tags })
    expect(mockApi.apiGet).toHaveBeenCalledWith('/api/planner/tags')
  })

  it('throws on failure', async () => {
    mockApi.apiGet.mockRejectedValue(new Error('tags failed'))
    await expect(fetchTags()).rejects.toThrow('tags failed')
  })
})

describe('fetchNotes', () => {
  it('calls GET /api/planner/notes', async () => {
    const notes = [{ id: 'n1', title: 'Note 1' }]
    mockApi.apiGet.mockResolvedValue({ notes })
    const result = await fetchNotes()
    expect(result).toEqual({ notes })
    expect(mockApi.apiGet).toHaveBeenCalledWith('/api/planner/notes')
  })

  it('throws on failure', async () => {
    mockApi.apiGet.mockRejectedValue(new Error('list failed'))
    await expect(fetchNotes()).rejects.toThrow('list failed')
  })
})

describe('createNote', () => {
  it('calls POST /api/planner/notes', async () => {
    const note = { id: 'n1', title: 'Test' }
    mockApi.apiPost.mockResolvedValue({ note })
    const result = await createNote({ title: 'Test', body: 'Content' })
    expect(result).toEqual({ note })
    expect(mockApi.apiPost).toHaveBeenCalledWith('/api/planner/notes', {
      title: 'Test',
      body: 'Content',
    })
  })

  it('throws on failure', async () => {
    mockApi.apiPost.mockRejectedValue(new Error('create failed'))
    await expect(createNote({ title: 'X' })).rejects.toThrow('create failed')
  })
})

describe('updateNote', () => {
  it('calls PUT /api/planner/notes/:id', async () => {
    const note = { id: 'n1', title: 'Updated' }
    mockApi.apiPut.mockResolvedValue({ note })
    const result = await updateNote('n1', { title: 'Updated' })
    expect(result).toEqual({ note })
    expect(mockApi.apiPut).toHaveBeenCalledWith('/api/planner/notes/n1', { title: 'Updated' })
  })

  it('throws on failure', async () => {
    mockApi.apiPut.mockRejectedValue(new Error('update failed'))
    await expect(updateNote('n1', { title: 'X' })).rejects.toThrow('update failed')
  })
})

describe('deleteNote', () => {
  it('calls DELETE /api/planner/notes/:id', async () => {
    mockApi.apiDelete.mockResolvedValue(undefined)
    await deleteNote('n1')
    expect(mockApi.apiDelete).toHaveBeenCalledWith('/api/planner/notes/n1')
  })

  it('throws on failure', async () => {
    mockApi.apiDelete.mockRejectedValue(new Error('delete failed'))
    await expect(deleteNote('n1')).rejects.toThrow('delete failed')
  })
})

describe('fetchStats', () => {
  it('calls GET /api/planner/stats', async () => {
    const stats = { total_cards: 10, byColumn: { todo: 5 }, columns: 4, total_notes: 3 }
    mockApi.apiGet.mockResolvedValue({ stats })
    const result = await fetchStats()
    expect(result).toEqual({ stats })
    expect(mockApi.apiGet).toHaveBeenCalledWith('/api/planner/stats')
  })

  it('throws on failure', async () => {
    mockApi.apiGet.mockRejectedValue(new Error('stats failed'))
    await expect(fetchStats()).rejects.toThrow('stats failed')
  })
})

describe('syncNotes', () => {
  it('calls POST /api/planner/sync', async () => {
    mockApi.apiPost.mockResolvedValue({ added: 2, updated: 0, total: 10 })
    const result = await syncNotes()
    expect(result).toEqual({ added: 2, updated: 0, total: 10 })
    expect(mockApi.apiPost).toHaveBeenCalledWith('/api/planner/sync', {})
  })

  it('throws on failure', async () => {
    mockApi.apiPost.mockRejectedValue(new Error('sync failed'))
    await expect(syncNotes()).rejects.toThrow('sync failed')
  })
})
