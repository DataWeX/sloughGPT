import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'

const mockFetch = vi.fn()

vi.mock('@/lib/auth', () => ({
  useAuthStore: (selector?: (s: { currentWorkspace: { id: string } | null }) => unknown) => {
    const state = { currentWorkspace: { id: 'ws-1' } }
    return selector ? selector(state) : state
  },
}))

vi.stubGlobal('fetch', mockFetch)

import KanbanPage from './page'

describe('KanbanPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/board')) return Promise.resolve({ ok: true, json: () => Promise.resolve({ board: null }) })
      if (url.includes('/notes')) return Promise.resolve({ ok: true, json: () => Promise.resolve({ notes: [] }) })
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) })
    })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders page with Planner title after loading', async () => {
    render(<KanbanPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Planner').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('fetches board and notes on mount', async () => {
    render(<KanbanPage />)
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith('/api/planner/board', expect.any(Object))
      expect(mockFetch).toHaveBeenCalledWith('/api/planner/notes', expect.any(Object))
    })
  })

  it('shows board and notes tabs after loading', async () => {
    render(<KanbanPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/board/i).length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText(/notes/i).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('renders column headers when board is loaded', async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/board')) return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          board: {
            name: 'Test Board',
            columns: [
              { name: 'todo', wip_limit: 5, order: 0 },
              { name: 'in_progress', wip_limit: 3, order: 1 },
            ],
            cards: [],
          },
        }),
      })
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ notes: [] }) })
    })
    render(<KanbanPage />)
    await waitFor(() => {
      expect(screen.getByText('To Do')).toBeTruthy()
      expect(screen.getAllByText('In Progress').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('renders cards when present', async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/board')) return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          board: {
            name: 'Test Board',
            columns: [{ name: 'todo', wip_limit: 5, order: 0 }],
            cards: [
              { id: 'c1', title: 'Test Card', description: 'A test card', column: 'todo', priority: 'high', tags: [], created_at: '2024-01-01', updated_at: '2024-01-01', due_date: '', assignee: '', notes: [] },
            ],
          },
        }),
      })
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ notes: [] }) })
    })
    render(<KanbanPage />)
    await waitFor(() => {
      expect(screen.getByText('Test Card')).toBeTruthy()
    })
  })

  it('shows loading state initially', () => {
    mockFetch.mockReturnValue(new Promise(() => {}))
    render(<KanbanPage />)
    expect(screen.queryByText('Planner')).toBeNull()
  })

  it('shows error state on fetch failure', async () => {
    mockFetch.mockRejectedValue(new Error('Network error'))
    render(<KanbanPage />)
    await waitFor(() => {
      // Error message is passed to PageContainer's error prop which renders StatusBanner
      expect(screen.getByText(/network error/i)).toBeTruthy()
    })
  })
})
