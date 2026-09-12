import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'

const { mockApiGet, mockApiPost, mockApiPut, mockApiDelete } = vi.hoisted(() => ({
  mockApiGet: vi.fn(),
  mockApiPost: vi.fn(),
  mockApiPut: vi.fn(),
  mockApiDelete: vi.fn(),
}))

const authState = { token: 'test-token', currentWorkspace: { id: 'ws-1' } }

vi.mock('@/lib/auth', () => ({
  useAuthStore: Object.assign(
    (selector?: (s: typeof authState) => unknown) => selector ? selector(authState) : authState,
    { getState: () => authState },
  ),
}))

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
  apiPost: (...args: unknown[]) => mockApiPost(...args),
  apiPut: (...args: unknown[]) => mockApiPut(...args),
  apiDelete: (...args: unknown[]) => mockApiDelete(...args),
}))

import KanbanPage from './page'

describe('KanbanPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApiGet.mockImplementation((url: string) => {
      if (url.includes('/board')) return Promise.resolve({ board: null })
      if (url.includes('/notes')) return Promise.resolve({ notes: [] })
      return Promise.resolve({})
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
      expect(mockApiGet).toHaveBeenCalledWith('/api/planner/board', undefined, expect.any(Object))
      expect(mockApiGet).toHaveBeenCalledWith('/api/planner/notes', undefined, expect.any(Object))
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
    mockApiGet.mockImplementation((url: string) => {
      if (url.includes('/board')) return Promise.resolve({
        board: {
          name: 'Test Board',
          columns: [
            { name: 'todo', wip_limit: 5, order: 0 },
            { name: 'in_progress', wip_limit: 3, order: 1 },
          ],
          cards: [],
        },
      })
      return Promise.resolve({ notes: [] })
    })
    render(<KanbanPage />)
    await waitFor(() => {
      expect(screen.getByText('To Do')).toBeTruthy()
      expect(screen.getAllByText('In Progress').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('renders cards when present', async () => {
    mockApiGet.mockImplementation((url: string) => {
      if (url.includes('/board')) return Promise.resolve({
        board: {
          name: 'Test Board',
          columns: [{ name: 'todo', wip_limit: 5, order: 0 }],
          cards: [
            { id: 'c1', title: 'Test Card', description: 'A test card', column: 'todo', priority: 'high', tags: [], created_at: '2024-01-01', updated_at: '2024-01-01', due_date: '', assignee: '', notes: [] },
          ],
        },
      })
      return Promise.resolve({ notes: [] })
    })
    render(<KanbanPage />)
    await waitFor(() => {
      expect(screen.getByText('Test Card')).toBeTruthy()
    })
  })

  it('shows loading state initially', () => {
    mockApiGet.mockReturnValue(new Promise(() => {}))
    render(<KanbanPage />)
    expect(screen.queryByText('Planner')).toBeNull()
  })

  it('shows error state on fetch failure', async () => {
    mockApiGet.mockRejectedValue(new Error('Network error'))
    render(<KanbanPage />)
    await waitFor(() => {
      expect(screen.getByText(/network error/i)).toBeTruthy()
    })
  })
})
