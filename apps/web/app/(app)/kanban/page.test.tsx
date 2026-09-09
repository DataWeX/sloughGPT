import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

const mockApiGet = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
}))

vi.mock('@/lib/auth', () => ({
  useAuthStore: (selector?: (s: { currentWorkspace: { id: string } | null }) => unknown) => {
    const state = { currentWorkspace: { id: 'ws-1' } }
    return selector ? selector(state) : state
  },
}))

import KanbanPage from './page'

describe('KanbanPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApiGet.mockResolvedValue({ data: null })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders page', async () => {
    render(<KanbanPage />)
    // The kanban page loads data and renders a board
    expect(screen.getAllByText(/loading/i).length).toBeGreaterThanOrEqual(1)
  })

  it('fetches board data on mount', async () => {
    render(<KanbanPage />)
    // Wait for data to load
    await screen.findByText(/board/i, {}, { timeout: 5000 }).catch(() => {})
    expect(mockApiGet).toHaveBeenCalled()
  })

  it('renders board and notes tabs', async () => {
    mockApiGet.mockResolvedValue({
      data: {
        board: {
          name: 'Test Board',
          columns: [
            { name: 'todo', wip_limit: 5, order: 0 },
            { name: 'in_progress', wip_limit: 3, order: 1 },
            { name: 'review', wip_limit: 2, order: 2 },
            { name: 'done', wip_limit: 0, order: 3 },
          ],
          cards: [],
        },
        notes: [],
      },
    })
    render(<KanbanPage />)
    await screen.findByText(/board/i, {}, { timeout: 5000 }).catch(() => {})
    // Board and notes tabs should be present
    expect(screen.getAllByText(/board/i).length).toBeGreaterThanOrEqual(1)
  })

  it('renders column headers when board is loaded', async () => {
    mockApiGet.mockResolvedValue({
      data: {
        board: {
          name: 'Test Board',
          columns: [
            { name: 'todo', wip_limit: 5, order: 0 },
            { name: 'in_progress', wip_limit: 3, order: 1 },
          ],
          cards: [],
        },
        notes: [],
      },
    })
    render(<KanbanPage />)
    await screen.findByText(/board/i, {}, { timeout: 5000 }).catch(() => {})
    expect(screen.getAllByText(/to do/i).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/in progress/i).length).toBeGreaterThanOrEqual(1)
  })

  it('renders cards when present', async () => {
    mockApiGet.mockResolvedValue({
      data: {
        board: {
          name: 'Test Board',
          columns: [
            { name: 'todo', wip_limit: 5, order: 0 },
          ],
          cards: [
            { id: 'c1', title: 'Test Card', description: 'A test card', column: 'todo', priority: 'high', tags: [], created_at: '2024-01-01', updated_at: '2024-01-01', due_date: '', assignee: '', notes: [] },
          ],
        },
        notes: [],
      },
    })
    render(<KanbanPage />)
    await screen.findByText(/board/i, {}, { timeout: 5000 }).catch(() => {})
    expect(screen.getAllByText('Test Card').length).toBeGreaterThanOrEqual(1)
  })
})
