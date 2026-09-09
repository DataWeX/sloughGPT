import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

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

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) =>
    selector({ addToast: vi.fn() }),
}))

vi.mock('@/components/icons/NavIcons', () => ({
  IconSearch: (props: Record<string, unknown>) => <svg data-testid="icon-search" {...props} />,
}))

import WorkspaceSearchPage from './page'

describe('WorkspaceSearchPage', () => {
  const mockSearchResults = {
    data: {
      results: {
        members: [
          { id: 'u1', type: 'member', title: 'alice', detail: 'alice@test.com' },
        ],
        training_jobs: [
          { id: 't1', type: 'training', title: 'Fine-tune LLaMA', detail: 'completed' },
        ],
        datasets: [
          { id: 'd1', type: 'dataset', title: 'training-data', detail: '1000 rows' },
        ],
        knowledge: [],
      },
      total: 3,
    },
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mockApiGet.mockResolvedValue(mockSearchResults)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders page header', async () => {
    render(<WorkspaceSearchPage />)
    await screen.findAllByText('Workspace Search')
    expect(screen.getAllByText('Workspace Search').length).toBeGreaterThanOrEqual(1)
  })

  it('shows search input', async () => {
    render(<WorkspaceSearchPage />)
    await screen.findAllByText('Workspace Search')
    expect(screen.getByPlaceholderText(/search/i)).toBeTruthy()
  })

  it('shows initial prompt', async () => {
    render(<WorkspaceSearchPage />)
    await screen.findAllByText('Workspace Search')
    expect(screen.getAllByText(/type to search/i).length).toBeGreaterThanOrEqual(1)
  })

  it('performs search on input', async () => {
    const user = userEvent.setup()
    render(<WorkspaceSearchPage />)
    await screen.findAllByText('Workspace Search')

    const searchInput = screen.getByPlaceholderText(/search/i)
    await user.type(searchInput, 'alice')

    // Wait for debounce
    await screen.findByText(/result/i)
    expect(mockApiGet).toHaveBeenCalled()
  })

  it('shows empty state when no results', async () => {
    mockApiGet.mockResolvedValue({
      data: { results: { members: [], training_jobs: [], datasets: [], knowledge: [] }, total: 0 },
    })
    const user = userEvent.setup()
    render(<WorkspaceSearchPage />)
    await screen.findAllByText('Workspace Search')

    const searchInput = screen.getByPlaceholderText(/search/i)
    await user.type(searchInput, 'zzznonexistent')

    await screen.findByText(/no results/i)
  })
})
